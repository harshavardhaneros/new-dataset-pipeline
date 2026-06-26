"""Gemma-3 vision captioner (eros_caption_video architecture)."""

from __future__ import annotations

import gc
import json
import logging
import queue
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from common.gpu_info import log_service_gpus, resolve_gpu_ids
from common.paths import models_root
from common.screen_position import frame_position_label, known_actor_names

logger = logging.getLogger(__name__)

CAPTION_SYSTEM_PROMPT = (
    "Output MUST be a valid JSON object only. No markdown or extra text.\n\n"
    "Rules:\n"
    "- Be precise and avoid repetition.\n"
    "- No hallucination. Only visible or strongly implied details.\n"
    "- Avoid generic phrases (e.g., \"a group of people\").\n"
    "- For humans, describe from THEIR perspective (not the viewer's).\n"
    "- Prioritise culturally significant visual elements when present.\n"
    "- Include actor names and positions while explaining object actions.\n\n"
    "Indian Cultural Details (include ONLY if visible):\n"
    "- attire: women: saree (silk/cotton), half-saree, salwar, blouse color/design,\n"
    "  embroidery (Zardozi, Chikankari). men: veshti/dhoti, kurta, shirt, traditional wear\n"
    "- accessories: jhumka, nose ring, choker, chain, bangles, anklets, kundan, bindi/sindoor\n"
    "- regional_identity: Tamil, Punjabi, Bengali, etc. (ONLY if clearly inferable)\n"
    "- cultural_context: temple, wedding, ritual, festival, street market, rural/urban India\n"
    "- architecture_landmarks: gopuram, heritage buildings (if visible)\n"
    "- food_elements: traditional dishes (if present)\n\n"
    "Text: Include ONLY clearly visible text. If none → return [].\n\n"
    "JSON structure:\n"
    "{ \"short_description\": \"\",\n"
    "  \"objects\": [{ \"description\":\"\",\"location\":\"\",\"relative_size\":\"\","
    "\"shape_color\":\"\",\"texture\":\"\",\"appearance_details\":\"\","
    "\"relationship\":\"\",\"orientation\":\"\",\"Indian_cultural_details\":{},"
    "\"pose\":\"\",\"expression\":\"\",\"clothing\":\"\","
    "\"actor_name_and_action\":\"\",\"gender\":\"\",\"skin_tone_texture\":\"\" }],\n"
    "  \"background_setting\":\"\",\n"
    "  \"lighting\":{\"conditions\":\"\",\"direction\":\"\",\"shadows\":\"\"},\n"
    "  \"aesthetics\":{\"composition\":\"\",\"color_scheme\":\"\",\"mood_atmosphere\":\"\"},\n"
    "  \"photographic_characteristics\":{\"depth_of_field\":\"\",\"focus\":\"\","
    "\"camera_angle\":\"\",\"camera_movement\":\"\",\"lens_focal_length\":\"\"},\n"
    "  \"style_medium\":\"\",\n"
    "  \"text_render\":[{\"text\":\"\",\"location\":\"\",\"size\":\"\","
    "\"color\":\"\",\"font\":\"\",\"appearance_details\":\"\"}] }"
)


def gemma_caption_model_path(config: Dict[str, Any]) -> Path:
    cc = config.get("models", {}).get("gemma_caption", {})
    pcfg = config.get("pipeline", {}).get("captioner", {})
    path = cc.get("model_path") or pcfg.get("model_path")
    if path:
        return Path(path)
    return models_root(config) / "gemma-3-4b-it"


def to_single_line_json(text: str) -> str:
    cleaned = text.strip()
    if "{" in cleaned and "}" in cleaned:
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        try:
            obj = json.loads(cleaned[start_idx : end_idx + 1])
            return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        obj = json.loads(cleaned)
        return json.dumps(obj, ensure_ascii=False)
    except json.JSONDecodeError:
        return re.sub(r"\s+", " ", cleaned)


def parse_caption_json(text: str) -> Dict[str, Any]:
    line = to_single_line_json(text)
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"short_description": line, "_parse_error": True}


def build_caption_user_text(rec: Dict[str, Any]) -> str:
    clip_actors = rec.get("clip_actors") or known_actor_names(rec.get("actors") or [])
    return (
        f"Actors present: {clip_actors}\n"
        f"Frame 1: {rec.get('actors_f1', '[]')} | {rec.get('pos_f1', 'unknown')}\n"
        f"Frame 2: {rec.get('actors_f2', '[]')} | {rec.get('pos_f2', 'unknown')}\n"
        f"Frame 3: {rec.get('actors_f3', '[]')} | {rec.get('pos_f3', 'unknown')}\n"
        "You are a Visual Art Director generating structured, "
        "high-quality captions for the video frames."
    )


def pick_caption_frame(rec: Dict[str, Any], frames_dir: Path) -> Optional[Path]:
    clip_id = rec["clip_id"]
    for idx in (2, 1, 3):
        p = frames_dir / f"{clip_id}.{idx}.jpg"
        if p.exists():
            return p
    legacy = frames_dir / f"{clip_id}.jpg"
    return legacy if legacy.exists() else None


class GemmaCaptionService:
    """Gemma-3-4B-IT captioner with eros-style batching."""

    _shared: Optional["GemmaCaptionService"] = None

    def __init__(self, config: Dict[str, Any]):
        cc = config.get("models", {}).get("gemma_caption", {})
        pcfg = config.get("pipeline", {}).get("captioner", {})
        self.model_path = str(gemma_caption_model_path(config))
        self.gpu_ids = resolve_gpu_ids(
            [int(g) for g in cc.get("gpu_ids", pcfg.get("gpu_ids", [0]))]
        )
        self.gpu_id = self.gpu_ids[0] if self.gpu_ids else 0
        self.device = f"cuda:{self.gpu_id}"
        self.batch_size = int(cc.get("batch_size", pcfg.get("batch_size", 8)))
        self.max_new_tokens = int(cc.get("max_tokens", pcfg.get("max_tokens", 1000)))
        self._model = None
        self._processor = None

    @classmethod
    def acquire(cls, config: Dict[str, Any]) -> "GemmaCaptionService":
        if cls._shared is None:
            cls._shared = cls(config)
        return cls._shared

    @classmethod
    def release(cls) -> None:
        if cls._shared:
            cls._shared.cleanup()
        cls._shared = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not Path(self.model_path).joinpath("config.json").exists():
            raise FileNotFoundError(
                f"Gemma caption model not found: {self.model_path}\n"
                "Download: hf download google/gemma-3-4b-it --local-dir "
                f"{self.model_path}"
            )
        log_service_gpus(
            "s8",
            "Gemma-3-4B-IT caption (eros-style JSON)",
            self.model_path,
            self.gpu_ids,
        )
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()

    def _build_messages(self, rec: Dict[str, Any], frame_path: Path) -> tuple[list | None, Image.Image | None]:
        if not frame_path.exists():
            return None, None
        try:
            img = Image.open(frame_path).convert("RGB")
        except Exception as exc:
            logger.warning("Cannot open %s: %s", frame_path, exc)
            return None, None
        user_text = build_caption_user_text(rec)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": CAPTION_SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text},
            ]},
        ]
        return messages, img

    def _infer_single(self, messages: list) -> str:
        import torch

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            gen_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return self._processor.decode(
            gen_ids[0][input_len:], skip_special_tokens=True
        ).strip()

    def caption_records(
        self,
        items: List[tuple[Dict[str, Any], Path]],
    ) -> List[str]:
        """Caption a list of (metadata_record, frame_path) pairs."""
        import torch

        self.load()
        if not items:
            return []

        results = [""] * len(items)
        q: queue.Queue = queue.Queue(maxsize=16)
        SENTINEL = object()

        def _producer():
            for i, (rec, fp) in enumerate(items):
                msgs, img = self._build_messages(rec, fp)
                q.put((i, msgs, img))
            q.put(SENTINEL)

        producer = threading.Thread(target=_producer, daemon=True)
        producer.start()

        def _flush(batch: list) -> None:
            if not batch:
                return
            try:
                msgs_list = [m for _, m, _ in batch]
                inputs = self._processor.apply_chat_template(
                    msgs_list,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device, dtype=torch.bfloat16)
                input_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    gen_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )
                caps = self._processor.batch_decode(
                    gen_ids[:, input_len:], skip_special_tokens=True
                )
                for (i, _, _), cap in zip(batch, caps):
                    results[i] = cap.strip()
            except Exception as exc:
                logger.warning("Batch failed (%s) — single-item fallback", exc)
                for i, msgs, _img in batch:
                    try:
                        results[i] = self._infer_single(msgs)
                    except Exception as e2:
                        logger.warning("Single caption failed: %s", e2)
            finally:
                for _, _, img in batch:
                    try:
                        img.close()
                    except Exception:
                        pass
                torch.cuda.empty_cache()

        batch: list = []
        while True:
            item = q.get()
            if item is SENTINEL:
                if batch:
                    _flush(batch)
                break
            i, msgs, img = item
            if msgs is not None and img is not None:
                batch.append((i, msgs, img))
                if len(batch) >= self.batch_size:
                    _flush(batch)
                    batch = []

        producer.join()
        gc.collect()
        torch.cuda.empty_cache()
        return results

    def cleanup(self) -> None:
        import torch

        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def enrich_record_actor_fields(
    rec: Dict[str, Any],
    frame_assignments: Dict[int, List[Dict[str, Any]]],
    frame_paths: Dict[int, Path],
) -> None:
    """Populate eros-style actor fields on metadata record."""
    all_names: List[str] = []
    for idx in (1, 2, 3):
        actors = frame_assignments.get(idx, [])
        names = known_actor_names(actors)
        rec[f"actors_f{idx}"] = names
        hw = None
        if actors and actors[0].get("_img_hw"):
            hw = actors[0]["_img_hw"]
        elif frame_paths.get(idx) and frame_paths[idx].exists():
            import cv2
            img = cv2.imread(str(frame_paths[idx]))
            if img is not None:
                hw = (img.shape[0], img.shape[1])
        rec[f"pos_f{idx}"] = frame_position_label(actors, hw)
        for n in names:
            if n not in all_names:
                all_names.append(n)
    rec["clip_actors"] = all_names
    rec["frame1"] = str(frame_paths.get(1, ""))
    rec["frame2"] = str(frame_paths.get(2, ""))
    rec["frame3"] = str(frame_paths.get(3, ""))
    if frame_assignments.get(2):
        rec["actors"] = frame_assignments[2]
    elif frame_assignments.get(1):
        rec["actors"] = frame_assignments[1]
    else:
        rec["actors"] = []
