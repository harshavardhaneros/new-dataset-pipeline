"""Lightweight Qwen2.5-VL bucket classifier (7B default)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from common.buckets import (
    load_classify_prompt,
    normalize_bucket,
    parse_classify_result,
)
from common.paths import models_root, qwen_classify_model_path
from common.vlm_service import parse_vlm_json

logger = logging.getLogger(__name__)

# s5 classification prompt (15 named buckets) — prompts/updated_prompt.txt.
CLASSIFY_JSON_PROMPT = load_classify_prompt()


def _normalize_bucket(raw: str, valid: List[str]) -> str:
    return normalize_bucket(raw, valid)


class QwenClassifyWorker:
    def __init__(self, config: Dict[str, Any], device: str = "cuda:0"):
        self._config = config
        s5 = config.get("pipeline", {}).get("s5", {})
        mp = config["pipeline"].get("master_pipeline", {})
        path = s5.get("classify_model_path") or mp.get("classify_model_path")
        self.model_path = str(path or qwen_classify_model_path(config))
        self.device = device
        self.max_new_tokens = int(s5.get("max_tokens", mp.get("max_new_tokens", 512)))
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not Path(self.model_path).joinpath("config.json").exists():
            raise FileNotFoundError(f"Classify model not found: {self.model_path}")

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from common.attn_backend import resolve_attn_implementation

        attn = resolve_attn_implementation()

        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation=attn,
        ).eval()

    def generate(self, image: Image.Image, prompt: str) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        self.load()
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
            min_pixels=256 * 28 * 28,
            max_pixels=640 * 28 * 28,
        ).to(self._model.device)
        with torch.no_grad():
            gen_ids = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = gen_ids[:, inputs.input_ids.shape[1] :]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def classify_clip(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from common.frame_sampler import sample_keyframes
        from common.video_time import clip_local_range

        rec = payload["record"]
        config = payload["config"]
        valid = payload["valid_buckets"]
        fractions = payload.get("fractions", [0.5])

        clip_id = rec["clip_id"]
        if not rec.get("keep", True):
            return {"clip_id": clip_id, "skipped": True}

        video_path = payload.get("video_path")
        if not video_path:
            return {
                "clip_id": clip_id,
                "bucket": valid[0],
                "bucket_confidence": 0.0,
                "reject": True,
                "reject_reason": "no_video",
            }

        start, end = clip_local_range(rec, config)
        frames = sample_keyframes(
            video_path, start, end, fractions=fractions,
            crop_box=rec.get("crop_box", ""),
        )
        if not frames:
            return {
                "clip_id": clip_id,
                "bucket": valid[0],
                "bucket_confidence": 0.0,
                "reject": True,
                "reject_reason": "no_keyframes",
            }

        parsed_frames: List[Dict[str, Any]] = []
        reject_votes = 0
        for bgr in frames:
            img = Image.fromarray(bgr[:, :, ::-1])
            raw = self.generate(img, CLASSIFY_JSON_PROMPT)
            data = parse_vlm_json(raw, {})
            row = parse_classify_result(data, valid)
            parsed_frames.append(row)
            if row["reject"]:
                reject_votes += 1

        if reject_votes >= max(1, len(frames) // 2 + 1):
            first = parsed_frames[0] if parsed_frames else {}
            return {
                "clip_id": clip_id,
                "bucket": first.get("bucket", valid[0]),
                "bucket_confidence": 0.0,
                "reject": True,
                "reject_reason": first.get("reject_reason") or "majority_reject",
                "attributes": first.get("attributes", {}),
            }

        from collections import Counter

        bucket_votes = [r["bucket"] for r in parsed_frames]
        counter = Counter(bucket_votes)
        winner, count = counter.most_common(1)[0]
        vote_conf = count / len(bucket_votes)
        confidences = [r["bucket_confidence"] for r in parsed_frames]
        attrs = next(
            (r["attributes"] for r in parsed_frames if r["bucket"] == winner), {}
        )
        return {
            "clip_id": clip_id,
            "bucket": winner,
            "bucket_confidence": round(sum(confidences) / len(confidences) * vote_conf, 4),
            "reject": False,
            "reject_reason": None,
            "attributes": attrs,
        }

    def cleanup(self) -> None:
        import gc
        import torch

        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
