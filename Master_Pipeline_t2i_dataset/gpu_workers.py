#!/usr/bin/env python3
"""
Dedicated GPU worker processes for the streaming pipeline.

Each worker runs as a multiprocessing.Process with CUDA_VISIBLE_DEVICES pinned
to a single GPU. Models are loaded once in setup() and reused across all batches.

Workers:
    - ActorTagWorker (GPU 4): YOLO face detection + InsightFace recognition
    - CLIPScoreWorker (GPU 5): CLIP image-text alignment scoring
    - GDINOScoreWorker (GPU 6): GroundingDINO noun grounding (ICR scoring)
    - AODScoreWorker (CPU): spaCy adjective-per-noun scoring (no GPU needed)
"""

import json
import logging
import multiprocessing as mp
import os
import signal
import traceback
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

# Sentinel value to signal worker shutdown
SHUTDOWN = None


class GPUWorker(ABC):
    """Base class for a long-lived GPU worker process.

    Runs in a separate Process with CUDA_VISIBLE_DEVICES pinned.
    Reads batches from input_queue, processes them, writes results to output_queue.
    Receives SHUTDOWN sentinel to exit.
    """

    def __init__(self, gpu_id: int, input_queue: mp.Queue, output_queue: mp.Queue,
                 name: str = "GPUWorker"):
        self.gpu_id = gpu_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.name = name
        self._process: mp.Process | None = None

    def start(self):
        """Start the worker process."""
        self._process = mp.Process(target=self._run, name=self.name, daemon=True)
        self._process.start()
        logger.info(f"{self.name} started (pid={self._process.pid}, gpu={self.gpu_id})")

    def _run(self):
        """Main worker loop. Runs in the child process."""
        # Pin to specific GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # Ignore SIGINT in workers — let the main process handle it
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            self.setup()
            logger.info(f"{self.name}: setup complete on GPU {self.gpu_id}")
        except Exception as e:
            logger.error(f"{self.name}: setup failed: {e}")
            traceback.print_exc()
            self.output_queue.put(("error", f"setup_failed: {e}"))
            return

        while True:
            try:
                item = self.input_queue.get()
                if item is SHUTDOWN:
                    break

                batch_id, batch_data = item
                try:
                    result = self.process_batch(batch_data)
                    self.output_queue.put(("result", batch_id, result))
                except Exception as e:
                    logger.error(f"{self.name}: batch {batch_id} failed: {e}")
                    traceback.print_exc()
                    self.output_queue.put(("error", batch_id, str(e)))
            except Exception as e:
                logger.error(f"{self.name}: unexpected error in loop: {e}")
                traceback.print_exc()

        try:
            self.teardown()
        except Exception:
            pass
        logger.info(f"{self.name}: shutdown complete")

    def stop(self):
        """Signal the worker to stop and wait for exit."""
        if self._process and self._process.is_alive():
            self.input_queue.put(SHUTDOWN)
            self._process.join(timeout=30)
            if self._process.is_alive():
                self._process.terminate()

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @abstractmethod
    def setup(self):
        """Load models. Called once in the child process."""

    @abstractmethod
    def process_batch(self, batch_data) -> object:
        """Process a single batch. Returns result to be sent via output_queue."""

    def teardown(self):
        """Clean up resources. Called before exit."""
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Actor Tagging Worker (GPU 4) ─────────────────────────────────────────────

class ActorTagWorker(GPUWorker):
    """Face detection + InsightFace actor recognition.

    Input batch: list of image path strings
    Output: dict mapping image_path_str -> list of actor dicts
    """

    def __init__(self, gpu_id: int, input_queue: mp.Queue, output_queue: mp.Queue,
                 actor_embeddings_dir: str, yolo_model_path: str,
                 actor_images_dir: str, similarity_threshold: float = 0.35):
        super().__init__(gpu_id, input_queue, output_queue, name="ActorTagWorker")
        self.actor_embeddings_dir = actor_embeddings_dir
        self.yolo_model_path = yolo_model_path
        self.actor_images_dir = actor_images_dir
        self.similarity_threshold = similarity_threshold
        self._yolo = None
        self._insightface_app = None
        self._actor_matrix = None
        self._actor_names = None

    def setup(self):
        import torch
        from actor_tagger import build_actor_embeddings

        emb_dir = Path(self.actor_embeddings_dir)
        if not emb_dir.exists() or not list(emb_dir.glob("*.pkl")):
            logger.info(f"{self.name}: Building actor embeddings...")
            build_actor_embeddings(
                self.actor_images_dir,
                self.actor_embeddings_dir,
                gpu_id=0,  # CUDA_VISIBLE_DEVICES already set → device 0 is our physical GPU
            )

        # Load YOLO
        from ultralytics import YOLO
        self._yolo = YOLO(self.yolo_model_path)
        self._yolo.to("cuda:0")

        # Load InsightFace
        import insightface
        self._insightface_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._insightface_app.prepare(ctx_id=0, det_size=(640, 640))

        # Load actor embeddings matrix
        import pickle
        import numpy as np
        self._actor_names = []
        embeddings = []
        for pkl_path in sorted(emb_dir.glob("*.pkl")):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self._actor_names.append(data["actor_name"])
            embeddings.append(np.array(data["embedding"], dtype=np.float32))

        if embeddings:
            self._actor_matrix = np.stack(embeddings)  # (N, 512)
        else:
            self._actor_matrix = np.zeros((0, 512), dtype=np.float32)

        logger.info(f"{self.name}: Loaded {len(self._actor_names)} actors")

    def process_batch(self, batch_data: list[str]) -> dict:
        """Process a batch of image paths for actor tagging.

        Returns dict: {image_path_str: [{"actor": slug, "display_name": name,
                        "similarity": float, "bbox": [x1,y1,x2,y2]}]}
        """
        import cv2
        import numpy as np

        results = {}
        if not self._actor_names:
            return {p: [] for p in batch_data}

        valid_imgs = []
        valid_paths = []
        for p_str in batch_data:
            bgr = cv2.imread(p_str)
            if bgr is not None:
                valid_imgs.append(bgr)
                valid_paths.append(p_str)
            else:
                results[p_str] = []

        if not valid_imgs:
            return results

        # YOLO batch detection
        yolo_results = self._yolo.predict(
            source=valid_imgs, imgsz=640, conf=0.5,
            half=True, verbose=False, stream=False,
        )

        for img_bgr, p_str, yolo_res in zip(valid_imgs, valid_paths, yolo_results):
            actors = []
            yolo_boxes = yolo_res.boxes.xyxy.cpu().numpy() if len(yolo_res.boxes) else []
            yolo_confs = yolo_res.boxes.conf.cpu().numpy() if len(yolo_res.boxes) else []

            if len(yolo_boxes) == 0:
                results[p_str] = []
                continue

            # InsightFace
            faces = self._insightface_app.get(img_bgr)

            for ybox, yconf in zip(yolo_boxes, yolo_confs):
                best_face = None
                best_iou = 0.0
                for face in faces:
                    fb = face.bbox
                    iou = self._compute_iou(ybox, fb)
                    if iou > best_iou:
                        best_iou = iou
                        best_face = face

                if best_face is None or not hasattr(best_face, "normed_embedding"):
                    continue

                emb = best_face.normed_embedding.reshape(1, -1)
                sims = emb @ self._actor_matrix.T
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[0, best_idx])

                if best_sim >= self.similarity_threshold:
                    import re
                    slug = re.sub(r"[^a-z0-9]+", "_", self._actor_names[best_idx].lower()).strip("_")
                    actors.append({
                        "actor": slug,
                        "display_name": self._actor_names[best_idx],
                        "similarity": round(best_sim, 4),
                        "bbox": [round(float(c), 1) for c in ybox],
                        "yolo_confidence": round(float(yconf), 4),
                    })

            results[p_str] = actors

        return results

    @staticmethod
    def _compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def teardown(self):
        del self._yolo, self._insightface_app
        self._yolo = None
        self._insightface_app = None
        super().teardown()


# ── CLIP Score Worker (GPU 5) ────────────────────────────────────────────────

class CLIPScoreWorker(GPUWorker):
    """CLIP image-text alignment scoring.

    Input batch: list of dicts with keys "image_path" and "caption"
    Output: list of float scores (same order)
    """

    def __init__(self, gpu_id: int, input_queue: mp.Queue, output_queue: mp.Queue,
                 model_name: str = "openai/clip-vit-large-patch14", batch_size: int = 16):
        super().__init__(gpu_id, input_queue, output_queue, name="CLIPScoreWorker")
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def setup(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._model = CLIPModel.from_pretrained(self.model_name).to("cuda:0").eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        logger.info(f"{self.name}: CLIP model loaded ({self.model_name})")

    def process_batch(self, batch_data: list[dict]) -> list[float]:
        """Score a batch of (image_path, caption) pairs.

        Returns list of cosine similarity scores (0-1).
        """
        import torch
        from PIL import Image

        scores = []
        for start in range(0, len(batch_data), self.batch_size):
            sub_batch = batch_data[start:start + self.batch_size]
            images = []
            captions = []
            valid_idx = []

            for i, item in enumerate(sub_batch):
                try:
                    img = Image.open(item["image_path"]).convert("RGB")
                    images.append(img)
                    captions.append(str(item["caption"])[:77])
                    valid_idx.append(i)
                except Exception as e:
                    logger.warning(f"CLIP: could not load {item['image_path']}: {e}")

            batch_scores = [0.0] * len(sub_batch)

            if images:
                inputs = self._processor(
                    text=captions, images=images, return_tensors="pt",
                    padding=True, truncation=True,
                ).to("cuda:0")

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    img_embeds = outputs.image_embeds
                    txt_embeds = outputs.text_embeds
                    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                    txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)
                    sims = (img_embeds * txt_embeds).sum(dim=-1).cpu().tolist()

                for j, idx in enumerate(valid_idx):
                    batch_scores[idx] = sims[j]

            scores.extend(batch_scores)

        return scores

    def teardown(self):
        del self._model, self._processor
        self._model = None
        self._processor = None
        super().teardown()



# ── AOD Score Worker (CPU — no GPU needed) ───────────────────────────────────

class AODScoreWorker:
    """Average Object Detailness scoring via spaCy (CPU only).

    Not a GPUWorker — runs in a ThreadPoolExecutor in the main process.
    Stateless: loads spaCy model on first call.
    """

    def __init__(self):
        self._nlp = None

    def setup(self):
        import spacy
        self._nlp = spacy.load("en_core_web_sm")

    def process_batch(self, batch_data: list[dict]) -> tuple[list[float], list[int]]:
        """Score a batch of captions for adjective richness.

        Input: list of dicts with key "caption"
        Returns: (scores, noun_counts)
        """
        if self._nlp is None:
            self.setup()

        scores = []
        noun_counts = []

        for item in batch_data:
            caption = str(item.get("caption", ""))
            doc = self._nlp(caption)

            nouns = [tok for tok in doc if tok.pos_ == "NOUN"]
            noun_counts.append(len(nouns))
            if not nouns:
                scores.append(0.0)
                continue

            total_modifiers = 0
            for noun in nouns:
                for child in noun.children:
                    if child.dep_ in ("amod", "acomp") or child.pos_ == "ADJ":
                        total_modifiers += 1
                if noun.dep_ == "compound":
                    head = noun.head
                    for child in head.children:
                        if child.dep_ in ("amod", "acomp") or child.pos_ == "ADJ":
                            total_modifiers += 0.5

            aod = total_modifiers / len(nouns)
            scores.append(round(aod, 4))

        return scores, noun_counts
