#!/usr/bin/env python3
"""
Watermark detection, removal, and verification.

Hybrid pipeline:
  1. Detect watermark (SigLIP2 or LAION ConvNeXt classifier) → score 0-1
  2. If light watermark (0.3-0.8): YOLO11x bbox → mask → LaMA inpainting
  3. Verify: re-run detector on cleaned image
  4. If still watermarked after removal → quarantine

Also handles letterbox border cropping for movie frames.

Models (auto-downloaded on first use):
  - Watermark classifier: ultralytics YOLO for detection
  - LaMA: simple-lama-inpainting (pip install simple-lama-inpainting)
"""

import json
import logging
import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

# Set HF cache to writable location before any HuggingFace imports
os.environ.setdefault("HF_HOME", "/data/kl_dev/models/hf_cache")

logger = logging.getLogger(__name__)


# ── Letterbox Border Cropping ─────────────────────────────────────────────────

def crop_letterbox(image_path: Path, overwrite: bool = True) -> Path:
    """Detect and crop black letterbox borders (top/bottom).

    Scans rows from top and bottom, finds first row with mean brightness > 15.
    Crops to content area. Only crops if borders are > 20px (to avoid
    false positives on dark scenes).

    Returns path to cropped image (same path if overwrite=True).
    """
    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        h, w, _ = arr.shape
        gray = np.mean(arr, axis=2)
        row_means = np.mean(gray, axis=1)

        # Find top border
        top = 0
        for i in range(h // 3):  # never crop more than 1/3
            if row_means[i] > 15:
                break
            top = i + 1

        # Find bottom border
        bottom = h
        for i in range(h - 1, h - h // 3, -1):
            if row_means[i] > 15:
                break
            bottom = i

        # Only crop if borders are significant (> 20px each)
        if top > 20 or (h - bottom) > 20:
            cropped = img.crop((0, top, w, bottom))
            out_path = image_path if overwrite else image_path.with_stem(image_path.stem + "_cropped")
            cropped.save(out_path, quality=95)
            logger.debug(f"  Cropped borders: {image_path.name} {h}→{bottom-top}px (top={top}, bottom={h-bottom})")
            return out_path

        return image_path
    except Exception as e:
        logger.warning(f"crop_letterbox failed for {image_path}: {e}")
        return image_path



# ── Watermark Detection ───────────────────────────────────────────────────────

class WatermarkDetector:
    """YOLO-based watermark detection — returns bounding boxes + confidence."""

    # Default to local path; falls back to HuggingFace download
    _DEFAULT_MODEL = str(Path(__file__).parent / "actors" / "joy_caption_watermark.pt")

    def __init__(self, model_name: str | None = None,
                 conf_threshold: float = 0.3, use_gpu: bool = False):
        self.model_name = model_name or self._DEFAULT_MODEL
        self.conf_threshold = conf_threshold
        self.use_gpu = use_gpu
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from ultralytics import YOLO
        logger.info(f"Loading watermark detector: {self.model_name}")
        try:
            self._model = YOLO(self.model_name)
        except Exception:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(self.model_name, "best.pt")
            self._model = YOLO(path)
        # NOTE: Do NOT call .to("cpu") — it corrupts CUDA state (device_count→0).
        # When running in main pipeline process, use device="cpu" in predict().
        # When running in watermark subprocess, CUDA is safe to use.

    def detect(self, image_path: Path) -> dict:
        """Detect watermarks in an image.

        Returns:
            {
                "has_watermark": bool,
                "confidence": float (0-1, max confidence across detections),
                "bboxes": list of [x1, y1, x2, y2, conf],
                "count": int
            }
        """
        self._load()
        try:
            device = "0" if self.use_gpu else "cpu"
            results = self._model.predict(
                source=str(image_path),
                conf=self.conf_threshold,
                device=device,
                verbose=False,
            )
            bboxes = []
            max_conf = 0.0
            # Get image dimensions for size filtering
            from PIL import Image as _PILImage
            _img = _PILImage.open(image_path)
            img_w, img_h = _img.size
            img_area = img_w * img_h
            _img.close()

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    # Filter out false positives: real watermarks are small logos
                    # (< 15% of image area). Large detections are art patterns/textures.
                    bw = xyxy[2] - xyxy[0]
                    bh = xyxy[3] - xyxy[1]
                    bbox_area = bw * bh
                    if bbox_area > img_area * 0.15:
                        continue  # too large — probably not a watermark
                    bboxes.append(xyxy + [conf])
                    max_conf = max(max_conf, conf)

            return {
                "has_watermark": len(bboxes) > 0,
                "confidence": round(max_conf, 4),
                "bboxes": bboxes,
                "count": len(bboxes),
            }
        except Exception as e:
            logger.warning(f"Watermark detection failed for {image_path}: {e}")
            return {"has_watermark": False, "confidence": 0.0, "bboxes": [], "count": 0}

    def detect_batch(self, image_paths: list[Path], batch_size: int = 32) -> list[dict]:
        """Detect watermarks in a batch using YOLO native batch inference."""
        self._load()
        all_results = []
        device = "0" if self.use_gpu else "cpu"

        for start in range(0, len(image_paths), batch_size):
            batch = image_paths[start:start + batch_size]
            try:
                yolo_results = self._model.predict(
                    source=[str(p) for p in batch],
                    conf=self.conf_threshold,
                    device=device,
                    verbose=False,
                )
                for p, r in zip(batch, yolo_results):
                    # Get image dimensions for size filtering
                    from PIL import Image as _PILImage
                    _img = _PILImage.open(p)
                    img_w, img_h = _img.size
                    img_area = img_w * img_h
                    _img.close()

                    bboxes = []
                    max_conf = 0.0
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        bw = xyxy[2] - xyxy[0]
                        bh = xyxy[3] - xyxy[1]
                        if bw * bh > img_area * 0.15:
                            continue
                        bboxes.append(xyxy + [conf])
                        max_conf = max(max_conf, conf)

                    all_results.append({
                        "has_watermark": len(bboxes) > 0,
                        "confidence": round(max_conf, 4),
                        "bboxes": bboxes,
                        "count": len(bboxes),
                    })
            except Exception as e:
                # Fallback: mark all as clean
                for p in batch:
                    all_results.append({"has_watermark": False, "confidence": 0.0, "bboxes": [], "count": 0})
                logger.warning(f"Batch watermark detection failed: {e}")

        return all_results


# ── Watermark Removal (LaMA inpainting) ──────────────────────────────────────

class WatermarkRemover:
    """LaMA-based inpainting to remove watermarks given bounding boxes.

    Uses simple-lama-inpainting. Key requirements:
    - Image must be PIL RGB
    - Mask must be PIL L (grayscale), same dimensions as image
    - Both are auto-padded to multiples of 8 internally
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self._lama = None

    def _load(self):
        if self._lama is not None:
            return
        import torch
        from simple_lama_inpainting import SimpleLama
        dev = torch.device(self.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading LaMA inpainting model on {dev}")
        self._lama = SimpleLama(device=dev)

    def remove(self, image_path: Path, bboxes: list, output_path: Path | None = None,
               padding: int = 20) -> Path:
        """Remove watermarks by inpainting bounding box regions with LaMA.

        Args:
            image_path: Input image
            bboxes: List of [x1, y1, x2, y2, conf] from detector
            output_path: Where to save cleaned image (default: overwrite)
            padding: Extra pixels around each bbox for cleaner inpainting

        Returns: Path to cleaned image
        """
        self._load()
        try:
            # Load image as RGB PIL — required by LaMA
            img = Image.open(image_path).convert("RGB")
            w, h = img.size

            # Build mask as L (grayscale) PIL — MUST be same size as image
            mask = Image.new("L", (w, h), 0)
            mask_arr = np.array(mask)
            for bbox in bboxes:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                mask_arr[y1:y2, x1:x2] = 255
            mask = Image.fromarray(mask_arr, mode="L")

            # Verify dimensions match (critical for LaMA)
            assert img.size == mask.size, f"Size mismatch: img={img.size}, mask={mask.size}"

            # Run LaMA inpainting
            result = self._lama(img, mask)

            # LaMA may pad to multiple of 8 — crop back to original size
            if result.size != (w, h):
                result = result.crop((0, 0, w, h))

            out = output_path or image_path
            result.save(out, quality=95)
            return out
        except Exception as e:
            logger.error(f"LaMA inpaint failed for {image_path}: {e}")
            return image_path


