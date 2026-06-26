#!/usr/bin/env python3
"""
Computational image quality filters — fast, deterministic, no GPU needed.

Runs BEFORE the VLM to reject obviously bad frames (blur, dark, text-heavy).
The VLM then only handles semantic filters (CBFC, credits, transitions, etc.)
and bucketing.

Usage:
    from image_filters import compute_filters
    filters = compute_filters("/path/to/image.jpg")
    # {"blurry": True, "dark_underexposed": False, "text_heavy": False,
    #  "metrics": {"sharpness": 28.5, "brightness": 45.2, ...}}
"""

import logging
import numpy as np
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

# Thresholds — tuned on Bollywood movie frames + Indian cultural images
#
# Blur: Face-aware + grid Laplacian (replaces saliency which was unreliable).
# Two-tier: grid Laplacian catches general blur, face detection rescues
# portrait shots with intentional DOF bokeh (sharp face, soft background).
# Tested on English Vinglish (Sridevi/Ajith scenes):
#   grid=25 + face_rescue=18 → 0 truly-blurry leaked, 21 DOF portraits rescued.
BLUR_THRESHOLD = 25        # Grid Laplacian threshold (same as before)
FACE_SHARP_RESCUE = 18     # Face Laplacian var: if face is this sharp, image is OK
# Dark: 40 catches truly dark (near-black). Frames 40-50 are indoor/night
# scenes common in Bollywood — keep them.
DARK_THRESHOLD = 40        # Mean brightness below this = too dark
BRIGHT_THRESHOLD = 245     # Mean brightness above this = overexposed
CONTRAST_THRESHOLD = 12    # Std dev below this = flat/washed out
# Haze: shot-through-glass, foggy, desaturated frames.
# Low saturation + low local contrast = glass/haze diffusion.
# Tested on English Vinglish immigration scenes vs good portraits.
HAZE_SAT_THRESHOLD = 60    # Mean HSV saturation below this
HAZE_LC_THRESHOLD = 11     # Median local contrast (64px patch std) below this

# Haar cascade paths (loaded once per process)
_face_cascades = None

def _get_face_cascades():
    """Lazy-load Haar cascades (frontal + profile)."""
    global _face_cascades
    if _face_cascades is None:
        import cv2
        _face_cascades = (
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml"),
        )
    return _face_cascades


def compute_sharpness(gray: np.ndarray) -> tuple[float, float]:
    """Face-aware sharpness: grid Laplacian + face rescue for DOF portraits.

    Returns (grid_sharpness, face_sharpness):
        grid_sharpness: max Laplacian variance from 7×7 subject-aware tiles
        face_sharpness: max Laplacian variance across detected face ROIs (0 if no face)

    Blur decision (in compute_filters):
        NOT blurry if grid_sharpness >= BLUR_THRESHOLD
        NOT blurry if face_sharpness >= FACE_SHARP_RESCUE (portrait rescue)
        BLURRY otherwise
    """
    import cv2

    h, w = gray.shape
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    # ── Step 1: Grid Laplacian (7×7) ─────────────────────────────────────
    # Use saliency to weight subject tiles higher (same as before)
    try:
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        _, saliency_map = saliency.computeSaliency(
            cv2.cvtColor(np.stack([gray_u8]*3, axis=2), cv2.COLOR_RGB2BGR)
        )
        saliency_map = (saliency_map * 255).astype(np.uint8)
        _, subject_mask = cv2.threshold(saliency_map, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        subject_mask = np.zeros((h, w), dtype=np.uint8)
        subject_mask[h//4:h*3//4, w//4:w*3//4] = 255

    grid = 7
    tile_h, tile_w = h // grid, w // grid
    subject_scores = []
    bg_scores = []

    for row in range(grid):
        for col in range(grid):
            y1, y2 = row * tile_h, (row + 1) * tile_h
            x1, x2 = col * tile_w, (col + 1) * tile_w
            tile = gray[y1:y2, x1:x2]
            mask_tile = subject_mask[y1:y2, x1:x2]

            lap = np.abs(
                tile[:-2, 1:-1] + tile[2:, 1:-1] +
                tile[1:-1, :-2] + tile[1:-1, 2:] -
                4 * tile[1:-1, 1:-1]
            )
            score = float(np.var(lap))

            if np.mean(mask_tile > 0) > 0.3:
                subject_scores.append(score)
            else:
                bg_scores.append(score)

    if subject_scores:
        grid_sharp = max(subject_scores)
    elif bg_scores:
        grid_sharp = max(bg_scores) * 0.5
    else:
        grid_sharp = 0.0

    # ── Step 2: Face detection + face ROI sharpness ──────────────────────
    # Haar cascade (fast, no GPU): detects frontal + profile faces.
    # If a face is sharp (Laplacian var >= FACE_SHARP_RESCUE), the image
    # is a portrait with intentional DOF — NOT blurry.
    face_sharp = 0.0
    try:
        frontal_cascade, profile_cascade = _get_face_cascades()
        min_face = (w // 12, h // 12)
        faces_f = frontal_cascade.detectMultiScale(gray_u8, 1.1, 4, minSize=min_face)
        faces_p = profile_cascade.detectMultiScale(gray_u8, 1.1, 4, minSize=min_face)
        all_faces = list(faces_f) + list(faces_p)

        for (x, y, fw, fh) in all_faces:
            roi = gray_u8[y:y+fh, x:x+fw]
            lap = cv2.Laplacian(roi, cv2.CV_64F)
            face_sharp = max(face_sharp, float(np.var(lap)))
    except Exception:
        pass  # face detection failure = no rescue, grid score decides

    return grid_sharp, face_sharp


def compute_haze(rgb_arr: np.ndarray, gray: np.ndarray) -> tuple[float, float]:
    """Detect haze/glass-shot frames via saturation + local contrast.

    Glass barriers, fog, and haze desaturate colors and flatten local contrast.
    Returns (mean_saturation, median_local_contrast).
    """
    import cv2

    # Mean saturation from HSV
    bgr = cv2.cvtColor(rgb_arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))

    # Median local contrast (std in 64×64 patches)
    h, w = gray.shape
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    patch_size = 64
    patch_stds = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray_u8[y:y + patch_size, x:x + patch_size]
            patch_stds.append(float(np.std(patch)))
    median_lc = float(np.median(patch_stds)) if patch_stds else 0.0

    return mean_sat, median_lc


def compute_brightness(gray: np.ndarray) -> float:
    """Mean pixel value (0-255)."""
    return float(np.mean(gray))


def compute_contrast(gray: np.ndarray) -> float:
    """Std deviation of pixel values."""
    return float(np.std(gray))


def compute_text_ratio(gray: np.ndarray) -> float:
    """Detect text-heavy frames using connected component analysis.

    Text has a distinctive pattern: many small, high-contrast connected
    components arranged in rows. Natural images have fewer, larger regions.
    This avoids false positives on detailed but non-text images (buildings,
    food textures) that fooled the old edge-density heuristic.
    """
    h, w = gray.shape

    # Binary threshold — text is typically high contrast
    binary = (gray > np.mean(gray) + 40).astype(np.uint8)

    # Count transitions per row (text rows have many black↔white transitions)
    transitions = np.abs(np.diff(binary.astype(int), axis=1))
    transitions_per_row = np.sum(transitions, axis=1)

    # A "text row" has many transitions (> 10% of width)
    text_row_threshold = w * 0.10
    text_rows = np.sum(transitions_per_row > text_row_threshold)
    text_row_ratio = text_rows / h

    # Also check: text frames have very uniform row-level variance
    # (all rows look similar because they're all text)
    row_means = np.mean(gray, axis=1)
    row_variance = np.std(row_means)

    # High text_row_ratio + low row_variance = text screen (credits, warnings)
    # High text_row_ratio + high row_variance = detailed image (paintings, streets)
    # Key insight: real text screens have VERY uniform brightness across rows
    # (row_variance < 20) while detailed images have high row_variance (> 30)
    if text_row_ratio > 0.5 and row_variance < 20:
        return text_row_ratio  # very likely text/credits screen
    elif text_row_ratio > 0.7 and row_variance < 30:
        return text_row_ratio * 0.6  # probably text
    else:
        return text_row_ratio * 0.1  # likely NOT text — detailed image


def compute_filters(image_path: str | Path) -> dict:
    """Compute all quality filters for a single image.

    Returns:
        {
            "blurry": bool,
            "dark_underexposed": bool,
            "text_heavy": bool,
            "overexposed": bool,
            "low_contrast": bool,
            "metrics": {
                "sharpness": float,
                "brightness": float,
                "contrast": float,
                "text_ratio": float,
            }
        }
    """
    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        gray = np.mean(arr, axis=2)

        grid_sharp, face_sharp = compute_sharpness(gray)
        brightness = compute_brightness(gray)
        contrast = compute_contrast(gray)
        text_ratio = compute_text_ratio(gray.astype(np.uint8))
        mean_sat, median_lc = compute_haze(arr.astype(np.uint8), gray)

        # Blur decision: grid check + face rescue for DOF portraits
        grid_pass = grid_sharp >= BLUR_THRESHOLD
        face_rescue = face_sharp >= FACE_SHARP_RESCUE
        is_blurry = not (grid_pass or face_rescue)

        # Haze: shot through glass/fog — desaturated + flat local contrast
        is_hazy = mean_sat < HAZE_SAT_THRESHOLD and median_lc < HAZE_LC_THRESHOLD

        sharpness = grid_sharp

        return {
            "blurry": bool(is_blurry or is_hazy),
            "dark_underexposed": bool(brightness < DARK_THRESHOLD),
            "text_heavy": bool(text_ratio > 0.45 and grid_sharp < 500),
            "overexposed": bool(brightness > BRIGHT_THRESHOLD),
            "low_contrast": bool(contrast < CONTRAST_THRESHOLD),
            "metrics": {
                "sharpness": round(float(sharpness), 1),
                "brightness": round(float(brightness), 1),
                "contrast": round(float(contrast), 1),
                "text_ratio": round(float(text_ratio), 3),
                "face_sharpness": round(float(face_sharp), 1),
                "saturation": round(float(mean_sat), 1),
                "local_contrast": round(float(median_lc), 1),
            },
        }
    except Exception as e:
        logger.warning(f"compute_filters failed for {image_path}: {e}")
        return {
            "blurry": False,
            "dark_underexposed": False,
            "text_heavy": False,
            "overexposed": False,
            "low_contrast": False,
            "metrics": {"sharpness": 0, "brightness": 0, "contrast": 0, "text_ratio": 0},
        }


def batch_compute_filters(image_paths: list[str | Path], max_workers: int = 8) -> list[dict]:
    """Compute filters for a batch of images in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(compute_filters, image_paths))
