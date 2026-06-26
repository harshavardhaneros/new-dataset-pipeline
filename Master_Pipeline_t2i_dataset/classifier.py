#!/usr/bin/env python3
"""
Track D — VLM-based frame filter & cultural tagger.

Uses Qwen2.5-VL-32B-Instruct in a single pass per frame to:
  1. Detect certificate/warning/disclaimer screens (CBFC, tobacco, anti-piracy)
  2. Detect blurry / motion-blur frames
  3. Detect dark / underexposed frames
  4. Detect text-heavy frames (subtitles covering large area)
  5. Detect transition frames (fades, wipes, mid-cuts)
  6. Detect blank / black screens with no content
  7. Assess whether frame has meaningful cultural content
  8. If accepted, classify into one of 12 cultural buckets

Uses Qwen2.5-VL-32B-Instruct for classification.
EROS production logos are explicitly kept (not rejected).

Usage:
    python classifier.py                               # process all frames
    python classifier.py --image-dir Englishv_frames   # specific dir
    python classifier.py --no-skip                      # re-process existing
"""

import argparse
import csv
import gc
import json
import logging
import time
from pathlib import Path

import torch
from PIL import Image

from common import IMAGE_EXTS, MODEL_ID, parse_llm_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
LOCAL_MODEL_PATH = None  # Override via --model-path CLI arg
OUTPUT_DIR = BASE_DIR / "vlm_results"

IMAGE_DIRS = [
    BASE_DIR / "Englishv_frames",
    BASE_DIR / ".." / "track-A" / "indictest-dataset",
]

CLASSIFICATION_PROMPT = """Analyze this image frame. Answer EVERY filter question honestly as true or false.

**FILTER CHECKS** — set each to true if the condition IS present:
1. cbfc_certificate: Is this a CBFC / censor board rating card or certificate screen? (true/false)
2. tobacco_warning: Is this a tobacco or alcohol statutory warning on a plain background? (true/false)
3. anti_piracy: Is this an anti-piracy notice or "piracy is a crime" screen? (true/false)
4. production_credits: Is this opening/closing credits, "all characters are fictitious" disclaimer, or production logo card? (Exception: Eros logo frames → false) (true/false)
5. blurry: Is the frame blurry, soft, out-of-focus, or motion-blurred? Even if the scene is recognizable, mark true if edges are not crisp, faces are not sharp, or the image looks like it was shot through fog/glass. A frame suitable for T2I training must have SHARP details. (true/false)
6. dark_underexposed: Is the frame so dark that no meaningful content is visible? (true/false)
7. text_heavy: Do subtitles, titles, or credits text cover more than ~30% of the frame? (true/false)
8. transition_frame: Is this a partial fade, dissolve, wipe, or mid-cut with no clear scene? (true/false)
9. blank_screen: Is this a solid black, white, or plain colored screen? (true/false)
10. no_useful_content: Does the frame lack any meaningful visual scene? (true/false)
11. has_watermark: Does the image have any watermark, logo overlay, copyright text, website URL, or creator name text overlaid on it? (true/false)

IMPORTANT: Be strict. If the frame is even slightly blurry, dark, or text-heavy, mark it true. Do NOT default everything to false.

**CLASSIFICATION** — answer even if some filters are true:
- t2i_suitable: Would this frame be useful for training a text-to-image AI model? (true/false)
- category: The single BEST category from the list below. Pick ONE only.
- description: 10-15 word description of what is visible.

**12 BUCKETS** (choose exactly ONE):
- people_portraits: Individuals or groups, faces, posed or candid (if person is the MAIN subject)
- clothing_textiles: Traditional garments, jewelry, fabric close-ups (only if NO face visible)
- architecture: Buildings, temples, monuments, interiors, structural details (if building is MAIN subject)
- landscape_nature: Natural scenery, gardens, rivers, mountains, countryside
- urban_street: City streets, traffic, markets, shops, modern infrastructure
- rural_village: Village scenes, farming, rural homes, traditional occupations
- food_drink: Food, cooking, beverages, kitchen scenes, street food stalls
- festivals_rituals: Ceremonies, celebrations, rituals, prayers, processions
- objects_artifacts: Handicrafts, pottery, tools, musical instruments, cultural objects
- animals_wildlife: Animals, birds, livestock, pets, wildlife
- art_design: Paintings, murals, rangoli, mehndi, sculptures, graphic art
- abstract_texture: Patterns, textures, close-up materials, geometric designs

**DISAMBIGUATION RULES** (pick the ONE that fits best):
- Person wearing traditional clothing → people_portraits (person is the subject)
- Close-up of clothing with no face → clothing_textiles
- Building/temple with people visible → architecture (building is the subject)
- Street scene with food stalls → urban_street
- Festival with animals → festivals_rituals

Respond with ONLY this JSON — no markdown fences, no extra text:
{"filters": {"cbfc_certificate": false, "tobacco_warning": false, "anti_piracy": false, "production_credits": false, "blurry": false, "dark_underexposed": false, "text_heavy": false, "transition_frame": false, "blank_screen": false, "no_useful_content": false, "has_watermark": false}, "t2i_suitable": true, "category": "bucket_name", "description": "10-15 word description"}"""


WATERMARK_DETECT_PROMPT = """Look at this image carefully. Is there ANY watermark, channel logo, copyright text, website URL, or creator/studio name overlaid on the image?

Common examples: "EROS", "T-Series", "YRF", "Dharma", YouTube channel names, semi-transparent text in corners, small logos.

These are often semi-transparent, in a corner (top-left, top-right, bottom-right), and easy to miss.

If YES: identify the text and give the bounding box as percentages of image width/height (0-100).
If NO: set has_watermark to false and all bbox values to 0.

Respond with ONLY this JSON — no markdown fences, no extra text:
{"has_watermark": true, "watermark_text": "EROS", "bbox_x1_pct": 0, "bbox_y1_pct": 0, "bbox_x2_pct": 12, "bbox_y2_pct": 10}"""


def get_all_image_paths(image_dirs: list[Path]) -> list[Path]:
    paths = []
    for d in image_dirs:
        d = d.resolve()
        if not d.exists():
            logger.warning(f"{d} does not exist, skipping")
            continue
        paths.extend(sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS))
    logger.info(f"Found {len(paths)} images across {len(image_dirs)} directories")
    return paths


def load_model(model_path: str | None = None, backend: str = "transformers",
               gpu_ids: list[int] | None = None):
    """Load VLM model. Returns a VLMBackend instance."""
    from vlm_backend import create_backend
    b = create_backend(backend, model_path=model_path or LOCAL_MODEL_PATH or MODEL_ID,
                       gpu_ids=gpu_ids, max_new_tokens=200)
    b.load()
    return b


def prepare_image(path: Path, max_dim: int = 512) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
    return img


def classify_frame(backend, _processor_unused, pil_image: Image.Image, prompt: str) -> str:
    """Classify a frame using VLM backend."""
    return backend.generate(pil_image, prompt)


def parse_vlm_json(raw: str) -> dict:
    """Parse VLM classification output into structured dict."""
    fallback = {"rejected": False, "reasons": ["PARSE_ERROR"], "accept_reason": "",
                "t2i_suitable": False, "category": "none", "description": raw.strip()[:100]}
    return parse_llm_json(raw, fallback=fallback)


def run_inference(image_dirs: list[Path], output_dir: Path, skip_existing: bool = True):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = get_all_image_paths(image_dirs)

    if not image_paths:
        logger.info("No images found. Exiting.")
        return []

    backend = load_model()

    results = []
    for idx, img_path in enumerate(image_paths, 1):
        result_path = output_dir / f"{img_path.stem}.json"

        if skip_existing and result_path.exists():
            with open(result_path) as f:
                results.append(json.load(f))
            continue

        try:
            pil = prepare_image(img_path)
            t0 = time.time()
            raw = classify_frame(backend, None, pil, CLASSIFICATION_PROMPT)
            elapsed = round(time.time() - t0, 3)

            parsed = parse_vlm_json(raw)
            rec = {
                "image": str(img_path),
                "image_name": img_path.name,
                "source_dir": img_path.parent.name,
                "rejected": parsed.get("rejected", False),
                "reasons": parsed.get("reasons", []),
                "accept_reason": parsed.get("accept_reason", ""),
                "t2i_suitable": parsed.get("t2i_suitable", False),
                "category": parsed.get("category", "none"),
                "description": parsed.get("description", ""),
                "inference_time_s": elapsed,
            }
            with open(result_path, "w") as f:
                json.dump(rec, f, indent=2)

            results.append(rec)
            status = "REJECT" if rec["rejected"] else f"ACCEPT -> {rec['category']}"
            reason_tag = f"  [{', '.join(rec['reasons'])}]" if rec["rejected"] and rec["reasons"] else ""
            logger.info(f"[{idx}/{len(image_paths)}] {img_path.name}  {status}{reason_tag}  ({elapsed}s)")

        except Exception as e:
            logger.exception(f"Traceback for {img_path.name}:")
            rec = {
                "image": str(img_path), "image_name": img_path.name,
                "source_dir": img_path.parent.name,
                "rejected": False, "reasons": [f"ERROR: {e}"],
                "accept_reason": "", "t2i_suitable": False,
                "category": "none", "description": "", "inference_time_s": 0,
            }
            results.append(rec)
            logger.error(f"[{idx}/{len(image_paths)}] {img_path.name}  ERROR: {e}")

    backend.cleanup()

    logger.info(f"VLM classification complete: {len(results)} frames processed")
    accepted = sum(1 for r in results if not r["rejected"])
    logger.info(f"Accepted: {accepted}  |  Rejected: {len(results) - accepted}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Track D — VLM frame filter & tagger")
    parser.add_argument("--image-dirs", nargs="+",
                        default=[str(d) for d in IMAGE_DIRS],
                        help="Directories containing frames")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--no-skip", action="store_true", help="Re-process existing results")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local model path (default: uses MODEL_ID from HuggingFace)")
    parser.add_argument("--backend", type=str, default="transformers",
                        choices=["transformers", "vllm"],
                        help="VLM inference backend (default: transformers)")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs")
    args = parser.parse_args()

    global LOCAL_MODEL_PATH
    if args.model_path:
        LOCAL_MODEL_PATH = args.model_path

    dirs = [Path(d) for d in args.image_dirs]
    results = run_inference(dirs, Path(args.output_dir), skip_existing=not args.no_skip)

    csv_path = Path(args.output_dir) / "vlm_results.csv"
    fields = ["image_name", "source_dir", "rejected", "reasons", "accept_reason",
              "t2i_suitable", "category", "description", "inference_time_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {**r, "reasons": "|".join(r.get("reasons", []))}
            w.writerow(row)

    logger.info(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()
