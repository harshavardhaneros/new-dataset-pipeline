#!/usr/bin/env python3
"""
Standalone LaMA inpainting from VLM-detected watermark bounding boxes.

Takes a JSON file with [{image, bbox, text}, ...] and inpaints each bbox.
Runs as subprocess for CUDA isolation.

Usage:
    python3 run_lama_inpaint.py --detections /path/to/detections.json --gpu 2
"""

import os
import sys

# Set GPU before any torch import
_gpu = "0"
for i, a in enumerate(sys.argv):
    if a == "--gpu" and i + 1 < len(sys.argv):
        _gpu = sys.argv[i + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
os.environ.setdefault("HF_HOME", "/data/kl_dev/models/hf_cache")

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True, help="JSON file with detection list")
    parser.add_argument("--gpu", default="0", help="GPU ID")
    parser.add_argument("--padding", type=int, default=15, help="Padding around bbox")
    args = parser.parse_args()

    with open(args.detections) as f:
        items = json.load(f)

    if not items:
        print("No items to inpaint")
        return

    # Load LaMA
    from simple_lama_inpainting import SimpleLama
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading LaMA on {device}")
    lama = SimpleLama(device=device)

    cleaned = 0
    for item in items:
        p = Path(item["image"])
        if not p.exists():
            continue
        x1, y1, x2, y2 = item["bbox"]
        text = item.get("text", "")

        img = Image.open(p).convert("RGB")
        w, h = img.size

        # Build mask with padding
        mask_arr = np.zeros((h, w), dtype=np.uint8)
        pad = args.padding
        mask_arr[max(0, y1 - pad):min(h, y2 + pad),
                 max(0, x1 - pad):min(w, x2 + pad)] = 255
        mask = Image.fromarray(mask_arr, mode="L")

        result = lama(img, mask)
        if result.size != (w, h):
            result = result.crop((0, 0, w, h))
        result.save(str(p), quality=95)
        cleaned += 1
        print(f"  Cleaned: '{text}' from {p.name}")

    print(f"Done: {cleaned}/{len(items)} images inpainted")


if __name__ == "__main__":
    main()
