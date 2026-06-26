#!/usr/bin/env python3
"""
Caption Quality Evaluation Pipeline

Scores captions on three axes:
  1. CLIP Score  - image-text alignment (cosine similarity)
  2. AOD Score   - Average Object Detailness (adjective richness per noun)
  3. ICR Score   - Image Coverage Rate (caption noun coverage via GroundingDINO)

Usage:
  python scorer.py --input-csv subset/captions.csv --output results/scores.csv
  python scorer.py --input-dir subset/ --output results/scores.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Caption quality evaluation pipeline")
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input-csv", type=str, help="Path to captions CSV (image_path,caption,model)")
    inp.add_argument("--input-dir", type=str, help="Directory with *_caption.json sidecar files")

    _dir = str(Path(__file__).resolve().parent)
    p.add_argument("--output", type=str, default=f"{_dir}/results/scores.csv")
    p.add_argument("--clip-model", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--gdino-config", type=str,
                   default=f"{_dir}/groundingdino/GroundingDINO_SwinT_OGC.py")
    p.add_argument("--gdino-checkpoint", type=str,
                   default=f"{_dir}/groundingdino/groundingdino_swint_ogc.pth")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--skip-icr", action="store_true", help="Skip ICR (GroundingDINO) if not installed")
    p.add_argument("--gdino-box-threshold", type=float, default=0.25)
    p.add_argument("--gdino-text-threshold", type=float, default=0.20)
    return p.parse_args()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert {"image_path", "caption", "model"}.issubset(df.columns), \
        f"CSV must have columns: image_path, caption, model. Got: {list(df.columns)}"
    return df


def load_from_dir(dir_path: str) -> pd.DataFrame:
    rows = []
    d = Path(dir_path)
    for cap_file in sorted(d.glob("*_caption.json")):
        with open(cap_file) as f:
            data = json.load(f)
        image_path = data.get("image", "")
        if not Path(image_path).exists():
            stem = cap_file.stem.replace("_caption", "")
            candidates = list(d.glob(f"{stem}.*"))
            candidates = [c for c in candidates if c.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
            if candidates:
                image_path = str(candidates[0])
            else:
                continue
        rows.append({
            "image_path": image_path,
            "caption": data["caption"],
            "model": data.get("model", "unknown"),
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Module 1: CLIP Score
# ---------------------------------------------------------------------------

def compute_clip_scores(df: pd.DataFrame, model_name: str, batch_size: int,
                        gpu_id: int | None = None) -> list[float]:
    from transformers import CLIPModel, CLIPProcessor

    print(f"\n--- CLIP Score ({model_name}) ---")
    device = (f"cuda:{gpu_id}" if gpu_id is not None
              else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    scores = []
    for start in tqdm(range(0, len(df), batch_size), desc="CLIP"):
        batch = df.iloc[start : start + batch_size]
        images = []
        captions = []
        valid_idx = []
        for i, row in batch.iterrows():
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                images.append(img)
                captions.append(str(row["caption"])[:77])  # CLIP max token context
                valid_idx.append(i)
            except Exception as e:
                print(f"  WARN: Could not load {row['image_path']}: {e}")

        if not images:
            scores.extend([0.0] * len(batch))
            continue

        inputs = processor(
            text=captions, images=images, return_tensors="pt",
            padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Per-pair cosine similarity (diagonal of the similarity matrix)
            img_embeds = outputs.image_embeds  # (B, D)
            txt_embeds = outputs.text_embeds   # (B, D)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
            txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)
            sims = (img_embeds * txt_embeds).sum(dim=-1).cpu().tolist()

        # Map back scores, filling 0.0 for failed images
        batch_scores = [0.0] * len(batch)
        for j, idx in enumerate(valid_idx):
            pos = idx - batch.index[0]
            batch_scores[pos] = sims[j]
        scores.extend(batch_scores)

    # Clean up GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores

# ---------------------------------------------------------------------------
# Module 2: AOD (Average Object Detailness)
# ---------------------------------------------------------------------------

COLOR_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "grey", "gray", "golden", "silver", "beige", "crimson",
    "scarlet", "maroon", "turquoise", "teal", "ivory", "cream", "amber",
    "saffron", "copper", "bronze", "dark", "light", "bright", "pale", "deep",
    "vibrant", "vivid",
}
SIZE_WORDS = {
    "large", "small", "big", "tiny", "huge", "enormous", "little", "tall",
    "short", "wide", "narrow", "thick", "thin", "long", "miniature", "massive",
}
TEXTURE_WORDS = {
    "smooth", "rough", "soft", "hard", "silky", "fuzzy", "bumpy", "coarse",
    "crispy", "crunchy", "flaky", "creamy", "glossy", "matte", "shiny",
    "glistening", "grainy", "powdery", "spongy", "chewy", "tender", "crusty",
}
MATERIAL_WORDS = {
    "wooden", "metal", "metallic", "glass", "ceramic", "plastic", "stone",
    "clay", "brass", "steel", "iron", "copper", "porcelain", "fabric",
    "leather", "bamboo", "terracotta",
}


def categorize_adjective(token_text: str) -> str:
    w = token_text.lower()
    if w in COLOR_WORDS:
        return "color"
    if w in SIZE_WORDS:
        return "size"
    if w in TEXTURE_WORDS:
        return "texture"
    if w in MATERIAL_WORDS:
        return "material"
    return "other"


def compute_aod_scores(df: pd.DataFrame) -> tuple[list[float], list[int]]:
    import spacy
    print("\n--- AOD Score (spaCy) ---")
    nlp = spacy.load("en_core_web_sm")

    scores = []
    noun_counts = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="AOD"):
        caption = str(row["caption"])
        doc = nlp(caption)

        nouns = [tok for tok in doc if tok.pos_ == "NOUN"]
        noun_counts.append(len(nouns))
        if not nouns:
            scores.append(0.0)
            continue

        total_modifiers = 0
        for noun in nouns:
            # Count adjectival modifiers from dependency tree
            for child in noun.children:
                if child.dep_ in ("amod", "acomp") or child.pos_ == "ADJ":
                    total_modifiers += 1
            # Also count compound adjective heads if noun is part of a compound
            if noun.dep_ == "compound":
                head = noun.head
                for child in head.children:
                    if child.dep_ in ("amod", "acomp") or child.pos_ == "ADJ":
                        total_modifiers += 0.5  # partial credit for shared modifiers

        aod = total_modifiers / len(nouns)
        scores.append(round(aod, 4))

    return scores, noun_counts

# ---------------------------------------------------------------------------
# Module 3: ICR (Image Coverage Rate) via GroundingDINO
# ---------------------------------------------------------------------------

def extract_nouns(caption: str, nlp) -> list[str]:
    """Extract unique noun lemmas from caption using spaCy."""
    doc = nlp(caption)
    nouns = set()
    for tok in doc:
        if tok.pos_ == "NOUN" and len(tok.text) > 2:
            nouns.add(tok.lemma_.lower())
    return sorted(nouns)


def fuzzy_match_nouns(detected_labels: list[str], caption_nouns: list[str]) -> int:
    """Count how many caption nouns are covered by detected labels (fuzzy substring match)."""
    detected_text = " ".join(detected_labels).lower()
    matched = 0
    for noun in caption_nouns:
        # Direct substring match in detected labels
        if noun in detected_text:
            matched += 1
            continue
        # Check if any detected label is a substring of the noun or vice versa
        for label in detected_labels:
            label_lower = label.lower()
            if noun in label_lower or label_lower in noun:
                matched += 1
                break
    return matched


def _patch_groundingdino_compat():
    """Patch GroundingDINO for compatibility with newer transformers.

    Fixes two issues:
    1. BertModel.get_head_mask was removed in newer transformers
    2. BertModelWarper passes 'device' as 'dtype' to get_extended_attention_mask
       (old API had device as 3rd param, new API changed it to dtype)
    """
    from transformers import BertModel as _BertModel

    # Fix 1: Restore get_head_mask
    if not hasattr(_BertModel, 'get_head_mask'):
        def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
                if is_attention_chunked:
                    head_mask = head_mask.unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask
        _BertModel.get_head_mask = _get_head_mask

    # Fix 2: Patch get_extended_attention_mask to ignore device passed as dtype
    import inspect
    from transformers.modeling_utils import PreTrainedModel
    _orig_get_ext_mask = PreTrainedModel.get_extended_attention_mask

    sig = inspect.signature(_orig_get_ext_mask)
    params = list(sig.parameters.keys())
    # New API has 'dtype' as 3rd kwarg; old had 'device'
    if 'dtype' in params and 'device' not in params:
        def _patched_get_extended_attention_mask(self, attention_mask, input_shape, dtype=None):
            import torch
            # If caller passed a device as dtype (GroundingDINO compat), ignore it
            if dtype is not None and isinstance(dtype, torch.device):
                dtype = None
            if dtype is not None and not isinstance(dtype, torch.dtype):
                dtype = None
            return _orig_get_ext_mask(self, attention_mask, input_shape, dtype=dtype)
        PreTrainedModel.get_extended_attention_mask = _patched_get_extended_attention_mask


def compute_icr_scores(
    df: pd.DataFrame,
    config_path: str,
    checkpoint_path: str,
    box_threshold: float,
    text_threshold: float,
    gpu_id: int | None = None,
) -> list[float]:
    import spacy
    _patch_groundingdino_compat()
    try:
        from groundingdino.util.inference import load_model, predict
        import groundingdino.datasets.transforms as T
    except ImportError:
        print("  ERROR: GroundingDINO not available. Returning 0s for ICR.")
        return [0.0] * len(df)

    print(f"\n--- ICR Score (GroundingDINO) ---")

    if not os.path.exists(checkpoint_path):
        print(f"  ERROR: Checkpoint not found: {checkpoint_path}")
        print("  Run install_deps.sh first. Returning 0s for ICR.")
        return [0.0] * len(df)

    if not os.path.exists(config_path):
        print(f"  ERROR: Config not found: {config_path}")
        print("  Run install_deps.sh first. Returning 0s for ICR.")
        return [0.0] * len(df)

    device = (f"cuda:{gpu_id}" if gpu_id is not None
              else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(config_path, checkpoint_path, device=device)
    nlp = spacy.load("en_core_web_sm")

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ICR"):
        try:
            image_path = row["image_path"]
            caption = str(row["caption"])

            # Extract nouns from caption
            caption_nouns = extract_nouns(caption, nlp)
            if not caption_nouns:
                scores.append(0.0)
                continue

            # Build detection prompt from caption nouns (GroundingDINO uses ". " separated)
            # Limit to top 20 nouns to stay within prompt length limits
            detection_prompt = " . ".join(caption_nouns[:20]) + " ."

            # Load and transform image
            pil_image = Image.open(image_path).convert("RGB")
            image_tensor, _ = transform(pil_image, None)

            # Run GroundingDINO
            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor,
                caption=detection_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device,
            )

            if len(phrases) == 0:
                # No detections - caption nouns not grounded
                scores.append(0.0)
                continue

            detected_labels = list(set(phrases))

            # ICR = matched caption nouns / total caption nouns
            matched = fuzzy_match_nouns(detected_labels, caption_nouns)
            icr = matched / len(caption_nouns) if caption_nouns else 0.0
            scores.append(round(min(icr, 1.0), 4))

        except Exception as e:
            print(f"  WARN: ICR failed for {row.get('image_path', '?')}: {e}")
            scores.append(0.0)

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores

# ---------------------------------------------------------------------------
# Combined scoring & summary
# ---------------------------------------------------------------------------

CLIP_WEIGHT = 0.55
ICR_WEIGHT = 0.0
AOD_WEIGHT = 0.45

# Normalization: AOD scores can be > 1, so we cap for the combined score.
# Real captions max ~0.72 AOD. Old value of 3.0 crushed all scores to 7-24%.
AOD_CAP = 1.0


MIN_NOUNS = 5  # captions with fewer nouns get ICR/AOD dampened


def compute_combined(row, clip_w=0.40, icr_w=0.35, aod_w=0.25):
    """Compute weighted combined score from CLIP, ICR, and AOD components.

    Args:
        row: DataFrame row with clip_score, icr_score, aod_score, noun_count.
        clip_w: Weight for CLIP score (default 0.40).
        icr_w: Weight for ICR score (default 0.35).
        aod_w: Weight for AOD score (default 0.25).
    """
    clip = row["clip_score"]
    icr = row["icr_score"]
    aod_norm = min(row["aod_score"], AOD_CAP) / AOD_CAP  # normalize to 0-1

    # Penalize lazy captions with very few nouns
    noun_count = row.get("noun_count", MIN_NOUNS)
    noun_penalty = min(noun_count / MIN_NOUNS, 1.0)
    icr_adj = icr * noun_penalty
    aod_adj = aod_norm * noun_penalty

    return round(clip_w * clip + icr_w * icr_adj + aod_w * aod_adj, 4)


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("CAPTION QUALITY EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\nTotal images evaluated: {len(df)}")
    print(f"\nOverall statistics:")
    for col in ["clip_score", "icr_score", "aod_score", "combined_score"]:
        if col in df.columns:
            print(f"  {col:16s}  mean={df[col].mean():.4f}  "
                  f"std={df[col].std():.4f}  "
                  f"min={df[col].min():.4f}  max={df[col].max():.4f}")

    # Hallucination flag
    halluc = df[df["clip_score"] < 0.2]
    if len(halluc) > 0:
        print(f"\n  WARNING: {len(halluc)} captions with CLIP < 0.2 (potential hallucination)")

    # Per-model breakdown
    if df["model"].nunique() > 1:
        print(f"\nPer-model breakdown:")
        grouped = df.groupby("model")[["clip_score", "icr_score", "aod_score", "combined_score"]].mean()
        print(grouped.to_string())
    else:
        model_name = df["model"].iloc[0]
        print(f"\nAll captions from model: {model_name}")

    # Top & bottom 5
    print(f"\nTop 5 captions (by combined_score):")
    top = df.nlargest(5, "combined_score")
    for _, row in top.iterrows():
        print(f"  [{row['combined_score']:.3f}] {Path(row['image_path']).name}: "
              f"{row['caption'][:80]}...")

    print(f"\nBottom 5 captions (by combined_score):")
    bot = df.nsmallest(5, "combined_score")
    for _, row in bot.iterrows():
        print(f"  [{row['combined_score']:.3f}] {Path(row['image_path']).name}: "
              f"{row['caption'][:80]}...")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load data
    if args.input_csv:
        df = load_from_csv(args.input_csv)
    else:
        df = load_from_dir(args.input_dir)

    print(f"Loaded {len(df)} image-caption pairs")
    if len(df) == 0:
        print("No data to evaluate.")
        sys.exit(1)

    # Module 1: CLIP Score
    df["clip_score"] = compute_clip_scores(df, args.clip_model, args.batch_size)

    # Module 2: AOD Score
    df["aod_score"], df["noun_count"] = compute_aod_scores(df)

    # Module 3: ICR Score
    if args.skip_icr:
        print("\n--- ICR Score SKIPPED (--skip-icr) ---")
        df["icr_score"] = 0.0
    else:
        df["icr_score"] = compute_icr_scores(
            df, args.gdino_config, args.gdino_checkpoint,
            args.gdino_box_threshold, args.gdino_text_threshold,
        )

    # Combined score
    df["combined_score"] = df.apply(compute_combined, axis=1)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_cols = ["image_path", "model", "caption", "clip_score", "icr_score", "aod_score", "combined_score"]
    df[output_cols].to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print_summary(df)


if __name__ == "__main__":
    main()
