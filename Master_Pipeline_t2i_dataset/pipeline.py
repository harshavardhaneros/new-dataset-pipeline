#!/usr/bin/env python3
"""
Master Pipeline — Indic Cultural Image Dataset

10-step pipeline from raw YouTube videos to training-ready dataset:

  1. discover     — Scan video directories, build manifest
  2. extract      — Scene detection + frame extraction from videos
  3. dedup_intra  — pHash deduplication within each video
  4. classify     — VLM 12-bucket classification + filter
  5. dedup_cross  — pHash deduplication across all sources
  6. caption      — Structured captioning with bucket-specific prompts
  7. score        — Quality scoring (CLIP + ICR + AOD)
  8. gate         — Threshold gating: final / review / discard
  9. export       — Build training-ready dataset (images + captions)
 10. report       — Generate summary statistics and HTML report

Usage:
    python pipeline.py --config pipeline_config.yaml
    python pipeline.py --step classify --input-dir /path/to/frames
    python pipeline.py --from-step score
    python pipeline.py --movie-list movies.csv
    python pipeline.py --dry-run
    python pipeline.py --force --from-step classify
    python pipeline.py --backend vllm --gpus 0,1
"""

import argparse
import csv
import gc
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# All imports are local to this directory
MASTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MASTER_DIR))

from common import IMAGE_EXTS, VIDEO_EXTS
from config import (
    BUCKETS, STEPS, STEP_ALIASES, SOURCE_TYPES,
    BKTree, compute_phash_worker, hamming_distance,
    PipelineConfig, load_config,
    ensure_dir, step_done, mark_done,
)


# BK-tree, pHash worker, and hamming_distance imported from config.py
# Aliases for backward compatibility within this file
_BKTree = BKTree
_compute_phash_worker = compute_phash_worker


# PipelineConfig, load_config, ensure_dir, step_done, mark_done imported from config.py


def load_movie_list(csv_path: str) -> list[dict]:
    """Load movie list CSV with columns: path, source_type.

    Returns list of {"path": str, "source_type": str}.
    """
    entries = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("path", "").strip()
            source_type = row.get("source_type", "youtube").strip()
            if path:
                entries.append({"path": path, "source_type": source_type})
    return entries


# ── Step 1: Discover ─────────────────────────────────────────────────────────

def step_discover(cfg: PipelineConfig) -> dict:
    """Scan video and image directories, build manifest."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  STEP 1 : Discover — Scanning sources")
    logger.info("=" * 70)

    if not cfg.force and step_done(cfg, "discover"):
        mp = Path(cfg.work_dir) / "manifest.json"
        if mp.exists():
            logger.info("  Step already done (use --force to re-run)")
            with open(mp) as f:
                return json.load(f)

    work = ensure_dir(cfg.work_dir)
    # manifest tracks source_type per entry
    manifest = {"videos": [], "images": [], "sources": {}}

    video_exts = VIDEO_EXTS

    # Load from movie list CSV if provided
    if cfg.movie_list:
        entries = load_movie_list(cfg.movie_list)
        for entry in entries:
            p = Path(entry["path"])
            st = entry["source_type"]
            if not p.exists():
                logger.warning(f"  {p} does not exist, skipping")
                continue
            if st == "precaptioned" and p.is_dir():
                # Precaptioned: scan for image+txt pairs
                images = sorted(
                    img for img in p.rglob("*")
                    if img.is_file() and img.suffix.lower() in IMAGE_EXTS
                )
                for img in images:
                    txt_file = img.with_suffix(".txt")
                    entry = {
                        "path": str(img),
                        "source_type": "precaptioned",
                        "dataset": img.parent.name,
                        "precaptioned": True,
                    }
                    if txt_file.exists():
                        entry["caption_txt"] = str(txt_file)
                    manifest["images"].append(entry)
                manifest["sources"][str(p)] = {
                    "type": "images", "source_type": "precaptioned", "count": len(images),
                }
                logger.info(f"  Precaptioned: {p} -> {len(images)} images")
            elif p.is_file() and p.suffix.lower() in video_exts:
                manifest["videos"].append({
                    "path": str(p), "source_type": st,
                })
            elif p.is_dir():
                videos = sorted(
                    v for v in p.rglob("*")
                    if v.is_file() and v.suffix.lower() in video_exts
                )
                for v in videos:
                    manifest["videos"].append({
                        "path": str(v), "source_type": st,
                    })
                manifest["sources"][str(p)] = {"type": "video", "source_type": st, "count": len(videos)}
                logger.info(f"  Videos ({st}): {p} -> {len(videos)} files")
    else:
        # Scan video directories (default YouTube paths)
        for vdir in cfg.video_dirs:
            vd = Path(vdir)
            if not vd.exists():
                logger.warning(f"  {vd} does not exist, skipping")
                continue
            videos = sorted(p for p in vd.rglob("*") if p.is_file() and p.suffix.lower() in video_exts)
            for v in videos:
                manifest["videos"].append({
                    "path": str(v), "source_type": "youtube",
                })
            manifest["sources"][str(vd)] = {"type": "video", "source_type": "youtube", "count": len(videos)}
            logger.info(f"  Videos (youtube): {vd} -> {len(videos)} files")

    # Scan image directories with source_type
    for idir_cfg in cfg.extra_image_dirs:
        if isinstance(idir_cfg, str):
            idir_cfg = {"path": idir_cfg, "source_type": "internal"}
        idir = Path(idir_cfg["path"])
        st = idir_cfg.get("source_type", "internal")
        if not idir.exists():
            logger.warning(f"  {idir} does not exist, skipping")
            continue
        images = sorted(p for p in idir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        for img in images:
            manifest["images"].append({
                "path": str(img),
                "source_type": st,
                "dataset": img.parent.name,
            })
        manifest["sources"][str(idir)] = {"type": "images", "source_type": st, "count": len(images)}
        logger.info(f"  Images ({st}): {idir} -> {len(images)} files")

    manifest_path = work / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"  Total: {len(manifest['videos'])} videos, {len(manifest['images'])} images")
    logger.info(f"  Manifest -> {manifest_path}")
    mark_done(cfg, "discover")
    return manifest


# ── Step 2: Extract Frames ───────────────────────────────────────────────────

def step_extract(cfg: PipelineConfig, manifest: dict | None = None) -> Path:
    """Extract frames from videos using scene detection."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  STEP 2 : Extract — Scene detection + frame extraction")
    logger.info("=" * 70)

    work = ensure_dir(cfg.work_dir)
    frames_dir = ensure_dir(work / "frames")

    if not cfg.force and step_done(cfg, "extract"):
        logger.info("  Step already done (use --force to re-run)")
        return frames_dir

    if manifest is None:
        mp = work / "manifest.json"
        if mp.exists():
            with open(mp) as f:
                manifest = json.load(f)
        else:
            logger.error("  No manifest found. Run 'discover' first.")
            return frames_dir

    from frame_extractor import extract_frames_from_video

    videos = manifest.get("videos", [])
    if not videos:
        logger.info("  No videos to process.")
    else:
        total_frames = 0
        for i, ventry in enumerate(videos, 1):
            # Support both old format (string) and new format (dict with path+source_type)
            if isinstance(ventry, str):
                vp = Path(ventry)
                source_type = "youtube"
            else:
                vp = Path(ventry["path"])
                source_type = ventry.get("source_type", "youtube")

            if not vp.exists():
                logger.warning(f"  [{i}/{len(videos)}] SKIP {vp.name} (not found)")
                continue
            logger.info(f"  [{i}/{len(videos)}] {vp.name} ({source_type})")
            count = extract_frames_from_video(
                vp, frames_dir, cfg.scene_threshold, cfg.frames_per_scene,
                adaptive=cfg.adaptive_detector,
            )
            total_frames += count
            # Explicit GC between videos — PySceneDetect has a known memory
            # leak when processing multiple videos in a loop (issue #373).
            gc.collect()

        logger.info(f"  Extracted: {total_frames} frames from {len(videos)} videos")

    # Link extra images into frames dir with dataset-prefix subdirs to avoid collisions
    extra_count = 0
    for img_entry in manifest.get("images", []):
        if isinstance(img_entry, str):
            img_p = Path(img_entry)
            dataset = "unknown"
        else:
            img_p = Path(img_entry["path"])
            dataset = img_entry.get("dataset", img_p.parent.name)

        if not img_p.exists():
            continue

        # Use dataset-prefixed subdir to prevent filename collisions
        dst_dir = ensure_dir(frames_dir / f"_extra_{dataset}")
        dst = dst_dir / img_p.name
        if not dst.exists():
            try:
                dst.symlink_to(img_p)
                extra_count += 1
            except OSError:
                shutil.copy2(img_p, dst)
                extra_count += 1

        # Also link .txt caption files for precaptioned entries
        if isinstance(img_entry, dict) and img_entry.get("precaptioned"):
            txt_src = Path(img_entry.get("caption_txt", ""))
            if txt_src.exists():
                txt_dst = dst_dir / txt_src.name
                if not txt_dst.exists():
                    try:
                        txt_dst.symlink_to(txt_src)
                    except OSError:
                        shutil.copy2(txt_src, txt_dst)

    logger.info(f"  Extra images linked: {extra_count}")
    logger.info(f"  Frames dir -> {frames_dir}")
    mark_done(cfg, "extract")
    return frames_dir


# ── Step 3: Unified pHash Dedup (intra + cross) ─────────────────────────────

def step_dedup(cfg: PipelineConfig) -> int:
    """Unified dedup: intra-video + cross-source in one step.

    Phase 1: Intra-video (BK-tree per video subdir, threshold ≤ 8)
    Phase 2: Cross-source (BK-tree across ALL remaining frames, threshold ≤ 6)

    Both phases MOVE duplicates to _dupes/ folders. This runs BEFORE classify
    and watermark, so no VLM time is wasted on duplicate frames.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("  STEP 3 : Dedup (intra + cross) — pHash deduplication")
    logger.info("=" * 70)

    # Check for unified or legacy done markers
    if not cfg.force and (step_done(cfg, "dedup") or
                          (step_done(cfg, "dedup_intra") and step_done(cfg, "dedup_cross"))):
        logger.info("  Step already done (use --force to re-run)")
        return 0

    from multiprocessing import Pool, cpu_count

    frames_dir = Path(cfg.work_dir) / "frames"
    if not frames_dir.exists():
        logger.error("  frames/ not found. Run 'extract' first.")
        return 0

    def _hamming(a, b):
        return abs(a - b)

    workers = min(cpu_count(), 8)

    # ── Phase 1: Intra-video dedup ────────────────────────────────────────
    logger.info("  Phase 1: Intra-video dedup (threshold ≤ %d)", cfg.phash_intra_threshold)
    intra_dupes = 0
    for subdir in sorted(frames_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_extra"):
            continue

        images = sorted(p for p in subdir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            continue

        with Pool(workers) as pool:
            hash_results = pool.map(_compute_phash_worker, [str(p) for p in images])

        tree = _BKTree(_hamming)
        dupes = []
        for img_path, (path_str, h) in zip(images, hash_results):
            if h is None:
                continue
            if tree.find(h, cfg.phash_intra_threshold):
                dupes.append(img_path)
            else:
                tree.insert(h)

        if dupes:
            dupe_dir = subdir / "_dupes"
            dupe_dir.mkdir(exist_ok=True)
            for dp in dupes:
                dp.rename(dupe_dir / dp.name)
            intra_dupes += len(dupes)

    logger.info(f"  Intra-video duplicates removed: {intra_dupes}")

    # ── Phase 2: Cross-source dedup ───────────────────────────────────────
    logger.info("  Phase 2: Cross-source dedup (threshold ≤ %d)", cfg.phash_cross_threshold)

    # Collect ALL remaining frames (after intra dedup)
    all_images = sorted(
        p for p in frames_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        and "/_dupes/" not in str(p) and "\\_dupes\\" not in str(p)
    )
    logger.info(f"  Computing cross-source pHash for {len(all_images)} frames...")

    with Pool(workers) as pool:
        hash_results = pool.map(_compute_phash_worker, [str(p) for p in all_images])

    tree = _BKTree(_hamming)
    cross_dupes = 0
    cross_dupe_dir = ensure_dir(frames_dir / "_dupes_cross")
    for img_path, (path_str, h) in zip(all_images, hash_results):
        if h is None:
            continue
        if tree.find(h, cfg.phash_cross_threshold):
            # Move to cross-dedup folder
            dst = cross_dupe_dir / img_path.name
            if not dst.exists():
                img_path.rename(dst)
                cross_dupes += 1
        else:
            tree.insert(h)

    logger.info(f"  Cross-source duplicates removed: {cross_dupes}")
    logger.info(f"  Total dedup: {intra_dupes + cross_dupes} removed")
    mark_done(cfg, "dedup")
    return intra_dupes + cross_dupes


# ── Step 4: VLM Classification ───────────────────────────────────────────────

def step_classify(cfg: PipelineConfig) -> Path:
    """Run 12-bucket VLM classification on all frames."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  STEP 4 : Classify — VLM 12-bucket classification")
    logger.info("=" * 70)

    if not cfg.force and step_done(cfg, "classify"):
        logger.info("  Step already done (use --force to re-run)")
        return Path(cfg.work_dir) / "vlm_results"

    work = Path(cfg.work_dir)
    frames_dir = work / "frames"
    vlm_dir = ensure_dir(work / "vlm_results")

    if not frames_dir.exists():
        logger.error("  frames/ not found. Run 'extract' first.")
        return vlm_dir

    # Build source_type map from manifest
    source_type_map = {}
    manifest_path = work / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Video frames: frames/{video_stem}/ → source_type
        video_stem_map = {}
        for ventry in manifest.get("videos", []):
            if isinstance(ventry, dict):
                vp = Path(ventry["path"])
                video_stem_map[vp.stem] = ventry.get("source_type", "youtube")

        # Extra images: resolve symlinks to find original path → source_type
        image_path_map = {}
        for ientry in manifest.get("images", []):
            if isinstance(ientry, dict):
                image_path_map[ientry["path"]] = ientry.get("source_type", "internal")

    # Collect all non-dupe images from all subdirs
    all_images = []
    for subdir in sorted(frames_dir.iterdir()):
        if not subdir.is_dir() or subdir.name == "_dupes":
            continue
        for img in sorted(subdir.rglob("*")):
            if img.suffix.lower() in IMAGE_EXTS and "_dupes" not in str(img):
                all_images.append(img)

    # Build source_type for each image
    if manifest_path.exists():
        for img_path in all_images:
            parent_name = img_path.parent.name
            if parent_name.startswith("_extra_"):
                # Extra image — resolve symlink to find original path
                try:
                    real = str(img_path.resolve())
                    st = image_path_map.get(real)
                    # Also try one-level readlink (handles chained symlinks)
                    if st is None and img_path.is_symlink():
                        st = image_path_map.get(str(img_path.readlink()))
                    source_type_map[str(img_path)] = st or "internal"
                except Exception:
                    source_type_map[str(img_path)] = "internal"
            else:
                # Video frame — parent dir name is video stem
                source_type_map[str(img_path)] = video_stem_map.get(parent_name, "youtube")

    logger.info(f"  Found {len(all_images)} images to classify")

    from classifier import load_model, classify_frame, parse_vlm_json, prepare_image, CLASSIFICATION_PROMPT

    backend = load_model(model_path=cfg.model_path, backend=cfg.backend,
                         gpu_ids=cfg.gpu_ids)

    accepted = 0
    rejected = 0
    for i, img_path in enumerate(all_images, 1):
        # Use parent__stem naming to avoid filename collisions
        result_path = vlm_dir / f"{img_path.parent.name}__{img_path.stem}.json"
        if not cfg.force and result_path.exists():
            with open(result_path) as f:
                data = json.load(f)
            if not data.get("rejected", True):
                accepted += 1
            else:
                rejected += 1
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
                "source_type": source_type_map.get(str(img_path), ""),
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

            if rec["rejected"]:
                rejected += 1
            else:
                accepted += 1

            if i % 100 == 0:
                logger.info(f"  [{i}/{len(all_images)}] Accepted: {accepted} | Rejected: {rejected}")

        except Exception as e:
            logger.error(f"  [{i}] {img_path.name}: {e}")
            rejected += 1

    backend.cleanup()

    logger.info(f"  Classification complete: {accepted} accepted, {rejected} rejected")
    logger.info(f"  VLM results -> {vlm_dir}")
    mark_done(cfg, "classify")
    return vlm_dir


# step_dedup_cross removed — now part of unified step_dedup above


# ── Step 5b: Actor Tagging ────────────────────────────────────────────────────

def step_tag_actors(cfg: PipelineConfig) -> Path:
    """Detect and identify Indian film actors in accepted people_portraits frames.

    Runs after dedup_cross so only accepted, deduplicated frames are processed.
    Pipeline: YOLO face detection → InsightFace 512-d embedding → cosine similarity
    against pre-built actor .pkl files → writes sidecar _actors.json per tagged image.

    Output: work_dir/actor_tags/{parentdir}__{stem}_actors.json
    These are picked up by step_caption to inject actor names into prompts,
    producing captions like "Shah Rukh Khan in a white kurta…" instead of
    "A man in a white kurta…".
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("  STEP 5b: Actor Tagging — YOLO face + InsightFace recognition")
    logger.info("=" * 70)

    if not cfg.tag_actors_enabled:
        logger.info("  Actor tagging disabled (tag_actors_enabled=False) — skipping.")
        return Path(cfg.work_dir) / "actor_tags"

    if not cfg.force and step_done(cfg, "tag_actors"):
        logger.info("  Step already done (use --force to re-run)")
        return Path(cfg.work_dir) / "actor_tags"

    work = Path(cfg.work_dir)
    vlm_dir = work / "vlm_results"
    actor_tags_dir = ensure_dir(work / "actor_tags")

    if not vlm_dir.exists():
        logger.error("  vlm_results/ not found. Run 'classify' first.")
        mark_done(cfg, "tag_actors")
        return actor_tags_dir

    from actor_tagger import build_actor_embeddings, tag_frames

    # Build actor embeddings if not already done (one-time per actor database)
    actor_images_dir = Path(cfg.actor_images_dir)
    embeddings_dir = Path(cfg.actor_embeddings_dir)

    if actor_images_dir.exists():
        build_actor_embeddings(
            actor_images_dir=actor_images_dir,
            output_dir=embeddings_dir,
            gpu_id=cfg.gpu_ids[0] if cfg.gpu_ids else 0,
        )
    else:
        logger.warning(f"  actor_images_dir not found: {actor_images_dir}")

    if not embeddings_dir.exists() or not list(embeddings_dir.glob("*.pkl")):
        logger.warning("  No actor embeddings found — skipping actor tagging.")
        mark_done(cfg, "tag_actors")
        return actor_tags_dir

    # Collect accepted people_portraits frames only
    # (actor tagging on non-portrait buckets adds noise and wastes GPU time)
    accepted_portraits: list[Path] = []
    for jp in sorted(vlm_dir.glob("*.json")):
        with open(jp) as f:
            r = json.load(f)
        if r.get("rejected", True):
            continue
        if r.get("category", "") != "people_portraits":
            continue
        img = Path(r.get("image", ""))
        if img.exists():
            accepted_portraits.append(img)

    logger.info(f"  {len(accepted_portraits)} accepted people_portraits frames to tag")

    if not accepted_portraits:
        logger.info("  No people_portraits frames found — nothing to tag.")
        mark_done(cfg, "tag_actors")
        return actor_tags_dir

    yolo_model = Path(cfg.yolo_face_model)
    if not yolo_model.exists():
        logger.warning(f"  YOLO model not found: {yolo_model} — skipping actor tagging.")
        mark_done(cfg, "tag_actors")
        return actor_tags_dir

    results = tag_frames(
        image_paths=accepted_portraits,
        actor_embeddings_dir=embeddings_dir,
        output_dir=actor_tags_dir,
        yolo_model_path=yolo_model,
        gpu_id=cfg.gpu_ids[0] if cfg.gpu_ids else 0,
        similarity_threshold=cfg.actor_similarity_threshold,
    )

    logger.info(f"  {len(results)} frames tagged with actor names → {actor_tags_dir}")
    mark_done(cfg, "tag_actors")
    return actor_tags_dir


# ── Step 6: Structured Captioning ─────────────────────────────────────────────

def step_caption(cfg: PipelineConfig) -> Path:
    """Run structured captioning with bucket-specific prompts."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  STEP 6 : Caption — Structured captioning (12-bucket prompts)")
    logger.info("=" * 70)

    if not cfg.force and step_done(cfg, "caption"):
        logger.info("  Step already done (use --force to re-run)")
        return Path(cfg.work_dir) / "captions"

    work = Path(cfg.work_dir)
    vlm_dir = work / "vlm_results"
    caption_dir = ensure_dir(work / "captions")

    if not vlm_dir.exists():
        logger.error("  vlm_results/ not found. Run 'classify' first.")
        return caption_dir

    from captioner import run_captioning

    actor_tags_dir = work / "actor_tags"
    run_captioning(
        input_dir=work,
        prompt_dir=Path(cfg.prompt_dir),
        output_dir=caption_dir,
        gpu_ids=cfg.gpu_ids,
        max_new_tokens=cfg.max_new_tokens,
        model_path=cfg.model_path,
        backend=cfg.backend,
        actor_tags_dir=actor_tags_dir if actor_tags_dir.exists() else None,
    )

    logger.info(f"  Captions -> {caption_dir}")
    mark_done(cfg, "caption")
    return caption_dir


# ── Step 7: Quality Scoring ──────────────────────────────────────────────────

def step_score(cfg: PipelineConfig) -> Path:
    """Compute quality scores (CLIP + ICR + AOD)."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  STEP 7 : Score — Quality evaluation (CLIP + ICR + AOD)")
    logger.info("=" * 70)

    if not cfg.force and step_done(cfg, "score"):
        logger.info("  Step already done (use --force to re-run)")
        return Path(cfg.work_dir) / "scores.csv"

    import pandas as pd

    work = Path(cfg.work_dir)
    caption_dir = work / "captions"
    scores_path = work / "scores.csv"

    if not caption_dir.exists():
        logger.error("  captions/ not found. Run 'caption' first.")
        return scores_path

    # Build DataFrame from caption JSONs
    rows = []
    for cp in sorted(caption_dir.glob("*_caption.json")):
        with open(cp) as f:
            data = json.load(f)
        img_path = data.get("image", "")
        if not Path(img_path).exists():
            continue
        rows.append({
            "image_path": img_path,
            "caption": data.get("caption", ""),
            "model": data.get("model", "unknown"),
            "bucket": data.get("bucket", ""),
            "source_type": data.get("source_type", ""),
        })

    if not rows:
        logger.warning("  No captioned images found.")
        return scores_path

    df = pd.DataFrame(rows)
    logger.info(f"  Scoring {len(df)} image-caption pairs...")

    from scorer import (
        compute_clip_scores, compute_aod_scores, compute_icr_scores, compute_combined,
    )

    df["clip_score"] = compute_clip_scores(df, cfg.clip_model, cfg.clip_batch_size)
    df["aod_score"], df["noun_count"] = compute_aod_scores(df)
    df["icr_score"] = compute_icr_scores(
        df, cfg.gdino_config, cfg.gdino_checkpoint, 0.25, 0.20,
    )
    # Pass configurable weights to compute_combined
    cw, iw, aw = cfg.clip_weight, cfg.icr_weight, cfg.aod_weight
    df["combined_score"] = df.apply(
        lambda r: compute_combined(r, clip_w=cw, icr_w=iw, aod_w=aw), axis=1
    )

    df.to_csv(scores_path, index=False)
    logger.info(f"  Scores saved -> {scores_path}")
    logger.info(f"  Mean CLIP: {df['clip_score'].mean():.4f}")
    logger.info(f"  Mean ICR:  {df['icr_score'].mean():.4f}")
    logger.info(f"  Mean AOD:  {df['aod_score'].mean():.4f}")
    logger.info(f"  Mean Combined: {df['combined_score'].mean():.4f}")
    mark_done(cfg, "score")
    return scores_path


# ── Step 8: Quality Gate ─────────────────────────────────────────────────────

def step_gate(cfg: PipelineConfig) -> dict:
    """Apply threshold gating: final / review / discard."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  STEP 8 : Gate — Threshold classification")
    logger.info("=" * 70)

    if not cfg.force and step_done(cfg, "gate"):
        logger.info("  Step already done (use --force to re-run)")
        return {}

    import pandas as pd

    scores_path = Path(cfg.work_dir) / "scores.csv"
    if not scores_path.exists():
        logger.error("  scores.csv not found. Run 'score' first.")
        return {}

    df = pd.read_csv(scores_path)

    df["gate"] = df["combined_score"].apply(
        lambda s: "final" if s >= cfg.gate_final
        else ("review" if s >= cfg.gate_review else "discard")
    )

    gated_path = Path(cfg.work_dir) / "gated.csv"
    df.to_csv(gated_path, index=False)

    counts = df["gate"].value_counts().to_dict()
    logger.info(f"  Final:   {counts.get('final', 0)}")
    logger.info(f"  Review:  {counts.get('review', 0)}")
    logger.info(f"  Discard: {counts.get('discard', 0)}")
    logger.info(f"  Gated CSV -> {gated_path}")

    mark_done(cfg, "gate")
    return counts


# ── Step 9: Export ───────────────────────────────────────────────────────────

def step_export(cfg: PipelineConfig) -> Path:
    """Build training-ready dataset from gated results."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  STEP 9 : Export — Build training dataset")
    logger.info("=" * 70)

    if not cfg.force and step_done(cfg, "export"):
        logger.info("  Step already done (use --force to re-run)")
        return Path(cfg.work_dir) / "export"

    import pandas as pd

    work = Path(cfg.work_dir)
    gated_path = work / "gated.csv"
    if not gated_path.exists():
        logger.error("  gated.csv not found. Run 'gate' first.")
        return work / "export"

    df = pd.read_csv(gated_path)
    final = df[df["gate"] == "final"]

    export_dir = ensure_dir(work / "export")
    images_dir = ensure_dir(export_dir / "images")
    captions_dir = ensure_dir(export_dir / "captions")

    logger.info(f"  Exporting {len(final)} final images...")

    # Load vlm_results for caption mixing (original descriptions)
    vlm_descriptions = {}
    if cfg.caption_mix_ratio > 0:
        vlm_dir = work / "vlm_results"
        if vlm_dir.exists():
            for jp in vlm_dir.glob("*.json"):
                with open(jp) as f:
                    vdata = json.load(f)
                vlm_descriptions[vdata.get("image", "")] = vdata.get("description", "")

    metadata_rows = []
    for _, row in final.iterrows():
        src = Path(row["image_path"])
        if not src.exists():
            continue

        # Use parent__stem naming to avoid filename collisions across sources
        unique_stem = f"{src.parent.name}__{src.stem}"
        unique_name = f"{unique_stem}{src.suffix}"

        dst_img = images_dir / unique_name
        if not dst_img.exists():
            shutil.copy2(src, dst_img)

        # Copy caption (using collision-safe naming)
        caption_src = work / "captions" / f"{unique_stem}_caption.json"
        caption_text = ""
        caption_source = "qwen"

        if caption_src.exists():
            shutil.copy2(caption_src, captions_dir / caption_src.name)
            with open(caption_src) as f:
                cap_data = json.load(f)
            caption_text = cap_data.get("caption", "")

            # Caption mixing: for specified sources, deterministically select
            # a subset to use original VLM descriptions instead of Qwen captions
            source_type = row.get("source_type", "")
            dataset = row.get("bucket", "")
            if cfg.caption_mix_ratio > 0 and cfg.caption_mix_sources:
                should_mix = any(
                    s in str(source_type).lower() or s in str(dataset).lower()
                    for s in cfg.caption_mix_sources
                )
                if should_mix:
                    # Deterministic hash-based selection for reproducibility
                    h = int(hashlib.md5(src.name.encode()).hexdigest(), 16)
                    if h % 1000 < cfg.caption_mix_ratio * 1000:
                        original_desc = vlm_descriptions.get(str(src), "")
                        if original_desc:
                            caption_text = original_desc
                            caption_source = "original"

            # Write plain text caption
            txt_path = images_dir / f"{unique_stem}.txt"
            with open(txt_path, "w") as f:
                f.write(caption_text)

        metadata_rows.append({
            "image": unique_name,
            "image_path": str(src),
            "bucket": row.get("bucket", ""),
            "source_type": row.get("source_type", ""),
            "caption": caption_text,
            "combined_score": row.get("combined_score", 0),
            "clip_score": row.get("clip_score", 0),
            "icr_score": row.get("icr_score", 0),
            "aod_score": row.get("aod_score", 0),
            "noun_count": row.get("noun_count", 0),
            "gate": row.get("gate", ""),
            "model": row.get("model", ""),
            "caption_source": caption_source,
        })

    # Write metadata CSV
    meta_path = export_dir / "metadata.csv"
    if metadata_rows:
        keys = metadata_rows[0].keys()
        with open(meta_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(metadata_rows)

    logger.info(f"  Exported {len(metadata_rows)} final images to {export_dir}")
    logger.info(f"  Metadata -> {meta_path}")

    # Export review images
    review = df[df["gate"] == "review"]
    if not review.empty:
        review_dir = ensure_dir(export_dir / "review")
        review_img_dir = ensure_dir(review_dir / "images")
        review_count = 0
        for _, row in review.iterrows():
            src = Path(row["image_path"])
            if not src.exists():
                continue
            ustem = f"{src.parent.name}__{src.stem}"
            dst = review_img_dir / f"{ustem}{src.suffix}"
            if not dst.exists():
                shutil.copy2(src, dst)
            # Copy caption if exists
            cap_json = work / "captions" / f"{ustem}_caption.json"
            if cap_json.exists():
                shutil.copy2(cap_json, review_dir / f"{ustem}_caption.json")
            review_count += 1
        logger.info(f"  Review images: {review_count} saved to {review_dir}")

    # Export rejected images (gate=discard + classify-rejected)
    rejected_dir = ensure_dir(export_dir / "rejected")
    rejected_img_dir = ensure_dir(rejected_dir / "images")
    rejected_count = 0

    # 1) Gate-discarded (scored but below review threshold)
    discard = df[df["gate"] == "discard"]
    for _, row in discard.iterrows():
        src = Path(row["image_path"])
        if not src.exists():
            continue
        ustem = f"{src.parent.name}__{src.stem}"
        dst = rejected_img_dir / f"{ustem}{src.suffix}"
        if not dst.exists():
            shutil.copy2(src, dst)
        rejected_count += 1

    # 2) Classify-rejected (filtered out by VLM or computational filters)
    vlm_dir = work / "vlm_results"
    if vlm_dir.exists():
        for jp in vlm_dir.glob("*.json"):
            with open(jp) as f:
                vdata = json.load(f)
            if not vdata.get("rejected", False):
                continue
            img_path = Path(vdata.get("image", ""))
            if not img_path.exists():
                continue
            ustem = f"{img_path.parent.name}__{img_path.stem}"
            dst = rejected_img_dir / f"{ustem}{img_path.suffix}"
            if not dst.exists():
                shutil.copy2(img_path, dst)
            rejected_count += 1

    logger.info(f"  Rejected images: {rejected_count} saved to {rejected_dir}")

    mark_done(cfg, "export")
    return export_dir


# ── Step 10: Report ──────────────────────────────────────────────────────────

def step_report(cfg: PipelineConfig):
    """Generate summary statistics (text + JSON)."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  STEP 10 : Report — Summary statistics")
    logger.info("=" * 70)

    import pandas as pd

    work = Path(cfg.work_dir)

    # Load gated data
    gated_path = work / "gated.csv"
    if gated_path.exists():
        df = pd.read_csv(gated_path)
    else:
        logger.warning("  No gated.csv found.")
        return

    report = []
    report.append(f"Pipeline Report — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)

    # Overall stats
    report.append(f"\nTotal scored images: {len(df)}")

    gate_counts = df["gate"].value_counts()
    for gate, count in gate_counts.items():
        report.append(f"  {gate:10s}: {count}")

    # Per source_type breakdown
    if "source_type" in df.columns:
        report.append("\nPer source_type:")
        st_counts = df.groupby("source_type")["gate"].value_counts().unstack(fill_value=0)
        report.append(st_counts.to_string())

    # Per-bucket breakdown
    if "bucket" in df.columns:
        report.append("\nPer-bucket breakdown (final only):")
        final = df[df["gate"] == "final"]
        if len(final) > 0:
            bucket_counts = final["bucket"].value_counts()
            for bucket, count in bucket_counts.items():
                report.append(f"  {bucket:30s}: {count}")

    # Score distributions
    score_cols = ["combined_score", "clip_score", "icr_score", "aod_score"]
    for col in score_cols:
        if col in df.columns:
            report.append(f"\n{col}:")
            report.append(f"  mean={df[col].mean():.4f}  std={df[col].std():.4f}  "
                          f"min={df[col].min():.4f}  max={df[col].max():.4f}")

    report_text = "\n".join(report)
    logger.info(report_text)

    report_path = work / "report.txt"
    report_path.write_text(report_text)
    logger.info(f"  Report -> {report_path}")

    # JSON report for programmatic consumption
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_scored": len(df),
        "gate_counts": df["gate"].value_counts().to_dict(),
        "score_stats": {},
        "per_bucket": {},
        "per_source_type": {},
    }

    for col in score_cols:
        if col in df.columns:
            report_data["score_stats"][col] = {
                "mean": round(df[col].mean(), 4),
                "std": round(df[col].std(), 4),
                "min": round(df[col].min(), 4),
                "max": round(df[col].max(), 4),
                "median": round(df[col].median(), 4),
            }

    if "bucket" in df.columns:
        for bucket, group in df.groupby("bucket"):
            report_data["per_bucket"][bucket] = {
                "count": len(group),
                "gate_counts": group["gate"].value_counts().to_dict(),
                "mean_combined": round(group["combined_score"].mean(), 4) if "combined_score" in group.columns else 0,
            }

    if "source_type" in df.columns:
        for st, group in df.groupby("source_type"):
            report_data["per_source_type"][st] = {
                "count": len(group),
                "gate_counts": group["gate"].value_counts().to_dict(),
                "mean_combined": round(group["combined_score"].mean(), 4) if "combined_score" in group.columns else 0,
            }

    json_report_path = work / "report.json"
    with open(json_report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"  JSON Report -> {json_report_path}")

    mark_done(cfg, "report")


# ── Pipeline Runner ──────────────────────────────────────────────────────────

STEP_FUNCS = {
    "discover":    lambda cfg, **kw: step_discover(cfg),
    "extract":     lambda cfg, **kw: step_extract(cfg),
    "dedup":       lambda cfg, **kw: step_dedup(cfg),
    "classify":    lambda cfg, **kw: step_classify(cfg),
    "tag_actors":  lambda cfg, **kw: step_tag_actors(cfg),
    "caption":     lambda cfg, **kw: step_caption(cfg),
    "score":       lambda cfg, **kw: step_score(cfg),
    "export":      lambda cfg, **kw: step_export(cfg),
    "report":      lambda cfg, **kw: step_report(cfg),
    "dedup_intra": lambda cfg, **kw: step_dedup(cfg),
    "gate":        lambda cfg, **kw: step_gate(cfg),
}


def run_pipeline(cfg: PipelineConfig, step: str | None = None,
                 from_step: str | None = None, dry_run: bool = False):
    """Run the pipeline (all steps, single step, or from a step)."""
    # Add file handler for persistent logging
    work = ensure_dir(cfg.work_dir)
    fh = logging.FileHandler(work / "pipeline.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    # Resolve step aliases
    if step:
        step = STEP_ALIASES.get(step, step)
    if from_step:
        from_step = STEP_ALIASES.get(from_step, from_step)

    if step:
        steps_to_run = [step]
    elif from_step:
        idx = STEPS.index(from_step)
        steps_to_run = STEPS[idx:]
    else:
        steps_to_run = STEPS

    logger.info(f"Master Pipeline — Steps to run: {', '.join(steps_to_run)}")
    logger.info(f"Work directory: {cfg.work_dir}")
    if cfg.force:
        logger.info(f"Force mode: ON (ignoring .done markers)")

    if dry_run:
        logger.info("[DRY RUN] Would execute these steps:")
        for s in steps_to_run:
            done = step_done(cfg, s)
            status = " (already done, will skip)" if done and not cfg.force else ""
            logger.info(f"  -> {s}{status}")
        return

    t0 = time.time()
    for s in steps_to_run:
        step_t0 = time.time()
        try:
            STEP_FUNCS[s](cfg)
        except Exception as e:
            logger.error(f"  FATAL ERROR in step '{s}': {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"  Pipeline stopped. Resume with: --from-step {s}")
            sys.exit(1)
        elapsed = time.time() - step_t0
        logger.info(f"  Step '{s}' completed in {elapsed:.1f}s")

    total = time.time() - t0
    logger.info("=" * 70)
    logger.info(f"  PIPELINE COMPLETE — {len(steps_to_run)} steps in {total:.1f}s")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Master Pipeline — Indic Cultural Image Dataset")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    all_step_names = STEPS + list(STEP_ALIASES.keys())
    parser.add_argument("--step", choices=all_step_names,
                        help="Run only this step")
    parser.add_argument("--from-step", choices=all_step_names,
                        help="Run from this step onwards")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print steps without executing")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Override work directory")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Override GPU IDs (comma-separated)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-processing (ignore .done markers and existing results)")
    parser.add_argument("--movie-list", type=str, default=None,
                        help="CSV file with columns: path, source_type")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local VLM model path (e.g. /data/kl_dev/models/Qwen3-VL-32B-Instruct)")
    parser.add_argument("--backend", type=str, default="transformers",
                        choices=["transformers", "vllm"],
                        help="VLM inference backend (default: transformers)")

    # Streaming mode (8-GPU concurrent pipeline)
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming concurrent pipeline (8-GPU mode)")
    parser.add_argument("--vllm-gpus", type=str, default=None,
                        help="GPUs for vLLM server (comma-separated, default: 0,1,2,3)")
    parser.add_argument("--actor-gpu", type=int, default=None,
                        help="GPU for actor tagging (default: 4)")
    parser.add_argument("--clip-gpu", type=int, default=None,
                        help="GPU for CLIP scoring (default: 5)")
    parser.add_argument("--gdino-gpu", type=int, default=None,
                        help="GPU for GroundingDINO scoring (default: 6)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume streaming pipeline from checkpoint")

    # Simple input mode (alternative to config file)
    parser.add_argument("--input", type=str, default=None,
                        help="Input folder containing videos and/or images (simple mode)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output folder (default: {input}_output)")

    args = parser.parse_args()

    cfg = load_config(args.config)

    # Simple input mode: --input /path/to/data --output /path/to/output
    if args.input:
        from pathlib import Path
        input_dir = Path(args.input)
        if not input_dir.exists():
            print(f"Error: input directory {input_dir} does not exist")
            sys.exit(1)

        # Auto-detect videos and images in the input folder
        video_exts = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv"}
        image_exts = {".jpg", ".jpeg", ".png", ".webp"}

        # Scan for videos
        videos = [p for p in input_dir.rglob("*") if p.suffix.lower() in video_exts]
        # Scan for image directories (any folder containing images)
        image_dirs = set()
        for p in input_dir.rglob("*"):
            if p.suffix.lower() in image_exts:
                image_dirs.add(str(p.parent))

        # Set config
        cfg.video_dirs = []
        cfg.extra_image_dirs = [{"path": d, "source_type": "internal"} for d in sorted(image_dirs)]

        # Create movie list if videos found
        if videos:
            import tempfile, csv
            ml = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
            writer = csv.writer(ml)
            writer.writerow(["path", "source_type"])
            for v in videos:
                writer.writerow([str(v), "youtube"])
            ml.close()
            cfg.movie_list = ml.name

        # Set output
        if args.output:
            cfg.work_dir = args.output
        else:
            cfg.work_dir = str(input_dir) + "_output"

        cfg.streaming = True
        logger.info(f"Simple input mode: {len(videos)} videos, {len(image_dirs)} image dirs")
        logger.info(f"Output: {cfg.work_dir}")

    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.gpus:
        cfg.gpu_ids = [int(g) for g in args.gpus.split(",")]
    if args.force:
        cfg.force = True
    if args.movie_list:
        cfg.movie_list = args.movie_list
    if args.model_path:
        cfg.model_path = args.model_path
    if args.backend:
        cfg.backend = args.backend

    # Streaming mode overrides
    if args.streaming:
        cfg.streaming = True
    if args.vllm_gpus:
        cfg.vllm_gpu_ids = [int(g) for g in args.vllm_gpus.split(",")]
    if args.actor_gpu is not None:
        cfg.actor_tag_gpu_id = args.actor_gpu
    if args.clip_gpu is not None:
        cfg.clip_gpu_id = args.clip_gpu
    if args.gdino_gpu is not None:
        cfg.gdino_gpu_id = args.gdino_gpu

    if cfg.streaming:
        import asyncio
        from stream_orchestrator import StreamingPipeline
        asyncio.run(StreamingPipeline(cfg).run())
    else:
        run_pipeline(cfg, step=args.step, from_step=args.from_step, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
