# Indic Video Dataset Pipeline

A 12-service pipeline that takes Indic-language movies and produces a curated, captioned, quality-scored video clip dataset — organized by thematic buckets with actor recognition, structured captions, and multi-factor quality gating.

## How It Works

A movie enters the pipeline and flows sequentially through 12 services. Each service reads and updates a shared per-movie `metadata.jsonl` file (protected by file locks). Idempotency markers (`.done_s1` ... `.done_s12`) allow resuming from any point without reprocessing.

```
movie.mp4
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  S1  Extract            Scene detect → virtual clips + phash + crop │
│  S2  Dedup / Filter     Dedup (BK-tree) + motion + DOVER filter     │
│  S3  Band Removal       Flag letterbox/pillarbox (from S1 crop)     │
│  S4  Watermark          Detect corner watermarks, store bbox        │
│  S5  Classify           Bucket classification (Qwen2.5-VL)         │
│  S6  Verify             Route assignment + optional Gemma verify    │
│  S7  Actor Tagging      YOLO face + InsightFace actor matching      │
│  S8  Caption            Structured captions (Gemma / Qwen video)    │
│  S9  Quality Scoring    CLIP + DOVER + motion + bucket + caption    │
│  S10 Gate               Accept / Review / Discard by score          │
│  S11 Export             Package clips + captions + manifests        │
│  S12 Report             Aggregate stats + summary reports           │
└──────────────────────────────────────────────────────────────────────┘
  │
  ▼
export/
  captions.jsonl          Per-clip captions + metadata
  {movie}_captions.csv    Spreadsheet-friendly export
  by_bucket/              Per-bucket clip folders + manifests
reports/
  {video_id}_report.json  Aggregate stats
```

---

## Service Details

### S1 — Extract

Splits the input movie into virtual clips using **PySceneDetect** (ContentDetector, threshold=27). Each scene is subdivided into fixed-length clips (default 5 seconds). For every clip:

- Detects and records crop box (letterbox/pillarbox) via FFmpeg cropdetect
- Computes a perceptual hash (pHash) of the middle frame for dedup
- Assigns a unique `clip_id` (`{video_id}_{index:06d}`)

**Output fields:** `video_id`, `clip_id`, `scene_id`, `timestamp_start`, `timestamp_end`, `duration`, `phash`, `crop_box`

### S2 — Dedup / Filter

Three-stage filtering:

1. **Deduplication** — BK-tree on perceptual hashes (hamming distance ≤ 8). Duplicates are moved to `dups/`.
2. **Motion analysis** — UniMatch optical flow (70% weight) + VMAF temporal diff (30% weight). Static and excessive-motion clips are rejected with per-source percentile thresholds.
3. **DOVER quality** — Aesthetic + technical video quality scoring. Clips below `dover_score < 0.60` are rejected.

Supports Ray GPU actors for parallel motion and DOVER scoring.

**Output fields:** `keep`, `dup_of`, `motion_score`, `unimatch_motion`, `vmaf_motion`, `dover_score`, `aesthetic_score`, `technical_score`, `s2_reject_reason`

### S3 — Band Removal

Marks clips that have a crop box (detected in S1) with `band_removed=True`. No re-encoding — the crop is applied later during export.

### S4 — Watermark Detection

Movie-level watermark detection by analyzing corner regions across sampled frames:

- Divides frames into 4 corner patches (h/5 x w/5)
- Votes on corner activity (std > 40 threshold)
- Stores watermark bbox for FFmpeg delogo during export

Results cached in `movie_watermark.json`.

**Output fields:** `watermark` dict with `present`, `corner`, `bbox`

### S5 — Classify

Assigns each clip to one of **12 thematic buckets** using **Qwen2.5-VL** (7B or 32B):

| Bucket | Category |
|--------|----------|
| bucket_01 | People & Portraits |
| bucket_02 | Clothing & Textiles |
| bucket_03 | Architecture & Built Environment |
| bucket_04 | Landscape & Nature |
| bucket_05 | Urban & Street Life |
| bucket_06 | Rural & Village Life |
| bucket_07 | Food & Drink |
| bucket_08 | Festivals, Rituals & Events |
| bucket_09 | Objects & Artifacts |
| bucket_10 | Animals & Wildlife |
| bucket_11 | Art, Design & Creative |
| bucket_12 | Abstract, Texture & Pattern |

Multi-frame voting (1-3 frames per clip). Supports transformers backend (single/multi-GPU via Ray) or vLLM batched inference.

**Output fields:** `bucket`, `bucket_confidence`, `reject`, `reject_reason`

### S6 — Verify

Two modes:

- **bucket_route** (default, fast): Derives route directly from S5 bucket — clips with people-related buckets get route `"people"`, others get `"other"`. No model loaded.
- **verify** mode: Loads Gemma-3-4B-IT for a second-pass verification of bucket assignment.

**Output fields:** `verified`, `confidence`, `route`, `bucket_verified`

### S7 — Actor Tagging

3-frame face detection and actor identification:

1. Extracts frames (at 20%, 50%, 80% of clip duration)
2. Runs **YOLOv12n-face** for face detection
3. Matches against **108 pre-computed actor embeddings** via **InsightFace Buffalo-L** (cosine similarity)
4. Records per-frame actor names, bounding boxes, and screen positions

Clips on the `"people"` route get full tagging; others are marked `"not_applicable"`.

**Output fields:** `actor_status`, `actors`, `clip_actors`, `actors_f1/f2/f3`, `pos_f1/f2/f3`

### S8 — Caption

Generates structured prose captions per clip. Four backends:

| Backend | Model | Input | Use Case |
|---------|-------|-------|----------|
| `gemma` | Gemma-3-4B / Gemma-4-31B | 3 frames | Default structured |
| `qwen` | Qwen-VL-32B | 3 keyframes | Legacy |
| `vllm` | Qwen via vLLM engine | multi-frame batch | High throughput |
| `qwen_video` | Qwen2.5-VL-7B | native MP4 | Video-native |

Captions use **bucket-specific prompt packs** (12 prompt templates) that guide the model to describe Indian cultural details — attire (saree, kurta, dhoti), architecture, food names, etc. Actor names from S7 are injected into prompts and enforced in output text.

**Output fields:** `caption`, `caption_struct`, `generated_caption`, `prompt_version`

### S9 — Quality Scoring

Composite multi-factor quality score:

```
final_score = 0.25 * clip_score      (CLIP image-text similarity)
            + 0.30 * dover_score     (aesthetic + technical quality)
            + 0.20 * motion_score    (normalized from S2)
            + 0.15 * bucket_semantic (bucket confidence if verified)
            + 0.10 * caption_present (1.0 if caption exists, else 0.0)
```

CLIP scoring samples 5 frames per clip (at 10%, 30%, 50%, 70%, 90%) and measures cosine similarity between frame embeddings and caption text. Weights are configurable in `thresholds.yaml`.

**Output fields:** `clip_score`, `icr`, `aod`, `final_score`

### S10 — Gate

Applies quality thresholds to assign a verdict:

| Condition | Verdict |
|-----------|---------|
| `final_score < 0.10` or `keep=False` or `reject=True` | **DISCARD** |
| `final_score < 0.18` | **REVIEW** |
| `final_score >= 0.18` | **FINAL** |

**Output fields:** `verdict` (DISCARD / REVIEW / FINAL)

### S11 — Export

Packages accepted clips (FINAL + REVIEW by default):

- `export/captions.jsonl` — per-clip JSON with clip_id, caption, bucket, verdict, score, actors
- `export/{movie}_captions.csv` — spreadsheet with all fields
- `export/by_bucket/{bucket}/clips/` — clip MP4s organized by bucket
- `export/by_bucket/{bucket}/manifest.jsonl` — per-bucket manifests
- `export/bucket_index.json` — bucket summary with counts and paths

Clip MP4 export applies crop + delogo (watermark removal) via FFmpeg filters.

### S12 — Report

Generates summary statistics:

- Clip counts by verdict (FINAL / REVIEW / DISCARD)
- Bucket distribution
- Actor distribution
- Score statistics (min, max, mean)
- Total export duration
- Runtime per service

Output: `{video_id}_report.txt` (human-readable) + `{video_id}_report.json` (structured)

---

## Models Used

| Stage | Model | Purpose |
|-------|-------|---------|
| S1 | PySceneDetect (ContentDetector) | Scene boundary detection |
| S2 | UniMatch | Optical flow motion scoring |
| S2 | VMAF | Temporal difference motion |
| S2 | DOVER | Video quality (aesthetic + technical) |
| S5 | Qwen2.5-VL-7B / 32B | 12-bucket classification |
| S6 | Gemma-3-4B-IT (optional) | Bucket verification |
| S7 | YOLOv12n-face | Face detection |
| S7 | InsightFace Buffalo-L | Face embedding + actor matching |
| S8 | Gemma-3-4B / Gemma-4-31B / Qwen2.5-VL | Caption generation |
| S9 | CLIP ViT-B-32 | Image-text similarity scoring |

All model endpoints and paths are configured in `configs/models.yaml`. GPU assignments are per-service in `configs/pipeline*.yaml`.

---

## Configuration

Three YAML files control everything — nothing is hardcoded:

| File | Controls |
|------|----------|
| `configs/pipeline.yaml` | Output paths, service order, per-service config (GPU IDs, backends, batch sizes), Ray settings, prompt pack path |
| `configs/models.yaml` | Model paths, families, GPU IDs, batch sizes, max tokens |
| `configs/thresholds.yaml` | Scene detection threshold, clip length, dedup hamming distance, motion weights/bounds, DOVER floor, quality score weights, gate thresholds, watermark detection params |

**GPU-scaled presets:**
- `pipeline_v3_1gpu.yaml` — single GPU
- `pipeline_v3_2gpu.yaml` — 2 GPUs
- `pipeline_v3_8gpu.yaml` — 8 GPUs (full parallelism)
- `pipeline_v3_vllm_*.yaml` — vLLM-backed variants

---

## Usage

### Process a single movie

```bash
python run_pipeline.py --movie movie.mp4 --video-id my_movie
```

### Resume from a specific step

```bash
python run_pipeline.py --movie movie.mp4 --video-id my_movie --from-step s5
```

### Run up to a specific step

```bash
python run_pipeline.py --movie movie.mp4 --video-id my_movie --to-step s7
```

### Force full rerun (ignore .done markers)

```bash
python run_pipeline.py --movie movie.mp4 --video-id my_movie --force
```

### Use a GPU-scaled config

```bash
python run_pipeline.py \
    --movie movie.mp4 \
    --video-id my_movie \
    --pipeline-yaml configs/pipeline_v3_8gpu.yaml \
    --force
```

### Batch processing (multi-movie)

```bash
python run_pipeline.py \
    --movie-registry data/movie_registry.csv \
    --workers 8
```

### Test on a short segment

```bash
bash scripts/make_test_segment.sh /path/to/movie.mp4 100 130   # 30 min segment
python run_pipeline.py \
    --movie test_segments/movie_100_130.mp4 \
    --video-id test_run \
    --time-offset 6000 \
    --max-clips 50 \
    --force
```

---

## Project Structure

```
Indic_video_pipeline/
├── configs/
│   ├── models.yaml                 # Model paths, GPU IDs, batch sizes
│   ├── thresholds.yaml             # All scoring/gating thresholds
│   ├── pipeline.yaml               # Default pipeline config
│   └── pipeline_v3_*.yaml          # GPU-scaled presets (1/2/8 GPU, vLLM)
├── common/
│   ├── base_service.py             # Abstract base class (all services inherit)
│   ├── service_registry.py         # Dynamic service loading (s1→s12)
│   ├── metadata_manager.py         # JSONL metadata read/write
│   ├── metadata_lock.py            # File-lock for concurrent metadata access
│   ├── runtime_tracker.py          # Per-service timing + CSV summary
│   ├── prompt_manager.py           # Bucket prompt pack loader (zip or dir)
│   ├── bucket_prompts.py           # Per-bucket caption guidance text
│   ├── ffmpeg_utils.py             # Duration, crop detection
│   ├── clip_io.py                  # Frame extraction + clip MP4 export
│   ├── frame_sampler.py            # Keyframe sampling at fractional offsets
│   ├── paths.py                    # Path resolution (models, outputs, logs)
│   ├── qwen_classify.py            # Qwen2.5-VL classification worker
│   ├── qwen_vllm.py                # Shared vLLM engine (batched inference)
│   ├── qwen_video_caption.py       # Qwen native MP4 video captioning
│   ├── vllm_classify.py            # vLLM-batched classification
│   ├── vllm_caption.py             # vLLM-batched captioning
│   ├── gemma_verify.py             # Gemma bucket verification
│   ├── gemma_caption.py            # Gemma caption (structured/prose)
│   ├── vlm_service.py              # Qwen-VL-32B classify + caption service
│   ├── vlm_ray_actors.py           # Ray remote GPU actors (Qwen, Gemma)
│   ├── ray_pool.py                 # Ray init, parallel_map, shutdown
│   ├── gpu_actor_pool.py           # Multi-GPU actor dispatch
│   ├── motion_filter.py            # UniMatch + VMAF score fusion
│   ├── motion_unimatch.py          # UniMatch optical flow
│   ├── motion_vmaf.py              # VMAF temporal diff
│   ├── watermark_vf.py             # FFmpeg delogo filter
│   ├── caption_models.py           # Model selection catalog
│   ├── caption_text.py             # Caption normalization (JSON→prose)
│   ├── actor_caption.py            # Actor name injection into captions
│   ├── master_bridge.py            # External Master_Pipeline integration
│   ├── dedup_bktree.py             # BK-tree for perceptual hash dedup
│   ├── dover_client.py             # DOVER quality model (in model_clients/)
│   └── ...
├── model_clients/
│   ├── qwen_client.py              # Qwen3-VL via vLLM API
│   ├── gemma_client.py             # Gemma verification (stub)
│   ├── lama_client.py              # LaMa inpainting (stub)
│   ├── clip_client.py              # CLIP ViT-B-32 similarity scoring
│   ├── face_client.py              # Face detection (stub)
│   └── dover_client.py             # DOVER video quality scoring
├── services/
│   ├── service_01_extract/         # Scene detect + virtual clips
│   ├── service_02_dedup/           # Dedup + motion + DOVER filter
│   ├── service_03_band_removal/    # Crop flag
│   ├── service_04_watermark/       # Corner watermark detection
│   ├── service_05_classify/        # Qwen bucket classification
│   ├── service_06_verify/          # Route + optional Gemma verify
│   ├── service_07_actor_tagging/   # YOLO face + InsightFace actors
│   ├── service_08_caption/         # Multi-backend captioning
│   ├── service_09_quality_scoring/ # CLIP+DOVER+motion composite score
│   ├── service_10_gate/            # Accept/Review/Discard verdict
│   ├── service_11_export/          # Clip MP4s + captions + manifests
│   └── service_12_report/          # Summary statistics
├── prompts/                        # 12 bucket-specific caption prompts
├── scripts/                        # Setup, run, and review scripts
├── run_pipeline.py                 # Main entry point / orchestrator
├── environment.yml                 # Conda environment (Python 3.10)
└── requirements.txt                # pip dependencies
```

---

## Directory Layout (Runtime)

```
/mnt/data0/harsha/new_dataset_pipeline/
├── Indic_video_pipeline/                  # Source code (this repo)
├── models/                                # Shared model weights
│   ├── Qwen2.5-VL-7B-Instruct/
│   ├── Qwen2.5-VL-32B-Instruct/
│   └── yolov12n-face.pt
├── Master_Pipeline_t2i_dataset/           # Actor tagger + 108 actor embeddings
│   └── actors/actor_embeddings/*.pkl
└── pipeline_outputs/                      # (or v3_outputs/ for per-movie layout)
    ├── workspaces/{video_id}/
    │   ├── metadata.jsonl                 # All clip records
    │   ├── clips/                         # Extracted clip MP4s
    │   ├── actor_tags/                    # Per-clip actor detection
    │   ├── actor_frames/                  # Extracted face frames
    │   └── export/
    │       ├── captions.jsonl
    │       ├── {movie}_captions.csv
    │       ├── bucket_index.json
    │       └── by_bucket/{bucket}/
    │           ├── clips/*.mp4
    │           └── manifest.jsonl
    ├── logs/s1/ ... s12/                  # Per-service runtime logs
    └── reports/
        ├── {video_id}_report.json
        ├── {video_id}_report.txt
        └── runtime_summary.csv
```

---

## Environment Setup

```bash
conda create -n indic_video_pipeline python=3.10
conda activate indic_video_pipeline
pip install -r requirements.txt
bash scripts/fix_cuda_torch.sh        # Fix CUDA/PyTorch if needed
bash scripts/setup_all_models.sh      # Download Qwen + YOLO models
bash scripts/verify_setup.sh          # Validate everything is ready
```

---

## Parallelization

Services S2, S5, S6, S7, S8, and S9 support **Ray-based multi-GPU parallelism**:

| Service | Ray Actor | Parallelizable Work |
|---------|-----------|-------------------|
| S2 | MotionScoreActor, DoverScoreActor | Motion scoring, DOVER quality |
| S5 | QwenClassifyActor | Bucket classification |
| S6 | GemmaVerifyActor | Bucket verification |
| S7 | ActorTagActor | Face detection + matching |
| S8 | QwenVideoCaptionActor | Caption generation |
| S9 | ClipScoreActor | CLIP scoring |

GPU assignments and worker counts are configured per pipeline YAML preset. The 8-GPU config runs all heavy stages in parallel across GPUs.
