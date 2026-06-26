# Indic Cultural Image Dataset Pipeline

9-step GPU-accelerated pipeline that transforms raw YouTube videos and image datasets into a training-ready dataset for text-to-image (T2I) model fine-tuning on Indian cultural content.

**What it does:** Extracts frames from Bollywood movies, removes watermarks with LaMA inpainting, classifies into 12 cultural buckets using Qwen2.5-VL-32B, identifies Indian film actors by face recognition, generates rich structured captions with cultural terminology (biryani, lehenga, Kanjeevaram saree), scores quality, and exports gated images with metadata.

**Performance:** ~1.7 images/sec on 8× H100 GPUs. 256 images processed end-to-end in ~2.5 minutes (excluding one-time model load).

---

## Pipeline Flow

```
INPUT: YouTube videos + Image datasets
    │
    ▼
 1. DISCOVER     Scan video/image directories → manifest.json
    │
    ▼
 2. EXTRACT      PySceneDetect → 3 frames/scene → frames/
    │
    ▼
 3. DEDUP        pHash BK-tree (intra ≤ 8, cross ≤ 6) → removes near-duplicates
    │
    ▼
 4. WATERMARK    YOLO detect → LaMA inpaint → verify (GPU 7)
    +             Also: auto-crop letterbox black borders
    │
    ▼
 5. CLASSIFY     Hybrid: computational filters (blur/dark/text)
    │             + VLM 12-bucket classification + has_watermark check
    │             (GPUs 0-3 via vLLM)
    │
    ▼
 6. TAG ACTORS   YOLO face detect + InsightFace recognition (GPU 4)
    │             Only on people_portraits bucket
    │
    ▼
 7. CAPTION      Bucket-specific prompts + actor name injection
    │             + cultural label injection from paired .txt files
    │             (GPUs 0-3 via vLLM)
    │
    ▼
 8. SCORE        CLIP image-text alignment (GPU 5+6) + AOD richness (CPU)
    │
    ▼
 9. EXPORT       Gate threshold → copy images + captions + metadata.csv
    +             Report + interactive HTML dashboard
```

---

## GPU Allocation (8× H100)

| GPU | Model | Pipeline Step |
|-----|-------|---------------|
| 0-3 | Qwen2.5-VL-32B via vLLM (TP=4) | Classify + Caption |
| 4 | YOLO v12n-face + InsightFace buffalo_l | Actor Tagging |
| 5 | CLIP ViT-L/14 (scorer #1) | Quality Scoring |
| 6 | CLIP ViT-L/14 (scorer #2) | Quality Scoring (parallel) |
| 7 | YOLO watermark + LaMA inpainter | Watermark Detection + Removal |

All 8 GPUs active. Configurable via `pipeline_config.yaml`.

---

## 12 Cultural Buckets

| # | Bucket | What it captures |
|---|--------|-----------------|
| 1 | `people_portraits` | Individuals or groups, faces — **actor tagging runs here** |
| 2 | `clothing_textiles` | Traditional garments, jewelry, fabric details |
| 3 | `architecture` | Buildings, temples, monuments, interiors |
| 4 | `landscape_nature` | Natural scenery, gardens, rivers, mountains |
| 5 | `urban_street` | City streets, markets, shops, modern infrastructure |
| 6 | `rural_village` | Village scenes, farming, rural homes |
| 7 | `food_drink` | Food, cooking, beverages, street food stalls |
| 8 | `festivals_rituals` | Ceremonies, celebrations, rituals, processions |
| 9 | `objects_artifacts` | Handicrafts, pottery, tools, musical instruments |
| 10 | `animals_wildlife` | Animals, birds, livestock, pets |
| 11 | `art_design` | Paintings, murals, rangoli, mehndi, sculptures |
| 12 | `abstract_texture` | Patterns, textures, geometric designs |

---

## Quick Start

### Full pipeline (streaming mode, 8 GPUs)

```bash
python3 pipeline.py --config pipeline_config.yaml --streaming
```

### With a movie list CSV

```bash
# movies.csv format:
# path,source_type
# /data/videos/english_vinglish.mp4,youtube
# /data/images/cultural_photos/,internal

python3 pipeline.py \
    --config pipeline_config.yaml \
    --movie-list movies.csv \
    --streaming
```

### Run a single step

```bash
python3 pipeline.py --config pipeline_config.yaml --step classify
python3 pipeline.py --config pipeline_config.yaml --step caption
```

### Resume from a step

```bash
python3 pipeline.py --config pipeline_config.yaml --from-step caption --streaming
```

### Force re-run (ignore .done markers)

```bash
python3 pipeline.py --config pipeline_config.yaml --force --streaming
```

### Inspect results

```bash
python3 inspect_pipeline.py /data/kl_dev/runs/run_01
# Opens interactive HTML at run_01/inspector.html
```

---

## Configuration

All settings are in `pipeline_config.yaml`. Key sections:

### GPU Allocation

```yaml
# 8× H100 (default)
vllm_gpu_ids: [0, 1, 2, 3]    # TP=4 for Qwen2.5-VL-32B
actor_tag_gpu_id: 4
clip_gpu_id: 5
gdino_gpu_id: 6                # CLIP #2 when icr_weight=0
watermark_gpu_id: 7

# 4× GPU (smaller setup)
vllm_gpu_ids: [0, 1]           # TP=2
actor_tag_gpu_id: 2
clip_gpu_id: 3
gdino_gpu_id: 3
watermark_gpu_id: 3

# 2× GPU (minimal)
vllm_gpu_ids: [0]              # TP=1
actor_tag_gpu_id: 1
clip_gpu_id: 1
watermark_gpu_id: 1
```

### Quality Scoring

```yaml
clip_weight: 0.55              # CLIP image-text alignment
icr_weight: 0.0                # GroundingDINO (disabled — can't ground Indian terms)
aod_weight: 0.45               # Adjective richness per noun
gate_final: 0.25               # Score >= this → exported
gate_review: 0.15              # Score >= this → review tier
```

### Watermark Handling

```yaml
watermark_enabled: true
watermark_detect_threshold: 0.45   # YOLO confidence cutoff
watermark_reject_threshold: 0.9    # Too heavy to remove
crop_letterbox: true               # Auto-crop black borders
```

### Deduplication

```yaml
phash_intra_threshold: 8       # Hamming distance within same video
phash_cross_threshold: 6       # Hamming distance across sources
```

---

## Directory Structure

```
master/
├── pipeline.py                  Main orchestrator (sequential + CLI)
├── stream_orchestrator.py       Async streaming orchestrator (8-GPU concurrent)
├── config.py                    PipelineConfig, BK-tree, shared utilities
├── pipeline_config.yaml         All-in-one configuration file
│
├── classifier.py                VLM classification (11 filters + 12 buckets)
├── captioner.py                 Structured captioning with bucket prompts
├── actor_tagger.py              YOLO face + InsightFace actor recognition
├── scorer.py                    CLIP + AOD quality scoring
├── frame_extractor.py           PySceneDetect frame extraction
├── image_filters.py             Computational filters (blur/dark/text)
├── watermark_handler.py         Watermark detection + LaMA removal
├── run_watermark.py             Watermark subprocess (isolated CUDA)
│
├── vllm_server.py               vLLM server lifecycle manager
├── vllm_client.py               Async HTTP client for vLLM
├── vlm_backend.py               VLM backend abstraction (transformers/vLLM)
├── gpu_workers.py               GPU worker processes (CLIP, GDINO, actors)
│
├── schemas.py                   JSON schemas for vLLM guided decoding
├── common.py                    Constants + utilities
├── dashboard.py                 HTML dashboard generator
├── inspect_pipeline.py          Interactive pipeline inspector
├── README.md
│
├── prompts/                     12 bucket-specific captioning prompts
│   ├── bucket_01_people_portraits.txt
│   ├── bucket_02_clothing_textiles.txt
│   ├── ...
│   └── bucket_12_abstract_texture_pattern.txt
│
├── actors/                      Face recognition models + embeddings
│   ├── actor_images/            Reference photos (12 actors, ~50 each)
│   ├── actor_embeddings/        Pre-built .pkl embeddings
│   ├── joy_caption_watermark.pt YOLO watermark detector (110MB)
│   └── yolov12n-face.pt         YOLO face detector (5.3MB)
│
├── groundingdino/               GroundingDINO weights (for ICR scoring)
└── models -> /data/kl_dev/models/   Symlink to VLM weights
```

---

## Requirements

### Core

```bash
pip install torch transformers accelerate pillow tqdm imagehash \
    scenedetect[opencv] spacy pandas open_clip_torch qwen-vl-utils httpx
python -m spacy download en_core_web_sm
```

### Actor Tagging

```bash
pip install ultralytics insightface onnxruntime-gpu
```

### Watermark Removal

```bash
pip install simple-lama-inpainting
```

### vLLM (streaming mode)

```bash
pip install vllm>=0.18.0
```

### Optional — Flash Attention 2

```bash
pip install flash-attn --no-build-isolation
```

---

## Key Design Decisions

### Hybrid Classification
The VLM is bad at pixel-level quality assessment (blur, darkness — only 40-57% accuracy per Q-Bench). So we use **computational filters first** (saliency-aware blur detection, brightness check, text detection) and only send clean images to the VLM for semantic classification + bucketing. This saves ~30% of VLM calls.

### Double Watermark Defense
1. **YOLO (joycaption model)** detects watermarks → **LaMA inpaints** them out → **verifies** removal
2. **VLM `has_watermark` filter** catches what YOLO missed (YouTube overlays, creator text, small copyright)

Result: no watermarked images reach the final export.

### Subject-Aware Blur Detection
Full-frame Laplacian variance fails on cinematic frames with intentional bokeh (shallow DOF). We use **OpenCV saliency + grid Laplacian** — measures sharpness on the detected subject, not the whole frame. 91% accuracy on Bollywood content: keeps artistic DOF, rejects motion blur.

### Cultural Label Injection
Paired `.txt` files (≤ 100 chars) are injected as hard facts:
```
"IMPORTANT: This image contains: biryani. You MUST use this exact name."
```
Prevents the VLM from hallucinating Western food names for Indian dishes.

### ICR Removed
GroundingDINO can't ground Indian cultural terms (biryani, lehenga, rangoli). ICR weight set to 0. GPU 6 repurposed as a second CLIP worker for 2x scoring speed.

---

## Output Layout

```
{work_dir}/
├── manifest.json              Source inventory
├── frames/                    Extracted + deduped frames
├── watermark_results/         Per-image watermark sidecar JSONs
├── watermark_quarantine/      Failed watermark removals
├── vlm_results/               Per-image classification JSONs
├── actor_tags/                Per-image actor recognition JSONs
├── captions/                  Per-image structured caption JSONs
├── scores.csv                 All quality scores
├── gated.csv                  Scores + gate tier (final/review/discard)
├── export/
│   ├── images/                Final images (collision-safe names)
│   ├── captions/              Caption JSONs + plain text
│   └── metadata.csv           13-column metadata
├── report.txt                 Summary statistics
├── report.json                Machine-readable report
├── dashboard.html             Interactive web dashboard
├── inspector.html             Pipeline phase inspector
└── pipeline.log               Full run log
```

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | None | Path to YAML config file |
| `--streaming` | false | Use async streaming pipeline (recommended) |
| `--step` | None | Run only this step |
| `--from-step` | None | Resume from this step |
| `--work-dir` | config | Override output directory |
| `--gpus` | config | Override GPU IDs (comma-separated) |
| `--model-path` | config | Override VLM model path |
| `--force` | false | Ignore .done markers, reprocess everything |
| `--movie-list` | None | CSV with columns: path, source_type |
| `--dry-run` | false | Print steps without executing |

---

## Performance

Tested on 8× H100 (80GB each):

| Input | Images | Time | Rate |
|-------|--------|------|------|
| 30s video + 20 images | 33 | 2.3 min | 0.2 img/s |
| 2min video + 60 images | 116 | 2.1 min* | 1.7 img/s |
| 5min video + 120 images | 256 | 3.8 min* | 1.7 img/s |

*Excluding one-time vLLM startup (~80s)

### Bottleneck: VLM captioning (345ms/image)
- Caption generation is the slowest step (300 tokens per image)
- Scales linearly with more vLLM instances or nodes

### 10M Image Projection

| Setup | Throughput | Time |
|-------|-----------|------|
| 1 node, TP=4 | ~5 img/s | ~23 days |
| 4 nodes | ~20 img/s | ~6 days |
| 8 nodes | ~40 img/s | ~3 days |
