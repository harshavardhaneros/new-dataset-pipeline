# Indic Video Pipeline — Cursor Implementation Plan

## IMPORTANT

The architecture document provided by Harsha is the ONLY source of truth.

Follow it exactly.

Do NOT:

* change service order
* merge services
* remove services
* redesign metadata
* replace per-clip metadata.json
* replace physical clips with virtual clips
* change model choices
* change scoring formula
* change routing logic

The architecture specification already defines:

* Service responsibilities
* Inputs
* Outputs
* Metadata schema
* Models
* Routing
* Thresholds
* Storage architecture

Implement it exactly.

---

# Project Root

Everything must be implemented under:

```text
/mnt/data0/harsha/new_dataset_pipeline/Indic_video_pipeline
```

---

# Development Goals

## Today

Build all 12 services.

End-to-end execution must work on one movie.

Pipeline should process:

```bash
python run_pipeline.py --movie movie.mp4
```

through all services.

Actor tagging may be stubbed today.

---

## Tomorrow

Run many movies.

Example:

```bash
python run_pipeline.py \
    --movie-registry data/movie_registry.csv \
    --workers 8
```

---

# Conda Environment

Create one environment for all services:

```bash
conda create -n indic_video_pipeline python=3.10
```

Generate:

```text
environment.yml
requirements.txt
```

---

# Repository Structure

```text
Indic_video_pipeline/

configs/

    models.yaml
    thresholds.yaml
    pipeline.yaml

common/

    base_service.py

    metadata_manager.py
    metadata_lock.py

    runtime_tracker.py

    prompt_manager.py

    ffmpeg_utils.py

    service_registry.py

model_clients/

    qwen_client.py
    gemma_client.py
    lama_client.py
    clip_client.py
    face_client.py

services/

    service_01_extract/
    service_02_dedup/
    service_03_band_removal/
    service_04_watermark_removal/
    service_05_classify/
    service_06_verify/
    service_07_actor_tagging/
    service_08_caption/
    service_09_quality_scoring/
    service_10_gate/
    service_11_export/
    service_12_report/

data/

    movie_registry.csv

reports/
logs/

run_pipeline.py

README.md
```

---

# Base Service

Create:

```text
common/base_service.py
```

Every service must inherit from:

```python
BaseService
```

Required methods:

```python
process()
load_metadata()
save_metadata()
write_runtime()
```

This is for maintainability only.

Do not change architecture.

---

# Runtime Logging (MANDATORY)

Every service must write runtime logs.

Structure:

```text
logs/

    s1/
    s2/
    ...
    s12/
```

Example:

```text
logs/s5/ABCD_runtime.json
```

Example content:

```json
{
  "service": "s5_classify",
  "movie": "ABCD.mp4",
  "runtime_seconds": 82.1,
  "clips_processed": 1848,
  "status": "success",
  "errors": 0
}
```

---

# Runtime Tracker

Create:

```text
common/runtime_tracker.py
```

Functions:

```python
start_timer()
stop_timer()
write_runtime_log()
```

Every service must use same runtime utility.

---

# Runtime Summary

Generate:

```text
reports/runtime_summary.csv
```

Columns:

```text
movie,
s1,
s2,
s3,
s4,
s5,
s6,
s7,
s8,
s9,
s10,
s11,
s12,
total_runtime
```

---

# Metadata Locking

Before updating metadata:

```text
metadata.lock
```

must be created.

After update:

remove lock.

Required for future multi-worker execution.

---

# Idempotency

Architecture already specifies:

```text
.done_s1
.done_s2
...
.done_s12
```

Implement exactly.

Support:

```bash
--from-step
```

and

```bash
--force
```

---

# Service Registry

Create:

```text
common/service_registry.py
```

Dynamic registration of services.

No hardcoded orchestration.

---

# Prompt Pack Integration (S8)

IMPORTANT

Do NOT hardcode prompts.

Harsha will provide:

```text
prompt_pack.zip
```

Path will be supplied later.

Create:

```text
common/prompt_manager.py
```

Responsibilities:

* extract zip
* validate prompts
* build prompt registry
* bucket → prompt mapping

Prompt pack path must come from:

```yaml
configs/pipeline.yaml
```

Example:

```yaml
prompt_pack:
  zip_path: /path/to/prompt_pack.zip
```

No hardcoded paths.

---

# Actor Tagging

Today:

Implement service interface and metadata wiring.

Output may be placeholder.

Tomorrow:

Integrate existing actor tagging pipeline.

Existing stack:

* YOLO Face
* InsightFace Buffalo-L
* Actor embedding database
* Cosine similarity matching

Do not redesign actor recognition.

Wrap existing implementation.

---

# Models

Use architecture-defined model services exactly:

## Classification

```text
Qwen3-VL-32B
```

## Verification

```text
Gemma
```

## Caption

```text
Qwen3-VL-32B
```

## Watermark

```text
LaMa
```

## Face

```text
YOLOv12n-face
InsightFace Buffalo-L
```

## Scoring

```text
CLIP
GroundingDINO
spaCy
```

All endpoints configurable through:

```yaml
configs/models.yaml
```

No hardcoded endpoints.

---

# Configuration

Everything configurable.

Never hardcode:

* thresholds
* model endpoints
* paths
* prompt locations

Use:

```text
configs/models.yaml
configs/thresholds.yaml
configs/pipeline.yaml
```

---

# Pipeline Runner

Support:

```bash
python run_pipeline.py \
    --movie movie.mp4
```

Resume:

```bash
python run_pipeline.py \
    --movie movie.mp4 \
    --from-step s5
```

Force:

```bash
python run_pipeline.py \
    --movie movie.mp4 \
    --force
```

---

# Testing Requirement

Before end of day:

Run one movie through:

```text
S1
S2
S3
S4
S5
S6
S7
S8
S9
S10
S11
S12
```

Verify:

* metadata updates
* runtime logs
* resume functionality
* report generation
* export generation

---

# Success Criteria

Cursor must deliver:

✓ 12 services

✓ Shared metadata architecture

✓ Runtime logging

✓ Runtime summary report

✓ Prompt manager

✓ Service registry

✓ Base service

✓ Resume support

✓ Force rerun support

✓ Single conda environment

✓ One movie end-to-end execution

✓ Ready for multi-movie execution tomorrow

Most important rule:

Follow Harsha's architecture document exactly.
Do not redesign the pipeline.
