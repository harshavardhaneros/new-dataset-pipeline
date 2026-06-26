# Indic Cultural Video Dataset — Service Pipeline v2

## Architecture

* 13 independent services
* Shared state via `metadata.json` stored on Yotta PFS
* Services are stateless and communicate only through metadata updates
* CPU services run on VM nodes
* GPU services call shared model-service pool through HTTP
* Video clips and metadata remain colocated on Yotta PFS

---

# S1 — Extract Clips + Hash

### Purpose

Convert long movies into scene-level clips.

### Implementation

* PySceneDetect threshold = 27
* Scene boundary detection
* Export scene clips as MP4
* Compute perceptual hash (pHash) from middle keyframe

### Input

* Raw movies (`.mp4`, `.mkv`)
* movie-list.csv

### Output

```json
{
  "video_id": "...",
  "clip_id": "...",
  "source": "...",
  "timestamp_start": 0,
  "timestamp_end": 0,
  "duration": 0,
  "phash": "..."
}
```

---

# S2 — Dedup + Motion + Quality Filtering

## S2A — Deduplication

### Purpose

Remove duplicate clips before expensive processing.

### Implementation

* pHash on middle frame
* BK-tree search
* Hamming distance ≤ 8 considered duplicate

### Output

```json
{
  "dup_of": "...",
  "keep": true
}
```

Duplicates moved to:

```text
/dups
```

---

## S2B — Motion Filtering

### Purpose

Remove:

* Static clips
* Slideshows
* Frozen scenes
* Extremely shaky clips
* Unusable motion patterns

Maintain clips with moderate and meaningful motion suitable for video generation.

---

### Method 1 — UniMatch Optical Flow

Repository:

https://github.com/autonomousvision/unimatch

### How It Works

UniMatch estimates dense optical flow between two frames.

Pipeline:

1. Sample frames every 0.5 seconds
2. Resize frames to:

```text
320 × 576
```

3. Compute optical flow between consecutive sampled frames
4. Compute average flow magnitude

Formula:

```text
MotionScore =
mean(optical_flow_magnitude)
```

Interpretation:

* Very low score → static scene
* Moderate score → good motion
* Very high score → unstable camera / fast transitions

Store:

```json
{
  "unimatch_motion": 0.42
}
```

---

### Method 2 — VMAF Temporal Difference

Repository:

https://github.com/Netflix/vmaf

### How It Works

VMAF is used as a temporal consistency metric.

Pipeline:

1. Use FFmpeg
2. Compare consecutive frames
3. Measure frame-to-frame visual change
4. Normalize values

Interpretation:

* Near 0 → almost identical frames
* Moderate → healthy motion
* Extremely high → cuts or noisy footage

Store:

```json
{
  "vmaf_motion": 0.37
}
```

---

### Motion Decision Logic

Calculate:

```text
motion_score =
0.7 × UniMatch +
0.3 × VMAF
```

Each source may have different motion statistics.

For each source:

1. Compute motion distribution
2. Estimate p10 and p90
3. Keep clips within valid motion range

Example:

```text
0.15 ≤ motion_score ≤ 0.80
```

Store:

```json
{
  "motion_score": 0.51,
  "motion_pass": true
}
```

Rejected clips:

```text
/static_clips
/excessive_motion
```

---

## S2C — Aesthetic Filtering

### Purpose

Remove visually poor clips before captioning.

---

### DOVER

Repository:

https://github.com/VQAssessment/DOVER

### How It Works

DOVER evaluates overall video quality using multiple visual dimensions.

Outputs:

1. Aesthetic Score
2. Technical Score
3. Overall Score

We use:

```text
overall_score
```

as filtering metric.

---

### Pipeline

1. Pass entire clip to DOVER
2. Receive three scores
3. Store all scores
4. Filter using overall score

Example threshold:

```text
overall_score >= 0.60
```

Store:

```json
{
  "aesthetic_score": 0.73,
  "technical_score": 0.69,
  "dover_score": 0.71
}
```

Reject:

```text
/low_quality
```

---

## S2 Final Decision

Clip survives S2 only if:

```text
keep == true
AND
motion_pass == true
AND
dover_score >= threshold
```

Output:

```json
{
  "keep": true,
  "motion_score": 0.51,
  "motion_pass": true,
  "dover_score": 0.71
}
```

---

# S3 — Band Removal

### Purpose

Remove black letterbox bars.

### Implementation

* Detect bands from first keyframe
* FFmpeg crop filter
* Re-encode clip

Output:

```json
{
  "band_removed": true,
  "crop_box": [x,y,w,h]
}
```

---

# S4 — Watermark Removal

### Purpose

Remove Eros watermark.

### Implementation

* Generate mask from first frame
* LaMa inpainting
* Apply to all frames

Output:

```json
{
  "wm_removed": true
}
```

---

# S5 — Classification + Filtering

Model:

* Qwen3-VL-32B

Input:

* 3 representative keyframes

Output:

```json
{
  "bucket": "...",
  "reject": false,
  "reject_reason": null
}
```

12 cultural buckets.

---

# S6 — Verification

Model:

* Gemma

Purpose:

* Confirm bucket assignment
* Reduce classification errors

Output:

```json
{
  "verified": true,
  "confidence": 0.91,
  "route": "people"
}
```

---

# S7 — Actor Tagging

Model:

* YOLOv12n-face
* InsightFace buffalo_l

Database:

* 200 actor embeddings

Threshold:

```text
cosine >= 0.35
```

Output:

```json
{
  "actor_ids": [],
  "actor_names": []
}
```

---

# S8 — Captioning

Model:

* Qwen3-VL-32B

Input:

* Keyframes
* Bucket
* Actor names

Output:

```json
{
  "caption": "...",
  "caption_struct": {}
}
```

For future video generation training, append:

```text
Motion score: {motion_score}
```

to generated captions.

Example:

```text
A woman dancing in a traditional Bharatanatyam performance.
Motion score: 0.51
```

This allows CogVideoX/Wan2.1 to learn motion magnitude conditioning.

---

# S9 — Quality Scoring

Weighted score:

```text
0.35 × CLIP
+
0.40 × ICR
+
0.25 × AOD
```

Output:

```json
{
  "clip_score": 0.71,
  "icr": 0.68,
  "aod": 0.74,
  "score": 0.71
}
```

---

# S10 — Gate

```text
FINAL   >= 0.30
REVIEW  0.20–0.30
DISCARD < 0.20
```

---

# S11 — Export

Outputs:

* video.mp4
* caption.txt
* metadata.csv

Includes:

* 15% caption mixing
* collision-safe naming

---

# S12 — Report

Generate:

* per bucket counts
* source statistics
* motion statistics
* DOVER distributions
* gate distributions
* total clip hours

Outputs:

```text
report.txt
report.json
```

---

# Final Metadata Additions

```json
{
  "unimatch_motion": 0.42,
  "vmaf_motion": 0.37,
  "motion_score": 0.51,
  "motion_pass": true,

  "aesthetic_score": 0.73,
  "technical_score": 0.69,
  "dover_score": 0.71
}
```
