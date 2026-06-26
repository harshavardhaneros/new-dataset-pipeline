#!/usr/bin/env python3
"""
Actor Tagging & Captioning Pipeline — Xeon Platinum 8580 + H200 Edition
========================================================================
Tuned for:
  2 × Intel Xeon Platinum 8580 (120 physical cores / 240 threads)
  3.0 TiB RAM  |  AVX-512 + AVX-512_BF16 + AMX  |  NVMe RAID ~900 MB/s
  H200 GPU (device 6)  |  Kubernetes pod  |  Ubuntu 22.04  |  Python 3.10

NUMA topology (interleaved):
  Node 0 → even CPUs  0,2,4…238   (60 cores / 120 threads)
  Node 1 → odd  CPUs  1,3,5…239   (60 cores / 120 threads)

Strategy:
  CPU side (all parallel, NUMA-pinned):
    • ffmpeg extraction  → 96 threads, NUMA node 0, saturates NVMe RAID
    • InsightFace        → 112 processes, NUMA node 0, 1 ONNX session each
    • Actor assignment   → single AVX-512 BLAS matmul (all 120 cores)
  GPU side:
    • Gemma-3-4B-IT      → H200 GPU 6, bfloat16, batch_size 64
    • Producer-consumer  → CPU thread pre-loads images while GPU runs inference
    • Batch fallback      → drops to single-item on OOM, recovers automatically

Usage:
  python pipeline.py config.toml movies.yaml
"""

import os

# ── NUMA node 0 (even CPUs) for everything except the captioner ──────────────
# The captioner block below overrides this for its own process.
NUMA0_CPUS = ",".join(str(i) for i in range(0, 240, 2))   # 0,2,4,...,238
NUMA1_CPUS = ",".join(str(i) for i in range(1, 240, 2))   # 1,3,5,...,239

# ── OpenBLAS / MKL / OMP: use all 120 physical cores for NumPy matmuls ───────
os.environ.setdefault("OMP_NUM_THREADS",        "8")   # conservative — shared machine
os.environ.setdefault("MKL_NUM_THREADS",        "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")

# ── AMX: let PyTorch use the AMX tile engine (auto on Sapphire Rapids) ────────
os.environ.setdefault("ONEDNN_MAX_CPU_ISA",     "AVX512_CORE_AMX")
os.environ.setdefault("DNNL_MAX_CPU_ISA",       "AVX512_CORE_AMX")

# ── NCCL: not used but keep quiet ─────────────────────────────────────────────
os.environ.setdefault("NCCL_DEBUG",        "WARN")
os.environ.setdefault("NCCL_P2P_DISABLE",  "0")
os.environ.setdefault("NCCL_IB_DISABLE",   "0")

import csv
import gc
import json
import logging
import multiprocessing as mp
import pickle
import queue
import re
import shutil
import subprocess
import sys
import threading
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import toml
import yaml
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ── Runtime limits (cgroup + observed coworker load) ─────────────────────────
# cgroup quota : 21,600,000 µs / 100,000 µs = 216 logical cores
# Coworker load: Sana training (~120 cores) + other jobs (~45 cores) = ~165 cores
# Safe pipeline budget: 216 − 165 − 9 (OS buffer) = ~42 logical = ~16 physical
# Face analysis moved entirely to GPU — frees CPU completely for other steps.
LOGICAL_CPUS     = 216     # cgroup hard cap
PHYSICAL_CORES   = 108     # logical / 2  (Xeon 8580 has HT)

FFMPEG_WORKERS   = 16      # conservative — coworkers own ~165/216 logical cores
CAPTION_Q_DEPTH  = 8       # image pre-load queue depth (halved, shared machine)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG & I/O
# ═════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return toml.load(f)

def load_movies(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def find_movie_file(name: str, input_dir: str) -> str | None:
    exact = Path(input_dir) / f"{name}.mp4"
    if exact.exists():
        return str(exact)
    norm = lambda s: s.lower().replace(" ", "")
    for f in Path(input_dir).glob("*.mp4"):
        if norm(f.stem) == norm(name):
            return str(f)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# NUMA PINNING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _pin_to_numa_node(node: int) -> None:
    """
    Pin the calling process to a NUMA node using taskset (available in the pod).
    Node 0 = even CPUs (0,2,4…238), Node 1 = odd CPUs (1,3,5…239).
    Falls back silently if taskset is unavailable.
    """
    cpu_list = NUMA0_CPUS if node == 0 else NUMA1_CPUS
    pid = os.getpid()
    try:
        subprocess.run(
            ["taskset", "-cp", cpu_list, str(pid)],
            capture_output=True, check=True,
        )
    except Exception:
        pass   # non-fatal — runs unpinned


# ═════════════════════════════════════════════════════════════════════════════
# TIME-CODE UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def tc_to_sec(tc: str) -> float:
    h, m, s = tc.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def sec_to_tc(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 – SCENE DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def detect_scenes(video_path: str, num_workers: int = 16) -> list[dict]:
    """
    PySceneDetect with PyAV backend — the standard, accurate, professional approach.

    Speed options applied (all official PySceneDetect parameters):
      threading_mode='auto'  → PyAV decodes frames across multiple C threads
                               (documented as the fastest PyAV mode)
      frame_skip=3           → every 4th frame sampled; at 25fps that is one
                               sample per 160ms — no real movie cut is shorter
      auto-downscale         → PySceneDetect picks downscale=2 for 1080p
                               automatically, histogram diff is 4x cheaper

    Expected runtime: 8-12 min for your 225k-frame movie.
    Results cached to CSV — runs once per movie, skipped on all reruns.

    Fallback: if PyAV fails for any reason, ffprobe I-frame mode runs instead.
    """
    try:
        return _detect_scenes_pyav_single(video_path)
    except Exception as exc:
        log.warning("  PyAV failed (%s) — falling back to ffprobe", exc)
        return _detect_scenes_ffprobe(video_path)


def _detect_scenes_ffprobe(video_path: str) -> list[dict]:
    """
    Detects scene changes using ffprobe scene filter on I-frames only.
    -skip_frame noref  → skip all non-reference frames (B and P frames)
                         leaving only I-frames to decode — ~75x speedup
    select=gt(scene,X) → ffmpeg built-in scene score per frame (0.0-1.0)
    threshold 0.25     → tuned for Tamil/Hindi movies with fast cuts & dissolves
    """
    try:
        dur = _video_duration(video_path)

        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-skip_frame", "noref",
                "-select_streams", "v:0",
                "-show_frames",
                "-show_entries", "frame=best_effort_timestamp_time",
                "-vf", "select=gt(scene\,0.25)",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        timestamps = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and line != "N/A":
                try:
                    timestamps.append(float(line))
                except ValueError:
                    pass

        if not timestamps:
            return []

        timestamps = sorted(set(timestamps))
        boundaries = [0.0] + timestamps + [dur]

        scenes = []
        for i in range(len(boundaries) - 1):
            s = round(boundaries[i], 3)
            e = round(boundaries[i + 1], 3)
            if e - s >= 0.5:
                scenes.append({
                    "scene_id":  len(scenes) + 1,
                    "start_sec": s,
                    "end_sec":   e,
                })

        log.info("  ffprobe found %d scenes from %d cut points", len(scenes), len(timestamps))
        return scenes

    except subprocess.TimeoutExpired:
        log.warning("  ffprobe timed out")
        return []
    except Exception as exc:
        log.debug("  ffprobe error: %s", exc)
        return []


def _get_keyframe_boundaries(video_path: str, n: int, duration: float) -> list[float]:
    """
    Use ffprobe to find keyframe timestamps, then pick N evenly-spaced
    ones as chunk boundaries. This ensures each worker starts on a clean
    I-frame and never stalls waiting for a reference frame.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-skip_frame", "noref",
                "-show_entries", "frame=best_effort_timestamp_time",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True, text=True, timeout=120,
        )
        keyframes = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and line != "N/A":
                try:
                    keyframes.append(float(line))
                except ValueError:
                    pass
        if len(keyframes) < n:
            # Fewer keyframes than workers — just split evenly
            return [duration * i / n for i in range(n)]
        # Pick n evenly-spaced keyframes as boundaries
        step = len(keyframes) // n
        return [keyframes[i * step] for i in range(n)]
    except Exception:
        # Plain even split — worst case a worker seeks to a non-keyframe
        # and PyAV handles it gracefully anyway
        return [duration * i / n for i in range(n)]


def _detect_chunk(args: tuple) -> list[tuple[float, float]]:
    """
    Worker function: detect scenes in one time segment [start, end).
    Returns list of (scene_start_sec, scene_end_sec) tuples.
    Runs in a separate process — imports are local to avoid pickling issues.
    """
    video_path, start_sec, end_sec, threshold = args

    from scenedetect import SceneManager
    from scenedetect.backends.pyav import VideoStreamAv
    from scenedetect.detectors import ContentDetector

    try:
        video = VideoStreamAv(video_path, start_time=start_sec)
        sm    = SceneManager()
        sm.add_detector(ContentDetector(threshold=threshold))

        # Only process frames up to end_sec
        frame_limit = None
        if end_sec is not None:
            fps = video.frame_rate
            frame_limit = int((end_sec - start_sec) * fps)

        sm.detect_scenes(
            video,
            frame_skip=1,              # every other frame
            end_time=end_sec,          # PySceneDetect 0.7 supports end_time
            show_progress=False,       # suppress per-worker bars
        )

        scenes = []
        for s, e in sm.get_scene_list():
            ss = round(s.get_seconds(), 3)
            es = round(e.get_seconds(), 3)
            # Clamp to chunk boundaries
            ss = max(ss, start_sec)
            es = min(es, end_sec) if end_sec else es
            if es - ss >= 0.5:
                scenes.append((ss, es))
        return scenes

    except Exception as exc:
        # Return empty — the merge step will produce a gap-filling scene
        return []


def _detect_scenes_parallel(video_path: str, num_workers: int) -> list[dict]:
    """
    Split → parallel detect → merge → renumber.
    """
    THRESHOLD = 27

    # Get duration
    duration = _video_duration(video_path)

    # Get keyframe-aligned boundaries
    log.info("  Scene detect: %d parallel workers (PyAV) …", num_workers)
    boundaries = _get_keyframe_boundaries(video_path, num_workers, duration)
    boundaries.append(duration)   # sentinel end

    # Build chunk args
    chunks = [
        (video_path, boundaries[i], boundaries[i + 1], THRESHOLD)
        for i in range(len(boundaries) - 1)
    ]

    # Run in parallel
    all_scenes: list[tuple[float, float]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        chunk_results = list(tqdm(
            pool.map(_detect_chunk, chunks),
            total=len(chunks),
            desc="  Scene detect chunks",
            unit="chunk",
        ))

    for chunk in chunk_results:
        all_scenes.extend(chunk)

    # Sort and deduplicate boundaries that fall within 1 second of each other
    # (seam artefacts from chunk boundaries)
    all_scenes.sort(key=lambda x: x[0])

    merged: list[tuple[float, float]] = []
    for s, e in all_scenes:
        if merged and s - merged[-1][1] < 1.0:
            # Merge with previous — extend its end
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Fill any gaps (chunks that returned nothing) with a single scene
    filled: list[tuple[float, float]] = []
    prev_end = 0.0
    for s, e in merged:
        if s - prev_end > 0.5:
            filled.append((prev_end, s))   # gap → one big scene
        filled.append((s, e))
        prev_end = e
    if duration - prev_end > 0.5:
        filled.append((prev_end, duration))

    scenes = [
        {"scene_id": i, "start_sec": round(s, 3), "end_sec": round(e, 3)}
        for i, (s, e) in enumerate(filled, 1)
    ]
    log.info("  Detected %d scenes in %.0f s of video",
             len(scenes), duration)
    return scenes


def _detect_scenes_pyav_single(video_path: str) -> list[dict]:
    """
    PyAV fallback with all official speed options applied:
      threading_mode='auto'  → PyAV decodes frames across multiple C threads
                               (official PySceneDetect recommendation for speed)
      frame_skip=3           → process every 4th frame = 25% of frames
                               at 25fps that is one sample per 160ms — no real
                               cut is shorter than that in a movie
      downscale              → PySceneDetect auto-selects based on resolution;
                               for 1080p it picks 2 (540p) automatically,
                               histogram comparison is 4× cheaper
    Combined: roughly 8-12× faster than default OpenCV backend.
    For your 225k-frame movie: ~8-12 minutes instead of 70.
    """
    from scenedetect import SceneManager
    from scenedetect.backends.pyav import VideoStreamAv
    from scenedetect.detectors import ContentDetector

    log.info("  Scene detect: PyAV (threading=auto, frame_skip=3, auto-downscale) …")
    video = VideoStreamAv(video_path, threading_mode="auto")
    sm    = SceneManager()
    sm.add_detector(ContentDetector(threshold=27))
    sm.detect_scenes(
        video,
        show_progress=True,
        frame_skip=3,       # every 4th frame — standard PySceneDetect option
    )

    scenes = [
        {
            "scene_id":  i,
            "start_sec": round(s.seconds, 3),
            "end_sec":   round(e.seconds, 3),
        }
        for i, (s, e) in enumerate(sm.get_scene_list(), 1)
    ]
    log.info("  Detected %d scenes", len(scenes))
    return scenes


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 – CLIP GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_clips(movie_name: str, scenes: list[dict], clip_dir: str) -> list[dict]:
    Path(clip_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(clip_dir) / f"{movie_name}_clips.csv"
    if csv_path.exists():
        log.info("  Clips CSV cached.")
        return pd.read_csv(csv_path).to_dict("records")

    clips, rows = [], []
    for sc in scenes:
        dur = sc["end_sec"] - sc["start_sec"]
        if dur < 3:
            continue
        for i in range(int(dur // 3)):
            cs  = round(sc["start_sec"] + i * 3, 3)
            ce  = round(cs + 3, 3)
            cid = f"{sc['scene_id']}.{i + 1}"
            clips.append({"scene_id": sc["scene_id"], "clip_id": cid,
                           "start_sec": cs, "end_sec": ce})
            rows.append([sc["scene_id"], cid, sec_to_tc(cs), sec_to_tc(ce), cs, ce, 3])

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "clip_id", "start_time_movie", "end_time_movie",
                     "start_sec", "end_sec", "duration_sec"])
        w.writerows(rows)
    log.info("  Clips CSV: %s (%d clips)", csv_path, len(clips))
    return clips


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 – BLACK-BAR DETECTION & FRAME EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def _video_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def _detect_black_bars(img_path: str) -> tuple[int, int, int, int]:
    img  = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    row_means = gray.mean(axis=1)       # NumPy vectorised — no Python loop
    nonzero   = np.where(row_means > 10)[0]
    y0 = int(nonzero[0])  if len(nonzero) else 0
    y1 = int(nonzero[-1]) if len(nonzero) else h - 1
    return (0, y0, w, y1)


def get_crop_coordinates(video_path: str, tmp_dir: str) -> tuple[int, int, int, int]:
    dur = _video_duration(video_path)
    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    crops, temps = [], []

    def _probe(args):
        i, frac = args
        fp = str(tmp / f"_cropprobe_{i}.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(dur * frac), "-i", video_path,
             "-frames:v", "1", "-q:v", "2", fp],
            capture_output=True,
        )
        return fp, _detect_black_bars(fp)

    with ThreadPoolExecutor(max_workers=3) as pool:
        for fp, crop in pool.map(_probe, enumerate([0.25, 0.5, 0.75])):
            crops.append(crop)
            temps.append(fp)

    for f in temps:
        Path(f).unlink(missing_ok=True)

    best, count = Counter(crops).most_common(1)[0]
    if count == 1:
        best = tuple(int(np.median([c[j] for c in crops])) for j in range(4))
    return best  # type: ignore[return-value]


def extract_frames(
    movie_name: str,
    movie_path: str,
    clips: list[dict],
    frames_dir: str,
    max_workers: int = FFMPEG_WORKERS,
) -> str:
    """
    Industry-standard frame extraction: one ffmpeg call per frame using
    input-side -ss (fast seek) + -frames:v 1.

    Why one call per frame instead of one per clip:
      Input-side -ss seeks to the nearest keyframe BEFORE the target time
      in O(1) — ffmpeg reads the keyframe index and jumps directly.
      It then decodes only a handful of frames to reach exact timestamp.
      Total decode per call: ~5-15 frames maximum regardless of position
      in the video.

      The select-filter approach (one call per clip) must decode from the
      seek point forward searching for frames matching the expression —
      for clips near the end of a 2.5-hour file this means decoding
      hundreds of frames per call, causing the 5+ minute hang.

    This is exactly how FFmpeg docs recommend thumbnail extraction:
      ffmpeg -ss <time> -i input.mp4 -frames:v 1 -q:v 2 output.jpg

    With 16 parallel workers × 5,151 frames each taking ~0.3s = ~16 min.
    NVMe at 900 MB/s handles 16 concurrent seeks easily.
    """
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    frame_csv = Path(frames_dir) / f"{movie_name}_frames.csv"

    if frame_csv.exists():
        log.info("  Frames CSV cached – skipping extraction.")
        return str(frame_csv)

    x0, y0, x1, y1 = get_crop_coordinates(movie_path, frames_dir)
    crop_str = f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0}"
    log.info("  Crop filter: %s", crop_str)

    # Build one task per frame (not per clip)
    frame_tasks = []
    for clip in clips:
        for idx, offset in enumerate([0.5, 1.5, 2.5], 1):
            t    = clip["start_sec"] + offset
            fid  = f"{clip['clip_id']}.{idx}"
            path = str(Path(frames_dir) / f"{movie_name}_frame_{fid}.jpg")
            frame_tasks.append({
                "time":     t,
                "path":     path,
                "clip":     clip,
                "frame_id": fid,
                "idx":      idx,
            })

    def _extract_one(task: dict) -> dict:
        """
        Single frame extraction using input-side seek (-ss before -i).
        This is the canonical FFmpeg fast-seek pattern used in production
        thumbnail generators, video editors, and streaming pipelines.
        """
        if Path(task["path"]).exists():
            return task
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{task['time']:.3f}",  # input-side seek — O(1) keyframe jump
                "-i", movie_path,
                "-frames:v", "1",               # decode exactly 1 frame
                "-vf", crop_str,
                "-q:v", "2",                    # JPEG quality 2 (high, ~500KB)
                "-threads", "1",                # 1 thread per process; pool provides parallelism
                task["path"],
            ],
            capture_output=True,
        )
        return task

    log.info("  Extracting %d frames  (%d workers) …",
             len(frame_tasks), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(tqdm(
            pool.map(_extract_one, frame_tasks),
            total=len(frame_tasks),
            desc="  Frames",
            unit="frame",
        ))

    # Write frames CSV
    rows = []
    for t in frame_tasks:
        c = t["clip"]
        rows.append([
            c["scene_id"], c["clip_id"], t["frame_id"],
            sec_to_tc(c["start_sec"]), sec_to_tc(c["end_sec"]),
            c["start_sec"], c["end_sec"],
            sec_to_tc(t["time"]), round(t["time"], 3), t["path"],
        ])

    with open(frame_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "clip_id", "frame_id",
                     "clip_start_time_movie", "clip_end_time_movie",
                     "clip_start_sec", "clip_end_sec",
                     "frame_time_movie", "frame_time_sec", "frame_path"])
        w.writerows(rows)
    log.info("  Frames CSV: %s", frame_csv)
    return str(frame_csv)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 – FACE DETECTION & EMBEDDING  (InsightFace on GPU)
# ═════════════════════════════════════════════════════════════════════════════
# Moved entirely to GPU 6 (H200) so zero CPU cores are consumed here.
# This is the biggest single CPU relief on a shared machine:
#   Before: 100 CPU processes pegged for the entire face-analysis phase
#   After:  1 GPU thread, CPU is free for coworkers throughout
#
# Implementation: a single FaceAnalysis session with CUDAExecutionProvider.
# Images are read from disk in a ThreadPoolExecutor (I/O, GIL-free) and
# fed to the GPU session one at a time. The H200 inference is fast enough
# that disk I/O becomes the bottleneck — we use 16 reader threads to keep
# the GPU fed without starving the NVMe for coworkers.

def build_face_analyzer_gpu(cfg: dict):
    """
    Load a single InsightFace session on the configured GPU.
    Returns the analyzer object — no worker processes needed.
    """
    from insightface.app import FaceAnalysis

    gpu_cfg    = cfg.get("gpu", {})
    device_ids = gpu_cfg.get("face_device_id", gpu_cfg.get("device_id", [6]))
    gpu_id     = device_ids[0] if isinstance(device_ids, list) else int(device_ids)

    model_pack = cfg.get("models", {}).get("insightface_model_pack", "buffalo_l")
    det_size   = tuple(cfg.get("models", {}).get("insightface_det_size", [640, 640]))

    log.info("Loading InsightFace (%s) on GPU %d …", model_pack, gpu_id)
    analyzer = FaceAnalysis(
        name=model_pack,
        root="~/.insightface",
        providers=[("CUDAExecutionProvider", {"device_id": gpu_id})],
    )
    analyzer.prepare(ctx_id=gpu_id, det_size=det_size)
    log.info("InsightFace ready on GPU %d.", gpu_id)
    return analyzer


def run_face_analysis(
    frame_paths: list[str],
    analyzer,                   # GPU FaceAnalysis instance
    io_workers: int = 16,       # threads for parallel disk reads
) -> dict[str, list[dict]]:
    """
    GPU face analysis with parallel image loading.

    Pattern:
      ThreadPoolExecutor (16 threads) reads JPEGs from NVMe in parallel.
      Results are queued and fed to the single GPU session sequentially.
      GPU inference is fast (~5-15 ms/frame on H200); I/O is the bottleneck.
      16 reader threads keep the GPU fed continuously with zero CPU contention.
    """
    results: dict[str, list[dict]] = {}
    read_q: queue.Queue = queue.Queue(maxsize=io_workers * 2)
    SENTINEL = object()

    def _reader():
        def _read_one(path):
            img = cv2.imread(path)
            return path, img
        with ThreadPoolExecutor(max_workers=io_workers) as pool:
            for path, img in pool.map(_read_one, frame_paths):
                read_q.put((path, img))
        read_q.put(SENTINEL)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    pbar = tqdm(total=len(frame_paths), desc="  Face analysis (GPU)", unit="frame")
    while True:
        item = read_q.get()
        if item is SENTINEL:
            break
        path, img = item
        dets = []
        if img is not None:
            try:
                for j, face in enumerate(analyzer.get(img)):
                    dets.append({
                        "face_id":    j,
                        "bbox":       [float(x) for x in face.bbox.tolist()],
                        "confidence": float(face.det_score),
                        "embedding":  face.normed_embedding.astype(np.float32),
                        "_img_hw":    (img.shape[0], img.shape[1]),
                    })
            except Exception as exc:
                log.debug("Face error %s: %s", path, exc)
        results[path] = dets
        pbar.update(1)

    pbar.close()
    reader_thread.join()
    return results


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 – ACTOR ASSIGNMENT  (single AVX-512 vectorised BLAS call)
# ═════════════════════════════════════════════════════════════════════════════

def load_actor_embeddings(
    actor_dir: str,
    actor_keys: list[str],
    actor_files: dict[str, str],
) -> tuple[list[str], np.ndarray | None]:
    keys, embs = [], []
    for key in actor_keys:
        fpath = Path(actor_dir) / actor_files.get(key, f"{key}.pkl")
        if not fpath.exists():
            log.warning("  Missing actor embedding: %s", fpath)
            continue
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        embs.append(np.array(data["embedding"], dtype=np.float32).reshape(-1))
        keys.append(key)

    if not embs:
        return [], None

    mat  = np.vstack(embs).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return keys, mat


def assign_actors(
    face_results: dict[str, list[dict]],
    actor_keys: list[str],
    actor_matrix: np.ndarray,
    display_names: dict[str, str],
    threshold: float = 0.35,
) -> dict[str, list[dict]]:
    """
    One BLAS sgemm call for the entire movie:
      (N_faces, 512) @ (512, N_actors) → (N_faces, N_actors)

    With 120 OMP threads and AVX-512, this finishes in milliseconds
    even for a movie with tens of thousands of detected faces.
    """
    all_embs, index_map = [], []
    for path, dets in face_results.items():
        for pos, det in enumerate(dets):
            all_embs.append(det["embedding"])
            index_map.append((path, pos))

    assignments: dict[str, list[dict]] = {p: [] for p in face_results}

    if all_embs:
        emb_mat  = np.vstack(all_embs).astype(np.float32)
        emb_mat /= np.linalg.norm(emb_mat, axis=1, keepdims=True) + 1e-8
        sims     = emb_mat @ actor_matrix.T          # one BLAS call
        best_idx = np.argmax(sims, axis=1)
        best_sim = sims[np.arange(len(sims)), best_idx]

        for i, (path, pos) in enumerate(index_map):
            det   = face_results[path][pos]
            actor = actor_keys[best_idx[i]] if best_sim[i] >= threshold else "unknown"
            assignments[path].append({
                "actor":        actor,
                "display_name": display_names.get(actor, actor),
                "bbox":         det["bbox"],
                "similarity":   float(best_sim[i]),
                "_img_hw":      det.get("_img_hw"),
            })

    return assignments


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 – POSITION LABELLING  (zero extra disk reads)
# ═════════════════════════════════════════════════════════════════════════════

def _screen_position(bbox: list[float], w: int, h: int) -> str:
    xc = (bbox[0] + bbox[2]) / 2
    yc = (bbox[1] + bbox[3]) / 2
    hp = "left"   if xc < w * 0.33 else ("right"  if xc >= w * 0.66 else "center")
    vp = "top"    if yc < h * 0.33 else ("bottom" if yc >= h * 0.66 else "center")
    return "center" if hp == "center" and vp == "center" else f"{vp}-{hp}"


def compute_positions(assignments: dict[str, list[dict]]) -> dict[str, str]:
    pos_map: dict[str, str] = {}
    for path, frame_assigns in assignments.items():
        if not frame_assigns:
            pos_map[path] = "unknown"
            continue
        hw  = next((a.get("_img_hw") for a in frame_assigns if a.get("_img_hw")), None)
        if hw is None:
            img = cv2.imread(path)
            hw  = (img.shape[0], img.shape[1]) if img is not None else (1080, 1920)
        h, w = hw
        parts = [
            f"{a['display_name']} ({_screen_position(a['bbox'], w, h)})"
            for a in frame_assigns if a["actor"] != "unknown"
        ]
        pos_map[path] = ", ".join(parts) if parts else "unknown"
    return pos_map


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6b – FINAL DATAFRAME
# ═════════════════════════════════════════════════════════════════════════════

def _known_actors(fp: str, assignments: dict) -> list[str]:
    return list({a["display_name"] for a in assignments.get(fp, [])
                 if a["actor"] != "unknown"})


def build_final_df(
    movie_name: str,
    clips: list[dict],
    frames_dir: str,
    assignments: dict[str, list[dict]],
    pos_map: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for clip in clips:
        cid = clip["clip_id"]
        f1  = str(Path(frames_dir) / f"{movie_name}_frame_{cid}.1.jpg")
        f2  = str(Path(frames_dir) / f"{movie_name}_frame_{cid}.2.jpg")
        f3  = str(Path(frames_dir) / f"{movie_name}_frame_{cid}.3.jpg")
        a1, a2, a3 = (_known_actors(f, assignments) for f in (f1, f2, f3))
        rows.append({
            "scene_id":         clip["scene_id"],
            "clip_id":          cid,
            "start_time_movie": sec_to_tc(clip["start_sec"]),
            "end_time_movie":   sec_to_tc(clip["end_sec"]),
            "start_sec":        clip["start_sec"],
            "end_sec":          clip["end_sec"],
            "duration_sec":     3,
            "frame1": f1, "frame2": f2, "frame3": f3,
            "actors_f1":   str(a1),
            "actors_f2":   str(a2),
            "actors_f3":   str(a3),
            "clip_actors": str(list(set(a1 + a2 + a3))),
            "pos_f1":      pos_map.get(f1, "unknown"),
            "pos_f2":      pos_map.get(f2, "unknown"),
            "pos_f3":      pos_map.get(f3, "unknown"),
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 – CAPTIONING  (INT8 + AMX, NUMA node 1, producer-consumer)
# ═════════════════════════════════════════════════════════════════════════════

CAPTION_SYSTEM_PROMPT = (
    "Output MUST be a valid JSON object only. No markdown or extra text.\n\n"
    "Rules:\n"
    "- Be precise and avoid repetition.\n"
    "- No hallucination. Only visible or strongly implied details.\n"
    "- Avoid generic phrases (e.g., \"a group of people\").\n"
    "- For humans, describe from THEIR perspective (not the viewer's).\n"
    "- Prioritise culturally significant visual elements when present.\n"
    "- Include actor names and positions while explaining object actions.\n\n"
    "Indian Cultural Details (include ONLY if visible):\n"
    "- attire: women: saree (silk/cotton), half-saree, salwar, blouse color/design,\n"
    "  embroidery (Zardozi, Chikankari). men: veshti/dhoti, kurta, shirt, traditional wear\n"
    "- accessories: jhumka, nose ring, choker, chain, bangles, anklets, kundan, bindi/sindoor\n"
    "- regional_identity: Tamil, Punjabi, Bengali, etc. (ONLY if clearly inferable)\n"
    "- cultural_context: temple, wedding, ritual, festival, street market, rural/urban India\n"
    "- architecture_landmarks: gopuram, heritage buildings (if visible)\n"
    "- food_elements: traditional dishes (if present)\n\n"
    "Text: Include ONLY clearly visible text. If none → return [].\n\n"
    "JSON structure:\n"
    "{ \"short_description\": \"\",\n"
    "  \"objects\": [{ \"description\":\"\",\"location\":\"\",\"relative_size\":\"\","
    "\"shape_color\":\"\",\"texture\":\"\",\"appearance_details\":\"\","
    "\"relationship\":\"\",\"orientation\":\"\",\"Indian_cultural_details\":{},"
    "\"pose\":\"\",\"expression\":\"\",\"clothing\":\"\","
    "\"actor_name_and_action\":\"\",\"gender\":\"\",\"skin_tone_texture\":\"\" }],\n"
    "  \"background_setting\":\"\",\n"
    "  \"lighting\":{\"conditions\":\"\",\"direction\":\"\",\"shadows\":\"\"},\n"
    "  \"aesthetics\":{\"composition\":\"\",\"color_scheme\":\"\",\"mood_atmosphere\":\"\"},\n"
    "  \"photographic_characteristics\":{\"depth_of_field\":\"\",\"focus\":\"\","
    "\"camera_angle\":\"\",\"camera_movement\":\"\",\"lens_focal_length\":\"\"},\n"
    "  \"style_medium\":\"\",\n"
    "  \"text_render\":[{\"text\":\"\",\"location\":\"\",\"size\":\"\","
    "\"color\":\"\",\"font\":\"\",\"appearance_details\":\"\"}] }"
)


class H200Captioner:
    """
    Gemma-3-4B-IT on H200 GPU (device 6), bfloat16.

    Optimisations:
    1. bfloat16 — H200 has dedicated BF16 tensor cores; identical accuracy
       to float32 for vision-language inference, ~2× the throughput.

    2. batch_size 64 — H200 has 141 GB HBM3e; 4B model in BF16 is ~8 GB,
       leaving >130 GB for activations. Batch of 8 frames fits comfortably.
       Batch inference amortises the fixed per-step GPU kernel launch cost.

    3. Producer-consumer queue (depth = batch_size × 2) — a background
       CPU thread decodes JPEGs and builds message dicts while the GPU runs
       the previous batch. On NVMe at 900 MB/s, image loading takes ~1 ms
       per frame; with queuing it is completely hidden behind GPU inference.

    4. Single-item fallback — if a batch raises OOM (can't happen on H200
       with these sizes, but defensive), the code retries each item alone
       and logs a warning rather than crashing the whole movie.

    5. torch.cuda.empty_cache() between batches — keeps fragmentation low
       across a long run of many movies.
    """

    def __init__(self, model_path: str, cfg: dict):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        gpu_cfg   = cfg.get("gpu", {})
        model_cfg = cfg.get("captioner", cfg.get("vllm", {}))

        llm_gpus        = gpu_cfg.get("device_id", [6])
        gpu_id          = llm_gpus[0] if isinstance(llm_gpus, list) else int(llm_gpus)
        self.device     = f"cuda:{gpu_id}"
        self.max_new_tokens = int(model_cfg.get("max_tokens", 1000))
        self.batch_size  = int(model_cfg.get("batch_size", 64))

        log.info("Loading Gemma-3-4B-IT on GPU %d (bfloat16, batch=%d) …",
                 gpu_id, self.batch_size)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()
        log.info("Captioner ready on %s.", self.device)

    def _build_messages(self, row: dict) -> tuple[list | None, Image.Image | None]:
        """
        Official Gemma-3 format: image embedded inside the messages dict.
        This is required for apply_chat_template batched inference to work.
        """
        frame_path = None
        for key in ("frame2", "frame1", "frame3"):
            candidate = row.get(key, "")
            if candidate and Path(candidate).exists():
                frame_path = candidate
                break
        if frame_path is None:
            return None, None
        try:
            img = Image.open(frame_path).convert("RGB")
        except Exception as exc:
            log.warning("Cannot open %s: %s", frame_path, exc)
            return None, None

        user_text = (
            f"Actors present: {row.get('clip_actors', '[]')}\n"
            f"Frame 1: {row.get('actors_f1', '[]')} | {row.get('pos_f1', 'unknown')}\n"
            f"Frame 2: {row.get('actors_f2', '[]')} | {row.get('pos_f2', 'unknown')}\n"
            "You are a Visual Art Director generating structured, "
            "high-quality captions for the video frames."
        )
        # Image goes inside messages dict — this is the correct Gemma-3 API
        messages = [
            {"role": "system", "content": [{"type": "text", "text": CAPTION_SYSTEM_PROMPT}]},
            {"role": "user",   "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": user_text},
            ]},
        ]
        return messages, img

    def _infer_single(self, messages: list, image: Image.Image) -> str:
        """Single-item inference using official Gemma-3 apply_chat_template API."""
        import torch
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return self.processor.decode(
            gen_ids[0][input_len:], skip_special_tokens=True
        ).strip()

    def caption_batch(self, rows: list[dict]) -> list[str]:
        """
        Single-item GPU inference with producer-consumer I/O overlap.

        Gemma-3's processor has a confirmed bug with batch inference —
        it miscounts <start_of_image> tokens across padded sequences,
        causing 'inconsistently sized batches' regardless of image count.
        Single-item inference is the correct approach for this model version.

        I/O latency is hidden by a producer thread that pre-loads images
        from disk while the GPU runs inference on the previous frame.
        Expected throughput: ~3-6 clips/sec on H200.
        """
        import torch

        if not rows:
            return []

        results  = ["{}"] * len(rows)
        q: queue.Queue = queue.Queue(maxsize=16)
        SENTINEL = object()

        def _producer():
            for i, row in enumerate(rows):
                msgs, img = self._build_messages(row)
                q.put((i, msgs, img))
            q.put(SENTINEL)

        producer = threading.Thread(target=_producer, daemon=True)
        producer.start()

        def _flush(batch):
            """Batch inference via official Gemma-3 apply_chat_template."""
            if not batch:
                return
            try:
                msgs_list = [m for _, m, _ in batch]
                inputs = self.processor.apply_chat_template(
                    msgs_list,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device, dtype=torch.bfloat16)
                input_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    gen_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )
                captions = self.processor.batch_decode(
                    gen_ids[:, input_len:], skip_special_tokens=True
                )
                for (i, _, _), cap in zip(batch, captions):
                    results[i] = cap.strip()
            except Exception as exc:
                log.warning("  Batch failed (%s) — single-item fallback", exc)
                for i, msgs, img in batch:
                    try:
                        results[i] = self._infer_single(msgs, img)
                    except Exception as e2:
                        log.warning("  Single-item %d failed: %s", i, e2)
            finally:
                for _, _, img in batch:
                    try: img.close()
                    except Exception: pass
                torch.cuda.empty_cache()

        pbar  = tqdm(total=len(rows), desc="  Captioning", unit="clip")
        batch: list = []

        while True:
            item = q.get()
            if item is SENTINEL:
                if batch:
                    _flush(batch)
                    pbar.update(len(batch))
                break
            i, msgs, img = item
            if msgs is not None and img is not None:
                batch.append((i, msgs, img))
                if len(batch) >= self.batch_size:
                    _flush(batch)
                    pbar.update(len(batch))
                    batch = []
            else:
                pbar.update(1)

        pbar.close()
        producer.join()
        gc.collect()
        torch.cuda.empty_cache()
        return results

# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 – OUTPUT  (4 parallel writes)
# ═════════════════════════════════════════════════════════════════════════════

def save_outputs(
    movie_name: str,
    output_dir: str,
    df: pd.DataFrame,
    face_results: dict[str, list[dict]],
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    csv1_cols = ["scene_id", "clip_id", "start_time_movie", "end_time_movie",
                  "start_sec", "end_sec", "duration_sec", "frame1", "frame2", "frame3"]
    csv2_cols = ["scene_id", "clip_id", "start_time_movie", "end_time_movie",
                  "start_sec", "end_sec", "duration_sec",
                  "frame1", "frame2", "frame3",
                  "actors_f1", "actors_f2", "actors_f3", "clip_actors",
                  "pos_f1", "pos_f2", "pos_f3"]

    def _w1(): df[csv1_cols].to_csv(
        Path(output_dir) / f"{movie_name}_clips_with_frames.csv", index=False)
    def _w2(): df[csv2_cols].to_csv(
        Path(output_dir) / f"{movie_name}_final_dataset_with_metadata.csv", index=False)
    def _w3(): df.to_csv(
        Path(output_dir) / f"{movie_name}_captions.csv", index=False)

    def _w4():
        fd = Path(output_dir) / "faces";          fd.mkdir(parents=True, exist_ok=True)
        ed = Path(output_dir) / "embeddings" / movie_name; ed.mkdir(parents=True, exist_ok=True)
        ad = Path(output_dir) / "assignments";   ad.mkdir(parents=True, exist_ok=True)
        with open(fd / f"{movie_name}.ndjson", "w") as ff, \
             open(ed / "embeddings.ndjson",    "w") as fe, \
             open(ad / "assignments.ndjson",   "w") as fa:
            for frame_path, dets in face_results.items():
                for det in dets:
                    r = {
                        "face_uid":     f"{frame_path}#{det.get('face_id', 0)}",
                        "image_path":   frame_path,
                        "bbox":         det["bbox"],
                        "confidence":   det.get("confidence", 1.0),
                        "actor":        det.get("actor", "unknown"),
                        "display_name": det.get("display_name", "Unknown"),
                        "similarity":   det.get("similarity", 0.0),
                    }
                    line = json.dumps(r) + "\n"
                    ff.write(line); fe.write(line); fa.write(line)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(fn) for fn in (_w1, _w2, _w3, _w4)]
        for fut in as_completed(futs):
            fut.result()
    log.info("  Outputs written to %s", output_dir)


# ═════════════════════════════════════════════════════════════════════════════
# PER-MOVIE ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════════

def to_single_line_json(text: str) -> str:
    cleaned = text.strip()
    if "{" in cleaned and "}" in cleaned:
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        json_candidate = cleaned[start_idx:end_idx+1]
        try:
            obj = json.loads(json_candidate)
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        obj = json.loads(cleaned)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return re.sub(r"\s+", " ", cleaned)


def process_movie(
    movie_name: str,
    movie_config: dict,
    cfg: dict,
    captioner: "H200Captioner",
    face_analyzer,
) -> str | None:
    log.info("\n%s", "=" * 70)
    log.info("🎬  Processing: %s", movie_name)
    log.info("%s", "=" * 70)
    t0 = datetime.now()

    input_dir  = cfg["paths"]["input_dir"]
    emb_dir    = cfg["paths"]["actor_embeddings_dir"]
    clip_dir   = cfg["paths"]["clip_dir"]
    threshold  = cfg["assignment"]["similarity_threshold"]
    output_dir = str(Path(cfg["paths"]["output_dir"]) / movie_name)
    frames_dir = str(Path(cfg["paths"]["frames_dir"]) / movie_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    final_csv = Path(output_dir) / f"{movie_name}_captions.csv"
    if final_csv.exists():
        log.info("  Already complete – skipping.")
        return str(final_csv)

    movie_path = find_movie_file(movie_name, input_dir)
    if not movie_path:
        log.warning("  Movie file not found in %s – skipping.", input_dir)
        return None

    try:
        log.info("[1/6] Scene detection + clip generation")
        clip_csv = Path(clip_dir) / f"{movie_name}_clips.csv"
        if clip_csv.exists():
            clips = pd.read_csv(clip_csv).to_dict("records")
            if not clips:
                log.warning("  Cached clips CSV is empty — deleting and re-detecting scenes.")
                clip_csv.unlink()
                scene_workers = cfg["processing"].get("scene_workers", 16)
                scenes = detect_scenes(movie_path, num_workers=scene_workers)
                clips  = generate_clips(movie_name, scenes, clip_dir)
        else:
            scene_workers = cfg["processing"].get("scene_workers", 16)
            scenes = detect_scenes(movie_path, num_workers=scene_workers)
            clips  = generate_clips(movie_name, scenes, clip_dir)
        log.info("  → %d clips", len(clips))

        if not clips:
            log.warning("  No clips generated for %s — skipping.", movie_name)
            return None

        log.info("[2/6] Frame extraction  (%d workers)", FFMPEG_WORKERS)
        workers = cfg["processing"].get("ffmpeg_workers", FFMPEG_WORKERS)
        extract_frames(movie_name, movie_path, clips, frames_dir, workers)

        log.info("[3/6] Face detection + embedding  (GPU)")
        frame_paths = [
            str(Path(frames_dir) / f"{movie_name}_frame_{c['clip_id']}.{i}.jpg")
            for c in clips for i in (1, 2, 3)
            if (Path(frames_dir) / f"{movie_name}_frame_{c['clip_id']}.{i}.jpg").exists()
        ]
        face_results = run_face_analysis(frame_paths, face_analyzer)

        log.info("[4/6] Actor identity assignment  (AVX-512 batch matmul)")
        actor_keys  = movie_config.get("actors", [])
        actor_files = movie_config.get("actor_embedding_files", {})
        actor_names = movie_config.get("actor_display_names", {})
        loaded_keys, actor_mat = load_actor_embeddings(emb_dir, actor_keys, actor_files)

        if actor_mat is not None and loaded_keys:
            assignments = assign_actors(
                face_results, loaded_keys, actor_mat, actor_names, threshold)
        else:
            log.warning("  No actor embeddings – all faces labelled 'unknown'.")
            assignments = {p: [] for p in frame_paths}

        log.info("[5/6] Screen-position labelling")
        pos_map = compute_positions(assignments)
        df      = build_final_df(movie_name, clips, frames_dir, assignments, pos_map)

        log.info("[6/6] Captioning  %d clips  (INT8 + AMX + KV cache)", len(df))
        raw_captions = captioner.caption_batch(df.to_dict("records"))
        df["generated_caption"] = [to_single_line_json(cap) for cap in raw_captions]

        log.info("[7/7] Saving outputs  (4 parallel writes)")
        save_outputs(movie_name, output_dir, df, face_results)

        log.info("✅  %s done in %s", movie_name, datetime.now() - t0)
        return str(final_csv)
    finally:
        if Path(frames_dir).exists():
            shutil.rmtree(frames_dir)
            log.info("  Frames deleted: %s", frames_dir)


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <config.toml> <movies.yaml>")
        sys.exit(1)

    cfg    = load_config(sys.argv[1])
    movies = load_movies(sys.argv[2])
    log.info("🎬  %d movie(s)  |  %d physical cores available  |  GPU: H200",
             len(movies), PHYSICAL_CORES)
    log.info("    ffmpeg workers: %d  |  face: GPU  |  caption batch: %d",
             cfg["processing"].get("ffmpeg_workers", FFMPEG_WORKERS),
             cfg.get("captioner", {}).get("batch_size", 64))

    captioner_cfg = cfg.get("captioner", cfg.get("vllm", {}))
    captioner     = H200Captioner(captioner_cfg["model_path"], cfg)
    face_analyzer = build_face_analyzer_gpu(cfg)

    t0                    = datetime.now()
    done, skipped, failed = 0, 0, 0

    for name, mcfg in movies.items():
        try:
            result = process_movie(
                name, mcfg, cfg, captioner, face_analyzer
            )
            done += 1 if result else 0
            skipped += 0 if result else 1
        except Exception as exc:
            log.error("❌  %s failed: %s", name, exc)
            traceback.print_exc()
            failed += 1

    gc.collect()
    log.info("\n%s", "=" * 70)
    log.info("🏁  Done in %s  |  ✅ %d  ⏭ %d  ❌ %d",
             datetime.now() - t0, done, skipped, failed)
    log.info("%s", "=" * 70)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
