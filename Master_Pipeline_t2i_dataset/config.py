#!/usr/bin/env python3
"""
Pipeline configuration and shared data structures.

Extracted from pipeline.py so both the sequential and streaming orchestrators
can share PipelineConfig, BK-tree, and pHash utilities.
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MASTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MASTER_DIR))

from common import IMAGE_EXTS  # noqa: E402

# ── 12 Buckets ────────────────────────────────────────────────────────────────

BUCKETS = [
    "people_portraits",
    "clothing_textiles",
    "architecture",
    "landscape_nature",
    "urban_street",
    "rural_village",
    "food_drink",
    "festivals_rituals",
    "objects_artifacts",
    "animals_wildlife",
    "art_design",
    "abstract_texture",
]

STEPS = [
    "discover", "extract", "dedup", "classify",
    "tag_actors", "caption", "score", "export", "report",
]

# Backward compat aliases for old step names
STEP_ALIASES = {
    "dedup_intra": "dedup",
    "dedup_cross": "dedup",
    "gate": "export",
}

# Source types for mixing config
SOURCE_TYPES = {"youtube", "internal", "external", "huggingface", "precaptioned"}


# ── BK-tree for O(n log n) hamming-distance dedup ─────────────────────────────

class BKTree:
    """BK-tree indexed by a distance function for efficient nearest-neighbour
    queries in metric spaces (here: hamming distance on perceptual hashes).

    insert() is O(log n) amortized; find() prunes subtrees whose distance
    band cannot contain matches, giving O(n^α) with α < 1 for small thresholds.
    """

    def __init__(self, distance_fn):
        self._dist = distance_fn
        self._root = None

    def insert(self, item):
        if self._root is None:
            self._root = (item, {})
            return
        node = self._root
        while True:
            d = self._dist(item, node[0])
            if d == 0:
                return  # exact duplicate
            if d in node[1]:
                node = node[1][d]
            else:
                node[1][d] = (item, {})
                return

    def find(self, item, threshold):
        """Return list of (stored_item, distance) within threshold."""
        if self._root is None:
            return []
        matches = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            d = self._dist(item, node[0])
            if d <= threshold:
                matches.append((node[0], d))
            for k, child in node[1].items():
                if abs(k - d) <= threshold:
                    stack.append(child)
        return matches


def compute_phash_worker(img_path_str: str) -> tuple[str, object]:
    """Compute pHash for a single image. Module-level for multiprocessing pickle.

    Returns (path_str, hash_or_None).
    """
    try:
        import imagehash
        from PIL import Image
        h = imagehash.phash(Image.open(img_path_str))
        return img_path_str, h
    except Exception:
        return img_path_str, None


def hamming_distance(a, b):
    """Hamming distance between two imagehash objects."""
    return abs(a - b)


# ── Pipeline Configuration ────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Configuration for the master pipeline."""
    # Directories
    video_dirs: list[str] = field(default_factory=lambda: [
        "/eos/youtube_videos/festivals",
        "/eos/youtube_videos/food",
        "/eos/youtube_videos/travel",
    ])
    work_dir: str = str(MASTER_DIR / "pipeline_output")
    prompt_dir: str = str(MASTER_DIR / "prompts")

    # Movie list CSV (columns: path, source_type)
    movie_list: str | None = None

    # Extra image dirs (non-video sources) — list of {"path": str, "source_type": str}
    extra_image_dirs: list[dict] = field(default_factory=lambda: [
        {"path": "/data/kl_dev/dataset/indian_cultural", "source_type": "internal"},
    ])

    # Scene detection
    scene_threshold: float = 27.0
    frames_per_scene: int = 3
    adaptive_detector: bool = False

    # Dedup
    phash_intra_threshold: int = 8
    phash_cross_threshold: int = 6

    # VLM
    gpu_ids: list[int] = field(default_factory=lambda: [0, 1])
    max_new_tokens: int = 300
    model_path: str | None = None
    backend: str = "transformers"

    # Quality scoring
    clip_model: str = "openai/clip-vit-large-patch14"
    gdino_config: str = str(MASTER_DIR / "groundingdino" / "GroundingDINO_SwinT_OGC.py")
    gdino_checkpoint: str = str(MASTER_DIR / "groundingdino" / "groundingdino_swint_ogc.pth")
    # ICR (GroundingDINO) removed — returns 0.0 for Indian cultural terms
    # (biryani, lehenga, rangoli etc. outside its vocabulary).
    # Redistributed: CLIP 0.55 + AOD 0.45
    clip_weight: float = 0.55
    icr_weight: float = 0.0
    aod_weight: float = 0.45

    # Gate thresholds — lowered for Indian cultural content where CLIP+ICR
    # scores are systematically lower due to Western training bias
    gate_final: float = 0.25
    gate_review: float = 0.15

    # Batch sizes
    clip_batch_size: int = 64  # H100 has plenty of VRAM for larger CLIP batches

    # Caption mixing
    caption_mix_ratio: float = 0.15
    caption_mix_sources: list[str] = field(default_factory=lambda: ["triveni", "drishtikon"])

    # Actor tagging
    actor_images_dir: str = str(MASTER_DIR / "actors" / "actor_images")
    actor_embeddings_dir: str = str(MASTER_DIR / "actors" / "actor_embeddings")
    yolo_face_model: str = str(MASTER_DIR / "actors" / "yolov12n-face.pt")
    actor_similarity_threshold: float = 0.35
    tag_actors_enabled: bool = True

    # Force re-processing
    force: bool = False

    # ── Watermark handling ────────────────────────────────────────────────────
    watermark_enabled: bool = True
    watermark_detect_threshold: float = 0.45  # raised from 0.3 for joycaption model (reduces false positives)
    watermark_reject_threshold: float = 0.9   # above this = reject (too heavy)
    watermark_gpu_id: int = 7                 # GPU for LaMA inpainting
    crop_letterbox: bool = True               # auto-crop black borders on video frames

    # ── Streaming mode fields ─────────────────────────────────────────────────
    streaming: bool = False
    vllm_port: int = 8100
    vllm_gpu_ids: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    actor_tag_gpu_id: int = 4
    clip_gpu_id: int = 5
    gdino_gpu_id: int = 6
    micro_batch_size: int = 500
    vllm_max_concurrent: int = 128  # H100 vLLM handles 256+, 128 is safe


def load_config(config_path: str | None) -> PipelineConfig:
    """Load config from YAML or return defaults."""
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return PipelineConfig(**{k: v for k, v in data.items()
                                  if k in PipelineConfig.__dataclass_fields__})
    return PipelineConfig()


# ── Shared utilities ──────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def step_done(cfg: PipelineConfig, step_name: str) -> bool:
    """Check if a step has a .done marker."""
    marker = Path(cfg.work_dir) / f".done_{step_name}"
    return marker.exists()


def mark_done(cfg: PipelineConfig, step_name: str):
    """Write a .done marker for a step."""
    marker = Path(cfg.work_dir) / f".done_{step_name}"
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def atomic_write_json(path: Path, data: dict):
    """Write JSON atomically via tmp + rename to prevent corruption on crash."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)
