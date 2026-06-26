"""Single source of truth for the s5 classification taxonomy.

The taxonomy switched from 12 numbered buckets (bucket_01..bucket_12) to 15
named buckets defined by prompts/updated_prompt.txt. This module centralises the
bucket list, people-routing, the rich classification schema, the prompt loader,
and a tolerant parser so s5 (classify) and s6 (verify) stay in sync.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

# 15 named buckets (order matters: BUCKETS[0] is the safe fallback).
BUCKETS: List[str] = [
    "portrait_closeup",
    "two_shot",
    "group",
    "crowd",
    "song_dance",
    "action_fight",
    "interior_domestic",
    "street_urban",
    "rural_village",
    "religious_festival_ritual",
    "landscape_nature",
    "architecture_monument",
    "object_food_artifact",
    "text_poster_graphic",
    "intimate_suggestive",
]

# Buckets dominated by people -> route to actor tagging in s6/s7.
PEOPLE_BUCKETS = {
    "portrait_closeup",
    "two_shot",
    "group",
    "crowd",
    "song_dance",
    "action_fight",
    "intimate_suggestive",
}

# Extra attributes the new prompt emits; captured into metadata by s5.
ATTR_FIELDS: List[str] = [
    "dominant_shot_scale",
    "dominant_people_count",
    "has_face",
    "has_text",
    "intimate_flag",
    "motion_type",
    "camera_movement",
    "scene_transition",
    "quality_issues",
    "tags",
    "era_hint",
    "dominant_actions",
    "temporal_consistency",
    "occlusion_level",
]

# Map legacy / near-miss labels onto the new taxonomy so stray model output or
# old metadata still resolves to a valid bucket instead of the fallback.
_ALIASES = {
    "people_portraits": "portrait_closeup",
    "portrait": "portrait_closeup",
    "closeup": "portrait_closeup",
    "bucket_01": "portrait_closeup",
    "clothing_textiles": "object_food_artifact",
    "bucket_02": "object_food_artifact",
    "architecture": "architecture_monument",
    "bucket_03": "architecture_monument",
    "bucket_04": "landscape_nature",
    "urban_street": "street_urban",
    "bucket_05": "street_urban",
    "bucket_06": "rural_village",
    "food_drink": "object_food_artifact",
    "bucket_07": "object_food_artifact",
    "festivals_rituals": "religious_festival_ritual",
    "bucket_08": "religious_festival_ritual",
    "objects_artifacts": "object_food_artifact",
    "bucket_09": "object_food_artifact",
    "bucket_10": "landscape_nature",
    "art_design": "text_poster_graphic",
    "bucket_11": "text_poster_graphic",
    "abstract_texture": "text_poster_graphic",
    "bucket_12": "text_poster_graphic",
}

_FALLBACK_PROMPT = (
    "Classify this Indian-film video clip for a text-to-video dataset.\n"
    "Return exactly one JSON object.\n"
    'Set "keep": false with a "reject_reason" of certificate, statutory_warning, '
    "anti_piracy, disclaimer, or non_content for non-film content; otherwise "
    '"keep": true and "reject_reason": "none".\n'
    "Choose one primary_bucket from: " + ", ".join(BUCKETS) + ".\n"
    "People categories take priority over location categories.\n"
    'Return JSON: {"keep": true, "reject_reason": "none", "primary_bucket": "", '
    '"bucket_confidence": 0.9}'
)


def _prompt_path() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts" / "updated_prompt.txt"


@lru_cache(maxsize=1)
def load_classify_prompt() -> str:
    """Load the s5 classification prompt from prompts/updated_prompt.txt."""
    try:
        text = _prompt_path().read_text(encoding="utf-8").strip()
        if len(text) > 50:
            return text
    except Exception:
        pass
    return _FALLBACK_PROMPT


def normalize_bucket(raw: str, valid: List[str] | None = None) -> str:
    """Resolve a raw model label to a valid bucket id."""
    valid = valid or BUCKETS
    key = (raw or "").strip().lower()
    if key in valid:
        return key
    if key in _ALIASES and _ALIASES[key] in valid:
        return _ALIASES[key]
    return valid[0] if valid else BUCKETS[0]


def route_for_bucket(bucket: str) -> str:
    """Actor-tagging route: 'people' for people-dominated buckets, else 'other'."""
    b = normalize_bucket(str(bucket or ""))
    return "people" if b in PEOPLE_BUCKETS else "other"


def parse_classify_result(data: Dict[str, Any], valid: List[str] | None = None) -> Dict[str, Any]:
    """Normalise a parsed classify JSON (new or legacy schema) into a flat row.

    Returns: bucket, bucket_confidence, reject, reject_reason, attributes.
    """
    valid = valid or BUCKETS
    raw_bucket = data.get("primary_bucket") or data.get("bucket") or ""
    bucket = normalize_bucket(str(raw_bucket), valid)

    keep = data.get("keep")
    if keep is None:
        reject = bool(data.get("reject", False))
    else:
        reject = not bool(keep)

    reason = data.get("reject_reason")
    if isinstance(reason, str) and reason.strip().lower() in ("none", "null", ""):
        reason = None
    if not reject:
        reason = None

    try:
        conf = float(data.get("bucket_confidence", 0.5) or 0.5)
    except (TypeError, ValueError):
        conf = 0.5

    attributes = {f: data[f] for f in ATTR_FIELDS if f in data}

    return {
        "bucket": bucket,
        "bucket_confidence": round(conf, 4),
        "reject": reject,
        "reject_reason": reason,
        "attributes": attributes,
    }
