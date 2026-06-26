#!/usr/bin/env python3
"""
JSON schemas for vLLM guided decoding.

When using vLLM's OpenAI-compatible API with guided_decoding, these schemas
enforce structured output — eliminating the fragile parse_llm_json() regex
fallback for classification and captioning.
"""

from config import BUCKETS

FILTER_NAMES = [
    "cbfc_certificate", "tobacco_warning", "anti_piracy", "production_credits",
    "blurry", "dark_underexposed", "text_heavy", "transition_frame",
    "blank_screen", "no_useful_content",
    "has_watermark",
]

# ── Classification Schema ─────────────────────────────────────────────────────

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "filters": {
            "type": "object",
            "properties": {k: {"type": "boolean"} for k in FILTER_NAMES},
            "required": FILTER_NAMES,
            "additionalProperties": False,
        },
        "t2i_suitable": {
            "type": "boolean",
        },
        "category": {
            "type": "string",
            "enum": BUCKETS + ["none"],
        },
        "description": {
            "type": "string",
        },
    },
    "required": ["filters", "t2i_suitable", "category", "description"],
    "additionalProperties": False,
}

# ── Caption Schema ────────────────────────────────────────────────────────────

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {
            "type": "string",
        },
        "tags": {
            "type": "object",
            "properties": {
                "setting": {"type": "string"},
                "lighting": {"type": "string"},
                "composition": {"type": "string"},
                "mood": {"type": "string"},
                "color_palette": {"type": "string"},
                "image_angle": {"type": "string"},
                "subject_focus": {"type": "string"},
                "time_of_day": {"type": "string"},
                "era_style": {"type": "string"},
            },
            "required": [
                "setting", "lighting", "composition", "mood",
                "color_palette", "image_angle", "subject_focus",
                "time_of_day", "era_style",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["caption", "tags"],
    "additionalProperties": False,
}

# Wrapped for vLLM /v1/chat/completions response_format parameter
CLASSIFICATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "classification",
        "strict": True,
        "schema": CLASSIFICATION_SCHEMA,
    },
}

CAPTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "caption",
        "strict": True,
        "schema": CAPTION_SCHEMA,
    },
}

# ── Watermark Detection Schema ────────────────────────────────────────────────

WATERMARK_SCHEMA = {
    "type": "object",
    "properties": {
        "has_watermark": {"type": "boolean"},
        "watermark_text": {"type": "string"},
        "bbox_x1_pct": {"type": "number"},
        "bbox_y1_pct": {"type": "number"},
        "bbox_x2_pct": {"type": "number"},
        "bbox_y2_pct": {"type": "number"},
    },
    "required": ["has_watermark", "watermark_text",
                  "bbox_x1_pct", "bbox_y1_pct", "bbox_x2_pct", "bbox_y2_pct"],
    "additionalProperties": False,
}

WATERMARK_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "watermark_detection",
        "strict": True,
        "schema": WATERMARK_SCHEMA,
    },
}
