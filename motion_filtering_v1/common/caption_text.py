"""Normalize caption field from VLM JSON to plain string."""

from __future__ import annotations

from typing import Any


def caption_to_str(caption: Any) -> str:
    """Coerce caption from str, list, or dict to a single string for CLIP/export."""
    if caption is None:
        return ""
    if isinstance(caption, str):
        text = caption.strip()
        if text.startswith("{") and "short_description" in text:
            try:
                import json
                obj = json.loads(text)
                if obj.get("short_description"):
                    return str(obj["short_description"]).strip()
            except json.JSONDecodeError:
                pass
        return text
    if isinstance(caption, list):
        parts = [str(x).strip() for x in caption if x]
        return " ".join(parts)
    if isinstance(caption, dict):
        if caption.get("short_description"):
            return str(caption["short_description"]).strip()
        inner = caption.get("caption") or caption.get("text")
        return caption_to_str(inner)
    return str(caption).strip()
