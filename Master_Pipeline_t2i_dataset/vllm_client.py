#!/usr/bin/env python3
"""
Async HTTP client for the persistent vLLM OpenAI-compatible server.

Handles image encoding, request formatting, and concurrent batching.
vLLM's continuous batching engine handles server-side scheduling — the client
just fires concurrent requests throttled by a semaphore.

Usage:
    client = VLLMClient("http://localhost:8100", max_concurrent=64)
    result = await client.classify(Path("image.jpg"))
    results = await client.classify_batch([Path("a.jpg"), Path("b.jpg")])
    await client.close()
"""

import asyncio
import base64
import io
import json
import logging
import time
from pathlib import Path

import httpx
from PIL import Image

from classifier import CLASSIFICATION_PROMPT, WATERMARK_DETECT_PROMPT
from common import parse_llm_json
from schemas import (CLASSIFICATION_RESPONSE_FORMAT, CAPTION_RESPONSE_FORMAT,
                     WATERMARK_RESPONSE_FORMAT)

logger = logging.getLogger(__name__)


def _encode_image(image_path: Path, max_dim: int = 512) -> str:
    """Open, resize, and convert image to base64 JPEG data URI.

    Pre-resizing before sending to vLLM avoids wasting server-side
    preprocessing on oversized images.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class VLLMClient:
    """Async HTTP client for vLLM's OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8100",
        max_concurrent: int = 64,
        model_name: str = "default",
        classify_max_tokens: int = 200,
        caption_max_tokens: int = 300,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._sem = asyncio.Semaphore(max_concurrent)
        self.model_name = model_name
        self.classify_max_tokens = classify_max_tokens
        self.caption_max_tokens = caption_max_tokens
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=30.0),
            limits=httpx.Limits(
                max_connections=max_concurrent + 10,
                max_keepalive_connections=max_concurrent,
            ),
        )
        self._request_count = 0
        self._error_count = 0

    async def _chat_completion(
        self,
        image_data_uri: str,
        prompt: str,
        max_tokens: int,
        response_format: dict | None = None,
    ) -> str:
        """Send a single chat completion request to the vLLM server."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }

        if response_format:
            payload["response_format"] = response_format

        async with self._sem:
            self._request_count += 1
            resp = await self._client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def classify(self, image_path: Path) -> dict:
        """Classify a single image using the VLM.

        Returns dict with: filters (10 booleans), rejected (computed),
        reasons (list), t2i_suitable, primary_bucket, secondary_bucket,
        category (alias for primary_bucket), description.
        """
        data_uri = _encode_image(image_path, max_dim=512)
        try:
            raw = await self._chat_completion(
                data_uri,
                CLASSIFICATION_PROMPT,
                max_tokens=self.classify_max_tokens,
                response_format=CLASSIFICATION_RESPONSE_FORMAT,
            )
            result = parse_llm_json(raw, fallback=None)
            if result is None or "_parse_error" in result:
                return self._classify_fallback("parse_error")

            # Compute rejected deterministically from filter booleans
            filters = result.get("filters", {})
            reasons = [k for k, v in filters.items() if v]
            rejected = len(reasons) > 0

            return {
                "filters": filters,
                "rejected": rejected,
                "reasons": reasons,
                "t2i_suitable": result.get("t2i_suitable", False) and not rejected,
                "category": result.get("category", "none"),
                "description": result.get("description", ""),
            }
        except Exception as e:
            self._error_count += 1
            logger.warning(f"classify failed for {image_path.name}: {e}")
            return self._classify_fallback("inference_error", str(e))

    @staticmethod
    def _classify_fallback(reason: str, error: str = "") -> dict:
        """Return a safe fallback classification result."""
        empty_filters = {
            "cbfc_certificate": False, "tobacco_warning": False,
            "anti_piracy": False, "production_credits": False,
            "blurry": False, "dark_underexposed": False,
            "text_heavy": False, "transition_frame": False,
            "blank_screen": False, "no_useful_content": False,
            "has_watermark": False,
        }
        result = {
            "filters": empty_filters,
            "rejected": True,
            "reasons": [reason],
            "t2i_suitable": False,
            "category": "none",
            "description": "",
        }
        if error:
            result["_error"] = error
        return result

    async def caption(self, image_path: Path, prompt: str) -> str:
        """Caption a single image with a bucket-specific prompt.

        Returns raw text output (caller parses JSON).
        """
        data_uri = _encode_image(image_path, max_dim=1024)
        try:
            return await self._chat_completion(
                data_uri,
                prompt,
                max_tokens=self.caption_max_tokens,
                response_format=CAPTION_RESPONSE_FORMAT,
            )
        except Exception as e:
            self._error_count += 1
            logger.warning(f"caption failed for {image_path.name}: {e}")
            return json.dumps({
                "caption": "",
                "tags": {
                    "setting": "", "lighting": "", "composition": "",
                    "mood": "", "color_palette": "", "image_angle": "",
                    "subject_focus": "", "time_of_day": "", "era_style": "",
                },
            })

    async def classify_batch(self, image_paths: list[Path]) -> list[dict]:
        """Classify a batch of images concurrently.

        Pre-encodes all images in a thread pool (CPU-bound PIL work),
        then fires concurrent HTTP requests to vLLM.
        """
        import concurrent.futures

        # Pre-encode all images in parallel threads (releases GIL)
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            data_uris = await asyncio.gather(*[
                loop.run_in_executor(pool, _encode_image, p, 512)
                for p in image_paths
            ])

        # Fire all HTTP requests concurrently with pre-encoded images
        tasks = [self._classify_with_uri(uri, p) for uri, p in zip(data_uris, image_paths)]
        return await asyncio.gather(*tasks)

    async def _classify_with_uri(self, data_uri: str, image_path: Path) -> dict:
        """Classify using a pre-encoded image data URI."""
        try:
            raw = await self._chat_completion(
                data_uri,
                CLASSIFICATION_PROMPT,
                max_tokens=self.classify_max_tokens,
                response_format=CLASSIFICATION_RESPONSE_FORMAT,
            )
            result = parse_llm_json(raw, fallback=None)
            if result is None or "_parse_error" in result:
                return self._classify_fallback("parse_error")

            filters = result.get("filters", {})
            reasons = [k for k, v in filters.items() if v]
            rejected = len(reasons) > 0

            return {
                "filters": filters,
                "rejected": rejected,
                "reasons": reasons,
                "t2i_suitable": result.get("t2i_suitable", False) and not rejected,
                "category": result.get("category", "none"),
                "description": result.get("description", ""),
            }
        except Exception as e:
            self._error_count += 1
            logger.warning(f"classify failed for {image_path.name}: {e}")
            return self._classify_fallback("inference_error", str(e))

    async def caption_batch(
        self, items: list[tuple[Path, str]]
    ) -> list[str]:
        """Caption a batch of (image_path, prompt) pairs concurrently.

        Pre-encodes images in thread pool, then fires HTTP requests.
        """
        import concurrent.futures

        loop = asyncio.get_event_loop()
        paths = [p for p, _ in items]
        prompts = [prompt for _, prompt in items]

        # Pre-encode all images in parallel (1024px for caption)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            data_uris = await asyncio.gather(*[
                loop.run_in_executor(pool, _encode_image, p, 1024)
                for p in paths
            ])

        # Fire all HTTP requests concurrently
        tasks = [self._caption_with_uri(uri, prompt, p)
                 for uri, prompt, p in zip(data_uris, prompts, paths)]
        return await asyncio.gather(*tasks)

    async def _caption_with_uri(self, data_uri: str, prompt: str, image_path: Path) -> str:
        """Caption using a pre-encoded image data URI."""
        try:
            return await self._chat_completion(
                data_uri, prompt,
                max_tokens=self.caption_max_tokens,
                response_format=CAPTION_RESPONSE_FORMAT,
            )
        except Exception as e:
            self._error_count += 1
            logger.warning(f"caption failed for {image_path.name}: {e}")
            return json.dumps({
                "caption": "", "tags": {
                    "setting": "", "lighting": "", "composition": "",
                    "mood": "", "color_palette": "", "image_angle": "",
                    "subject_focus": "", "time_of_day": "", "era_style": "",
                },
            })

    async def detect_watermark_batch(self, image_paths: list[Path]) -> list[dict]:
        """Detect watermarks via VLM — returns bbox in percentages.

        New flow: VLM sees semi-transparent logos YOLO misses (EROS, T-Series).
        Returns list of {has_watermark, watermark_text, bbox_x1_pct, ...}.
        """
        import concurrent.futures

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            data_uris = await asyncio.gather(*[
                loop.run_in_executor(pool, _encode_image, p, 512)
                for p in image_paths
            ])

        tasks = [self._detect_wm_with_uri(uri, p)
                 for uri, p in zip(data_uris, image_paths)]
        return await asyncio.gather(*tasks)

    async def _detect_wm_with_uri(self, data_uri: str, image_path: Path) -> dict:
        """Detect watermark using a pre-encoded image data URI."""
        fallback = {"has_watermark": False, "watermark_text": "",
                    "bbox_x1_pct": 0, "bbox_y1_pct": 0,
                    "bbox_x2_pct": 0, "bbox_y2_pct": 0}
        try:
            raw = await self._chat_completion(
                data_uri,
                WATERMARK_DETECT_PROMPT,
                max_tokens=100,
                response_format=WATERMARK_RESPONSE_FORMAT,
            )
            result = parse_llm_json(raw, fallback=None)
            if result is None or "_parse_error" in result:
                return fallback
            return result
        except Exception as e:
            self._error_count += 1
            logger.warning(f"watermark detect failed for {image_path.name}: {e}")
            return fallback

    @property
    def stats(self) -> dict:
        return {
            "requests": self._request_count,
            "errors": self._error_count,
            "error_rate": (self._error_count / max(self._request_count, 1)),
        }

    async def close(self):
        """Close the HTTP client and release connections."""
        await self._client.aclose()
        logger.info(f"VLLMClient closed — {self._request_count} requests, "
                     f"{self._error_count} errors")
