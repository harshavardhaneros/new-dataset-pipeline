#!/usr/bin/env python3
"""
Streaming concurrent pipeline orchestrator.

Replaces the sequential run_pipeline() with an asyncio-based concurrent
orchestrator that streams micro-batches through parallel GPU stages.

GPU allocation:
    GPUs 0-3: Qwen2.5-VL-32B via vLLM (TP=4) — classify + caption
    GPU 4:    YOLO + InsightFace — actor tagging
    GPU 5:    CLIP — image-text alignment scoring
    GPU 6:    GroundingDINO — noun grounding (ICR scoring)

Usage:
    from stream_orchestrator import StreamingPipeline
    from config import PipelineConfig
    import asyncio

    cfg = PipelineConfig(streaming=True, model_path="models/Qwen2.5-VL-32B-Instruct")
    asyncio.run(StreamingPipeline(cfg).run())
"""

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from common import IMAGE_EXTS, unique_stem, parse_llm_json
from config import (
    PipelineConfig, BKTree, compute_phash_worker, hamming_distance,
    BUCKETS, STEPS,
    ensure_dir, step_done, mark_done, atomic_write_json,
)

logger = logging.getLogger(__name__)


# ── Micro-batch ──────────────────────────────────────────────────────────────

@dataclass
class MicroBatch:
    batch_id: int
    image_paths: list[Path]
    # Per-image metadata keyed by str(image_path)
    vlm_results: dict[str, dict] = field(default_factory=dict)
    actor_tags: dict[str, list] = field(default_factory=dict)
    captions: dict[str, dict] = field(default_factory=dict)
    scores: dict[str, dict] = field(default_factory=dict)
    source_types: dict[str, str] = field(default_factory=dict)


# ── Streaming Pipeline ───────────────────────────────────────────────────────

class StreamingPipeline:
    """Asyncio-based concurrent pipeline orchestrator."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.work_dir = Path(cfg.work_dir)

        # Linear queue chain: classify → tag_actors → caption → score
        self._classify_q: asyncio.Queue[MicroBatch | None] = asyncio.Queue(maxsize=4)
        self._tag_actors_q: asyncio.Queue[MicroBatch | None] = asyncio.Queue(maxsize=4)
        self._caption_q: asyncio.Queue[MicroBatch | None] = asyncio.Queue(maxsize=4)
        self._score_q: asyncio.Queue[MicroBatch | None] = asyncio.Queue(maxsize=4)

        # Accumulated results for final report
        self._all_scores_rows: list[dict] = []
        self._gate_counts = {"final": 0, "review": 0, "discard": 0}

        # GPU worker processes and queues
        self._worker_input_qs: dict[str, mp.Queue] = {}
        self._worker_output_qs: dict[str, mp.Queue] = {}
        self._workers: dict[str, object] = {}

        # vLLM server and client
        self._vllm_server = None
        self._vllm_client = None

        # Prompts for captioning (loaded once)
        self._bucket_prompts: dict[str, str] = {}

        # Source type map (built during discover)
        self._source_type_map: dict[str, str] = {}

        # Stats
        self._stage_times: dict[str, float] = {}
        self._images_processed = 0

    async def run(self):
        """Main entry point. Runs the full streaming pipeline."""
        t0 = time.time()
        work = ensure_dir(self.cfg.work_dir)

        # Set up logging
        fh = logging.FileHandler(work / "pipeline.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(fh)

        logger.info("=" * 70)
        logger.info("  STREAMING PIPELINE — 8-GPU Concurrent Mode")
        logger.info("=" * 70)
        logger.info(f"  Work directory: {self.cfg.work_dir}")
        logger.info(f"  vLLM GPUs: {self.cfg.vllm_gpu_ids} (TP={len(self.cfg.vllm_gpu_ids)})")
        logger.info(f"  Actor tag GPU: {self.cfg.actor_tag_gpu_id}")
        logger.info(f"  CLIP GPU: {self.cfg.clip_gpu_id}")
        logger.info(f"  GroundingDINO GPU: {self.cfg.gdino_gpu_id}")
        logger.info(f"  Micro-batch size: {self.cfg.micro_batch_size}")

        try:
            # Phase 1: Sequential CPU stages (must complete before GPU work)
            await self._run_cpu_stages()

            # Phase 2: Collect all images
            all_images = self._collect_images()
            if not all_images:
                logger.info("No images to process after CPU stages.")
                return

            # Phase 2b: Letterbox crop only (no watermark detection yet)
            if self.cfg.crop_letterbox:
                self._run_letterbox_crop(all_images)

            logger.info(f"  Total images for GPU stages: {len(all_images)}")

            # Phase 3: Start GPU infrastructure (vLLM + workers)
            await self._start_infrastructure()

            # Phase 3b: VLM watermark detection → LaMA inpaint → YOLO verify
            # VLM catches semi-transparent logos (EROS, T-Series) that YOLO misses.
            if self.cfg.watermark_enabled:
                all_images = await self._vlm_watermark_cleanup(all_images)

            # Phase 4: Stream micro-batches through concurrent GPU stages
            await self._run_gpu_stages(all_images)

            # Phase 5: Write final outputs and report
            self._write_scores_csv()
            self._run_export()  # gate threshold applied inline
            self._run_report()

            total = time.time() - t0
            logger.info("=" * 70)
            logger.info(f"  STREAMING PIPELINE COMPLETE — {total:.1f}s total")
            logger.info(f"  Images processed: {self._images_processed}")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            self._write_checkpoint()
            raise
        finally:
            await self._shutdown_infrastructure()

    # ── Phase 1: Sequential CPU stages ────────────────────────────────────────

    async def _run_cpu_stages(self):
        """Run discover, extract, dedup_intra synchronously."""
        import pipeline as seq

        # Step 1: Discover
        if not self.cfg.force and step_done(self.cfg, "discover"):
            logger.info("Step 'discover' already done, loading manifest...")
        else:
            seq.step_discover(self.cfg)

        # Step 2: Extract
        if not self.cfg.force and step_done(self.cfg, "extract"):
            logger.info("Step 'extract' already done, skipping...")
        else:
            seq.step_extract(self.cfg)

        # Step 3: Unified dedup (intra + cross)
        if not self.cfg.force and step_done(self.cfg, "dedup"):
            logger.info("Step 'dedup' already done, skipping...")
        else:
            seq.step_dedup(self.cfg)

        # Build source_type_map from manifest
        self._build_source_type_map()

    def _build_source_type_map(self):
        """Build mapping of image_path -> source_type from manifest."""
        manifest_path = self.work_dir / "manifest.json"
        if not manifest_path.exists():
            return

        with open(manifest_path) as f:
            manifest = json.load(f)

        frames_dir = self.work_dir / "frames"

        # Map video frames → source_type via parent directory
        for ventry in manifest.get("videos", []):
            if isinstance(ventry, str):
                st = "youtube"
                vp = Path(ventry)
            else:
                st = ventry.get("source_type", "youtube")
                vp = Path(ventry["path"])
            # Frames are in frames/{video_stem}/
            video_stem = vp.stem
            video_frame_dir = frames_dir / video_stem
            if video_frame_dir.exists():
                for img in video_frame_dir.iterdir():
                    if img.suffix.lower() in IMAGE_EXTS:
                        self._source_type_map[str(img)] = st

        # Map extra images → source_type
        for img_entry in manifest.get("images", []):
            if isinstance(img_entry, dict):
                st = img_entry.get("source_type", "internal")
                img_path = img_entry.get("path", "")
                self._source_type_map[str(img_path)] = st
                # Also map the symlinked path in frames/
                dataset = img_entry.get("dataset", Path(img_path).parent.name)
                linked = frames_dir / f"_extra_{dataset}" / Path(img_path).name
                self._source_type_map[str(linked)] = st

    def _collect_images(self) -> list[Path]:
        """Collect all images from frames/ after CPU stages."""
        frames_dir = self.work_dir / "frames"
        if not frames_dir.exists():
            return []

        images = []
        for p in sorted(frames_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                # Skip dedup'd images
                if "/_dupes" in str(p) or "\\_dupes" in str(p):
                    continue
                images.append(p)
        return images

    def _run_letterbox_crop(self, all_images: list[Path]):
        """Crop black letterbox borders from video frames (CPU-only)."""
        from watermark_handler import crop_letterbox

        video_frames = [p for p in all_images if "_extra_" not in str(p)]
        if video_frames:
            logger.info(f"  Cropping letterbox borders on {len(video_frames)} video frames...")
            for p in video_frames:
                crop_letterbox(p, overwrite=True)
            logger.info(f"  Cropped {len(video_frames)} video frames")

    async def _vlm_watermark_cleanup(self, all_images: list[Path]) -> list[Path]:
        """VLM → LaMA → YOLO watermark pipeline.

        Flow:
          1. VLM scans ALL images for watermark + bounding box
          2. LaMA inpaints images where VLM found watermark
          3. YOLO verifies: if YOLO still detects watermark → reject image
        """
        import subprocess, sys

        logger.info(f"  [watermark] VLM scanning {len(all_images)} images...")
        t0 = time.time()

        # ── Step 1: VLM detect watermarks with bounding boxes ────────────
        detections = await self._vllm_client.detect_watermark_batch(all_images)

        wm_found = []
        wm_results_dir = ensure_dir(self.work_dir / "watermark_results")
        for p, det in zip(all_images, detections):
            has_wm = det.get("has_watermark", False)
            # Write sidecar for inspector
            sidecar = {
                "image": str(p), "image_name": p.name,
                "vlm_detected": has_wm,
                "watermark_text": det.get("watermark_text", ""),
                "bbox_pct": [det.get("bbox_x1_pct", 0), det.get("bbox_y1_pct", 0),
                             det.get("bbox_x2_pct", 0), det.get("bbox_y2_pct", 0)],
                "status": "clean" if not has_wm else "pending_inpaint",
            }
            atomic_write_json(wm_results_dir / f"{unique_stem(p)}_wm.json", sidecar)
            if has_wm:
                wm_found.append((p, det))

        logger.info(f"  [watermark] VLM found {len(wm_found)}/{len(all_images)} with watermarks")

        if not wm_found:
            logger.info(f"  [watermark] Done in {time.time()-t0:.1f}s (all clean)")
            return all_images

        # ── Step 2: LaMA inpaint VLM-detected regions ────────────────────
        # Convert pct bboxes to pixel bboxes
        inpaint_list = []
        for p, det in wm_found:
            from PIL import Image as PILImage
            img = PILImage.open(p)
            w, h = img.size
            img.close()
            x1 = int(det.get("bbox_x1_pct", 0) / 100 * w)
            y1 = int(det.get("bbox_y1_pct", 0) / 100 * h)
            x2 = int(det.get("bbox_x2_pct", 0) / 100 * w)
            y2 = int(det.get("bbox_y2_pct", 0) / 100 * h)
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            inpaint_list.append({
                "image": str(p), "bbox": [x1, y1, x2, y2],
                "text": det.get("watermark_text", ""),
            })

        if inpaint_list:
            logger.info(f"  [watermark] LaMA inpainting {len(inpaint_list)} images...")
            wm_vlm_file = self.work_dir / "vlm_watermark_detections.json"
            with open(wm_vlm_file, "w") as f:
                json.dump(inpaint_list, f)

            # LaMA subprocess (CUDA isolation) — uses run_lama_inpaint.py
            lama_cmd = [
                sys.executable,
                str(Path(__file__).parent / "run_lama_inpaint.py"),
                "--detections", str(wm_vlm_file),
                "--gpu", str(self.cfg.watermark_gpu_id),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.cfg.watermark_gpu_id)
            proc = subprocess.run(
                lama_cmd,
                capture_output=True, text=True, env=env,
            )
            if proc.returncode == 0:
                logger.info(f"  [watermark] LaMA cleaned {len(inpaint_list)} images")
            else:
                logger.warning(f"  [watermark] LaMA error: {proc.stderr[-300:]}")

        # ── Step 3: YOLO verify — check if watermark is still present ────
        # Only verify the images that were inpainted
        inpainted_paths = [Path(item["image"]) for item in inpaint_list]
        if inpainted_paths:
            logger.info(f"  [watermark] YOLO verifying {len(inpainted_paths)} cleaned images...")
            yolo_reject = []
            try:
                # Import YOLO detector (CPU-only to avoid CUDA issues)
                from watermark_handler import WatermarkDetector
                detector = WatermarkDetector(
                    conf_threshold=self.cfg.watermark_detect_threshold,
                    use_gpu=False,  # CPU for verification
                )
                for p in inpainted_paths:
                    result = detector.detect(p)
                    if result["has_watermark"]:
                        yolo_reject.append(p)
                        # Update sidecar
                        sidecar_path = wm_results_dir / f"{unique_stem(p)}_wm.json"
                        if sidecar_path.exists():
                            with open(sidecar_path) as f:
                                sc = json.load(f)
                            sc["status"] = "rejected_yolo_verify"
                            atomic_write_json(sidecar_path, sc)
                del detector
            except Exception as e:
                logger.warning(f"  [watermark] YOLO verify failed: {e}")

            if yolo_reject:
                logger.info(f"  [watermark] YOLO rejected {len(yolo_reject)} (watermark still present)")
                reject_set = set(str(p) for p in yolo_reject)
                all_images = [p for p in all_images if str(p) not in reject_set]

        # Update sidecars for cleaned images
        for item in inpaint_list:
            p = Path(item["image"])
            sidecar_path = wm_results_dir / f"{unique_stem(p)}_wm.json"
            if sidecar_path.exists():
                with open(sidecar_path) as f:
                    sc = json.load(f)
                if sc.get("status") == "pending_inpaint":
                    sc["status"] = "cleaned"
                    atomic_write_json(sidecar_path, sc)

        logger.info(f"  [watermark] Done in {time.time()-t0:.1f}s — "
                     f"{len(all_images)} images remaining")
        return all_images

    # ── Phase 3: Infrastructure ───────────────────────────────────────────────

    async def _start_infrastructure(self):
        """Start vLLM server and GPU worker processes."""
        from vllm_server import VLLMServer
        from vllm_client import VLLMClient
        from gpu_workers import ActorTagWorker, CLIPScoreWorker
        import httpx

        model_path = self.cfg.model_path
        if not model_path:
            from common import MODEL_ID
            model_path = MODEL_ID

        # Check if a vLLM server is already running on the configured port
        base_url = f"http://localhost:{self.cfg.vllm_port}"
        server_already_running = False
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{base_url}/health", timeout=5.0)
                if resp.status_code == 200:
                    server_already_running = True
                    logger.info(f"Found existing vLLM server at {base_url}, reusing it")
        except Exception:
            pass

        if not server_already_running:
            self._vllm_server = VLLMServer(
                model_path=model_path,
                gpu_ids=self.cfg.vllm_gpu_ids,
                port=self.cfg.vllm_port,
            )
            log_path = str(self.work_dir / "vllm_server.log")
            await self._vllm_server.start(log_path=log_path)
            await self._vllm_server.wait_ready()

        # Resolve model name — query the server for its model ID
        vllm_model_name = model_path
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{base_url}/v1/models", timeout=10.0)
                if resp.status_code == 200:
                    models = resp.json().get("data", [])
                    if models:
                        vllm_model_name = models[0]["id"]
                        logger.info(f"  vLLM model name: {vllm_model_name}")
        except Exception:
            pass

        # Create vLLM client
        self._vllm_client = VLLMClient(
            base_url=base_url,
            max_concurrent=self.cfg.vllm_max_concurrent,
            model_name=vllm_model_name,
        )

        # Load bucket prompts for captioning
        from captioner import load_prompts
        self._bucket_prompts = load_prompts(Path(self.cfg.prompt_dir))

        # Start GPU workers
        self._start_worker(
            "actor_tag",
            ActorTagWorker(
                gpu_id=self.cfg.actor_tag_gpu_id,
                input_queue=mp.Queue(maxsize=8),
                output_queue=mp.Queue(maxsize=8),
                actor_embeddings_dir=self.cfg.actor_embeddings_dir,
                yolo_model_path=self.cfg.yolo_face_model,
                actor_images_dir=self.cfg.actor_images_dir,
                similarity_threshold=self.cfg.actor_similarity_threshold,
            ),
        )
        self._start_worker(
            "clip",
            CLIPScoreWorker(
                gpu_id=self.cfg.clip_gpu_id,
                input_queue=mp.Queue(maxsize=8),
                output_queue=mp.Queue(maxsize=8),
                model_name=self.cfg.clip_model,
                batch_size=self.cfg.clip_batch_size,
            ),
        )
        # Second CLIP worker for faster scoring
        self._start_worker(
            "clip2",
            CLIPScoreWorker(
                gpu_id=self.cfg.gdino_gpu_id,
                input_queue=mp.Queue(maxsize=8),
                output_queue=mp.Queue(maxsize=8),
                model_name=self.cfg.clip_model,
                batch_size=self.cfg.clip_batch_size,
            ),
        )

    def _start_worker(self, name: str, worker):
        """Register and start a GPU worker."""
        self._workers[name] = worker
        self._worker_input_qs[name] = worker.input_queue
        self._worker_output_qs[name] = worker.output_queue
        worker.start()

    async def _shutdown_infrastructure(self):
        """Shut down vLLM server and all GPU workers."""
        # Stop GPU workers
        for name, worker in self._workers.items():
            try:
                worker.stop()
                logger.info(f"Worker '{name}' stopped")
            except Exception as e:
                logger.warning(f"Error stopping worker '{name}': {e}")

        # Close vLLM client
        if self._vllm_client:
            await self._vllm_client.close()

        # Shut down vLLM server
        if self._vllm_server:
            await self._vllm_server.shutdown()

    # ── Phase 4: Concurrent GPU stages ────────────────────────────────────────

    async def _run_gpu_stages(self, all_images: list[Path]):
        """Stream micro-batches through concurrent stages."""
        # Launch all stage coroutines
        tasks = [
            asyncio.create_task(self._stage_classify(), name="classify"),
            asyncio.create_task(self._stage_tag_actors(), name="tag_actors"),
            asyncio.create_task(self._stage_caption(), name="caption"),
            asyncio.create_task(self._stage_score(), name="score"),
        ]

        # Feed micro-batches to classify only (linear chain, no fan-out)
        batch_size = self.cfg.micro_batch_size
        batch_id = 0
        for start in range(0, len(all_images), batch_size):
            chunk = all_images[start:start + batch_size]
            mb = MicroBatch(batch_id=batch_id, image_paths=chunk)
            for p in chunk:
                mb.source_types[str(p)] = self._source_type_map.get(str(p), "")
            await self._classify_q.put(mb)
            batch_id += 1
            logger.info(f"  Fed micro-batch {batch_id}/{(len(all_images) + batch_size - 1) // batch_size}")

        await self._classify_q.put(None)
        await asyncio.gather(*tasks)
        logger.info("All GPU stages complete")

    # ── Stage: Classify ───────────────────────────────────────────────────────

    async def _stage_classify(self):
        """Hybrid classify: computational filters first, then VLM for semantics + bucketing."""
        from image_filters import batch_compute_filters

        vlm_dir = ensure_dir(self.work_dir / "vlm_results")
        t0 = time.time()
        total = 0
        comp_rejected = 0

        while True:
            mb = await self._classify_q.get()
            if mb is None:
                await self._tag_actors_q.put(None)
                break

            logger.info(f"  [classify] Batch {mb.batch_id}: {len(mb.image_paths)} images")

            # Check for already-done sidecars (idempotency)
            to_classify = []
            for p in mb.image_paths:
                ustem = unique_stem(p)
                sidecar = vlm_dir / f"{ustem}.json"
                if sidecar.exists() and not self.cfg.force:
                    with open(sidecar) as f:
                        mb.vlm_results[str(p)] = json.load(f)
                else:
                    to_classify.append(p)

            if to_classify:
                # Stage 1: Computational filters (fast, <5ms per image)
                comp_results = batch_compute_filters(to_classify)

                # Split: images that pass computational filters → send to VLM
                # Images that fail → reject immediately, skip VLM (saves cost)
                vlm_candidates = []
                for p, comp in zip(to_classify, comp_results):
                    comp_reasons = [k for k in ("blurry", "dark_underexposed", "text_heavy",
                                                 "overexposed", "low_contrast") if comp.get(k)]
                    if comp_reasons:
                        # Reject by computational filter — no VLM needed
                        ustem = unique_stem(p)
                        sidecar_data = {
                            "image": str(p),
                            "image_name": p.name,
                            "source_dir": p.parent.name,
                            "source_type": mb.source_types.get(str(p), ""),
                            "filters": {
                                "cbfc_certificate": False, "tobacco_warning": False,
                                "anti_piracy": False, "production_credits": False,
                                "blurry": comp.get("blurry", False),
                                "dark_underexposed": comp.get("dark_underexposed", False),
                                "text_heavy": comp.get("text_heavy", False),
                                "transition_frame": False, "blank_screen": False,
                                "no_useful_content": False,
                                "overexposed": comp.get("overexposed", False),
                                "low_contrast": comp.get("low_contrast", False),
                            },
                            "filter_source": "computational",
                            "metrics": comp.get("metrics", {}),
                            "rejected": True,
                            "reasons": comp_reasons,
                            "t2i_suitable": False,
                            "category": "none",
                            "description": "",
                        }
                        atomic_write_json(vlm_dir / f"{ustem}.json", sidecar_data)
                        mb.vlm_results[str(p)] = sidecar_data
                        comp_rejected += 1
                    else:
                        vlm_candidates.append((p, comp))

                # Stage 2: VLM for semantic filters + bucketing (only clean images)
                if vlm_candidates:
                    vlm_paths = [p for p, _ in vlm_candidates]
                    vlm_results = await self._vllm_client.classify_batch(vlm_paths)

                    for (p, comp), result in zip(vlm_candidates, vlm_results):
                        ustem = unique_stem(p)
                        # Merge: computational metrics + VLM semantic results
                        vlm_filters = result.get("filters", {})
                        # Override blur/dark/text with computational ground truth
                        vlm_filters["blurry"] = comp.get("blurry", False)
                        vlm_filters["dark_underexposed"] = comp.get("dark_underexposed", False)
                        vlm_filters["text_heavy"] = comp.get("text_heavy", False)

                        all_reasons = [k for k, v in vlm_filters.items() if v]
                        rejected = len(all_reasons) > 0

                        sidecar_data = {
                            "image": str(p),
                            "image_name": p.name,
                            "source_dir": p.parent.name,
                            "source_type": mb.source_types.get(str(p), ""),
                            "filters": vlm_filters,
                            "filter_source": "hybrid",
                            "metrics": comp.get("metrics", {}),
                            "rejected": rejected,
                            "reasons": all_reasons,
                            "t2i_suitable": result.get("t2i_suitable", False) and not rejected,
                            "category": result.get("category", "none"),
                            "description": result.get("description", ""),
                        }
                        atomic_write_json(vlm_dir / f"{ustem}.json", sidecar_data)
                        mb.vlm_results[str(p)] = sidecar_data

            total += len(mb.image_paths)
            await self._tag_actors_q.put(mb)

        self._stage_times["classify"] = time.time() - t0
        logger.info(f"  [classify] Done: {total} images in {self._stage_times['classify']:.1f}s")

    # ── Stage: Cross-Dedup ────────────────────────────────────────────────────

    # dedup_cross removed — now part of unified step_dedup in CPU stages

    # ── Stage: Tag Actors (linear, after classify) ────────────────────────────

    async def _stage_tag_actors(self):
        """Tag actors on people_portraits only, then push to caption queue.

        Linear chain: receives classified micro-batches from classify stage,
        filters to people_portraits, tags actors, pushes to caption.
        """
        t0 = time.time()
        actor_tags_dir = ensure_dir(self.work_dir / "actor_tags")
        loop = asyncio.get_event_loop()

        if not self.cfg.tag_actors_enabled:
            # Pass through — no tagging
            while True:
                mb = await self._tag_actors_q.get()
                if mb is None:
                    await self._caption_q.put(None)
                    break
                await self._caption_q.put(mb)
            self._stage_times["tag_actors"] = time.time() - t0
            return

        worker = self._workers["actor_tag"]

        while True:
            mb = await self._tag_actors_q.get()
            if mb is None:
                await self._caption_q.put(None)
                break

            # Filter to people_portraits only (from classify results)
            to_tag = []
            tag_results: dict[str, list] = {}
            for p in mb.image_paths:
                r = mb.vlm_results.get(str(p), {})
                if r.get("rejected", True):
                    continue
                if r.get("category", "") != "people_portraits":
                    continue
                ustem = unique_stem(p)
                sidecar = actor_tags_dir / f"{ustem}_actors.json"
                if sidecar.exists() and not self.cfg.force:
                    try:
                        with open(sidecar) as f:
                            data = json.load(f)
                        tag_results[str(p)] = data.get("actors", [])
                    except Exception:
                        to_tag.append(p)
                else:
                    to_tag.append(p)

            if to_tag:
                batch_paths = [str(p) for p in to_tag]
                worker.input_queue.put((mb.batch_id, batch_paths))
                result = await loop.run_in_executor(None, worker.output_queue.get)
                if result[0] == "result":
                    _, bid, worker_results = result
                    for p_str, actors in worker_results.items():
                        tag_results[p_str] = actors
                        p = Path(p_str)
                        ustem = unique_stem(p)
                        atomic_write_json(
                            actor_tags_dir / f"{ustem}_actors.json",
                            {"image": p_str, "image_name": p.name, "actors": actors},
                        )
                elif result[0] == "error":
                    logger.warning(f"Actor tag batch {mb.batch_id} failed: {result}")

            mb.actor_tags = tag_results
            await self._caption_q.put(mb)

        self._stage_times["tag_actors"] = time.time() - t0
        logger.info(f"  [tag_actors] Done in {self._stage_times['tag_actors']:.1f}s")

    # ── Stage: Caption ────────────────────────────────────────────────────────

    async def _stage_caption(self):
        """Caption accepted images via vLLM, write sidecars."""
        from captioner import _build_prompt, normalize_bucket
        caption_dir = ensure_dir(self.work_dir / "captions")
        t0 = time.time()
        total = 0

        while True:
            mb = await self._caption_q.get()
            if mb is None:
                await self._score_q.put(None)
                break

            # Filter to accepted, non-duplicate images
            to_caption = []
            for p in mb.image_paths:
                r = mb.vlm_results.get(str(p), {})
                if r.get("rejected", True):
                    continue

                ustem = unique_stem(p)
                sidecar = caption_dir / f"{ustem}_caption.json"
                if sidecar.exists() and not self.cfg.force:
                    try:
                        with open(sidecar) as f:
                            mb.captions[str(p)] = json.load(f)
                        continue
                    except Exception:
                        pass

                category = normalize_bucket(r.get("category", "none"))
                actors = mb.actor_tags.get(str(p), [])
                item = {
                    "image_path": p,
                    "category": category,
                    "actors": actors,
                }
                # Check for paired .txt file with cultural label
                # (resolves symlinks for _extra_ images to find the original .txt)
                real_path = p.resolve() if p.is_symlink() else p
                txt_path = real_path.with_suffix(".txt")
                if txt_path.exists():
                    try:
                        item["original_caption"] = txt_path.read_text(
                            encoding="utf-8", errors="replace"
                        ).strip()
                    except Exception:
                        pass
                prompt = _build_prompt(item, self._bucket_prompts)
                to_caption.append((p, prompt, category, r))

            if to_caption:
                # Send to vLLM
                caption_items = [(p, prompt) for p, prompt, _, _ in to_caption]
                raw_results = await self._vllm_client.caption_batch(caption_items)

                for (p, prompt, category, r), raw in zip(to_caption, raw_results):
                    parsed = parse_llm_json(raw, fallback={
                        "caption": "",
                        "tags": {},
                    })
                    ustem = unique_stem(p)
                    caption_data = {
                        "image": str(p),
                        "image_name": p.name,
                        "bucket": category,
                        "source_type": mb.source_types.get(str(p), ""),
                        "caption": parsed.get("caption", ""),
                        "tags": parsed.get("tags", {}),
                        "model": "vllm_streaming",
                    }
                    atomic_write_json(caption_dir / f"{ustem}_caption.json", caption_data)
                    mb.captions[str(p)] = caption_data

                    # Also write plain text caption
                    txt_path = caption_dir / f"{ustem}_recaptioned.txt"
                    txt_path.write_text(parsed.get("caption", ""))

            total += len([p for p in mb.image_paths
                          if not mb.vlm_results.get(str(p), {}).get("rejected", True)])
            await self._score_q.put(mb)

        self._stage_times["caption"] = time.time() - t0
        logger.info(f"  [caption] Done: {total} images in {self._stage_times['caption']:.1f}s")

    # ── Stage: Score ──────────────────────────────────────────────────────────

    async def _stage_score(self):
        """Score captioned images using CLIP, GroundingDINO, and AOD workers."""
        from scorer import compute_combined
        from gpu_workers import AODScoreWorker

        t0 = time.time()
        loop = asyncio.get_event_loop()
        aod_worker = AODScoreWorker()
        total = 0

        while True:
            mb = await self._score_q.get()
            if mb is None:
                break

            # Collect items that have captions
            score_items = []
            for p in mb.image_paths:
                cap_data = mb.captions.get(str(p))
                if cap_data and cap_data.get("caption"):
                    score_items.append({
                        "image_path": str(p),
                        "caption": cap_data["caption"],
                        "bucket": cap_data.get("bucket", ""),
                        "source_type": cap_data.get("source_type", ""),
                        "model": cap_data.get("model", ""),
                    })

            if not score_items:
                continue

            # Fan out scoring in parallel
            has_clip2 = "clip2" in self._workers
            has_gdino = "gdino" in self._workers

            if has_clip2:
                # Split work between two CLIP workers (GPU 5 + GPU 6)
                mid = len(score_items) // 2
                items_a = score_items[:mid]
                items_b = score_items[mid:]
                self._workers["clip"].input_queue.put((mb.batch_id, items_a))
                self._workers["clip2"].input_queue.put((mb.batch_id, items_b))
            else:
                self._workers["clip"].input_queue.put((mb.batch_id, score_items))

            if has_gdino:
                self._workers["gdino"].input_queue.put((mb.batch_id, score_items))

            # AOD on CPU concurrently
            aod_future = loop.run_in_executor(None, aod_worker.process_batch, score_items)

            # Collect CLIP results
            if has_clip2:
                clip_result_a = await loop.run_in_executor(None, self._workers["clip"].output_queue.get)
                clip_result_b = await loop.run_in_executor(None, self._workers["clip2"].output_queue.get)
                scores_a = clip_result_a[2] if clip_result_a[0] == "result" else [0.0] * len(items_a)
                scores_b = clip_result_b[2] if clip_result_b[0] == "result" else [0.0] * len(items_b)
                clip_scores = scores_a + scores_b
            else:
                clip_result = await loop.run_in_executor(None, self._workers["clip"].output_queue.get)
                clip_scores = clip_result[2] if clip_result[0] == "result" else [0.0] * len(score_items)

            if has_gdino:
                gdino_result = await loop.run_in_executor(None, self._workers["gdino"].output_queue.get)
                icr_scores = gdino_result[2] if gdino_result[0] == "result" else [0.0] * len(score_items)
            else:
                icr_scores = [0.0] * len(score_items)
            aod_scores, noun_counts = await aod_future

            # Combine scores
            cw, iw, aw = self.cfg.clip_weight, self.cfg.icr_weight, self.cfg.aod_weight
            for i, item in enumerate(score_items):
                row = {
                    "image_path": item["image_path"],
                    "caption": item["caption"],
                    "model": item["model"],
                    "bucket": item["bucket"],
                    "source_type": item["source_type"],
                    "clip_score": clip_scores[i],
                    "aod_score": aod_scores[i],
                    "noun_count": noun_counts[i],
                    "icr_score": icr_scores[i],
                }
                row["combined_score"] = compute_combined(row, clip_w=cw, icr_w=iw, aod_w=aw)
                self._all_scores_rows.append(row)

            total += len(score_items)

        self._stage_times["score"] = time.time() - t0
        logger.info(f"  [score] Done: {total} images in {self._stage_times['score']:.1f}s")

    # ── Phase 5: Final outputs ────────────────────────────────────────────────

    def _write_scores_csv(self):
        """Write accumulated scores to scores.csv."""
        import pandas as pd
        scores_path = self.work_dir / "scores.csv"
        df = pd.DataFrame(self._all_scores_rows)
        if not df.empty:
            df.to_csv(scores_path, index=False)
            logger.info(f"  Wrote {len(df)} rows to {scores_path}")
            logger.info(f"  Mean scores: CLIP={df['clip_score'].mean():.3f}, "
                         f"ICR={df['icr_score'].mean():.3f}, AOD={df['aod_score'].mean():.3f}, "
                         f"Combined={df['combined_score'].mean():.3f}")
        mark_done(self.cfg, "score")

    def _run_gate(self):
        """Apply threshold gating."""
        import pandas as pd
        scores_path = self.work_dir / "scores.csv"
        if not scores_path.exists():
            return

        df = pd.read_csv(scores_path)
        df["gate"] = df["combined_score"].apply(
            lambda s: "final" if s >= self.cfg.gate_final
            else ("review" if s >= self.cfg.gate_review else "discard")
        )
        gated_path = self.work_dir / "gated.csv"
        df.to_csv(gated_path, index=False)

        counts = df["gate"].value_counts().to_dict()
        self._gate_counts = counts
        logger.info(f"  Gate results: {counts}")
        self._images_processed = len(df)
        mark_done(self.cfg, "gate")

    def _run_export(self):
        """Apply gate threshold + export final images, captions, and metadata."""
        import pandas as pd

        scores_path = self.work_dir / "scores.csv"
        if not scores_path.exists():
            return

        # Apply gate threshold inline
        df = pd.read_csv(scores_path)
        df["gate"] = df["combined_score"].apply(
            lambda s: "final" if s >= self.cfg.gate_final
            else ("review" if s >= self.cfg.gate_review else "discard")
        )
        gated_path = self.work_dir / "gated.csv"
        df.to_csv(gated_path, index=False)

        counts = df["gate"].value_counts().to_dict()
        self._gate_counts = counts
        self._images_processed = len(df)
        logger.info(f"  Gate results: {counts}")

        final_df = df[df["gate"] == "final"]

        export_dir = ensure_dir(self.work_dir / "export")
        img_dir = ensure_dir(export_dir / "images")
        cap_dir = ensure_dir(export_dir / "captions")
        caption_dir = self.work_dir / "captions"

        # Load VLM descriptions for caption mixing
        vlm_descriptions = {}
        if self.cfg.caption_mix_ratio > 0:
            vlm_dir = self.work_dir / "vlm_results"
            if vlm_dir.exists():
                for jp in vlm_dir.glob("*.json"):
                    with open(jp) as f:
                        vdata = json.load(f)
                    vlm_descriptions[vdata.get("image", "")] = vdata.get("description", "")

        metadata_rows = []
        for _, row in final_df.iterrows():
            src = Path(row["image_path"])
            if not src.exists():
                continue

            ustem = unique_stem(src)
            dst_name = f"{ustem}{src.suffix}"
            dst = img_dir / dst_name

            if not dst.exists():
                shutil.copy2(src, dst)

            # Caption
            caption_text = str(row.get("caption", ""))
            caption_source = "qwen"

            # Caption mixing
            source_type = row.get("source_type", "")
            bucket = row.get("bucket", "")
            if self.cfg.caption_mix_ratio > 0 and self.cfg.caption_mix_sources:
                should_mix = any(
                    s in str(source_type).lower() or s in str(bucket).lower()
                    for s in self.cfg.caption_mix_sources
                )
                if should_mix:
                    h = int(hashlib.md5(src.name.encode()).hexdigest(), 16)
                    if h % 1000 < self.cfg.caption_mix_ratio * 1000:
                        original_desc = vlm_descriptions.get(str(src), "")
                        if original_desc:
                            caption_text = original_desc
                            caption_source = "original"

            # Write text caption
            (cap_dir / f"{ustem}.txt").write_text(caption_text)

            # Copy caption JSON if exists
            src_json = caption_dir / f"{ustem}_caption.json"
            if src_json.exists():
                shutil.copy2(src_json, cap_dir / f"{ustem}_caption.json")

            metadata_rows.append({
                "image": dst_name,
                "image_path": str(src),
                "bucket": bucket,
                "source_type": source_type,
                "caption": caption_text,
                "combined_score": row.get("combined_score", 0),
                "clip_score": row.get("clip_score", 0),
                "icr_score": row.get("icr_score", 0),
                "aod_score": row.get("aod_score", 0),
                "noun_count": row.get("noun_count", 0),
                "gate": "final",
                "model": row.get("model", ""),
                "caption_source": caption_source,
            })

        if metadata_rows:
            pd.DataFrame(metadata_rows).to_csv(export_dir / "metadata.csv", index=False)

        logger.info(f"  Exported {len(metadata_rows)} final images to {export_dir}")

        # Export review images to separate folder for human inspection
        review_df = df[df["gate"] == "review"]
        if not review_df.empty:
            review_dir = ensure_dir(export_dir / "review")
            review_img_dir = ensure_dir(review_dir / "images")
            review_count = 0
            for _, row in review_df.iterrows():
                src = Path(row["image_path"])
                if not src.exists():
                    continue
                ustem = unique_stem(src)
                dst = review_img_dir / f"{ustem}{src.suffix}"
                if not dst.exists():
                    shutil.copy2(src, dst)
                # Copy caption if exists
                cap_json = caption_dir / f"{ustem}_caption.json"
                if cap_json.exists():
                    shutil.copy2(cap_json, review_dir / f"{ustem}_caption.json")
                review_count += 1
            logger.info(f"  Review images: {review_count} saved to {review_dir}")

        # Export rejected images (gate=discard + classify-rejected)
        rejected_dir = ensure_dir(export_dir / "rejected")
        rejected_img_dir = ensure_dir(rejected_dir / "images")
        rejected_count = 0

        # 1) Gate-discarded (scored but below review threshold)
        discard_df = df[df["gate"] == "discard"]
        for _, row in discard_df.iterrows():
            src = Path(row["image_path"])
            if not src.exists():
                continue
            ustem = unique_stem(src)
            dst = rejected_img_dir / f"{ustem}{src.suffix}"
            if not dst.exists():
                shutil.copy2(src, dst)
            rejected_count += 1

        # 2) Classify-rejected (filtered out by VLM or computational filters)
        vlm_dir = self.work_dir / "vlm_results"
        if vlm_dir.exists():
            for jp in vlm_dir.glob("*.json"):
                with open(jp) as f:
                    vdata = json.load(f)
                if not vdata.get("rejected", False):
                    continue
                img_path = Path(vdata.get("image", ""))
                if not img_path.exists():
                    continue
                ustem = unique_stem(img_path)
                dst = rejected_img_dir / f"{ustem}{img_path.suffix}"
                if not dst.exists():
                    shutil.copy2(img_path, dst)
                rejected_count += 1

        logger.info(f"  Rejected images: {rejected_count} saved to {rejected_dir}")

        mark_done(self.cfg, "export")

    def _run_report(self):
        """Generate report.txt and report.json."""
        import pandas as pd

        gated_path = self.work_dir / "gated.csv"
        if not gated_path.exists():
            return

        df = pd.read_csv(gated_path)
        report_lines = [
            "=" * 70,
            "  PIPELINE REPORT (Streaming Mode)",
            "=" * 70,
            f"  Total scored: {len(df)}",
            f"  Gate counts: {df['gate'].value_counts().to_dict()}",
            "",
        ]

        # Score stats
        for col in ["combined_score", "clip_score", "icr_score", "aod_score"]:
            if col in df.columns:
                report_lines.append(
                    f"  {col}: mean={df[col].mean():.3f} std={df[col].std():.3f} "
                    f"min={df[col].min():.3f} max={df[col].max():.3f}"
                )

        # Per-bucket
        report_lines.append("\n  Per-bucket breakdown (final only):")
        final_df = df[df["gate"] == "final"]
        if "bucket" in final_df.columns:
            for bucket, grp in final_df.groupby("bucket"):
                report_lines.append(
                    f"    {bucket}: {len(grp)} images, "
                    f"mean_combined={grp['combined_score'].mean():.3f}"
                )

        report_text = "\n".join(report_lines)
        (self.work_dir / "report.txt").write_text(report_text)

        # JSON report
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "streaming",
            "total_scored": len(df),
            "gate_counts": df["gate"].value_counts().to_dict(),
            "score_stats": {},
            "per_bucket": {},
            "stage_times": self._stage_times,
        }

        for col in ["combined_score", "clip_score", "icr_score", "aod_score"]:
            if col in df.columns:
                report_data["score_stats"][col] = {
                    "mean": round(df[col].mean(), 4),
                    "std": round(df[col].std(), 4),
                    "min": round(df[col].min(), 4),
                    "max": round(df[col].max(), 4),
                    "median": round(df[col].median(), 4),
                }

        if "bucket" in df.columns:
            for bucket, grp in df.groupby("bucket"):
                report_data["per_bucket"][bucket] = {
                    "count": len(grp),
                    "gate_counts": grp["gate"].value_counts().to_dict(),
                    "mean_combined": round(grp["combined_score"].mean(), 4),
                }

        atomic_write_json(self.work_dir / "report.json", report_data)
        logger.info(f"  Report written to {self.work_dir / 'report.txt'}")

        # Generate dashboard
        try:
            from dashboard import generate_dashboard
            generate_dashboard(str(self.work_dir))
            logger.info(f"  Dashboard written to {self.work_dir / 'dashboard.html'}")
        except Exception as e:
            logger.warning(f"  Dashboard generation failed: {e}")

        mark_done(self.cfg, "report")

    def _write_checkpoint(self):
        """Write checkpoint for crash recovery."""
        checkpoint = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "images_processed": self._images_processed,
            "stage_times": self._stage_times,
            "vllm_stats": self._vllm_client.stats if self._vllm_client else {},
        }
        atomic_write_json(self.work_dir / "streaming_checkpoint.json", checkpoint)
        logger.info(f"  Checkpoint written to {self.work_dir / 'streaming_checkpoint.json'}")
