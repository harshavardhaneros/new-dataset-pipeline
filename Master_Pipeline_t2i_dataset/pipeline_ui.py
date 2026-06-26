#!/usr/bin/env python3
"""
Pipeline Control UI — Eros Universe Web Interface.

FastAPI backend that serves the HTML control panel, manages pipeline config,
launches pipeline runs, and streams logs to the browser.

Usage:
    python3 pipeline_ui.py                    # http://localhost:8080
    python3 pipeline_ui.py --port 9090        # custom port
    python3 pipeline_ui.py --host 0.0.0.0     # expose to network
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

MASTER_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Indic Cultural Image Pipeline")

# Pipeline process state
_pipeline_proc = None
_pipeline_log_path = None
_pipeline_config = {}
_pipeline_status = "idle"  # idle, running, completed, failed


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (MASTER_DIR / "ui" / "index.html").read_text()


@app.get("/api/gpu-info")
async def gpu_info():
    """Get current GPU status via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "id": int(parts[0]),
                    "name": parts[1],
                    "memory_used": int(parts[2]),
                    "memory_total": int(parts[3]),
                    "memory_free": int(parts[4]),
                    "utilization": int(parts[5]),
                })
        return {"gpus": gpus}
    except Exception as e:
        return {"gpus": [], "error": str(e)}


@app.get("/api/status")
async def status():
    """Get pipeline run status."""
    global _pipeline_status, _pipeline_proc
    if _pipeline_proc and _pipeline_proc.poll() is not None:
        _pipeline_status = "completed" if _pipeline_proc.returncode == 0 else "failed"
    return {
        "status": _pipeline_status,
        "config": _pipeline_config,
    }


@app.post("/api/run")
async def run_pipeline(request: Request):
    """Start a pipeline run with the given config."""
    global _pipeline_proc, _pipeline_log_path, _pipeline_config, _pipeline_status

    if _pipeline_status == "running":
        return JSONResponse({"error": "Pipeline already running"}, status_code=409)

    body = await request.json()
    _pipeline_config = body

    # Generate YAML config
    config_path = MASTER_DIR / "ui" / "_run_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(body, f, default_flow_style=False)

    # Set up log file
    work_dir = Path(body.get("work_dir", "/tmp/pipeline_output"))
    work_dir.mkdir(parents=True, exist_ok=True)
    _pipeline_log_path = work_dir / "pipeline_ui.log"

    # Launch pipeline as subprocess (-u for unbuffered output so logs stream)
    cmd = [sys.executable, "-u", str(MASTER_DIR / "pipeline.py"),
           "--config", str(config_path), "--streaming"]

    _pipeline_proc = subprocess.Popen(
        cmd,
        stdout=open(_pipeline_log_path, "w"),
        stderr=subprocess.STDOUT,
        cwd=str(MASTER_DIR),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    _pipeline_status = "running"

    return {"status": "started", "pid": _pipeline_proc.pid, "log": str(_pipeline_log_path)}


@app.get("/api/logs")
async def stream_logs():
    """Stream pipeline logs via Server-Sent Events (SSE)."""
    async def generate():
        if not _pipeline_log_path or not Path(_pipeline_log_path).exists():
            yield "data: Waiting for pipeline to start...\n\n"
            return

        with open(_pipeline_log_path, "r") as f:
            # Stream from beginning so UI shows full logs
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    if _pipeline_proc and _pipeline_proc.poll() is not None:
                        yield f"data: [PIPELINE FINISHED with code {_pipeline_proc.returncode}]\n\n"
                        break
                    await asyncio.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/stop")
async def stop_pipeline():
    """Stop the running pipeline."""
    global _pipeline_status, _pipeline_proc
    if _pipeline_proc and _pipeline_proc.poll() is None:
        _pipeline_proc.terminate()
        _pipeline_proc.wait(timeout=10)
        _pipeline_status = "stopped"
        return {"status": "stopped"}
    return {"status": "not_running"}


@app.get("/api/results")
async def get_results():
    """Get pipeline results summary."""
    work_dir = Path(_pipeline_config.get("work_dir", "/tmp/pipeline_output"))
    report_path = work_dir / "report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return {"error": "No results yet"}


@app.get("/api/defaults")
async def get_defaults():
    """Return default config values."""
    return {
        "work_dir": str(MASTER_DIR / "pipeline_output"),
        "model_path": "/data/kl_dev/models/Qwen2.5-VL-32B-Instruct",
        "vllm_gpu_ids": [0, 1, 2, 3],
        "actor_tag_gpu_id": 4,
        "clip_gpu_id": 5,
        "gdino_gpu_id": 6,
        "watermark_gpu_id": 7,
        "vllm_port": 8100,
        "vllm_max_concurrent": 128,
        "micro_batch_size": 500,
        "clip_batch_size": 64,
        "scene_threshold": 27.0,
        "frames_per_scene": 3,
        "phash_intra_threshold": 8,
        "phash_cross_threshold": 6,
        "watermark_enabled": True,
        "watermark_detect_threshold": 0.45,
        "watermark_reject_threshold": 0.9,
        "crop_letterbox": True,
        "clip_weight": 0.55,
        "icr_weight": 0.0,
        "aod_weight": 0.45,
        "gate_final": 0.25,
        "gate_review": 0.15,
        "tag_actors_enabled": True,
        "actor_similarity_threshold": 0.35,
        "caption_mix_ratio": 0.15,
        "streaming": True,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"\n  Pipeline Control UI: http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
