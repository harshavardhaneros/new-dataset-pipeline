#!/usr/bin/env python3
"""
Persistent vLLM OpenAI-compatible HTTP server manager.

Launches vLLM as a subprocess on dedicated GPUs with tensor parallelism,
prefix caching, and chunked prefill. Manages health checks and graceful
shutdown.

Usage:
    server = VLLMServer(model_path="/data/kl_dev/models/Qwen2.5-VL-32B-Instruct",
                        gpu_ids=[0,1,2,3], port=8100)
    await server.start()
    await server.wait_ready()
    # ... use the server via VLLMClient ...
    await server.shutdown()
"""

import asyncio
import atexit
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class VLLMServer:
    """Manages a persistent vLLM OpenAI-compatible API server as a subprocess."""

    def __init__(
        self,
        model_path: str,
        gpu_ids: list[int] | None = None,
        port: int = 8100,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
    ):
        self.model_path = model_path
        self.gpu_ids = gpu_ids or [0, 1, 2, 3]
        self.port = port
        self.max_model_len = max_model_len
        self.dtype = dtype
        self._process: subprocess.Popen | None = None
        self._log_file = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    async def start(self, log_path: str | None = None):
        """Spawn the vLLM server as a subprocess.

        Args:
            log_path: Optional path to write server stdout/stderr.
                      Defaults to /tmp/vllm_server_{port}.log
        """
        if self.is_running:
            logger.info(f"vLLM server already running on port {self.port} (pid={self._process.pid})")
            return

        cuda_devices = ",".join(str(g) for g in self.gpu_ids)
        tp_size = len(self.gpu_ids)

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--tensor-parallel-size", str(tp_size),
            "--dtype", self.dtype,
            "--max-model-len", str(self.max_model_len),
            "--enforce-eager",
            "--enable-prefix-caching",
            "--gpu-memory-utilization", "0.92",
            "--port", str(self.port),
            "--limit-mm-per-prompt", '{"image": 1}',
            "--trust-remote-code",
        ]


        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices

        if log_path is None:
            log_path = f"/tmp/vllm_server_{self.port}.log"

        self._log_file = open(log_path, "w")

        logger.info(f"Starting vLLM server: CUDA_VISIBLE_DEVICES={cuda_devices} TP={tp_size}")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Port: {self.port}, max_model_len={self.max_model_len}")
        logger.info(f"  Log: {log_path}")

        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # New process group for clean shutdown
        )

        # Register cleanup on interpreter exit
        atexit.register(self._cleanup_sync)

        logger.info(f"  vLLM server started (pid={self._process.pid})")

    async def wait_ready(self, timeout: float = 300, poll_interval: float = 2.0):
        """Poll /health endpoint until the server is ready.

        Args:
            timeout: Max seconds to wait (default 300s — Qwen2.5-VL-32B takes ~60-90s on 4x H100).
            poll_interval: Seconds between polls.

        Raises:
            TimeoutError: If server doesn't become ready within timeout.
            RuntimeError: If server process exits during wait.
        """
        import httpx

        url = f"{self.base_url}/health"
        logger.info(f"Waiting for vLLM server at {url} (timeout={timeout}s)...")

        elapsed = 0.0
        async with httpx.AsyncClient() as client:
            while elapsed < timeout:
                # Check if process has died
                if self._process and self._process.poll() is not None:
                    rc = self._process.returncode
                    raise RuntimeError(
                        f"vLLM server exited with code {rc} during startup. "
                        f"Check logs at /tmp/vllm_server_{self.port}.log"
                    )

                try:
                    resp = await client.get(url, timeout=5.0)
                    if resp.status_code == 200:
                        logger.info(f"  vLLM server ready after {elapsed:.0f}s")
                        return
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
                    pass

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

        raise TimeoutError(
            f"vLLM server not ready after {timeout}s. "
            f"Check logs at /tmp/vllm_server_{self.port}.log"
        )

    async def shutdown(self, grace_period: float = 10.0):
        """Gracefully stop the vLLM server.

        Sends SIGTERM, waits grace_period, then SIGKILL if still alive.
        """
        if not self.is_running:
            logger.info("vLLM server not running, nothing to shut down")
            return

        pid = self._process.pid
        logger.info(f"Shutting down vLLM server (pid={pid})...")

        # Send SIGTERM to process group
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            logger.info("  Process already exited")
            self._process = None
            return

        # Wait for graceful exit
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self._process.wait),
                timeout=grace_period,
            )
            logger.info(f"  vLLM server exited gracefully (code={self._process.returncode})")
        except asyncio.TimeoutError:
            logger.warning(f"  vLLM server did not exit in {grace_period}s, sending SIGKILL")
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                self._process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass

        self._process = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def _cleanup_sync(self):
        """Synchronous cleanup for atexit handler."""
        if self.is_running:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                self._process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except Exception:
                    pass
            if self._log_file:
                self._log_file.close()

    async def restart(self, log_path: str | None = None):
        """Restart the server (shutdown + start + wait_ready)."""
        await self.shutdown()
        await asyncio.sleep(2)  # Brief pause for port release
        await self.start(log_path=log_path)
        await self.wait_ready()
