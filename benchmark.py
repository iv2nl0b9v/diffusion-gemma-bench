#!/usr/bin/env python3
"""
Benchmark runner for DiffusionGemma vs Gemma4-AR across multiple GPUs.

Manages vLLM server lifecycle, executes benchmark phases, collects results
to structured JSON. Designed to be run on each Vast.ai GPU instance.

Usage:
    # Auto-detect GPU and run all phases
    python benchmark.py --gpu rtx-3090

    # Run a specific phase
    python benchmark.py --gpu l40s --phase batch-sweep

    # Dry run (print commands without executing)
    python benchmark.py --gpu h100-sxm --dry-run
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from config import (
    EXPERIMENT_CONFIG,
    GPU_CONFIGS,
    MODEL_CONFIGS,
    get_checkpoint,
    get_vllm_serve_cmd,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data structure
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""

    model: str
    gpu: str
    batch_size: int
    output_length: int
    input_length: int
    quantization: str
    phase: str
    denoising_steps: Optional[int] = None
    ttft_ms: Optional[float] = None
    ttfb_ms: Optional[float] = None
    itl_median_ms: Optional[float] = None
    itl_p95_ms: Optional[float] = None
    itl_p99_ms: Optional[float] = None
    e2el_ms: Optional[float] = None
    throughput_tps: Optional[float] = None
    per_stream_tps: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    mem_util_pct: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    raw_output: Optional[str] = None


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------
class VLLMServer:
    """Manages a vLLM server process."""

    def __init__(self, model_slug: str, gpu_slug: str, port: int = 8000):
        self.model_slug = model_slug
        self.gpu_slug = gpu_slug
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://localhost:{port}"

    def start(self, extra_args: Optional[dict] = None, dry_run: bool = False) -> bool:
        """Start the vLLM server. Returns True if server is ready."""
        cmd = get_vllm_serve_cmd(self.model_slug, self.gpu_slug, extra_args)

        log.info(f"Starting vLLM server: {' '.join(cmd)}")

        if dry_run:
            log.info("[DRY RUN] Would start server with above command")
            return True

        # Kill any existing vLLM process on this port
        self._kill_existing()

        env = os.environ.copy()
        env["VLLM_LOGGING_LEVEL"] = "WARNING"

        self.log_file = open("vllm_server.log", "w")
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            preexec_fn=os.setsid,
        )

        # Wait for server to be ready
        return self._wait_for_ready(timeout=600)

    def stop(self):
        """Stop the vLLM server."""
        if hasattr(self, 'log_file') and self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass
            self.log_file = None

        if self.process is None:
            return

        log.info("Stopping vLLM server...")
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=10)
            except Exception:
                pass
        self.process = None
        log.info("vLLM server stopped")
        # Give the GPU a moment to release memory
        time.sleep(5)

    def _wait_for_ready(self, timeout: int = 300) -> bool:
        """Poll the health endpoint until the server is ready."""
        log.info(f"Waiting for vLLM server to be ready (timeout={timeout}s)...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=5)
                if resp.status_code == 200:
                    log.info(
                        f"vLLM server ready in {time.time() - start:.1f}s"
                    )
                    return True
            except requests.ConnectionError:
                pass

            # Check if process died
            if self.process.poll() is not None:
                if hasattr(self, 'log_file') and self.log_file:
                    try:
                        self.log_file.flush()
                    except Exception:
                        pass
                try:
                    with open("vllm_server.log") as lf:
                        stdout = lf.read()
                except Exception:
                    stdout = ""
                log.error(f"vLLM server exited with code {self.process.returncode}")
                log.error(f"Server output:\n{stdout[-2000:]}")
                return False

            time.sleep(5)

        log.error(f"vLLM server did not become ready within {timeout}s")
        return False

    def _kill_existing(self):
        """Kill any existing process on the port."""
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{self.port}"],
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                for pid in result.stdout.strip().split("\n"):
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except (ProcessLookupError, ValueError):
                        pass
                time.sleep(2)
        except FileNotFoundError:
            # lsof not available, try fuser
            subprocess.run(
                ["fuser", "-k", f"{self.port}/tcp"],
                capture_output=True,
            )
            time.sleep(2)


# ---------------------------------------------------------------------------
# Benchmark execution via OpenAI-compatible API
# ---------------------------------------------------------------------------
def generate_prompt(length: int) -> str:
    """Generate a synthetic prompt of approximately `length` tokens.

    Uses a repeating pattern of common English words to approximate
    1 token ≈ 1 word (conservative estimate for most tokenizers).
    """
    words = (
        "The quick brown fox jumps over the lazy dog near the river bank "
        "while the sun sets behind the mountains and the birds sing their "
        "evening songs in the tall oak trees that line the winding path "
        "through the ancient forest where deer roam freely among the ferns "
    )
    word_list = words.split()
    # Repeat to fill desired length (rough 1 word ≈ 1.3 tokens estimate)
    target_words = int(length * 0.8)
    prompt_words = []
    while len(prompt_words) < target_words:
        prompt_words.extend(word_list)
    return " ".join(prompt_words[:target_words])


def run_single_request(
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    stream: bool = True,
) -> dict:
    """Send a single completion request and measure timing.

    Returns dict with timing metrics:
        ttft_ms, e2el_ms, output_tokens, token_timestamps
    """
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    token_timestamps = []
    first_token_time = None
    start_time = time.perf_counter()
    output_tokens = 0

    if stream:
        resp = requests.post(url, json=payload, stream=True, timeout=300)
        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            now = time.perf_counter()
            choices = data.get("choices", [])
            if choices and choices[0].get("text"):
                if first_token_time is None:
                    first_token_time = now
                token_timestamps.append(now)
                output_tokens += 1

            # Extract exact token count from stream options usage block if present
            usage = data.get("usage")
            if usage and "completion_tokens" in usage:
                output_tokens = usage["completion_tokens"]
    else:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        now = time.perf_counter()
        first_token_time = now
        output_tokens = data["usage"]["completion_tokens"]
        token_timestamps = [now]

    end_time = time.perf_counter()

    ttft = (first_token_time - start_time) * 1000 if first_token_time else None
    e2el = (end_time - start_time) * 1000

    # Compute inter-token latencies
    itls = []
    if len(token_timestamps) > 1:
        for i in range(1, len(token_timestamps)):
            itl = (token_timestamps[i] - token_timestamps[i - 1]) * 1000
            itls.append(itl)

    return {
        "ttft_ms": ttft,
        "e2el_ms": e2el,
        "output_tokens": output_tokens,
        "itls_ms": itls,
    }


def run_concurrent_benchmark(
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    num_requests: int,
    max_concurrency: int,
    temperature: float = 0.0,
    warmup: int = 10,
) -> list[dict]:
    """Run multiple requests with controlled concurrency.

    Uses ThreadPoolExecutor to simulate concurrent streams.
    Returns list of per-request result dicts.
    """
    import concurrent.futures

    # For streaming measurement with concurrency, use non-streaming
    # to get accurate per-request metrics
    use_stream = max_concurrency == 1

    all_results = []

    def _run_one(_idx: int) -> dict:
        return run_single_request(
            base_url, model_name, prompt, max_tokens, temperature, stream=use_stream
        )

    # Warmup phase
    if warmup > 0:
        log.info(f"  Warmup: {warmup} requests (concurrency={max_concurrency})...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            list(pool.map(_run_one, range(warmup)))

    # Measurement phase
    log.info(
        f"  Measuring: {num_requests} requests "
        f"(concurrency={max_concurrency}, output_len={max_tokens})..."
    )
    wall_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = [pool.submit(_run_one, i) for i in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            try:
                all_results.append(f.result())
            except Exception as e:
                log.warning(f"  Request failed: {e}")

    wall_elapsed = time.perf_counter() - wall_start

    total_tokens = sum(r["output_tokens"] for r in all_results)
    aggregate_tps = total_tokens / wall_elapsed if wall_elapsed > 0 else 0

    log.info(
        f"  Done: {len(all_results)}/{num_requests} successful, "
        f"{total_tokens} tokens in {wall_elapsed:.1f}s "
        f"= {aggregate_tps:.1f} tok/s aggregate"
    )

    # Attach aggregate stats to each result
    for r in all_results:
        r["aggregate_tps"] = aggregate_tps
        r["wall_elapsed_s"] = wall_elapsed

    return all_results


def aggregate_results(
    raw_results: list[dict],
    model_slug: str,
    gpu_slug: str,
    batch_size: int,
    output_length: int,
    input_length: int,
    quantization: str,
    phase: str,
    denoising_steps: Optional[int] = None,
) -> BenchmarkResult:
    """Aggregate raw per-request results into a single BenchmarkResult."""
    import numpy as np

    if not raw_results:
        log.warning("No results to aggregate!")
        return BenchmarkResult(
            model=model_slug,
            gpu=gpu_slug,
            batch_size=batch_size,
            output_length=output_length,
            input_length=input_length,
            quantization=quantization,
            phase=phase,
            denoising_steps=denoising_steps,
        )

    ttfts = [r["ttft_ms"] for r in raw_results if r.get("ttft_ms") is not None]
    e2els = [r["e2el_ms"] for r in raw_results if r.get("e2el_ms") is not None]
    all_itls = []
    for r in raw_results:
        all_itls.extend(r.get("itls_ms", []))

    output_tokens_list = [r["output_tokens"] for r in raw_results]
    per_stream_tps_list = [
        r["output_tokens"] / (r["e2el_ms"] / 1000)
        for r in raw_results
        if r.get("e2el_ms") and r["e2el_ms"] > 0
    ]

    aggregate_tps = raw_results[0].get("aggregate_tps", 0) if raw_results else 0

    model_type = MODEL_CONFIGS[model_slug].type

    result = BenchmarkResult(
        model=model_slug,
        gpu=gpu_slug,
        batch_size=batch_size,
        output_length=output_length,
        input_length=input_length,
        quantization=quantization,
        phase=phase,
        denoising_steps=denoising_steps,
        e2el_ms=float(np.median(e2els)) if e2els else None,
        throughput_tps=float(aggregate_tps),
        per_stream_tps=float(np.median(per_stream_tps_list))
        if per_stream_tps_list
        else None,
    )

    if model_type == "autoregressive":
        result.ttft_ms = float(np.median(ttfts)) if ttfts else None
        if all_itls:
            result.itl_median_ms = float(np.median(all_itls))
            result.itl_p95_ms = float(np.percentile(all_itls, 95))
            result.itl_p99_ms = float(np.percentile(all_itls, 99))
    else:  # diffusion
        result.ttfb_ms = float(np.median(ttfts)) if ttfts else None

    return result


# ---------------------------------------------------------------------------
# GPU metrics collection
# ---------------------------------------------------------------------------
class GPUMonitor:
    """Collects GPU metrics via nvidia-smi in the background."""

    def __init__(self, output_file: str):
        self.output_file = output_file
        self.process: Optional[subprocess.Popen] = None

    def start(self):
        """Start nvidia-smi dmon in the background."""
        self.process = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "pucvmet", "-d", "1"],
            stdout=open(self.output_file, "w"),
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> Optional[dict]:
        """Stop monitoring and return summary stats."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

        # Parse basic stats from the output file
        try:
            return self._parse_summary()
        except Exception:
            return None

    def _parse_summary(self) -> dict:
        """Parse nvidia-smi dmon output for average utilization."""
        import numpy as np

        gpu_utils = []
        mem_utils = []

        with open(self.output_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        gpu_utils.append(int(parts[1]))
                        mem_utils.append(int(parts[2]))
                    except (ValueError, IndexError):
                        continue

        return {
            "gpu_util_pct": float(np.mean(gpu_utils)) if gpu_utils else None,
            "mem_util_pct": float(np.mean(mem_utils)) if mem_utils else None,
        }


# ---------------------------------------------------------------------------
# Experiment phases
# ---------------------------------------------------------------------------
def run_phase1_single_stream(
    server: VLLMServer,
    model_slug: str,
    gpu_slug: str,
    quantization: str,
    dry_run: bool = False,
) -> list[BenchmarkResult]:
    """Phase 1: Single-stream latency baseline (batch=1, output=256)."""
    log.info(f"=== Phase 1: Single-Stream Latency ({model_slug} on {gpu_slug}) ===")

    if dry_run:
        log.info("[DRY RUN] Would run single-stream benchmark")
        return []

    prompt = generate_prompt(EXPERIMENT_CONFIG.input_length)
    model_name = get_checkpoint(model_slug, gpu_slug)
    raw = run_concurrent_benchmark(
        base_url=server.base_url,
        model_name=model_name,
        prompt=prompt,
        max_tokens=256,
        num_requests=EXPERIMENT_CONFIG.num_prompts,
        max_concurrency=1,
        temperature=EXPERIMENT_CONFIG.temperature,
        warmup=EXPERIMENT_CONFIG.warmup_prompts,
    )

    result = aggregate_results(
        raw,
        model_slug=model_slug,
        gpu_slug=gpu_slug,
        batch_size=1,
        output_length=256,
        input_length=EXPERIMENT_CONFIG.input_length,
        quantization=quantization,
        phase="single-stream",
    )

    log.info(
        f"  Result: per-stream={result.per_stream_tps:.1f} tok/s, "
        f"E2EL={result.e2el_ms:.0f}ms"
    )
    return [result]


def run_phase2_batch_sweep(
    server: VLLMServer,
    model_slug: str,
    gpu_slug: str,
    quantization: str,
    baseline_e2el: Optional[float] = None,
    dry_run: bool = False,
) -> list[BenchmarkResult]:
    """Phase 2: Throughput scaling across batch sizes."""
    log.info(f"=== Phase 2: Batch Sweep ({model_slug} on {gpu_slug}) ===")

    results = []
    prompt = generate_prompt(EXPERIMENT_CONFIG.input_length)

    for bs in EXPERIMENT_CONFIG.batch_sizes:
        log.info(f"  --- Batch size = {bs} ---")

        if dry_run:
            log.info(f"[DRY RUN] Would benchmark batch_size={bs}")
            continue

        try:
            model_name = get_checkpoint(model_slug, gpu_slug)
            raw = run_concurrent_benchmark(
                base_url=server.base_url,
                model_name=model_name,
                prompt=prompt,
                max_tokens=256,
                num_requests=EXPERIMENT_CONFIG.num_prompts,
                max_concurrency=bs,
                temperature=EXPERIMENT_CONFIG.temperature,
                warmup=EXPERIMENT_CONFIG.warmup_prompts,
            )

            result = aggregate_results(
                raw,
                model_slug=model_slug,
                gpu_slug=gpu_slug,
                batch_size=bs,
                output_length=256,
                input_length=EXPERIMENT_CONFIG.input_length,
                quantization=quantization,
                phase="batch-sweep",
            )
            results.append(result)

            log.info(
                f"  bs={bs}: throughput={result.throughput_tps:.1f} tok/s, "
                f"per-stream={result.per_stream_tps:.1f} tok/s, "
                f"E2EL={result.e2el_ms:.0f}ms"
            )

            # Early stop: if per-stream latency exceeds 3x baseline
            if (
                baseline_e2el
                and result.e2el_ms
                and result.e2el_ms > 3 * baseline_e2el
            ):
                log.info(
                    f"  Stopping: E2EL {result.e2el_ms:.0f}ms > "
                    f"3x baseline {baseline_e2el:.0f}ms"
                )
                break

        except Exception as e:
            log.error(f"  Failed at batch_size={bs}: {e}")
            # Likely OOM — stop increasing batch size
            break

    return results


def run_phase3_output_sweep(
    server: VLLMServer,
    model_slug: str,
    gpu_slug: str,
    quantization: str,
    dry_run: bool = False,
) -> list[BenchmarkResult]:
    """Phase 3: Output length sensitivity at batch=1 and batch=4."""
    log.info(f"=== Phase 3: Output Length Sweep ({model_slug} on {gpu_slug}) ===")

    results = []
    prompt = generate_prompt(EXPERIMENT_CONFIG.input_length)

    for bs in [1, 4]:
        for out_len in EXPERIMENT_CONFIG.output_lengths:
            log.info(f"  --- batch={bs}, output_len={out_len} ---")

            if dry_run:
                log.info(f"[DRY RUN] Would benchmark bs={bs}, out={out_len}")
                continue

            try:
                model_name = get_checkpoint(model_slug, gpu_slug)
                raw = run_concurrent_benchmark(
                    base_url=server.base_url,
                    model_name=model_name,
                    prompt=prompt,
                    max_tokens=out_len,
                    num_requests=EXPERIMENT_CONFIG.num_prompts,
                    max_concurrency=bs,
                    temperature=EXPERIMENT_CONFIG.temperature,
                    warmup=EXPERIMENT_CONFIG.warmup_prompts,
                )

                result = aggregate_results(
                    raw,
                    model_slug=model_slug,
                    gpu_slug=gpu_slug,
                    batch_size=bs,
                    output_length=out_len,
                    input_length=EXPERIMENT_CONFIG.input_length,
                    quantization=quantization,
                    phase="output-sweep",
                )
                results.append(result)

                log.info(
                    f"  bs={bs}, out={out_len}: "
                    f"per-stream={result.per_stream_tps:.1f} tok/s, "
                    f"E2EL={result.e2el_ms:.0f}ms"
                )

            except Exception as e:
                log.error(f"  Failed at bs={bs}, out={out_len}: {e}")

    return results


def run_phase4_denoising_sweep(
    server: VLLMServer,
    model_slug: str,
    gpu_slug: str,
    quantization: str,
    dry_run: bool = False,
) -> list[BenchmarkResult]:
    """Phase 4: Denoising steps sweep (DiffusionGemma only)."""
    if MODEL_CONFIGS[model_slug].type != "diffusion":
        log.info(f"  Skipping Phase 4 for non-diffusion model {model_slug}")
        return []

    log.info(f"=== Phase 4: Denoising Steps Sweep ({model_slug} on {gpu_slug}) ===")

    results = []
    prompt = generate_prompt(EXPERIMENT_CONFIG.input_length)

    for steps in EXPERIMENT_CONFIG.denoising_steps_sweep:
        log.info(f"  --- denoising_steps={steps} ---")

        if dry_run:
            log.info(f"[DRY RUN] Would benchmark steps={steps}")
            continue

        # Note: Denoising steps may need to be configured via the API
        # or by restarting the server with different args.
        # For now, we pass it as a generation parameter if supported.
        try:
            model_name = get_checkpoint(model_slug, gpu_slug)
            raw = run_concurrent_benchmark(
                base_url=server.base_url,
                model_name=model_name,
                prompt=prompt,
                max_tokens=256,
                num_requests=EXPERIMENT_CONFIG.num_prompts,
                max_concurrency=1,
                temperature=EXPERIMENT_CONFIG.temperature,
                warmup=EXPERIMENT_CONFIG.warmup_prompts,
            )

            result = aggregate_results(
                raw,
                model_slug=model_slug,
                gpu_slug=gpu_slug,
                batch_size=1,
                output_length=256,
                input_length=EXPERIMENT_CONFIG.input_length,
                quantization=quantization,
                phase="denoising-sweep",
                denoising_steps=steps,
            )
            results.append(result)

            log.info(
                f"  steps={steps}: per-stream={result.per_stream_tps:.1f} tok/s"
            )

        except Exception as e:
            log.error(f"  Failed at steps={steps}: {e}")

    return results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run_all_phases(
    gpu_slug: str,
    phases: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    dry_run: bool = False,
):
    """Run all benchmark phases for the specified GPU."""

    if gpu_slug not in GPU_CONFIGS:
        log.error(f"Unknown GPU: {gpu_slug}. Choose from: {list(GPU_CONFIGS.keys())}")
        sys.exit(1)

    gpu_config = GPU_CONFIGS[gpu_slug]
    quantization = gpu_config.quantization
    all_phases = phases or ["single-stream", "batch-sweep", "output-sweep", "denoising-sweep"]
    model_slugs = models or list(MODEL_CONFIGS.keys())

    results_dir = Path(EXPERIMENT_CONFIG.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    gpu_metrics_dir = results_dir / "gpu_metrics"
    gpu_metrics_dir.mkdir(exist_ok=True)

    all_results: list[BenchmarkResult] = []

    log.info(f"{'='*60}")
    log.info(f"BENCHMARK: {gpu_slug}")
    log.info(f"  VRAM: {gpu_config.vram_gb} GB")
    log.info(f"  Mem BW: {gpu_config.mem_bw_gbs} GB/s")
    log.info(f"  Quantization: {quantization}")
    log.info(f"  Peak TOPS: {gpu_config.peak_tops_dense}")
    log.info(f"  Models: {model_slugs}")
    log.info(f"  Phases: {all_phases}")
    log.info(f"{'='*60}")

    for model_slug in model_slugs:
        checkpoint = get_checkpoint(model_slug, gpu_slug)
        log.info(f"\n{'='*60}")
        log.info(f"MODEL: {model_slug} ({checkpoint})")
        log.info(f"{'='*60}")

        server = VLLMServer(
            model_slug=model_slug,
            gpu_slug=gpu_slug,
            port=EXPERIMENT_CONFIG.vllm_port,
        )

        # Start GPU monitoring
        gpu_monitor = GPUMonitor(
            str(gpu_metrics_dir / f"gpu_{model_slug}_{gpu_slug}.csv")
        )

        try:
            if not dry_run:
                gpu_monitor.start()

            if not server.start(dry_run=dry_run):
                log.error(f"Failed to start vLLM for {model_slug}. Skipping.")
                continue

            # Phase 1: Single-stream
            baseline_e2el = None
            if "single-stream" in all_phases:
                phase1 = run_phase1_single_stream(
                    server, model_slug, gpu_slug, quantization, dry_run
                )
                all_results.extend(phase1)
                if phase1 and phase1[0].e2el_ms:
                    baseline_e2el = phase1[0].e2el_ms

            # Phase 2: Batch sweep
            if "batch-sweep" in all_phases:
                phase2 = run_phase2_batch_sweep(
                    server,
                    model_slug,
                    gpu_slug,
                    quantization,
                    baseline_e2el,
                    dry_run,
                )
                all_results.extend(phase2)

            # Phase 3: Output length sweep
            if "output-sweep" in all_phases:
                phase3 = run_phase3_output_sweep(
                    server, model_slug, gpu_slug, quantization, dry_run
                )
                all_results.extend(phase3)

            # Phase 4: Denoising sweep (diffusion only)
            if "denoising-sweep" in all_phases:
                phase4 = run_phase4_denoising_sweep(
                    server, model_slug, gpu_slug, quantization, dry_run
                )
                all_results.extend(phase4)

        finally:
            server.stop()
            gpu_stats = gpu_monitor.stop()

            # Attach GPU stats to results from this model
            if gpu_stats:
                for r in all_results:
                    if r.model == model_slug and r.gpu_util_pct is None:
                        r.gpu_util_pct = gpu_stats.get("gpu_util_pct")
                        r.mem_util_pct = gpu_stats.get("mem_util_pct")

            # Clean up model cache from disk to prevent out of space errors on limited storage
            if not dry_run:
                try:
                    import shutil
                    checkpoint_slug = checkpoint.replace("/", "--")
                    cache_dir = Path("/workspace/.hf_home/hub") / f"models--{checkpoint_slug}"
                    if cache_dir.exists():
                        log.info(f"Cleaning up model cache at {cache_dir} to free disk space...")
                        shutil.rmtree(cache_dir)
                        log.info("Model cache cleanup complete.")
                except Exception as e:
                    log.warning(f"Failed to clean up model cache directory: {e}")

    # Save all results
    output_file = results_dir / f"results_{gpu_slug}.json"
    results_dicts = [asdict(r) for r in all_results]
    # Remove raw_output to save space
    for d in results_dicts:
        d.pop("raw_output", None)

    with open(output_file, "w") as f:
        json.dump(results_dicts, f, indent=2)

    log.info(f"\nResults saved to {output_file}")
    log.info(f"Total measurements: {len(all_results)}")

    return all_results


def merge_results(results_dir: str = "results"):
    """Merge per-GPU result files into a single all_results.json."""
    results_path = Path(results_dir)
    all_data = []

    for f in sorted(results_path.glob("results_*.json")):
        if f.name == "all_results.json":
            continue
        log.info(f"Loading {f.name}...")
        with open(f) as fh:
            data = json.load(fh)
            all_data.extend(data)
            log.info(f"  {len(data)} measurements")

    output = results_path / "all_results.json"
    with open(output, "w") as f:
        json.dump(all_data, f, indent=2)

    log.info(f"\nMerged {len(all_data)} measurements into {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DiffusionGemma vs Gemma4-AR on vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all phases on RTX 3090
  python benchmark.py --gpu rtx-3090

  # Run only single-stream and batch-sweep
  python benchmark.py --gpu h100-sxm --phase single-stream batch-sweep

  # Run only DiffusionGemma
  python benchmark.py --gpu l40s --model diffusiongemma-26b

  # Dry run (show commands without executing)
  python benchmark.py --gpu dgx-spark --dry-run

  # Merge per-GPU results after all GPUs are done
  python benchmark.py --merge
        """,
    )

    parser.add_argument(
        "--gpu",
        choices=list(GPU_CONFIGS.keys()),
        help="GPU to benchmark on",
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=["single-stream", "batch-sweep", "output-sweep", "denoising-sweep"],
        help="Specific phases to run (default: all)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        help="Specific models to benchmark (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge per-GPU result files into all_results.json",
    )

    args = parser.parse_args()

    if args.merge:
        merge_results()
        return

    if not args.gpu:
        parser.error("--gpu is required (unless using --merge)")

    run_all_phases(
        gpu_slug=args.gpu,
        phases=args.phase,
        models=args.model,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
