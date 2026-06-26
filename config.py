"""Configuration module for the DiffusionGemma 26B-A4B vs Gemma4 26B-A4B benchmark suite.

This module centralises every knob the benchmark needs:

* **GPU_CONFIGS** – hardware specs for the five target GPUs.
* **MODEL_CONFIGS** – model metadata and per-quantisation HuggingFace checkpoints.
* **EXPERIMENT_CONFIG** – sweep ranges, prompt counts, and vLLM runtime defaults.

Two helper functions translate config lookups into actionable values:

* ``get_checkpoint``  – resolves (model, GPU) → HuggingFace checkpoint ID.
* ``get_vllm_serve_cmd`` – builds a ready-to-exec ``vllm serve`` command list.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# GPU hardware configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUConfig:
    """Immutable descriptor for a single GPU SKU."""

    vram_gb: int
    mem_bw_gbs: int  # memory bandwidth in GB/s
    arch: str
    has_transformer_engine: bool
    quantization: str  # suggested quantisation for this GPU
    peak_tops_dense: int  # peak TOPS at the suggested quant (dense)
    ops_per_byte: int  # peak_tops_dense * 1000 / mem_bw_gbs, rounded


GPU_CONFIGS: Dict[str, GPUConfig] = {
    "dgx-spark": GPUConfig(
        vram_gb=128,
        mem_bw_gbs=273,
        arch="Blackwell",
        has_transformer_engine=True,
        quantization="fp4",
        peak_tops_dense=500,
        ops_per_byte=1832,
    ),
    "rtx-3090": GPUConfig(
        vram_gb=24,
        mem_bw_gbs=936,
        arch="Ampere",
        has_transformer_engine=False,
        quantization="int4-awq",
        peak_tops_dense=476,
        ops_per_byte=509,
    ),
    "l40": GPUConfig(
        vram_gb=48,
        mem_bw_gbs=864,
        arch="Ada Lovelace",
        has_transformer_engine=False,
        quantization="int4-awq",
        peak_tops_dense=724,
        ops_per_byte=838,
    ),
    "l40s": GPUConfig(
        vram_gb=48,
        mem_bw_gbs=864,
        arch="Ada Lovelace",
        has_transformer_engine=True,
        quantization="fp8",
        peak_tops_dense=733,
        ops_per_byte=848,
    ),
    "h100-sxm": GPUConfig(
        vram_gb=80,
        mem_bw_gbs=3350,
        arch="Hopper",
        has_transformer_engine=True,
        quantization="fp8",
        peak_tops_dense=1979,
        ops_per_byte=591,
    ),
}


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Immutable descriptor for a single model variant."""

    type: str  # 'diffusion' | 'autoregressive'
    total_params: str
    active_params: str
    checkpoints: Dict[str, str]  # quantisation → HuggingFace checkpoint ID
    default_denoising_steps: Optional[int] = None  # only for diffusion models
    vllm_extra_args: Dict[str, str] = field(default_factory=dict)


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "diffusiongemma-26b": ModelConfig(
        type="diffusion",
        total_params="26B",
        active_params="3.8B",
        checkpoints={
            "fp8": "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic",
            "int4-awq": "cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4",
            "fp4": "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic",  # fallback
        },
        default_denoising_steps=20,
        vllm_extra_args={"--max-num-seqs": "4"},
    ),
    "gemma4-26b": ModelConfig(
        type="autoregressive",
        total_params="26B",
        active_params="4B",
        checkpoints={
            "fp8": "RedHatAI/gemma-4-26b-a4b-it-FP8-dynamic",
            "int4-awq": "google/gemma-4-26b-a4b-it-qat-q4_0",  # community AWQ placeholder
            "fp4": "RedHatAI/gemma-4-26b-a4b-it-FP8-dynamic",  # fallback
        },
        vllm_extra_args={},
    ),
}


# ---------------------------------------------------------------------------
# Experiment / sweep configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters that govern the benchmark sweep."""

    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    output_lengths: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    input_length: int = 256
    num_prompts: int = 100
    warmup_prompts: int = 10
    temperature: float = 0.0
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.80
    denoising_steps_sweep: List[int] = field(
        default_factory=lambda: [10, 15, 20, 25, 30]
    )
    vllm_port: int = 18080
    results_dir: str = "results"


EXPERIMENT_CONFIG = ExperimentConfig()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_checkpoint(model_slug: str, gpu_slug: str) -> str:
    """Return the HuggingFace checkpoint ID for a *model + GPU* combination.

    The GPU determines which quantisation variant to use (via
    ``GPUConfig.quantization``), and the model config maps that quantisation
    to a concrete checkpoint string.

    Parameters
    ----------
    model_slug:
        Key into :data:`MODEL_CONFIGS` (e.g. ``'diffusiongemma-26b'``).
    gpu_slug:
        Key into :data:`GPU_CONFIGS` (e.g. ``'h100-sxm'``).

    Returns
    -------
    str
        A HuggingFace model identifier such as
        ``'RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic'``.

    Raises
    ------
    KeyError
        If ``model_slug`` or ``gpu_slug`` is unknown, or if the GPU's
        suggested quantisation has no checkpoint entry for the model.
    """
    if gpu_slug not in GPU_CONFIGS:
        raise KeyError(
            f"Unknown GPU slug {gpu_slug!r}. "
            f"Available: {sorted(GPU_CONFIGS)}"
        )
    if model_slug not in MODEL_CONFIGS:
        raise KeyError(
            f"Unknown model slug {model_slug!r}. "
            f"Available: {sorted(MODEL_CONFIGS)}"
        )

    gpu_cfg = GPU_CONFIGS[gpu_slug]
    model_cfg = MODEL_CONFIGS[model_slug]
    quant = gpu_cfg.quantization

    if quant not in model_cfg.checkpoints:
        raise KeyError(
            f"Model {model_slug!r} has no checkpoint for quantisation "
            f"{quant!r} (required by GPU {gpu_slug!r}). "
            f"Available quantisations: {sorted(model_cfg.checkpoints)}"
        )

    return model_cfg.checkpoints[quant]


def get_vllm_serve_cmd(
    model_slug: str,
    gpu_slug: str,
    extra_args: Optional[Dict[str, str]] = None,
) -> list[str]:
    """Build a ``vllm serve`` command as a list of strings.

    The command includes:

    * The resolved checkpoint for the model + GPU pair.
    * Standard runtime flags derived from :data:`EXPERIMENT_CONFIG`
      (``--max-model-len``, ``--gpu-memory-utilization``, ``--port``).
    * Model-specific vLLM flags from :pyattr:`ModelConfig.vllm_extra_args`.
    * Any caller-supplied *extra_args* (which override everything else).

    Parameters
    ----------
    model_slug:
        Key into :data:`MODEL_CONFIGS`.
    gpu_slug:
        Key into :data:`GPU_CONFIGS`.
    extra_args:
        Optional dict of additional ``--flag`` → ``value`` pairs that are
        appended last (and therefore override earlier flags with the same
        name in shell semantics).

    Returns
    -------
    list[str]
        A command suitable for :func:`subprocess.run` or similar.
    """
    checkpoint = get_checkpoint(model_slug, gpu_slug)
    model_cfg = MODEL_CONFIGS[model_slug]
    exp = EXPERIMENT_CONFIG

    # Accumulate args in insertion order; later duplicates override in
    # practice because vLLM uses argparse (last wins for most flags).
    args: Dict[str, str] = {
        "--max-model-len": str(exp.max_model_len),
        "--gpu-memory-utilization": str(exp.gpu_memory_utilization),
        "--port": str(exp.vllm_port),
    }

    # Model-specific overrides (e.g. --max-num-seqs for diffusion).
    args.update(model_cfg.vllm_extra_args)

    # Caller overrides.
    if extra_args:
        args.update(extra_args)

    cmd: list[str] = ["vllm", "serve", checkpoint]
    for flag, value in args.items():
        cmd.extend([flag, value])

    return cmd


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=== GPU Configs ===")
    for slug, cfg in GPU_CONFIGS.items():
        print(f"  {slug}: {cfg}")

    print("\n=== Model Configs ===")
    for slug, cfg in MODEL_CONFIGS.items():
        print(f"  {slug}: type={cfg.type}, active={cfg.active_params}")

    print("\n=== Experiment Config ===")
    print(f"  {EXPERIMENT_CONFIG}")

    print("\n=== Checkpoint resolution ===")
    for m in MODEL_CONFIGS:
        for g in GPU_CONFIGS:
            ckpt = get_checkpoint(m, g)
            print(f"  ({m}, {g}) → {ckpt}")

    print("\n=== Sample vLLM commands ===")
    for m in MODEL_CONFIGS:
        cmd = get_vllm_serve_cmd(m, "h100-sxm")
        print(f"  {m}: {' '.join(cmd)}")
