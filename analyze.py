#!/usr/bin/env python3
"""
analyze.py — Visualization and analysis for DiffusionGemma vs Gemma4-AR benchmarks.

Loads benchmark results from JSON, generates six publication-quality charts,
and prints a summary table to stdout.

Charts produced (saved as PNGs):
  1. Throughput vs Batch Size (per GPU)
  2. Per-Stream TPS vs Batch Size (per GPU)
  3. Speedup Heatmap (GPU × batch size)
  4. TPS vs Memory Bandwidth (scatter + linear fit)
  5. Latency Breakdown Waterfall (grouped bars)
  6. Denoising Steps vs TPS (per GPU)

Usage:
    python analyze.py --results-file results/all_results.json --output-dir results/charts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

matplotlib.use("Agg")  # non-interactive backend for headless rendering

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU specifications — canonical ordering by memory bandwidth (ascending)
# Attempt to import from the project config; fall back to embedded defaults.
# ---------------------------------------------------------------------------
_FALLBACK_GPU_SPECS: dict[str, dict[str, Any]] = {
    "dgx-spark": {
        "display_name": "DGX Spark",
        "mem_bw_gbs": 273,
        "vram_gb": 128,
        "arch": "Blackwell (Grace)",
    },
    "l40": {
        "display_name": "L40",
        "mem_bw_gbs": 864,
        "vram_gb": 48,
        "arch": "Ada Lovelace",
    },
    "l40s": {
        "display_name": "L40S",
        "mem_bw_gbs": 864,
        "vram_gb": 48,
        "arch": "Ada Lovelace",
    },
    "rtx-3090": {
        "display_name": "RTX 3090",
        "mem_bw_gbs": 936,
        "vram_gb": 24,
        "arch": "Ampere",
    },
    "h100-sxm": {
        "display_name": "H100 SXM",
        "mem_bw_gbs": 3350,
        "vram_gb": 80,
        "arch": "Hopper",
    },
}

try:
    from config import GPU_SPECS as _imported_specs  # type: ignore[import-untyped]

    GPU_SPECS: dict[str, dict[str, Any]] = _imported_specs
    log.info("Loaded GPU specs from config.py")
except (ImportError, AttributeError):
    GPU_SPECS = _FALLBACK_GPU_SPECS
    log.info("Using embedded fallback GPU specs (config.py not found)")

# Canonical GPU order: ascending memory bandwidth
GPU_ORDER: list[str] = sorted(GPU_SPECS.keys(), key=lambda g: GPU_SPECS[g]["mem_bw_gbs"])

# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------
COLOR_DIFFUSION = "#4A90D9"  # blue
COLOR_AR = "#E8913A"  # orange
MODEL_COLORS = {
    "diffusiongemma-26b": COLOR_DIFFUSION,
    "gemma4-26b": COLOR_AR,
}
MODEL_LABELS = {
    "diffusiongemma-26b": "DiffusionGemma",
    "gemma4-26b": "Gemma4-AR",
}
FIGURE_DPI = 180


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_results(path: Path) -> list[dict[str, Any]]:
    """Load and validate the benchmark results JSON."""
    if not path.exists():
        log.error("Results file not found: %s", path)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        log.error("Expected a JSON array at top level; got %s", type(data).__name__)
        sys.exit(1)
    log.info("Loaded %d result records from %s", len(data), path)
    return data


def _filter(
    records: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Return records matching all key=value filters.

    A filter value of ``None`` is treated as "match anything".
    """
    out: list[dict[str, Any]] = []
    for rec in records:
        match = True
        for key, val in kwargs.items():
            if val is not None and rec.get(key) != val:
                match = False
                break
        if match:
            out.append(rec)
    return out


def _get_ttft(rec: dict[str, Any]) -> float | None:
    """Unify ttft_ms / ttfb_ms into a single value."""
    return rec.get("ttft_ms") or rec.get("ttfb_ms")


def _gpu_display(gpu_key: str) -> str:
    """Human-readable GPU name."""
    return GPU_SPECS.get(gpu_key, {}).get("display_name", gpu_key)


def _gpu_bw(gpu_key: str) -> float:
    """Memory bandwidth in GB/s."""
    return GPU_SPECS.get(gpu_key, {}).get("mem_bw_gbs", 0.0)


# ---------------------------------------------------------------------------
# Chart 1: Throughput vs Batch Size
# ---------------------------------------------------------------------------
def chart_throughput_vs_batch(
    records: list[dict[str, Any]], output_dir: Path
) -> None:
    """One subplot per GPU — aggregate tokens/sec vs batch size."""
    subset = _filter(records, phase="batch-sweep", output_length=256)
    if not subset:
        log.warning("No data for chart 1 (throughput vs batch size); skipping.")
        return

    gpus = [g for g in GPU_ORDER if any(r["gpu"] == g for r in subset)]
    n = len(gpus)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=True, squeeze=False)
    axes = axes[0]

    for idx, gpu in enumerate(gpus):
        ax = axes[idx]
        for model, color in MODEL_COLORS.items():
            pts = _filter(subset, gpu=gpu, model=model)
            if not pts:
                continue
            pts.sort(key=lambda r: r["batch_size"])
            bs = [r["batch_size"] for r in pts]
            tps = [r["throughput_tps"] for r in pts]
            ax.plot(
                bs, tps,
                marker="o", markersize=5, linewidth=2,
                color=color, label=MODEL_LABELS[model],
            )
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel("Batch Size")
        ax.set_title(_gpu_display(gpu), fontsize=11, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Aggregate Tokens / sec")
        ax.grid(alpha=0.25)

    # Single legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10)

    fig.suptitle("Throughput vs Batch Size (output_length=256)", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    dest = output_dir / "01_throughput_vs_batch.png"
    fig.savefig(dest, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved chart 1 → %s", dest)


# ---------------------------------------------------------------------------
# Chart 2: Per-Stream TPS vs Batch Size
# ---------------------------------------------------------------------------
def chart_per_stream_vs_batch(
    records: list[dict[str, Any]], output_dir: Path
) -> None:
    """One subplot per GPU — per-stream tokens/sec vs batch size."""
    subset = _filter(records, phase="batch-sweep", output_length=256)
    if not subset:
        log.warning("No data for chart 2 (per-stream TPS); skipping.")
        return

    gpus = [g for g in GPU_ORDER if any(r["gpu"] == g for r in subset)]
    n = len(gpus)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=True, squeeze=False)
    axes = axes[0]

    for idx, gpu in enumerate(gpus):
        ax = axes[idx]
        for model, color in MODEL_COLORS.items():
            pts = _filter(subset, gpu=gpu, model=model)
            if not pts:
                continue
            pts.sort(key=lambda r: r["batch_size"])
            bs = [r["batch_size"] for r in pts]
            tps = [r["per_stream_tps"] for r in pts]
            ax.plot(
                bs, tps,
                marker="s", markersize=5, linewidth=2,
                color=color, label=MODEL_LABELS[model],
            )
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel("Batch Size")
        ax.set_title(_gpu_display(gpu), fontsize=11, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Per-Stream Tokens / sec")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10)

    fig.suptitle("Per-Stream TPS vs Batch Size (output_length=256)", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    dest = output_dir / "02_per_stream_vs_batch.png"
    fig.savefig(dest, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved chart 2 → %s", dest)


# ---------------------------------------------------------------------------
# Chart 3: Speedup Heatmap
# ---------------------------------------------------------------------------
def chart_speedup_heatmap(
    records: list[dict[str, Any]], output_dir: Path
) -> None:
    """Heatmap — DiffusionGemma / AR throughput ratio per GPU × batch size."""
    subset = _filter(records, phase="batch-sweep", output_length=256)
    if not subset:
        log.warning("No data for chart 3 (speedup heatmap); skipping.")
        return

    gpus = [g for g in GPU_ORDER if any(r["gpu"] == g for r in subset)]
    batch_sizes = sorted({r["batch_size"] for r in subset})

    if not gpus or not batch_sizes:
        return

    # Build ratio matrix (rows=GPU, cols=batch_size)
    ratio_matrix = np.full((len(gpus), len(batch_sizes)), np.nan)
    for gi, gpu in enumerate(gpus):
        for bi, bs in enumerate(batch_sizes):
            diff = _filter(subset, gpu=gpu, model="diffusiongemma-26b", batch_size=bs)
            ar = _filter(subset, gpu=gpu, model="gemma4-26b", batch_size=bs)
            if diff and ar and ar[0]["throughput_tps"] > 0:
                ratio_matrix[gi, bi] = diff[0]["throughput_tps"] / ar[0]["throughput_tps"]

    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(batch_sizes)), max(3.5, 0.9 * len(gpus))))

    # Center the colormap at 1.0 so green=diffusion wins, red=AR wins
    vmin = max(0.0, np.nanmin(ratio_matrix) - 0.1) if not np.all(np.isnan(ratio_matrix)) else 0.5
    vmax = np.nanmax(ratio_matrix) + 0.1 if not np.all(np.isnan(ratio_matrix)) else 2.0
    abs_max = max(abs(1.0 - vmin), abs(vmax - 1.0))
    norm = matplotlib.colors.TwoSlopeNorm(vmin=1.0 - abs_max, vcenter=1.0, vmax=1.0 + abs_max)

    im = ax.imshow(ratio_matrix, cmap="RdYlGn", norm=norm, aspect="auto")

    # Annotate cells
    for gi in range(len(gpus)):
        for bi in range(len(batch_sizes)):
            val = ratio_matrix[gi, bi]
            if not np.isnan(val):
                text_color = "black" if 0.7 < val < 1.4 else "white"
                ax.text(bi, gi, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_yticks(range(len(gpus)))
    ax.set_yticklabels([_gpu_display(g) for g in gpus])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("GPU (↑ mem bandwidth)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Speedup (DiffusionGemma / AR)")

    ax.set_title("Throughput Speedup: DiffusionGemma over Gemma4-AR", fontsize=12, fontweight="bold")
    fig.tight_layout()
    dest = output_dir / "03_speedup_heatmap.png"
    fig.savefig(dest, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved chart 3 → %s", dest)


# ---------------------------------------------------------------------------
# Chart 4: TPS vs Memory Bandwidth (scatter)
# ---------------------------------------------------------------------------
def chart_tps_vs_membw(
    records: list[dict[str, Any]], output_dir: Path
) -> None:
    """Scatter — per-stream TPS at batch=1 vs GPU memory bandwidth."""
    subset = _filter(records, phase="batch-sweep", batch_size=1, output_length=256)
    # Fallback: try single-stream phase
    if not subset:
        subset = _filter(records, phase="single-stream", batch_size=1, output_length=256)
    if not subset:
        subset = _filter(records, phase="single-stream", batch_size=1)
    if not subset:
        log.warning("No data for chart 4 (TPS vs mem BW); skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for model, color in MODEL_COLORS.items():
        pts = _filter(subset, model=model)
        if not pts:
            continue
        bws = [_gpu_bw(r["gpu"]) for r in pts]
        tps_vals = [r["per_stream_tps"] for r in pts]
        ax.scatter(bws, tps_vals, color=color, s=80, zorder=5,
                   label=MODEL_LABELS[model], edgecolors="white", linewidths=0.5)
        # Label each point with GPU name
        for r, bw, tps in zip(pts, bws, tps_vals):
            ax.annotate(
                _gpu_display(r["gpu"]),
                (bw, tps),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color=color,
                alpha=0.9,
            )

        # Linear fit for AR
        if model == "gemma4-26b" and len(bws) >= 2:
            bw_arr = np.array(bws, dtype=float)
            tps_arr = np.array(tps_vals, dtype=float)
            coeffs = np.polyfit(bw_arr, tps_arr, 1)
            fit_x = np.linspace(bw_arr.min() * 0.85, bw_arr.max() * 1.1, 100)
            fit_y = np.polyval(coeffs, fit_x)
            ax.plot(fit_x, fit_y, color=color, linestyle="--", linewidth=1.2,
                    alpha=0.55, label="AR linear fit")

    ax.set_xlabel("Memory Bandwidth (GB/s)")
    ax.set_ylabel("Per-Stream Tokens / sec (batch=1)")
    ax.set_title("Single-Stream TPS vs GPU Memory Bandwidth", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    dest = output_dir / "04_tps_vs_membw.png"
    fig.savefig(dest, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved chart 4 → %s", dest)


# ---------------------------------------------------------------------------
# Chart 5: Latency Breakdown Waterfall
# ---------------------------------------------------------------------------
def chart_latency_waterfall(
    records: list[dict[str, Any]], output_dir: Path
) -> None:
    """Grouped bar chart — latency breakdown for each GPU at batch=1, output=256."""
    subset = _filter(records, batch_size=1, output_length=256)
    if not subset:
        subset = _filter(records, batch_size=1)
    if not subset:
        log.warning("No data for chart 5 (latency waterfall); skipping.")
        return

    gpus = [g for g in GPU_ORDER if any(r["gpu"] == g for r in subset)]
    if not gpus:
        return

    bar_width = 0.35
    x = np.arange(len(gpus))

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(gpus)), 5))

    # --- AR bars (left) ---
    ar_ttft: list[float] = []
    ar_decode: list[float] = []
    for gpu in gpus:
        ar = _filter(subset, gpu=gpu, model="gemma4-26b")
        if ar:
            rec = ar[0]
            ttft = _get_ttft(rec) or 0.0
            # Decode time = ITL_median × output_length (approximate)
            itl = rec.get("itl_median_ms") or 0.0
            out_len = rec.get("output_length", 256)
            decode = itl * out_len
            ar_ttft.append(ttft)
            ar_decode.append(decode)
        else:
            ar_ttft.append(0.0)
            ar_decode.append(0.0)

    ax.bar(x - bar_width / 2, ar_ttft, bar_width,
           label="AR: TTFT", color=COLOR_AR, edgecolor="white", linewidth=0.5)
    ax.bar(x - bar_width / 2, ar_decode, bar_width, bottom=ar_ttft,
           label="AR: Decode (ITL × len)", color=COLOR_AR, alpha=0.55,
           edgecolor="white", linewidth=0.5)

    # --- Diffusion bars (right) ---
    diff_ttfb: list[float] = []
    for gpu in gpus:
        diff = _filter(subset, gpu=gpu, model="diffusiongemma-26b")
        if diff:
            ttfb = _get_ttft(diff[0]) or diff[0].get("e2el_ms", 0.0)
            diff_ttfb.append(ttfb)
        else:
            diff_ttfb.append(0.0)

    ax.bar(x + bar_width / 2, diff_ttfb, bar_width,
           label="Diffusion: TTFB", color=COLOR_DIFFUSION,
           edgecolor="white", linewidth=0.5)

    # Add value labels on top of bars
    for i in range(len(gpus)):
        total_ar = ar_ttft[i] + ar_decode[i]
        if total_ar > 0:
            ax.text(x[i] - bar_width / 2, total_ar + total_ar * 0.02,
                    f"{total_ar:.0f}", ha="center", va="bottom", fontsize=7,
                    color=COLOR_AR)
        if diff_ttfb[i] > 0:
            ax.text(x[i] + bar_width / 2, diff_ttfb[i] + diff_ttfb[i] * 0.02,
                    f"{diff_ttfb[i]:.0f}", ha="center", va="bottom", fontsize=7,
                    color=COLOR_DIFFUSION)

    ax.set_xticks(x)
    ax.set_xticklabels([_gpu_display(g) for g in gpus])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Breakdown — batch=1, output_length=256", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    dest = output_dir / "05_latency_waterfall.png"
    fig.savefig(dest, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved chart 5 → %s", dest)


# ---------------------------------------------------------------------------
# Chart 6: Denoising Steps vs TPS
# ---------------------------------------------------------------------------
def chart_denoising_sweep(
    records: list[dict[str, Any]], output_dir: Path
) -> None:
    """Line chart — per-stream TPS vs denoising steps (DiffusionGemma only)."""
    subset = _filter(records, phase="denoising-sweep", model="diffusiongemma-26b")
    if not subset:
        log.warning("No data for chart 6 (denoising sweep); skipping.")
        return

    gpus = [g for g in GPU_ORDER if any(r["gpu"] == g for r in subset)]
    if not gpus:
        return

    # Use a distinct color palette for per-GPU lines
    cmap = plt.cm.get_cmap("plasma", max(len(gpus), 4))

    fig, ax = plt.subplots(figsize=(7, 5))

    for gi, gpu in enumerate(gpus):
        pts = _filter(subset, gpu=gpu)
        if not pts:
            continue
        pts.sort(key=lambda r: (r.get("denoising_steps") or 0))
        steps = [r.get("denoising_steps", 0) for r in pts]
        tps = [r["per_stream_tps"] for r in pts]
        ax.plot(
            steps, tps,
            marker="D", markersize=5, linewidth=2,
            color=cmap(gi / max(len(gpus) - 1, 1)),
            label=_gpu_display(gpu),
        )

    ax.set_xlabel("Denoising Steps")
    ax.set_ylabel("Per-Stream Tokens / sec")
    ax.set_title("DiffusionGemma: TPS vs Denoising Steps", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    dest = output_dir / "06_denoising_steps_vs_tps.png"
    fig.savefig(dest, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved chart 6 → %s", dest)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary_table(records: list[dict[str, Any]]) -> None:
    """Print a plaintext summary table of key metrics per GPU to stdout."""
    # Use batch=1, output_length=256 data (or single-stream)
    candidates = _filter(records, batch_size=1, output_length=256)
    if not candidates:
        candidates = _filter(records, batch_size=1)
    if not candidates:
        log.warning("No batch=1 data available for summary table.")
        return

    sep = "─" * 108
    header = (
        f"{'GPU':<14} │ {'Model':<20} │ {'TTFT/B (ms)':>11} │ "
        f"{'E2E (ms)':>10} │ {'TPS':>10} │ {'Mem BW':>10} │ {'GPU%':>6} │ {'MEM%':>6}"
    )
    print("\n" + sep)
    print("  BENCHMARK SUMMARY  —  batch=1, output_length=256")
    print(sep)
    print(header)
    print(sep)

    for gpu in GPU_ORDER:
        gpu_recs = _filter(candidates, gpu=gpu)
        if not gpu_recs:
            continue
        for model in ("diffusiongemma-26b", "gemma4-26b"):
            mrecs = _filter(gpu_recs, model=model)
            if not mrecs:
                continue
            rec = mrecs[0]
            ttft = _get_ttft(rec)
            ttft_str = f"{ttft:>11.1f}" if ttft is not None else f"{'—':>11}"
            e2e = rec.get("e2el_ms")
            e2e_str = f"{e2e:>10.1f}" if e2e is not None else f"{'—':>10}"
            tps = rec.get("per_stream_tps")
            tps_str = f"{tps:>10.1f}" if tps is not None else f"{'—':>10}"
            bw_str = f"{_gpu_bw(gpu):>10.0f}"
            gpu_util = rec.get("gpu_util_pct")
            gpu_str = f"{gpu_util:>5.1f}%" if gpu_util is not None else f"{'—':>6}"
            mem_util = rec.get("mem_util_pct")
            mem_str = f"{mem_util:>5.1f}%" if mem_util is not None else f"{'—':>6}"
            print(
                f"{_gpu_display(gpu):<14} │ {MODEL_LABELS.get(model, model):<20} │ "
                f"{ttft_str} │ {e2e_str} │ {tps_str} │ {bw_str} │ {gpu_str} │ {mem_str}"
            )
        # Compute speedup
        diff_rec = _filter(gpu_recs, model="diffusiongemma-26b")
        ar_rec = _filter(gpu_recs, model="gemma4-26b")
        if diff_rec and ar_rec:
            d_tps = diff_rec[0].get("per_stream_tps", 0)
            a_tps = ar_rec[0].get("per_stream_tps", 0)
            if a_tps > 0:
                ratio = d_tps / a_tps
                winner = "Diffusion" if ratio > 1 else "AR"
                print(f"{'':>14} │ {'  ⤷ speedup':<20} │ {'':>11} │ {'':>10} │ "
                      f"{ratio:>9.2f}× │ {'':>10} │ {'':>6} │  {winner}")
        print(sep)

    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and visualize DiffusionGemma vs Gemma4-AR benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("results/all_results.json"),
        help="Path to the benchmark results JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/charts"),
        help="Directory where chart PNGs will be saved.",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", args.output_dir.resolve())

    # Load data
    records = load_results(args.results_file)

    # Apply dark theme
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "#1C1C1C",
        "axes.facecolor": "#2A2A2A",
        "savefig.facecolor": "#1C1C1C",
    })

    # Generate all charts
    chart_throughput_vs_batch(records, args.output_dir)
    chart_per_stream_vs_batch(records, args.output_dir)
    chart_speedup_heatmap(records, args.output_dir)
    chart_tps_vs_membw(records, args.output_dir)
    chart_latency_waterfall(records, args.output_dir)
    chart_denoising_sweep(records, args.output_dir)

    # Print summary table
    print_summary_table(records)

    log.info("All charts saved to %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
