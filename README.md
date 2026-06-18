# DiffusionGemma vs Gemma4-AR Benchmark Suite

Benchmark suite comparing **DiffusionGemma 26B-A4B** (diffusion LLM) against **Gemma4 26B-A4B** (autoregressive) across 5 GPU types with varying memory bandwidth.

## Hardware Matrix

| GPU | VRAM | Mem BW | Quantization | Peak TOPS (dense) |
|:----|:-----|:-------|:-------------|:------------------|
| DGX Spark (GB10) | 128 GB | 273 GB/s | FP4 (NVFP4) | 500 |
| RTX 3090 | 24 GB | 936 GB/s | INT4 (AWQ) | 476 |
| L40 | 48 GB | 864 GB/s | INT4 (AWQ) | 724 |
| L40S | 48 GB | 864 GB/s | FP8 | 733 |
| H100 SXM | 80 GB | 3,350 GB/s | FP8 | 1,979 |

## Quick Start

### 1. Set up Vast.ai instance

Rent the desired GPU on [Vast.ai](https://vast.ai). Use a PyTorch template with CUDA ≥ 12.x.

### 2. Clone and install

```bash
git clone https://github.com/<YOUR_USERNAME>/diffusion-gemma-bench.git
cd diffusion-gemma-bench
pip install -r requirements.txt
pip install vllm
```

### 3. Set HuggingFace token

```bash
export HUGGING_FACE_HUB_TOKEN=<your_token>
```

### 4. Run benchmarks

```bash
# Run all phases on your GPU (auto-selects quantization + checkpoints)
python benchmark.py --gpu rtx-3090

# Run specific phases
python benchmark.py --gpu h100-sxm --phase single-stream batch-sweep

# Run only DiffusionGemma
python benchmark.py --gpu l40s --model diffusiongemma-26b

# Dry run (see commands without executing)
python benchmark.py --gpu dgx-spark --dry-run
```

### 5. Collect results

Each GPU produces `results/results_<gpu>.json`. After all GPUs are done:

```bash
# Copy result files from each instance to a single machine
scp vast-instance-1:diffusion-gemma-bench/results/results_*.json results/
scp vast-instance-2:diffusion-gemma-bench/results/results_*.json results/
# ... etc

# Merge into a single file
python benchmark.py --merge
```

### 6. Generate charts

```bash
python analyze.py --results-file results/all_results.json --output-dir results/charts
```

## Experiment Phases

| Phase | Description | Batch Sizes | Output Lengths |
|:------|:------------|:------------|:---------------|
| **1. single-stream** | Baseline latency at batch=1 | 1 | 256 |
| **2. batch-sweep** | Throughput scaling | 1, 2, 4, 8, 16, 32 | 256 |
| **3. output-sweep** | Output length sensitivity | 1, 4 | 128, 256, 512, 1024 |
| **4. denoising-sweep** | Denoising steps (DiffusionGemma only) | 1 | 256 |

## Output Charts

1. **Throughput vs Batch Size** — aggregate tokens/sec scaling
2. **Per-Stream TPS vs Batch Size** — user-experience degradation
3. **Speedup Heatmap** — DiffusionGemma / Gemma4-AR ratio
4. **TPS vs Memory Bandwidth** — validates bandwidth-insensitivity hypothesis
5. **Latency Breakdown** — prefill + decode waterfall
6. **Denoising Steps vs TPS** — latency-quality tradeoff

## Project Structure

```
diffusion-gemma-bench/
├── benchmark.py        # Main benchmark runner
├── config.py           # Hardware + model + experiment configuration
├── analyze.py          # Visualization and analysis
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── results/            # Generated at runtime
    ├── results_<gpu>.json
    ├── all_results.json
    ├── charts/
    └── gpu_metrics/
```

## Estimated Runtime

~8 hours per GPU, ~10 wall-clock hours if all 5 GPUs run in parallel.

## Estimated Cost (Vast.ai)

~$45–65 total across all 5 GPU types.
