---
title: Multi-GPU Scaling
description: Tensor parallelism, NVIDIA Hopper/Blackwell optimizations, AMD MI300 support, and device selection
impact: CRITICAL
category: multigpu
tags: [tensor-parallel, nvidia, amd, hopper, mi300, device-selection]
error_patterns:
  - "tensor parallel"
  - "multi-GPU"
  - "device"
  - "GPU OOM"
  - "NCCL"
  - "communication"
  - "model too large"
scenarios:
  - "Deploy large model on multiple GPUs"
  - "Configure tensor parallelism"
  - "Select GPU devices"
  - "Optimize for NVIDIA Hopper"
  - "Deploy on AMD MI300X"
  - "Fix multi-GPU communication"
consolidates:
  - multigpu-tensor-parallel.md
  - multigpu-nvidia-hopper.md
  - multigpu-amd-mi300.md
  - multigpu-device-selection.md
---

# Multi-GPU Scaling

**Category:** multigpu | **Impact:** CRITICAL

Comprehensive patterns for multi-GPU deployment including tensor parallelism for large models (70B+), NVIDIA Hopper/Blackwell optimizations, AMD MI300X support, and proper device selection syntax.

---

## Core Concepts

### Tensor Parallelism

Tensor parallelism distributes model layers across multiple GPUs, enabling inference on models that don't fit on a single GPU.

> **WARNING:** Tensor parallelism works best with power-of-2 GPU counts (2, 4, 8).
> Odd counts (3, 5, 7) may fail or have degraded performance.

**Pattern:**

```bash
# Tensor parallelism across 4 GPUs
max serve --model meta-llama/Llama-3.3-70B-Instruct \
  --devices gpu:0,1,2,3 \
  --quantization-encoding bfloat16
```

**Don't:**
```bash
# Single GPU - model may not fit
max serve --model meta-llama/Llama-3.3-70B-Instruct
```

### Device Selection Syntax

Use `--devices gpu:0,1,2,3` format for GPU selection (v25.3+).

**Pattern:**

```bash
# v25.3+ format
max serve --model meta-llama/Llama-3.3-70B-Instruct \
  --devices gpu:0,1,2,3

# Single GPU
max serve --model ... --devices gpu:0
```

**Don't:**
```bash
# Old format (deprecated)
max serve --model ... --devices gpu-0,gpu-1
max serve --model ... --use-gpu
```

**Note:** `--use-gpu` and `--devices=gpu-N` deprecated in v25.3.

---

## Tensor Parallel vs Data Parallel Decision Guide

| Factor | Tensor Parallel (TP) | Data Parallel (DP) |
|--------|---------------------|-------------------|
| **Use when** | Model doesn't fit on one GPU | Model fits, need higher throughput |
| **Memory scaling** | Model weights split across GPUs | Model replicated on each GPU |
| **Batch handling** | Single batch, shared computation | Separate batches per replica |
| **Latency** | Lower (inter-GPU sync) | Same as single GPU |
| **Throughput** | Limited by sync overhead | Linear scaling with GPU count |
| **Communication** | High (every layer) | Low (inference only) |

### Decision Flowchart

```
Does the model fit on a single GPU?
  │
  ├─ NO → Use Tensor Parallel (TP)
  │       --devices gpu:0,1,2,3
  │
  └─ YES → Do you need maximum throughput?
           │
           ├─ YES → Use Data Parallel (DP)
           │        --data-parallel-degree 4
           │
           └─ NO → Single GPU is sufficient
                   --devices gpu:0
```

### Hybrid TP + DP

For very large deployments, combine both:

```bash
# 70B model on 8 GPUs: TP=4 (model sharding) + DP=2 (throughput)
max serve --model meta-llama/Llama-3.3-70B-Instruct \
  --devices gpu:0,1,2,3,4,5,6,7 \
  --data-parallel-degree 2
# Creates 2 replicas, each using 4 GPUs with tensor parallelism
```

**Capacity formula:** `total_batch_capacity = max_batch_size × data_parallel_degree`

### Batch Size Semantics with Data Parallelism

When using data parallelism, understand how batch size is interpreted:

- **`--max-batch-size` is PER REPLICA**, not total system capacity
- Each replica independently handles up to `max_batch_size` requests
- Total system capacity = `max_batch_size × data_parallel_degree`

**Example:**
```bash
# 4-way data parallelism with batch size 8 per replica
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --devices gpu:0,1,2,3 \
  --data-parallel-degree 4 \
  --max-batch-size 8
# Total capacity: 8 × 4 = 32 concurrent requests
```

> **Migration Note (v25.x users):** In earlier versions, batch size behavior varied.
> Starting with v25.3+, `--max-batch-size` consistently applies per replica.
> If migrating from v25.x and experiencing capacity differences, multiply your
> previous batch size by your data parallel degree to match prior behavior.

---

## Common Patterns

### NVIDIA Hopper/Blackwell Deployment

**When:** Deploying on H100, H200 (Hopper) or B200 (Blackwell) GPUs for maximum performance.

MAX is optimized for NVIDIA Hopper and Blackwell architectures with state-of-the-art matmul kernels.

**Supported Architectures:**
- H100/H200 (SM90) - Hopper
- B200 (SM100) - Blackwell
- A100/A10/L4/L40 - Ampere

**Pattern:**
```bash
# Hopper/Blackwell with bfloat16 (large model example)
max serve --model meta-llama/Llama-3.3-70B-Instruct \
  --devices gpu:0,1,2,3 \
  --quantization-encoding bfloat16

# Enable profiling for optimization
max benchmark --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu-profiling detailed \
  --trace
```

**Performance Features:**
- State-of-the-art matmul kernels (competitive with cuBLAS)
- Warp specialization for compute/memory overlap
- TMA loading for efficient memory access
- Flash Attention 3 implementation

**Driver Requirements:**
- CUDA 12.4+ / Driver 550+ (v25.3+)
- CUDA 12.6+ / Driver 580+ (v26.1.0.0.0+)

**Cross-reference:** See mojo-best-practices `gpu-tensor-core-sm90-sm100` for kernel patterns.

---

### AMD MI300X/MI325X/MI355X Deployment

**When:** Deploying on AMD datacenter GPUs with the same codebase as NVIDIA.

MAX supports AMD MI300X, MI325X, and MI355X datacenter GPUs with no code changes required.

**Pattern:**
```bash
# AMD GPU serving
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --devices gpu:0 \
  --quantization-encoding bfloat16

# Docker with AMD
docker run --device=/dev/kfd --device=/dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    modular/max-amd:latest \
    --model-path meta-llama/Llama-3.1-8B-Instruct
```

**Supported Hardware:**

| GPU | Architecture | Notes |
|-----|--------------|-------|
| MI300X | CDNA3 | Datacenter GPU |
| MI325X | CDNA3 | Datacenter GPU |
| MI355X | CDNA4 | Datacenter GPU (v25.6+) |
| Radeon 780m | RDNA3 | Consumer (limited support) |

**Features:**
- Same codebase as NVIDIA (no code changes)
- BFloat16 support on ARM hosts (GH200-style)
- `max benchmark --collect-gpu-stats` for AMD

**Note:** AMD support added in v25.4, production-ready in v25.6+.

---

### Multi-GPU Sizing Guide

**When to use multi-GPU:**

| Model Size | GPU Memory Needed | Recommended Setup |
|------------|-------------------|-------------------|
| 7-8B | ~16GB | 1x GPU (any) |
| 13B | ~26GB | 1x A100/H100 or 2x smaller |
| 34B | ~68GB | 2x H100 or 4x A10 |
| 70B+ | ~140GB | 4x H100 or 8x A10 |

**Pattern - 70B Model:**
```bash
# 4-way tensor parallelism for 70B model
max serve --model meta-llama/Llama-3.3-70B-Instruct \
  --devices gpu:0,1,2,3 \
  --quantization-encoding bfloat16 \
  --max-batch-size 8
```

**Pattern - 8B Model (single GPU):**
```bash
# Single GPU sufficient
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --devices gpu:0 \
  --quantization-encoding bfloat16 \
  --max-batch-size 32
```

---

### KV Cache Memory Considerations

The KV (key-value) cache stores attention states for each token in context. For long-context workloads, KV cache memory can significantly exceed model weight memory.

**KV Cache Size Formula:**
```
KV cache ≈ 2 × num_layers × num_heads × head_dim × max_seq_len × batch_size × bytes_per_element
```

**Example Calculation (Llama 70B, 32K context):**
```
2 × 80 layers × 64 heads × 128 dim × 32768 seq × 8 batch × 2 bytes (bf16)
≈ 171 GB per batch of 8 requests
```

**Impact by Context Length:**

| Context Length | KV Cache (per request, bf16) | Notes |
|----------------|------------------------------|-------|
| 4K | ~2.6 GB | Manageable on most GPUs |
| 8K | ~5.2 GB | Standard workloads |
| 32K | ~21 GB | May exceed model weights |
| 128K | ~84 GB | Requires memory optimization |

**Memory-Constrained Setup Pattern:**
```bash
# Enable KV cache swapping to host memory for long contexts
max serve --model meta-llama/Llama-3.1-8B-Instruct \
  --devices gpu:0 \
  --enable-kvcache-swapping-to-host \
  --max-cache-batch-size 4 \
  --max-length 32768
```

**Recommendations:**
- For 32K+ context: Use `--enable-kvcache-swapping-to-host` to offload inactive KV cache to CPU memory
- Monitor GPU memory usage during peak load, not just at startup
- Reduce `--max-batch-size` or `--max-cache-batch-size` if hitting OOM during inference
- Consider quantized KV cache (when available) for further memory reduction

---

### Python API Device Selection

**When:** Programmatically selecting devices in Python code.

```python
from max.driver import CPU, Accelerator, accelerator_count
from max.engine import InferenceSession

# Automatic GPU selection
if accelerator_count() > 0:
    device = Accelerator()  # Uses first available GPU
else:
    device = CPU()

# Multiple GPUs
devices = [Accelerator(i) for i in range(4)]
session = InferenceSession(devices=devices)
```

---

## Decision Guide

| Scenario | Approach | GPU Count |
|----------|----------|-----------|
| Small models (7-8B) | Single GPU | 1 |
| Medium models (13B) | Single large GPU or 2x | 1-2 |
| Large models (70B) | Tensor parallelism | 4 |
| Very large models (405B) | Tensor parallelism | 8 |
| Maximum throughput | More GPUs + batching | 4-8 |
| Cost optimization | Quantization (GPTQ, GGUF) | 1-2 |

---

## Quick Reference

- **Device syntax**: `--devices gpu:0,1,2,3` (comma-separated, v25.3+)
- **Tensor parallelism**: Distributes layers across GPUs automatically
- **NVIDIA**: H100/H200 (Hopper), B200 (Blackwell), A100/A10/L4 (Ampere)
- **AMD**: MI300X/MI325X/MI355X (same codebase, no changes)
- **Driver**: CUDA 12.6+ / Driver 580+ for latest features

---

## Hardware Comparison

| Feature | NVIDIA Hopper | AMD MI300X |
|---------|---------------|------------|
| Memory | 80GB (H100), 141GB (H200) | 192GB |
| Tensor Cores | 4th gen | Matrix cores |
| MAX Support | Full | Full (v25.6+) |
| Container | `max-nvidia-full` | `max-amd` |
| Docker flag | `--gpus all` | `--device=/dev/kfd --device=/dev/dri` |
| MAX flag | `--devices gpu:0,1,...` | `--devices gpu:0,1,...` |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `tensor parallel failed` | Model doesn't support TP | Check if model is supported, use data parallel |
| `NCCL error` | Inter-GPU communication failure | Check NVLink/PCIe topology, NCCL version |
| `model too large` | Model exceeds combined GPU memory | Add more GPUs or use quantization |
| `device 0,1 not found` | Invalid device syntax | Use `--devices gpu:0,1,2,3` format |
| `slow multi-GPU` | Communication bottleneck | Use NVLink for better bandwidth |
| `GPU utilization imbalance` | Uneven model sharding | Ensure even layer distribution |

---

## Version-Specific Features

### Stable (v26.1.0.0.0) vs Nightly (v26.2+)

| Feature | Stable (v26.1.0.0.0) | Nightly (v26.2+) |
|---------|----------------|--------------------|
| **Device syntax** | `--devices gpu:0,1,2,3` | `--devices gpu:0,1,2,3` (unchanged) |
| **Tensor parallelism** | Stable | Stable (unchanged) |
| **SM100 support** | Not available | Blackwell GPU support added |
| **Pipeline parallelism** | Limited | Enhanced in v26.2+ |
| `ops.fence` | Not available | Synchronization fence for multi-GPU coordination |
| `Tensor` | Standard tensor API | Enhanced tensor API with improved multi-device support |

**Stable (v26.1.0.0.0):**
```bash
# Tensor parallelism across 4 GPUs (H100)
max serve --model meta-llama/Llama-3.3-70B-Instruct \
  --devices gpu:0,1,2,3 \
  --quantization-encoding bfloat16
```

**Nightly (v26.2+):**
```bash
# Same syntax, now with Blackwell support
max serve --model meta-llama/Llama-3.3-70B-Instruct \
  --devices gpu:0,1,2,3 \
  --quantization-encoding bfloat16

# NEW: NVFP4 quantization on Blackwell
max serve --model meta-llama/Llama-3.3-70B-Instruct \
  --devices gpu:0,1,2,3 \
  --quantization-encoding float4_e2m1fnx2
```

**Notes:**
- Device selection syntax (`--devices gpu:0,1,2,3`) is stable
- Tensor parallelism for 70B+ models is stable
- Power-of-2 GPU count recommendation is stable
- SM100 (Blackwell) support added in v26.2+
- AMD MI300X support is stable across versions

---

## Related Patterns

- [`deploy-deployment.md`](deploy-deployment.md) — Container and cloud deployment
- [`perf-inference.md`](perf-inference.md) — Performance optimization
- [`engine-operations.md`](engine-operations.md) — Session configuration

---

## References

- [MAX Multi-GPU Documentation](https://docs.modular.com/max/serve)
- [NVIDIA GPU Support](https://docs.modular.com/max/faq/)
- [AMD GPU Support](https://docs.modular.com/max/faq/)
- mojo-best-practices: `gpu-tensor-core-sm90-sm100` for kernel patterns
