---
title: Quantization Configuration
description: Float8, GPTQ, and graph-level quantization patterns for memory-efficient inference
impact: HIGH
category: engine
tags: [quantization, float8, gptq, memory, performance]
error_patterns:
  - "quantization"
  - "float8"
  - "fp8"
  - "GPTQ"
  - "scale"
  - "precision loss"
  - "accuracy"
scenarios:
  - "Configure Float8 quantization"
  - "Use GPTQ 4-bit inference"
  - "Reduce model memory usage"
  - "Fix quantization accuracy issues"
  - "Enable H100 FP8"
  - "Configure scale granularity"
consolidates:
  - engine-float8-config.md
  - engine-gptq-config.md
  - graph-quantization.md
---

# Quantization Configuration

**Category:** engine | **Impact:** HIGH

Comprehensive guide to quantization in MAX Engine including Float8 configuration for H100 GPUs, GPTQ for 4-bit inference, and graph-level quantization patterns. Proper configuration can achieve up to 2x throughput improvement with minimal accuracy loss.

---

## Core Concepts

### Float8 Quantization (H100/Hopper)

Float8 enables high-performance inference on NVIDIA Hopper GPUs (H100) with minimal accuracy loss. Proper scale granularity and scaling origin configuration is critical.

**Scale Granularity Options:**

| Granularity | Input | Weight | Use Case |
|-------------|-------|--------|----------|
| `TENSOR` | One scale per tensor | One scale per tensor | Simple, lower accuracy |
| `COLWISE` | Per-token | N/A | Dynamic input activation |
| `ROWWISE` | N/A | Per-output-channel | Static weight quantization |
| `BLOCK` | (1, K) blocks | (N, K) blocks | Fine-grained, highest accuracy |

**Pattern - FBGEMM FP8 (Recommended for H100):**

```python
from max.nn.legacy.float8_config import (
    Float8Config,
    Float8InputScaleSpec,
    Float8WeightScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
)
from max.dtype import DType

# FBGEMM FP8: Dynamic column-wise input, static row-wise weights
input_spec = Float8InputScaleSpec(
    granularity=Float8ScaleGranularity.COLWISE,  # Per-token scaling
    origin=Float8ScaleOrigin.DYNAMIC,
    dtype=DType.float8_e4m3fn,
    activation_scale_ub=1200.0,  # Optional upper bound
)
weight_spec = Float8WeightScaleSpec(
    granularity=Float8ScaleGranularity.ROWWISE,  # Per-output-channel
    dtype=DType.float32,  # Scale factors are float32
)

float8_config = Float8Config(
    input_scale=input_spec,
    weight_scale=weight_spec,
    mlp_in_float8=set(range(num_layers)),
    attn_qkv_in_float8=set(range(num_layers)),
    embedding_output_dtype=DType.bfloat16,
    quant_method="fbgemm_fp8",
)
```

**Pattern - Block-wise FP8:**

```python
# Block-wise scaling: 128x128 blocks for weights, 1x128 for inputs
input_spec = Float8InputScaleSpec(
    granularity=Float8ScaleGranularity.BLOCK,
    origin=Float8ScaleOrigin.DYNAMIC,
    dtype=DType.float32,
    block_size=(1, 128),  # Required for BLOCK granularity
)
weight_spec = Float8WeightScaleSpec(
    granularity=Float8ScaleGranularity.BLOCK,
    dtype=DType.float32,
    block_size=(128, 128),
)

float8_config = Float8Config(
    input_scale=input_spec,
    weight_scale=weight_spec,
    quant_method="fp8",
)
```

**K-axis Constraint:**
```python
# The K-axis granularity MUST match between input and weight scales
# For matmul: output = input @ weight.T where K is the shared dimension
assert k_input_granularity == k_weight_granularity
```

### GPTQ Quantization (4-bit GPU)

GPTQ (Group-wise Post-Training Quantization) enables memory-efficient 4-bit inference on NVIDIA GPUs.

**Configuration:**

```python
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.nn.legacy import GPTQLinear
from max.dtype import DType

# Standard GPTQ configuration
gptq_config = QuantizationConfig(
    quant_method="gptq",
    bits=4,              # 4-bit quantization
    group_size=128,      # Common default
    desc_act=False,      # Faster inference
    sym=True,            # Symmetric (required)
)

# Create GPTQ linear layer
linear = GPTQLinear(
    in_dim=4096,
    out_dim=4096,
    dtype=DType.uint8,  # Weights stored as packed uint8
    device=DeviceRef.GPU(0),
    quantization_encoding=QuantizationEncoding.GPTQ,
    quantization_config=gptq_config,
)
```

**Group Size Impact:**

| Group Size | Accuracy | Speed | Memory | Use Case |
|------------|----------|-------|--------|----------|
| 32 | Highest | Slowest | Most scales | Quality-critical |
| 64 | High | Medium | Moderate | Balanced |
| 128 | Good | Fast | Fewer scales | Default |
| -1 (full) | Lowest | Fastest | Minimal | Maximum compression |

**desc_act Tradeoff:**

| desc_act | Accuracy | Speed | When to Use |
|----------|----------|-------|-------------|
| False | Good | Faster | Standard inference |
| True | Better | Slower | Accuracy-critical |

**Key Constraints:**
- GPTQ is GPU-only (use GGUF formats for CPU)
- Only `sym=True` (symmetric) is supported
- Requires float16 scales in HuggingFace config
- Cannot combine with Float8 quantization

### Graph-Level Quantization

Use MAX's quantization APIs for memory-efficient inference at the graph level. Quantization is applied through `QuantizationEncoding` on `Weight` objects and `ops.qmatmul()` for quantized matrix multiplication.

**Pattern — Quantized Weight:**

```python
from max.graph import Weight, DeviceRef
from max.graph.quantization import QuantizationEncoding, QuantizationConfig
from max.dtype import DType

# Create a quantized weight using QuantizationEncoding enum
quantized_weight = Weight(
    name="linear.weight",
    dtype=DType.uint8,  # Quantized weights stored as packed uint8
    shape=[4096, 4096],
    device=DeviceRef.GPU(0),
    quantization_encoding=QuantizationEncoding.Q4_K,
)

# Use qmatmul for quantized matmul
from max.graph.ops.quantized import qmatmul
result = qmatmul(
    encoding=QuantizationEncoding.Q4_K,
    config=None,  # or QuantizationConfig for GPTQ
    lhs=input_tensor,
    rhs=quantized_weight,
)
```

**Supported QuantizationEncoding Values:**

| Encoding | Bits | Elements/Block | Use Case |
|----------|------|----------------|----------|
| `QuantizationEncoding.Q4_0` | 4 | 32 | Basic 4-bit |
| `QuantizationEncoding.Q4_K` | 4 | 256 | K-quant 4-bit (recommended) |
| `QuantizationEncoding.Q5_K` | 5 | 256 | K-quant 5-bit |
| `QuantizationEncoding.Q6_K` | 6 | 256 | K-quant 6-bit |
| `QuantizationEncoding.Q8_0` | 8 | 32 | Graph-level input pre-quantization |
| `QuantizationEncoding.GPTQ` | 4 | per-group | Group-wise PTQ for LLMs |

---

## Common Patterns

### FBGEMM FP8 for Production

**When:** H100 GPU deployment requiring maximum throughput

**Do:**
```python
# FBGEMM FP8 with dynamic input scaling
input_spec = Float8InputScaleSpec(
    granularity=Float8ScaleGranularity.COLWISE,
    origin=Float8ScaleOrigin.DYNAMIC,
    dtype=DType.float8_e4m3fn,
)
weight_spec = Float8WeightScaleSpec(
    granularity=Float8ScaleGranularity.ROWWISE,
    dtype=DType.float32,
)
```

**Don't:**
```python
# nocompile
# ERROR: K-axis granularities must match
input_spec = Float8InputScaleSpec(
    granularity=Float8ScaleGranularity.TENSOR,  # K-axis: full tensor
)
weight_spec = Float8WeightScaleSpec(
    granularity=Float8ScaleGranularity.BLOCK,   # K-axis: block
)
# This may not work correctly for all workloads
```

### GPTQ for Memory-Constrained GPU

**When:** Large model on limited GPU memory

**Do:**
```bash
# Load GPTQ model from HuggingFace
max serve --model <your-gptq-model> \
  --quantization-encoding gptq
```

```python
# Standard configuration
gptq_config = QuantizationConfig(
    quant_method="gptq",
    bits=4,
    group_size=128,
    desc_act=False,
    sym=True,
)
```

**Don't:**
```python
# nocompile
# ERROR: GPTQ is GPU-only
config = PipelineConfig(
    model="<any-gptq-model>",
    quantization_encoding="gptq",
    device_specs=["cpu"],  # GPTQ not supported on CPU
)

# ERROR: sym=False not supported
gptq_config = QuantizationConfig(
    quant_method="gptq",
    sym=False,  # Will raise AssertionError
)
```

### GGUF for CPU Deployment

**When:** CPU-only inference with quantization

**Do:**
```python
from max.pipelines.lib import SupportedEncoding

# Use GGUF formats for CPU
if device_type == "cpu":
    encoding = SupportedEncoding.q4_k  # or q6_k for higher accuracy
else:
    encoding = SupportedEncoding.gptq
```

### High-Accuracy GPTQ

**When:** Quality-critical applications

**Do:**
```python
high_accuracy_config = QuantizationConfig(
    quant_method="gptq",
    bits=4,
    group_size=32,   # More fine-grained scaling
    desc_act=True,   # Activation-order descriptor
    sym=True,
)
```

---

## Decision Guide

| Scenario | Quantization | Configuration |
|----------|--------------|---------------|
| H100 GPU, max throughput | Float8 | FBGEMM FP8 |
| GPU, memory-constrained | GPTQ | 4-bit, group_size=128 |
| CPU deployment | GGUF | Q4_K or Q6_K |
| Quality-critical | GPTQ | group_size=32, desc_act=True |
| Max compression | GPTQ | group_size=-1, desc_act=False |

---

## Quick Reference

### Float8

- **FBGEMM FP8**: `COLWISE` input + `ROWWISE` weight (recommended)
- **Block-wise**: Fine-grained scaling with `block_size`
- **K-axis constraint**: Input and weight K-axis granularity must match
- **Static vs Dynamic**: STATIC for stable inputs, DYNAMIC for variable
- **GPU required**: H100/Hopper for hardware FP8 support

### GPTQ

- **GPU-only**: Use GGUF for CPU inference
- **Symmetric only**: `sym=True` required
- **Group size**: 128 default, 32 for accuracy, -1 for compression
- **desc_act**: True for accuracy, False for speed
- **Float16 scales**: Required in HuggingFace config

### Graph Quantization

- **QuantizationEncoding**: Enum with `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `GPTQ`
- **Weight**: Set `quantization_encoding=QuantizationEncoding.Q4_K` on `Weight` constructor
- **qmatmul**: `qmatmul(encoding, config, lhs, rhs)` for quantized matrix multiplication

---

## Supported Quantization Methods

| Method | Format | Input Scaling | Weight Scaling |
|--------|--------|---------------|----------------|
| `fbgemm_fp8` | FP8 E4M3 | Dynamic COLWISE | Static ROWWISE |
| `compressed-tensors` | FP8 E4M3 | Static/Dynamic | Static |
| `fp8` | FP8 E4M3 | Dynamic BLOCK | Static BLOCK |
| `gptq` | INT4 | N/A | Static per-group |
| `q4_k` | GGUF | N/A | K-quant blocks |
| `q6_k` | GGUF | N/A | K-quant blocks |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `quantization encoding not supported` | Model architecture doesn't support quantization | Check supported models; some layers can't be quantized |
| `scale factor mismatch` | Wrong scale for quantization type | Use per-tensor for FP8, per-group for GPTQ |
| `accuracy degradation too high` | Over-aggressive quantization | Use higher precision (FP8 > INT4); calibrate with representative data |
| `GPTQ loading failed` | Incompatible GPTQ format | Verify group_size matches; check model was quantized correctly |
| `float8 not supported` | Hardware doesn't support FP8 | FP8 requires H100+; use GPTQ for older GPUs |
| `dynamic quantization overhead` | Per-batch scale computation | Use static quantization for inference; dynamic for training |

---

## Version-Specific Features

### Stable (v26.1) vs Nightly (v26.2+)

| Feature | Stable (v26.1) | Nightly (v26.2+) |
|---------|----------------|--------------------|
| **Float8 formats** | `float8_e4m3fn` | `float8_e4m3fn`, `float8_e5m2` |
| **NVFP4** | Not available | `float4_e2m1fnx2` (Blackwell GPUs) |
| **GPTQ** | 4-bit with group_size | 4-bit with group_size (unchanged) |
| **Block-wise scaling** | `Float8ScaleGranularity.BLOCK` | `Float8ScaleGranularity.BLOCK` (unchanged) |

**Stable (v26.1):**
```python
from max.nn.legacy.float8_config import (
    Float8Config,
    Float8InputScaleSpec,
    Float8WeightScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
)

# FBGEMM FP8 configuration
input_spec = Float8InputScaleSpec(
    granularity=Float8ScaleGranularity.COLWISE,
    origin=Float8ScaleOrigin.DYNAMIC,
    dtype=DType.float8_e4m3fn,
)
```

**Nightly (v26.2+):**
```python
from max.nn.legacy.float8_config import (
    Float8Config,
    Float8InputScaleSpec,
    Float8WeightScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
)

# Same FBGEMM FP8 configuration
input_spec = Float8InputScaleSpec(
    granularity=Float8ScaleGranularity.COLWISE,
    origin=Float8ScaleOrigin.DYNAMIC,
    dtype=DType.float8_e4m3fn,
)

# NEW: NVFP4 for Blackwell GPUs (v26.2+)
# max serve --model ... --quantization-encoding float4_e2m1fnx2
```

**Notes:**
- FBGEMM FP8 and block-wise FP8 patterns are stable across versions
- GPTQ 4-bit quantization is stable across versions
- GGUF CPU quantization (q4_k, q6_k) is stable across versions
- NVFP4 (`float4_e2m1fnx2`) requires Blackwell GPUs and v26.2+
- K-axis constraint (input and weight K-axis granularity must match) is stable

---

## Related Patterns

- [`engine-weights.md`](engine-weights.md) — Weight management and sharding
- [`serve-kv-cache.md`](serve-kv-cache.md) — Memory management

---

## References

- [MAX Documentation](https://docs.modular.com/max/)
- Source: `max/nn/legacy/float8_config.py`, `max/graph/quantization.py`
