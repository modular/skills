---
title: Weight Management and Sharding
description: Weight sharding strategies, weight adapters, buffer transfers, and DLPack integration
impact: HIGH
category: engine
tags: [weights, sharding, adapters, buffers, dlpack]
error_patterns:
  - "weight"
  - "sharding"
  - "buffer"
  - "DLPack"
  - "transfer"
  - "device mismatch"
scenarios:
  - "Shard weights across GPUs"
  - "Create weight adapter"
  - "Transfer buffers between devices"
  - "Use DLPack interop"
  - "Configure weight sharding strategy"
consolidates:
  - engine-weight-sharding.md
  - engine-weight-adapter.md
  - engine-buffer-transfer.md
  - engine-dlpack.md
---

# Weight Management and Sharding

**Category:** engine | **Impact:** HIGH

Comprehensive guide to weight management in MAX Engine including sharding strategies for multi-GPU inference, weight adapters for different model formats, buffer transfers between devices, and DLPack interoperability.

---

## Core Concepts

### Weight Sharding Strategies

Use sharding strategies to distribute model weights across multiple GPUs for large models.

**Available Strategies:**

```python
from max.graph import Weight, ShardingStrategy
from max.dtype import DType

weight = Weight(
    name="model.weight",
    dtype=DType.bfloat16,
    shape=[4096, 4096],
    device=DeviceRef.GPU(0),
)

# Row-wise: splits output dimension (no AllReduce needed)
weight.sharding_strategy = ShardingStrategy.rowwise(num_devices=4)

# Column-wise: splits input dimension (requires AllReduce)
weight.sharding_strategy = ShardingStrategy.columnwise(num_devices=4)

# Replicate: copy to all devices (for small tensors)
weight.sharding_strategy = ShardingStrategy.replicate(num_devices=4)

# Head-aware: for attention with uneven head distribution
weight.sharding_strategy = ShardingStrategy.head_aware_columnwise(
    num_devices=4, num_heads=32, head_dim=128
)

# Stacked QKV: for fused QKV projections
weight.sharding_strategy = ShardingStrategy.stacked_qkv(
    num_devices=4, num_heads=32, head_dim=128
)

# Tensor parallel: for MLP layers (auto-shards gate/up/down)
# WARNING: Currently a placeholder — raises NotImplementedError at runtime.
# Use rowwise/columnwise on individual MLP weights instead (see MLP pattern below).
weight.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices=4)

# Expert parallel: for MoE expert layers
# WARNING: Currently a placeholder — raises NotImplementedError at runtime.
# Use rowwise/columnwise on individual expert weights instead.
weight.sharding_strategy = ShardingStrategy.expert_parallel(num_devices=4)

# Gate-up: for fused gate+up projections
weight.sharding_strategy = ShardingStrategy.gate_up(num_devices=4, axis=2)

# Create shards for each device
devices = [DeviceRef.GPU(i) for i in range(4)]
shards = weight.shard(devices)
```

**Strategy Selection Guide:**

| Strategy | Communication | Best For |
|----------|--------------|----------|
| `rowwise` | None | QKV projections, gate/up projections (first linear in a pair) |
| `columnwise` | AllReduce | Output projections, down projections (second linear in a pair) |
| `replicate` | None | Small tensors, embeddings |
| `head_aware_columnwise` | AllReduce | Attention QKV |
| `stacked_qkv` | AllReduce | Fused QKV with GQA |
| `tensor_parallel` | AllReduce | MLP layers (auto-shards gate/up/down) — **placeholder; raises `NotImplementedError`** |
| `expert_parallel` | AllReduce | MoE expert layers — **placeholder; raises `NotImplementedError`** |
| `gate_up` | AllReduce | Fused gate+up projections |

### Weight Adapters

Use `WeightsAdapter` functions to transform checkpoint state dicts to MAX weight naming conventions.

**Pattern - Name Mapping:**

```python
from max.graph.weights import WeightData, Weights, WeightsAdapter

# Define name mapping from checkpoint format to MAX format
MY_MODEL_SAFETENSOR_MAPPING = {
    "model.": "",  # Strip "model." prefix
    "layers.": "transformer.layers.",  # Rename layer prefix
    ".self_attn.q_proj": ".attention.wq",
    ".self_attn.k_proj": ".attention.wk",
    ".self_attn.v_proj": ".attention.wv",
    ".self_attn.o_proj": ".attention.wo",
    ".mlp.gate_proj": ".feed_forward.w1",
    ".mlp.up_proj": ".feed_forward.w3",
    ".mlp.down_proj": ".feed_forward.w2",
}

def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config,
    pipeline_config,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert HuggingFace safetensor weights to MAX format."""
    new_state_dict: dict[str, WeightData] = {}

    for hf_name, weight in state_dict.items():
        max_name = hf_name
        for before, after in MY_MODEL_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = weight.data()

    return new_state_dict
```

**Pattern - Filtering Weights:**

```python
def convert_with_filtering(state_dict: dict[str, Weights], **kwargs) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}
    for name, weight in state_dict.items():
        # Skip vision model weights if only loading language model
        if name.startswith("vision_model."):
            continue
        # Skip unused GPTQ keys
        if name.endswith(".qzeros") or name.endswith(".bias"):
            continue
        new_state_dict[name] = weight.data()
    return new_state_dict
```

### Buffer Transfers

Use explicit buffer management for efficient GPU memory usage and interoperability.

**Pattern - Buffer Creation and Transfer:**

```python
from max.driver import Buffer, CPU, Accelerator
import numpy as np

# Create buffer from numpy
np_array = np.random.randn(10, 10).astype(np.float32)
buffer = Buffer.from_numpy(np_array)

# Transfer to GPU
device_buffer = buffer.to(Accelerator())

# Execute model
results = model.execute(device_buffer)

# Transfer results back to CPU
result = results[0].to(CPU())
output_array = result.to_numpy()
```

### DLPack Interoperability

Use DLPack protocol for zero-copy data exchange between MAX and other ML frameworks.

**Pattern - PyTorch Integration:**

```python
from max.driver import Buffer
from max.tensor import Tensor
import torch

# PyTorch to MAX Buffer (zero-copy on same device)
torch_tensor = torch.randn(10, 10, device="cuda")
max_buffer = Buffer.from_dlpack(torch_tensor)

# PyTorch to MAX Tensor (zero-copy)
max_tensor = Tensor.from_dlpack(torch_tensor)

# MAX to PyTorch (zero-copy)
max_tensor = Tensor.zeros([10, 10])
torch_tensor = torch.from_dlpack(max_tensor)

# NumPy to MAX Buffer (zero-copy if contiguous)
np_array = np.random.randn(10, 10).astype(np.float32)
max_buffer = Buffer.from_dlpack(np_array)
```

**Key Constraints:**
- Arrays must be contiguous (C-order for NumPy)
- Producer retains ownership of underlying memory
- Device placement must match for zero-copy
- Read-only NumPy arrays may require explicit `copy=True`

---

## Common Patterns

### Multi-GPU Weight Distribution

**When:** Model doesn't fit on single GPU

**Do:**
```python
# Distribute across 4 GPUs
weight = Weight(name="model.weight", shape=[16384, 16384], device=DeviceRef.GPU(0))
weight.sharding_strategy = ShardingStrategy.columnwise(num_devices=4)
shards = weight.shard([DeviceRef.GPU(i) for i in range(4)])
```

**Don't:**
```python
# nocompile
# Model too large for single GPU - will OOM
weight = Weight(name="model.weight", shape=[16384, 16384], device=DeviceRef.GPU(0))
```

### MLP Tensor Parallel Pattern

**When:** Implementing tensor parallel for MLP layers

**Do:**
```python
from max.nn.linear import MLP
from max.graph.quantization import QuantizationEncoding

mlp = MLP(
    dtype=DType.bfloat16,
    quantization_encoding=None,  # Required parameter
    hidden_dim=4096,
    feed_forward_length=11008,
    devices=[DeviceRef.GPU(i) for i in range(4)],
    activation_function="silu",
)

# Set tensor parallel sharding
# WARNING: ShardingStrategy.tensor_parallel() is currently a placeholder and
# raises NotImplementedError at runtime. Instead, set sharding on individual
# weights: gate_proj → rowwise, up_proj → rowwise, down_proj → columnwise.
# Check the latest API docs for updates.
# - gate_proj: rowwise sharding
# - down_proj: columnwise sharding
# - up_proj: rowwise sharding
mlp.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices=4)
devices = [DeviceRef.GPU(i) for i in range(4)]
shards = mlp.shard(devices)  # Returns list[MLP], one per device
```

### Pre-Transfer for Batch Processing

**When:** Processing multiple batches with same device

**Do:**
```python
# Transfer once, execute many
device = Accelerator()
inputs = [Buffer.from_numpy(x).to(device) for x in batches]

for device_input in inputs:
    result = model.execute(device_input)  # No transfer needed
```

**Don't:**
```python
# nocompile
# Each execute() call transfers data (slow)
for batch in batches:
    np_input = process(batch)
    result = model.execute(np_input)  # Implicit H2D transfer
```

### Zero-Copy PyTorch Integration

**When:** Integrating MAX with PyTorch pipelines

**Do:**
```python
# PyTorch preprocessing -> MAX inference -> PyTorch postprocessing
def inference_pipeline(torch_input: torch.Tensor) -> torch.Tensor:
    # Zero-copy input transfer
    max_input = Buffer.from_dlpack(torch_input)

    # Run inference
    results = model.execute(max_input)

    # Zero-copy output transfer
    return torch.from_dlpack(results[0])
```

**Don't:**
```python
# nocompile
# BAD: Unnecessary explicit copy
torch_tensor = torch.randn(1000, 1000)
np_array = torch_tensor.numpy()  # Creates copy
max_buffer = Buffer.from_numpy(np_array)  # Another potential copy
```

### Handling Non-Contiguous Arrays

**When:** Working with transposed or sliced arrays

**Do:**
```python
# NumPy non-contiguous - make contiguous first
np_transposed = np.arange(12).reshape(3, 4).T
np_contiguous = np.ascontiguousarray(np_transposed)
buffer = Buffer.from_dlpack(np_contiguous)

# PyTorch non-contiguous - make contiguous first
torch_transposed = torch.arange(12).reshape(3, 4).T
torch_contiguous = torch_transposed.contiguous()
buffer = Buffer.from_dlpack(torch_contiguous)
```

**Don't:**
```python
# nocompile
# WRONG: Will raise ValueError
np_transposed = np.arange(12).reshape(3, 4).T
buffer = Buffer.from_dlpack(np_transposed)  # ValueError!
```

---

## Decision Guide

| Scenario | Strategy | Notes |
|----------|----------|-------|
| Single GPU | No sharding | Simplest approach |
| Model too large | Column/row-wise sharding | Distribute across GPUs |
| Attention layers | `head_aware_columnwise` | Handles GQA properly |
| Fused QKV | `stacked_qkv` | For combined projections |
| Small tensors | `replicate` | Avoid communication |
| PyTorch integration | DLPack | Zero-copy transfer |
| Batch processing | Pre-transfer | Transfer once, execute many |

---

## Quick Reference

- **Rowwise sharding**: No communication, use for QKV/gate/up projections (first linear in a pair)
- **Columnwise sharding**: Requires AllReduce, use for output/down projections (second linear in a pair)
- **Replicate**: For embeddings and layer norms (small tensors)
- **DLPack**: Zero-copy between MAX and PyTorch/JAX/NumPy
- **Contiguity**: Arrays must be contiguous for DLPack
- **Ownership**: Producer retains ownership, keep source alive

---

## Weight Adapter Registration

```python
from max.pipelines.lib.registry import SupportedArchitecture, PIPELINE_REGISTRY
from max.graph.weights import WeightsFormat

my_architecture = SupportedArchitecture(
    name="MyModelForCausalLM",
    example_repo_ids=["your-org/your-model"],
    # Register weight adapters for each supported format
    weight_adapters={
        WeightsFormat.safetensors: convert_safetensor_state_dict,
        WeightsFormat.gguf: convert_gguf_state_dict,
    },
    # ... other configuration
)

PIPELINE_REGISTRY.register(my_architecture)
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `weight shape mismatch` | Wrong tensor dimensions | Verify shape matches model architecture; check transpose |
| `DLPack conversion failed` | Incompatible memory layout | Ensure tensor is contiguous; check dtype compatibility |
| `shard index out of range` | Wrong shard count | Ensure shard indices 0 to n-1 for n GPUs |
| `adapter loading failed` | Incompatible LoRA format | Check adapter was trained for this base model |
| `out of memory loading weights` | Weights too large for GPU | Use sharding across GPUs; enable quantization |
| `weight dtype mismatch` | Model expects different dtype | Cast weights: `weights.to(torch.bfloat16)` before loading |

---

## Version-Specific Features

### Stable (v26.1) vs Nightly (v26.2+)

| Feature | Stable (v26.1) | Nightly (v26.2+) |
|---------|----------------|--------------------|
| **Sharding strategies** | `rowwise()`, `columnwise()`, `replicate()`, `stacked_qkv()`, `head_aware_columnwise()`, `tensor_parallel()`, `expert_parallel()`, `gate_up()` | Same (unchanged) |
| **Weight adapters** | `WeightsAdapter` functions | `WeightsAdapter` functions (unchanged) |
| **DLPack** | `Buffer.from_dlpack()`, `Tensor.from_dlpack()` | Same (unchanged) |
| **DeviceRef** | `DeviceRef.GPU(0)` | `DeviceRef.GPU(0)` (unchanged) |

**Stable (v26.1):**
```python
from max.graph import Weight, ShardingStrategy
from max.dtype import DType

# Create weight with sharding
weight = Weight(
    name="model.weight",
    dtype=DType.bfloat16,
    shape=[4096, 4096],
    device=DeviceRef.GPU(0),
)

# Column-wise sharding for tensor parallel
weight.sharding_strategy = ShardingStrategy.columnwise(num_devices=4)

# DLPack interop
from max.driver import Buffer
import torch
max_buffer = Buffer.from_dlpack(torch_tensor)
```

**Nightly (v26.2+):**
```python
from max.graph import Weight, ShardingStrategy
from max.dtype import DType

# Same API in v26.2+
weight = Weight(
    name="model.weight",
    dtype=DType.bfloat16,
    shape=[4096, 4096],
    device=DeviceRef.GPU(0),
)

weight.sharding_strategy = ShardingStrategy.columnwise(num_devices=4)

# DLPack interop unchanged
from max.driver import Buffer
import torch
max_buffer = Buffer.from_dlpack(torch_tensor)
```

**Notes:**
- Weight sharding strategies are stable across versions
- `WeightsAdapter` registration pattern is stable
- DLPack interoperability with PyTorch/NumPy is stable
- Buffer transfer APIs (`to(Accelerator())`, `to(CPU())`) are stable
- Zero-copy semantics for contiguous arrays are stable

---

## Related Patterns

- [`engine-quantization.md`](engine-quantization.md) — Quantization for weight compression
- [`serve-configuration.md`](serve-configuration.md) — Multi-GPU serving configuration

---

## References

- [MAX Multi-GPU](https://docs.modular.com/max/serve/)
- [MAX Engine API](https://docs.modular.com/max/api/python/engine)
- [MAX Driver Buffer](https://docs.modular.com/max/api/python/driver)
