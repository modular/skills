---
title: MAX Engine Operations
description: Custom operations, architecture registration, inference sessions, and graph management
impact: HIGH
category: engine
tags: [custom-ops, architecture, inference, caching, subgraphs]
error_patterns:
  - "DeviceRef has no attribute 'from_device'"
  - "custom() missing required argument 'device'"
  - "TensorType missing required argument 'device'"
  - "Could not find mojo kernel"
  - "unable to locate module 'max'"
  - "Failed to resolve module path for MOGGKernelAPI"
  - "kernel compilation failed"
scenarios:
  - "Create custom MAX operation"
  - "Register new model architecture"
  - "Fix custom ops API error"
  - "Configure inference session"
  - "Run offline batch inference"
  - "Build complete custom model with MAX"
  - "Integrate Mojo layers with Python serving"
  - "Create mixed Mojo/Python model project"
consolidates:
  - engine-custom-ops.md
  - engine-custom-op-integration.md
  - engine-architecture-registration.md
  - engine-inference-session.md
  - engine-offline-inference.md
  - engine-graph-caching.md
  - engine-subgraphs.md
---

# MAX Engine Operations

**Category:** engine | **Impact:** HIGH

> **WARNING: Breaking API Change Between Versions**
>
> The `foreach` callback signature differs between stable and nightly:
>
> - **v26.1.0.0.0 (stable):** `fn[width: Int, element_alignment: Int](idx: IndexList[rank]) -> SIMD[dtype, width]`
> - **v26.2+ (nightly):** `fn[width: Int](idx: IndexList[rank]) -> SIMD[dtype, width]`
>
> Using the wrong signature causes `no matching function in call to 'foreach'` errors.
> See the [Custom Operations](#custom-operations-with-inputtensoroutputtensor) section for version-specific patterns.

Comprehensive patterns for MAX Engine operations including custom operations with Mojo kernels, architecture registration for custom models, inference session management, offline batch inference, graph caching, and subgraph execution.

---

## Core Concepts

### Custom Operations with InputTensor/OutputTensor

Use `InputTensor` and `OutputTensor` types for custom ops (not deprecated `ManagedTensorSlice`).

**Critical: Version Alignment**

MAX Python package and Mojo must be version-aligned. Check with:

```bash
pip show max | grep Version   # e.g., 26.2.0
mojo --version                # Must match major.minor
```

**Pattern (Stable v26.1.0.0.0):**

```mojo
# kernels/my_op.mojo
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList

@compiler.register("my_op")
struct MyOp:
    @staticmethod
    fn execute[target: StaticString](
        output: OutputTensor,
        inp: InputTensor[dtype=output.dtype, rank=output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn op[width: Int, element_alignment: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, width]:
            return inp.load[width](idx) * 2

        foreach[op, target=target](output, ctx)
```

**Pattern (Nightly v26.2+):**

```mojo
# kernels/my_op.mojo
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList

@compiler.register("my_op")
struct MyOp:
    @staticmethod
    fn execute[target: StaticString](
        output: OutputTensor,
        inp: InputTensor[dtype=output.dtype, rank=output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn op[width: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, width]:
            return inp.load[width](idx) * 2

        foreach[op, target=target](output, ctx)
```

**Critical difference:** In stable v26.1.0.0.0, the `foreach` callback **must** have `element_alignment: Int` as a second compile-time parameter. In nightly v26.2+, only `width: Int` is required.

**Don't:**

```mojo
# nocompile - Anti-pattern examples (won't compile)
# Wrong imports (won't compile)
from max.tensor import InputTensor, OutputTensor  # Use: from tensor import ...
from max.driver import DeviceContextPtr           # Use: from runtime.asyncrt import ...

# Old API (deprecated)
@compiler.register("my_op", num_dps_outputs=1)
struct MyOp:
    fn execute(out: ManagedTensorSlice, inp: ManagedTensorSlice): ...
```

### Inference Session Configuration

Always specify devices explicitly when creating InferenceSession.

**Pattern:**

```python
from max.engine import InferenceSession
from max.driver import CPU, Accelerator

# CPU inference
session = InferenceSession(devices=[CPU()])

# GPU inference
session = InferenceSession(devices=[Accelerator()])
```

**Don't:**

```python
# No device specified - may default unexpectedly
session = InferenceSession()
```

---

## Common Patterns

### Custom Op Integration Pipeline

**When:** Implementing operations not available in MAX ops, fusing operations for performance, or adding hardware-specific optimizations.

The complete custom op integration pipeline connects Mojo kernels to MAX graphs through `@compiler.register`.

```
Mojo Kernel          @compiler.register        ops.custom()         MAX Graph
   (*.mojo)    --->      decorator      --->   Python call   --->   execution
```

**Step 1 - Mojo Kernel with Registration:**

**Stable (v26.1.0.0.0):**

```mojo
# nocompile - v26.1.0.0.0 only (element_alignment parameter removed in v26.2+)
# kernels/my_custom_op.mojo
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList

@compiler.register("my_custom_op")
struct MyCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu" - set at compile time
    ](
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        # NOTE: v26.1.0.0.0 requires element_alignment parameter
        fn elementwise_op[width: Int, element_alignment: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, width]:
            return x.load[width](idx) * 2  # Example: double values

        foreach[elementwise_op, target=target](output, ctx)
```

**Step 2 - Python Graph Integration:**

**Stable (v26.1.0.0.0):**

```python
from pathlib import Path
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

# Build kernel package first: mojo package kernels -o kernels.mojopkg
# Kernel directory must have __init__.mojo
mojo_kernels = Path(__file__).parent / "kernels"
device = Accelerator() if accelerator_count() > 0 else CPU()
device_ref = DeviceRef.from_device(device)

graph = Graph(
    "my_graph",
    forward=lambda x: ops.custom(
        name="my_custom_op",
        device=device_ref,  # device is REQUIRED
        values=[x],
        out_types=[TensorType(dtype=x.dtype, shape=x.tensor.shape, device=device_ref)],
    )[0].tensor,
    input_types=[TensorType(DType.float32, shape=[rows, cols], device=device_ref)],
    custom_extensions=[mojo_kernels],  # Must point to compiled .mojopkg or directory with __init__.mojo
)

session = InferenceSession(devices=[device])
model = session.load(graph)
result = model.execute(Buffer.from_numpy(input_array).to(device))[0]
```

**Nightly (v26.2+):**

```python
import numpy as np
from pathlib import Path
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

# Build kernel package first: mojo package kernels -o kernels.mojopkg
# Kernel directory must have __init__.mojo
mojo_kernels = Path(__file__).parent / "kernels"

# DeviceRef.from_device() works across all versions (NOT deprecated)
# DeviceRef.CPU() / DeviceRef.GPU() are convenient static constructors
device = Accelerator() if accelerator_count() > 0 else CPU()
device_ref = DeviceRef.from_device(device)  # Or DeviceRef.GPU() / DeviceRef.CPU()

def forward(x):
    # ops.custom signature: custom(name, device, values, out_types, parameters=None)
    # device accepts Device | DeviceRef
    return ops.custom(
        name="my_custom_op",
        device=device_ref,  # Required
        values=[x],
        out_types=[TensorType(dtype=x.dtype, shape=x.tensor.shape, device=device_ref)],
    )[0].tensor

# custom_extensions MUST be passed during Graph construction
# Must point to compiled .mojopkg or directory with __init__.mojo
graph = Graph(
    "my_graph",
    forward=forward,
    input_types=[TensorType(DType.float32, [rows, cols], device=device_ref)],
    custom_extensions=[mojo_kernels],  # Required at construction time
)

session = InferenceSession(devices=[device])
model = session.load(graph)

# Execute with Buffer (v26.2+ uses Buffer, previously Tensor)
from max.driver import Buffer
input_data = np.random.randn(rows, cols).astype(np.float32)
input_buffer = Buffer.from_numpy(input_data).to(device)
result = model.execute(input_buffer)

# Convert output Buffer to numpy
output = result[0].to(CPU()).to_numpy()  # Transfer to CPU first, then to numpy
```

**Common Errors:**

- `DeviceRef has no attribute 'from_device'` → Ensure `from max.graph import DeviceRef` (not from `max.driver`)
- `custom() missing required argument 'device'` → `device` is now positional before `values`
- `TensorType missing required argument 'device'` → Add `device=device_ref`
- `Could not find mojo kernel` → Ensure `custom_extensions` passed in Graph constructor
- `Failed to resolve module path for MOGGKernelAPI` → Environment issue; reinstall MAX with pixi

**Apple Silicon Notes:**

- Custom ops work on CPU using `DeviceRef.CPU()` and `CPU()` device
- In v26.1.0.0.0+, `accelerator_count()` returns non-zero on Apple Silicon but custom GPU ops may require NVIDIA/AMD
- For development, use CPU device: `device = CPU()` and `device_ref = DeviceRef.CPU()`

**Step 3 - Parameterized Kernels (v26.2+):**

```mojo
# kernels/add_constant.mojo
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList

@compiler.register("add_constant")
struct AddConstant[value: Int]:  # Compile-time parameter
    @staticmethod
    fn execute[target: StaticString](
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn add_const[width: Int](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return x.load[width](idx) + Self.value

        foreach[add_const, target=target](output, ctx)
```

**Step 3 - Parameterized Kernels (v26.1.0.0.0):**

```mojo
# nocompile - v26.1.0.0.0 only (element_alignment parameter removed in v26.2+)
# kernels/add_constant.mojo
@compiler.register("add_constant")
struct AddConstant[value: Int]:  # Compile-time parameter
    @staticmethod
    fn execute[target: StaticString](
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        # NOTE: v26.1.0.0.0 requires element_alignment parameter
        fn add_const[width: Int, element_alignment: Int](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return x.load[width](idx) + Self.value

        foreach[add_const, target=target](output, ctx)
```

**v26.1.0.0.0:**

```python
# nocompile - v26.1.0.0.0 only
# Pass compile-time parameters to kernel
result = ops.custom(
    name="add_constant",
    device=DeviceRef.from_device(device),
    values=[x],
    parameters={"value": 5},  # Maps to struct parameter [value: Int]
    out_types=[TensorType(dtype=x.dtype, shape=x.tensor.shape, device=DeviceRef.from_device(device))],
)[0].tensor
```

**v26.2+ (current):**

```python
# Use DeviceRef.from_device(), DeviceRef.CPU(), or DeviceRef.GPU()
result = ops.custom(
    name="add_constant",
    device=DeviceRef.from_device(device),  # or DeviceRef.GPU()
    values=[x],
    parameters={"value": 5},
    out_types=[TensorType(dtype=x.dtype, shape=x.tensor.shape, device=DeviceRef.from_device(device))],
)[0].tensor
```

Supported parameter types: `bool`, `int`, `str`, `DType`.

---

### Architecture Registration

**When:** Adding new model architectures to MAX pipelines, enabling quantization support for custom models.

**Pattern:**

```python
from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    PIPELINE_REGISTRY,
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from .model import MyPipelineModel
from .model_config import MyModelConfig
from . import weight_adapters

my_arch = SupportedArchitecture(
    name="MyModelForCausalLM_Legacy",  # HuggingFace name + "_Legacy" suffix for legacy Module architectures
    example_repo_ids=["my-org/my-model-7b", "my-org/my-model-13b"],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        SupportedEncoding.q4_k: [KVCacheStrategy.PAGED],
        SupportedEncoding.float8_e4m3fn: [KVCacheStrategy.PAGED],
    },
    pipeline_model=MyPipelineModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    rope_type=RopeType.normal,
    multi_gpu_supported=True,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=MyModelConfig,
)

PIPELINE_REGISTRY.register(my_arch)
```

**Required Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | HuggingFace `architectures` field name + `_Legacy` suffix for legacy Module architectures |
| `example_repo_ids` | `list[str]` | HuggingFace repos for testing/validation |
| `default_encoding` | `SupportedEncoding` | Default quantization when not specified |
| `supported_encodings` | `dict` | Encoding-to-cache-strategy mapping |
| `pipeline_model` | `type[PipelineModel]` | Model class with `load_model()` and `execute()` |
| `task` | `PipelineTask` | Task type (TEXT_GENERATION, EMBEDDINGS_GENERATION, etc.) |

**Encoding/Strategy Compatibility:**

- GPU only: `bfloat16`, `float8_e4m3fn`, `float4_e2m1fnx2`, `gptq`
- CPU only: `q4_k`, `q4_0`, `q6_k` (GGUF quantization)
- Universal: `float32`

---

### Offline Batch Inference

**When:** Processing large datasets, running data pipelines, generating training data, or batch evaluation.

**Pattern:**

```python
from max.entrypoints.llm import LLM
from max.pipelines import PipelineConfig

def batch_process(prompts: list[str]) -> list[str]:
    pipeline_config = PipelineConfig(model_path="modularai/Llama-3.1-8B-Instruct-GGUF")
    llm = LLM(pipeline_config)

    # Batch all prompts in a single call - internally parallelized
    return list(llm.generate(prompts, max_new_tokens=50))
```

**Don't:**

```python
# Anti-pattern: HTTP overhead for batch processing
import requests

def batch_process(prompts: list[str]) -> list[str]:
    results = []
    for prompt in prompts:
        response = requests.post(
            "http://localhost:8000/v1/completions",
            json={"prompt": prompt, "max_tokens": 50}
        )
        results.append(response.json()["choices"][0]["text"])
    return results
```

**Memory Optimization for Large Batches:**

```python
from max.pipelines import PipelineConfig
from max.pipelines.kv_cache import KVCacheStrategy

pipeline_config = PipelineConfig(
    model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
    max_batch_size=32,
    cache_strategy=KVCacheStrategy.PAGED,
    device_memory_utilization=0.85,
    enable_chunked_prefill=True,
    max_batch_input_tokens=8192,
)

llm = LLM(pipeline_config)
```

---

### Graph Caching

**When:** Development workflows, CI/CD pipelines, production deployments.

MAX caches compiled kernels for up to 28% faster compilation on subsequent runs.

**Pattern:**

```python
from max.engine import InferenceSession

# First run: full compilation
session = InferenceSession(devices=[Accelerator()])
model = session.load(graph)

# Subsequent runs: cached kernels loaded automatically
# No code changes needed - caching is automatic
```

**Benefits:**

- 28% faster compilation for iterative development
- Shared cache between similar model architectures
- Automatic invalidation when custom ops change

**Note:** Cache is architecture-specific and not portable across targets.

---

### Subgraphs for Device-Aware Scheduling

**When:** Multi-device execution, hybrid CPU/GPU pipelines, conditional computation paths.

**Pattern:**

```python
from max.graph import Graph, TensorType, ops

# Create main graph
main_graph = Graph(TensorType(DType.float32, "batch", 128))

# Add subgraph with device specification
subgraph = main_graph.add_subgraph(
    inputs=[TensorType(DType.float32, "batch", 128)],
    devices=[Accelerator(0)]  # Run on GPU 0
)

# Build subgraph operations
x = subgraph[0]
result = ops.relu(x)
subgraph.output(result)

# Call subgraph from main graph
output = ops.call(subgraph, main_graph[0])
main_graph.output(output)
```

**Note:** Added in v26.1.0.0.0 with `devices` argument.

---

### Complete Custom Model Project Structure

**When:** Building a complete custom model architecture with Mojo layers and MAX serving.

This example shows the recommended project structure for a mixed Mojo/Python model project:

```
my-custom-model/
├── pixi.toml                    # Mojo/MAX dependencies
├── pyproject.toml               # Python dependencies
├── README.md
│
├── model/                       # Pure Mojo model code
│   ├── __init__.mojo
│   ├── config.mojo              # @fieldwise_init struct for model config
│   ├── state.mojo               # Model state with UnsafePointer
│   ├── model.mojo               # Full model assembly
│   └── layers/
│       ├── __init__.mojo
│       ├── attention.mojo       # Attention layer
│       ├── feedforward.mojo     # FFN layer
│       ├── layer_norm.mojo      # Normalization
│       └── embedding.mojo       # Token embeddings
│
├── kernels/                     # Custom GPU ops
│   ├── __init__.mojo
│   └── fused_attention.mojo     # @compiler.register kernels
│
├── serving/                     # Python MAX integration
│   ├── __init__.py
│   ├── architecture.py          # PIPELINE_REGISTRY registration
│   ├── pipeline_model.py        # PipelineModel implementation
│   ├── graph_builder.py         # MAX Graph with custom ops
│   └── tokenizer.py             # Tokenizer wrapper
│
├── weights/                     # Weight conversion
│   └── convert_hf.py            # HuggingFace → MAX format
│
└── tests/
    ├── test_layers.mojo         # Mojo layer tests
    └── test_serving.py          # Python serving tests
```

**pixi.toml (Mojo dependencies):**

```toml
[project]
name = "my-custom-model"
version = "0.1.0"
channels = ["conda-forge", "https://conda.modular.com/max"]
platforms = ["osx-arm64", "linux-64"]

[dependencies]
max = ">=25.1.0"

[tasks]
test = "mojo test tests/"
```

**pyproject.toml (Python dependencies):**

```toml
[project]
name = "my-custom-model"
version = "0.1.0"
dependencies = [
    "torch>=2.0",
    "transformers>=4.30",
    "safetensors",
]

[project.optional-dependencies]
dev = ["pytest"]
```

**Key Integration Points:**

1. **Model layers (Mojo)**: Use `@fieldwise_init` with `(Copyable, Movable)` traits for config, `UnsafePointer` for weights
2. **Custom kernels (Mojo)**: Use `@compiler.register` with `InputTensor`/`OutputTensor`
3. **Graph builder (Python)**: Pass kernel directory via `custom_extensions=[Path("kernels")]`
4. **Architecture registration (Python)**: Register with `PIPELINE_REGISTRY` for MAX Serve

**Cross-Reference:** See `mojo-best-practices` patterns:

- `struct-design.md` (mojo-best-practices) — Model config structs
- `memory-ownership.md` (mojo-best-practices) — Weight management with UnsafePointer
- `gpu-fundamentals.md` (mojo-best-practices) — Custom GPU kernels

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Custom GPU kernel | Use `@compiler.register` with `InputTensor`/`OutputTensor` | `gpu-fundamentals.md` (mojo-best-practices) |
| New model architecture | Register with `PIPELINE_REGISTRY` | [`graph-construction.md`](graph-construction.md) |
| Batch data processing | Use `LLM` class for offline inference | [`perf-inference.md`](perf-inference.md) |
| Multi-device execution | Use subgraphs with device specification | [`multigpu-scaling.md`](multigpu-scaling.md) |
| Faster development iteration | Leverage automatic graph caching | — |

---

## Quick Reference

- **Version alignment**: MAX Python and Mojo versions must match (check with `pip show max` and `mojo --version`)
- **Custom ops**: Use `InputTensor`/`OutputTensor`, not `ManagedTensorSlice`
- **Kernel imports**: `from tensor import InputTensor, OutputTensor, foreach` (not `from max.tensor`)
- **foreach callback (v26.1.0.0.0)**: Must include `element_alignment: Int` parameter: `fn[width: Int, element_alignment: Int](idx)`
- **foreach callback (v26.2+)**: Simpler signature: `fn[width: Int](idx)`
- **DeviceRef**: Use `DeviceRef.from_device(device)`, `DeviceRef.CPU()`, or `DeviceRef.GPU()` (all work in all versions)
- **ops.custom**: `device` parameter is REQUIRED in both v26.1.0.0.0 and v26.2
- **custom_extensions**: Must pass during Graph construction; requires compiled `.mojopkg` or directory with `__init__.mojo`
- **Build workflow**: `mojo package kernels -o kernels.mojopkg` to compile kernel package
- **InferenceSession**: Always specify `devices=[CPU()]` or `devices=[Accelerator()]`
- **Architecture names**: HuggingFace name + `_Legacy` suffix for legacy Module architectures (registry appends `_Legacy` automatically when resolving)
- **Offline inference**: `LLM` class avoids HTTP overhead for batch jobs
- **Parameter types**: Custom ops support `bool`, `int`, `str`, `DType` parameters
- **Graph caching**: Automatic, provides ~28% faster compilation

---

## Version-Specific Features

### v26.1.0.0.0 (Stable): Custom Ops API

**DeviceRef:** Use `DeviceRef.from_device()`

```python
from max.driver import CPU, Accelerator, accelerator_count
from max.graph import DeviceRef

device = Accelerator() if accelerator_count() > 0 else CPU()
device_ref = DeviceRef.from_device(device)
```

**ops.custom:** Device is required (pass as keyword arg)

```python
ops.custom(
    name="my_kernel",
    device=device_ref,  # Required in v26.1.0.0.0 too
    values=[x],
    out_types=[TensorType(dtype=x.dtype, shape=x.tensor.shape, device=device_ref)],
)[0].tensor
```

**TensorType:** Device parameter is optional

```python
TensorType(DType.float32, shape=[batch, seq], device=device_ref)
```

**Buffer to numpy:** Use `np.array()`

```python
output = np.array(result.to(CPU()))
```

### v26.2+ (Nightly): Custom Ops API

**DeviceRef:** All three methods work (from_device is NOT deprecated)

```python
from max.graph import DeviceRef

device_ref = DeviceRef.CPU()    # For CPU
device_ref = DeviceRef.GPU()    # For GPU
device_ref = DeviceRef.GPU(1)   # Specific GPU
device_ref = DeviceRef.from_device(device)  # From driver Device — still works
```

**ops.custom:** Device is required positional argument

```python
ops.custom(
    name="my_kernel",
    device=device_ref,  # REQUIRED - before values
    values=[x],
    out_types=[TensorType(dtype=x.dtype, shape=x.tensor.shape, device=device_ref)],
)[0].tensor
```

**TensorType:** Device parameter is required

```python
TensorType(DType.float32, [batch, seq], device=device_ref)  # device required
```

**Buffer to numpy:** Use `.to_numpy()`

```python
output = result[0].to(CPU()).to_numpy()  # Transfer to CPU, then to_numpy()
```

**Kernel imports:** Use `from tensor import ...`

```mojo
# Correct (nightly)
from tensor import InputTensor, OutputTensor, foreach
from runtime.asyncrt import DeviceContextPtr

# Wrong - old imports
from max.tensor import ...  # Doesn't exist
```

### Version Comparison

| Task | Stable (v26.1.0.0.0) | Nightly (v26.2+) |
|------|----------------|------------------|
| Create DeviceRef | `DeviceRef.from_device(device)` | `DeviceRef.CPU()` / `DeviceRef.GPU()` / `DeviceRef.from_device()` (all work) |
| Call custom op | `ops.custom(name, device=..., values, out_types)` | `ops.custom(name, device, values, out_types)` |
| TensorType device | Required | Required |
| Buffer to numpy | `np.array(buf.to(CPU()))` | `buf.to(CPU()).to_numpy()` |
| Kernel imports | `from tensor import ...` | `from tensor import ...` |
| **foreach callback** | `fn[width: Int, element_alignment: Int](idx)` | `fn[width: Int](idx)` |
| `ops.neg()` | Use `ops.neg()` | Use `ops.negate()` |
| `ops.reduce_mean()` | Use `ops.reduce_mean()` | Use `ops.mean()` (different API) |
| `ops.clamp()` | N/A | N/A - use `ops.max(ops.min(x, high), low)` |
| `ops.gather()` | Yes | Yes (changed) |
| `ops.fence` | No | Yes |
| `max.nn` | Yes | Yes (changed) |
| `max.nn.moe` | No | Yes |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `no matching function in call to 'foreach'` | Version mismatch between MAX and Mojo | Run version alignment check; use `pixi shell` for consistency |
| `custom op not found` | `@compiler.register` not visible to graph | Ensure Mojo kernel is in correct path and compiled |
| `DeviceRef not found` | Wrong import | Use `from max.graph import DeviceRef` — `from_device()`, `CPU()`, `GPU()` all work |
| `TensorType device required` | Missing device argument in nightly | Add `device=DeviceRef.GPU()` to TensorType in v26.2+ |
| `architecture not registered` | Pipeline not in PIPELINE_REGISTRY | Add model to registry with `@register_arch` decorator |
| `InferenceSession creation failed` | Invalid model path or format | Verify model file exists; check supported formats in model-loading.md |

---

## Related Patterns

- [`graph-construction.md`](graph-construction.md) — Building MAX graphs
- [`multigpu-scaling.md`](multigpu-scaling.md) — Multi-GPU deployment
- [`perf-inference.md`](perf-inference.md) — Performance optimization

---

## References

- [MAX Graph API](https://docs.modular.com/max/graph)
- [MAX Pipeline Development](https://docs.modular.com/max/api/python/pipelines/)
- [Offline Inference Documentation](https://docs.modular.com/max/serve/offline-inference)
