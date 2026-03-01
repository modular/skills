---
title: MAX Graph Construction
description: Graph building patterns including lazy context, modules, symbolic dimensions, and pipeline models
impact: HIGH
category: graph
tags: [graph, construction, modules, symbolic-dims, pipeline]
error_patterns:
  - "Graph"
  - "TensorType"
  - "symbolic dimension"
  - "shape mismatch"
  - "compilation"
  - "module"
scenarios:
  - "Build MAX Graph from scratch"
  - "Use symbolic dimensions"
  - "Create reusable modules"
  - "Implement pipeline model"
  - "Fix graph compilation errors"
  - "Configure lazy context"
consolidates:
  - graph-construction.md
  - graph-lazy-context.md
  - graph-modules.md
  - graph-symbolic-dims.md
  - graph-pipeline-model.md
---

# MAX Graph Construction

**Category:** graph | **Impact:** HIGH

Comprehensive patterns for MAX Graph construction including proper type specifications, lazy context for efficient compilation, reusable modules, symbolic dimensions for dynamic shapes, and pipeline model implementation.

---

## Core Concepts

### Complete Import Reference

**Import Reference (all versions):**
```python
# Graph construction
from max.graph import Graph, TensorType, ops, DeviceRef
from max.graph import Dim, SymbolicDim  # Symbolic dimensions
from max.graph import Weight, ShardingStrategy  # Weight management
from max.dtype import DType

# Neural network modules (new API)
from max.nn import Module, Linear, Embedding, Sequential, ModuleList, module_dataclass
from max.nn import RMSNorm, GemmaRMSNorm, GroupNorm  # Exported from max.nn directly
from max.nn import RotaryEmbedding, TransposedRotaryEmbedding
from max.nn import Conv2d
from max.tensor import Tensor
from max import functional as F  # F.lazy() context, @F.functional decorator

# Graph modules (for pipeline architectures)
from max.nn import Module, LayerList
from max.nn.legacy.linear import Linear, MLP, ColumnParallelLinear
from max.nn.norm import RMSNorm

# Engine and execution
from max.engine import InferenceSession

# Pipeline development
from max.pipelines.lib.interfaces import PipelineModel, KVCacheMixin
from max.pipelines.lib.registry import SupportedArchitecture, PIPELINE_REGISTRY

# Device handling
from max.driver import Device, DeviceSpec, CPU, Accelerator, Buffer, accelerator_count
```

**Notes:**
- `Dim` is stable and available in all versions
- Strings can also be used for symbolic dimensions (implicitly converted to `SymbolicDim`)
- `TensorType` requires a `device` parameter (has been required for ~1 year)
- `TensorType` accepts `Device | DeviceRef` for the `device` param (internally calls `DeviceRef.from_device()`)
- `graph.verify()` is called automatically during compilation - no need to call directly
- `DeviceRef.from_device()` is NOT deprecated — it works across all versions

### Graph Type Specifications

Specify input types explicitly when constructing graphs. The `Graph` constructor takes a `name` string and `input_types` list.

**Pattern (using `with` statement - most common):**

```python
from max.graph import Graph, TensorType, ops, DeviceRef
from max.dtype import DType

device = DeviceRef.GPU()

# Graph constructor: Graph(name, input_types=[...])
with Graph(
    "my_graph",
    input_types=[
        TensorType(DType.float32, ("batch", 128), device=device),
    ],
) as graph:
    # Access inputs via graph.inputs property
    x = graph.inputs[0]

    # Add operations
    result = ops.relu(x.tensor @ weight)

    # Define outputs
    graph.output(result)
```

**Pattern (using `forward` callable):**

```python
from max.graph import Graph, TensorType, ops, DeviceRef
from max.dtype import DType

def forward(x):
    return ops.relu(x.tensor)

# Forward is called automatically with graph.inputs as args
graph = Graph(
    "my_graph",
    forward=forward,
    input_types=[TensorType(DType.float32, ("batch", 128), device=DeviceRef.GPU())],
)
```

**Don't:**
```python
# WRONG: No name, passing TensorType as positional arg
graph = Graph(TensorType(DType.float32, "batch", 128))

# WRONG: graph.input() method does not exist
x = graph.input(input_type)

# WRONG: indexing graph directly
x = graph[0]
```

### Symbolic Dimensions

Use named dimensions for dynamic shapes that must match across tensors.

**Pattern (strings - simplest):**

```python
from max.graph import Graph, TensorType, DeviceRef
from max.dtype import DType

device = DeviceRef.GPU()

# Strings are implicitly converted to SymbolicDim
# Same string "batch" enforces matching across inputs
with Graph(
    "my_model",
    input_types=[
        TensorType(DType.float32, ("batch", 128), device=device),  # [batch, 128]
        TensorType(DType.float32, ("batch", 256), device=device),  # [batch, 256] - same batch!
    ],
) as graph:
    x, y = graph.inputs
```

**Pattern (Dim - for expressions):**

```python
from max.graph import Graph, TensorType, Dim, DeviceRef
from max.dtype import DType

batch = Dim("batch")
device = DeviceRef.GPU()

# Named dims that must match
with Graph(
    "my_model",
    input_types=[
        TensorType(DType.float32, (batch, 128), device=device),  # [batch, 128]
        TensorType(DType.float32, (batch, 256), device=device),  # [batch, 256] - same batch!
    ],
) as graph:
    x, y = graph.inputs

# Algebraic expressions - use Dim when you need math
seq_len = Dim("seq")
padded = seq_len + 4  # Compile-time expression
```

**With Device Reference:**

```python
from max.graph import Graph, TensorType, DeviceRef, Dim
from max.dtype import DType

# Create DeviceRef - required for TensorType
device_ref = DeviceRef.GPU()           # Default GPU
# device_ref = DeviceRef.CPU()         # CPU device

batch = Dim("batch")
seq_len = Dim("seq_len")

# TensorType with required device parameter
input_type = TensorType(
    dtype=DType.float32,
    shape=[batch, seq_len, 768],
    device=device_ref
)

with Graph(
    "transformer",
    input_types=[
        TensorType(DType.float32, [batch, 128], device=device_ref),
        TensorType(DType.float32, [batch, 256], device=device_ref),
    ],
) as graph:
    x, y = graph.inputs
```

**Don't:**
```python
# WRONG: Passing TensorType as positional args to Graph
graph = Graph(
    TensorType(DType.float32, ("batch", 128), device=device),
    TensorType(DType.float32, ("batch", 256), device=device),
)

# WRONG: Using graph.input() method
x = graph.input(input_type)
```

**Note:** Algebraic expressions simplify canonically: `-x - 4 == -(x + 4)`.

---

## Common Patterns

### Lazy Context for Graph Building

**When:** Building complex neural network graphs, implementing custom model architectures, optimizing memory allocation during model loading.

Use `F.lazy()` context manager to defer tensor creation during model construction.

**Pattern:**
```python
from max import functional as F
from max.nn import Module

class GptOssModel(PipelineModel):
    def load_model(self):
        # Build input types
        tokens_type = TensorType(DType.int64, shape=["total_seq_len"], device=device_ref)

        # Lazy context defers tensor realization
        with F.lazy():
            nn_model = GptOss(model_config, self.kv_manager)
            nn_model.to(self.devices[0])

        # Compile with concrete types
        compiled_model = nn_model.compile(
            tokens_type,
            return_n_logits_type,
            input_row_offsets_type,
            *kv_types,
            weights=state_dict,
        )
        return compiled_model
```

**Don't:**
```python
# Tensors allocated immediately (may not be optimal for compilation)
model = MyModel()
model.to(device)  # Immediate transfer
```

**Correct:**
```python
# Tensors deferred until compilation
with F.lazy():
    model = MyModel()
    model.to(device)  # Recorded, not executed

# Compile triggers optimized allocation
compiled = model.compile(input_types, weights=weights)
```

**How it works:**
1. Within `F.lazy()`, tensor operations are recorded, not executed
2. Model structure is captured as a graph
3. `compile()` triggers optimization and allocation
4. Weights can be loaded externally via `weights=` parameter
5. Parameters automatically become graph weights

---

### Module-Based Architecture

**When:** Custom model architectures, reusable layer libraries, weight management.

Use `max.nn.Module` base class for building reusable neural network layers.

**Pattern:**
```python
from max.nn import Module, Linear, Sequential

class MLP(Module):
    def __init__(self, hidden_dim: int, output_dim: int):
        self.fc1 = Linear(hidden_dim)
        self.fc2 = Linear(output_dim)

    def forward(self, x):
        return self.fc2(ops.gelu(self.fc1(x)))

# Use with state_dict for weight loading
mlp = MLP(512, 256)
mlp.load_state_dict(weights)
```

**Don't:**
```python
# Ad-hoc layer construction
def create_mlp(x, w1, w2):
    return ops.gelu(x @ w1) @ w2
```

**Module Compilation Pattern:**
```python
from max.nn import Module, module_dataclass
from max.tensor import Tensor
from max.graph import TensorType

@module_dataclass
class Linear(Module):
    weight: Tensor
    bias: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias

# Create module with initial weights
linear = Linear(
    weight=Tensor.zeros([10, 5]),
    bias=Tensor.zeros([10])
)

# Compile with input type specification
input_type = TensorType(DType.float32, [3, 5], device=device)
compiled = linear.compile(input_type)

# Execute compiled model
result = compiled(Tensor.ones([3, 5]))
```

**Note:** `Module` ensures unique weight names and supports `state_dict()`.

---

### PipelineModel Implementation

**When:** Implementing new model architectures for MAX Serve, adding custom models, building production inference pipelines.

Use the `PipelineModel` base class with `KVCacheMixin` for text generation models.

**Pattern:**
```python
from max.pipelines.lib import PipelineModel, KVCacheMixin, PipelineConfig
from max.engine import InferenceSession

class MyModel(PipelineModel[TextContext], KVCacheMixin):
    """PipelineModel constructor receives: pipeline_config, session,
    devices, kv_cache_config, weights, adapter, return_logits,
    return_hidden_states."""

    @classmethod
    def calculate_max_seq_len(cls, pipeline_config, huggingface_config) -> int:
        """Calculate maximum sequence length. (abstract, must implement)"""
        return pipeline_config.model.max_length or huggingface_config.max_position_embeddings

    @classmethod
    def get_kv_params(cls, huggingface_config, pipeline_config, devices, kv_cache_config, cache_dtype) -> KVCacheParams:
        """Configure KV cache parameters. (from KVCacheMixin)"""
        return MyConfig.construct_kv_params(...)

    def load_model(self) -> Callable[..., Any]:
        """Build and compile the model graph. (abstract, must implement)"""
        # self.device_refs is set automatically from self.devices
        device_ref = self.device_refs[0]
        tokens_type = TensorType(DType.int64, shape=["total_seq_len"], device=device_ref)

        with F.lazy():
            nn_model = MyNN(model_config, self.kv_manager)
            nn_model.to(self.devices[0])

        compiled_model = nn_model.compile(
            tokens_type,
            return_n_logits_type,
            input_row_offsets_type,
            *kv_types,
            weights=state_dict,
        )
        return compiled_model

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute model and return outputs. (abstract, must implement)"""
        model_outputs = self.model(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            model_inputs.input_row_offsets,
            *model_inputs.kv_cache_inputs,
        )
        return ModelOutputs(logits=model_outputs[0].driver_tensor)
```

**Don't:**
```python
# Not compatible with MAX Serve, no KV cache management
class MyModel:
    def generate(self, prompt):
        return self.model(prompt)
```

**Required Abstract Methods:**

| Method | Purpose |
|--------|---------|
| `load_model()` | Build and compile the model graph (from `ComponentModel`) |
| `calculate_max_seq_len()` | Determine max context length (classmethod) |
| `execute()` | Run inference |
| `prepare_initial_token_inputs()` | Prepare input buffers for prefill |
| `prepare_next_token_inputs()` | Prepare input buffers for decode |

---

### Architecture Registration

**When:** Adding models to MAX Serve, enabling quantization support.

```python
from max.pipelines.lib.registry import SupportedArchitecture, PIPELINE_REGISTRY

my_architecture = SupportedArchitecture(
    name="MyModelForCausalLM",  # Must match HuggingFace architectures
    example_repo_ids=["your-org/your-model"],
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16": ["paged"],
        "q4_k": ["paged"],
    },
    pipeline_model=MyModel,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    rope_type=RopeType.none,
    weight_adapters={
        WeightsFormat.safetensors: convert_safetensor_state_dict,
    },
    multi_gpu_supported=True,
    task=PipelineTask.TEXT_GENERATION,
    context_type=TextContext,
)

PIPELINE_REGISTRY.register(my_architecture)
```

**Pipeline Factory Usage:**
```python
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig, PipelineTask

# Retrieve tokenizer and pipeline factory
tokenizer, pipeline_factory = PIPELINE_REGISTRY.retrieve_factory(
    pipeline_config,
    task=PipelineTask.TEXT_GENERATION,
)

# Create pipeline (lazy construction)
pipeline = pipeline_factory()

# Generate text
response = pipeline.generate(
    prompt="Hello, world!",
    max_tokens=100,
)
```

---

### Causal Mask Construction

**When:** Building decoder-only transformers (GPT-2, Llama, etc.) that need causal attention masks. MAX does not provide `F.tril` or `F.triu`.

**Pattern (Eager API — using F.band_part):**

```python
from max import functional as F
from max.tensor import Tensor
from max.graph import Dim, DimLike
from max.dtype import DType

@F.functional
def causal_mask(seq_len: DimLike, *, dtype: DType | None = None, device=None) -> Tensor:
    """Causal mask: 0 for attend, -inf for mask (upper triangle)."""
    n = Dim(seq_len)
    mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)
    mask = F.broadcast_to(mask, shape=(n, n))
    return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)
```

**Pattern (Graph API — using range comparison):**

```python
from max import functional as F
from max.graph import ops
from max.dtype import DType

def causal_mask(seq_len, dtype=DType.float32, device=None):
    """Causal mask using range comparison (works with symbolic dims)."""
    rows = F.arange(0, seq_len, step=1, out_dim=seq_len, dtype=DType.int64, device=device)
    cols = F.arange(0, seq_len, step=1, out_dim=seq_len, dtype=DType.int64, device=device)
    # Compare: mask is True where cols > rows (upper triangle)
    is_masked = F.greater(cols.unsqueeze(0), rows.unsqueeze(1))
    # Use ops.where to avoid NaN from 0 * -inf:
    # where masked -> -inf, where not masked -> 0.0
    zero = Tensor.constant(0.0, dtype=dtype, device=device)
    neg_inf = Tensor.constant(float("-inf"), dtype=dtype, device=device)
    return ops.where(is_masked, neg_inf, zero)
```

**Note:** `F.broadcast_to` does NOT support dynamic/symbolic dimensions, so the mask cannot be explicitly broadcast to `[n_heads, seq, seq]`. Instead, rely on implicit broadcasting when adding the mask to attention weights.

---

### Manual Graph Construction with realization_context

**When:** `model.compile()` fails (e.g., GPU context issues on Apple Silicon) or you need fine-grained control over graph construction with Module tracing.

**Pattern:**

```python
from max.graph import Graph, TensorType
from max._realization_context import (
    GraphRealizationContext,
    realization_context,
    as_weight,  # Converts Tensor parameters to graph Weights
)
from max.tensor import Tensor
from max.driver import CPU, Buffer
from max.engine import InferenceSession
from max.dtype import DType

# 1. Create graph with explicit input types
graph = Graph("my_model", input_types=[
    TensorType(DType.int64, ("batch", "seq_len"), device=CPU())
])

# 2. Set up realization context for Module tracing
with realization_context(GraphRealizationContext(graph)) as ctx, ctx:
    sym_input = Tensor.from_graph_value(graph.inputs[0])
    # _mapped_parameters converts eager Tensor weights to graph Weight nodes
    with model._mapped_parameters(as_weight):
        outputs = model(sym_input)
    graph.output(outputs)

# 3. Build weights registry from model parameters
cpu_weights = {name: param for name, param in model.parameters}

# 4. Compile and execute
session = InferenceSession(devices=[CPU()])
compiled = session.load(graph, weights_registry=cpu_weights)
result = compiled.execute(Buffer.from_numpy(input_np))
```

**Key differences from `with Graph(...) as graph:`:**

| Pattern | Module Tracing | ContextVar Setup | Use Case |
|---------|---------------|-----------------|----------|
| `with Graph(...) as graph:` | Not supported | Not set up | Simple op-level graph construction |
| `realization_context(GraphRealizationContext(graph))` | Supported | Sets `_CONTEXT` ContextVar | Module-based model tracing |

**Common error:** Using `with Graph(...) as graph:` with `Tensor.from_graph_value()` raises `LookupError: ContextVar '_CONTEXT'`. Use `realization_context` instead.

> **[Temporary]** This pattern uses internal APIs (`_realization_context`, `_mapped_parameters`, `as_weight`) and may change between versions. Prefer `model.compile()` when possible.

---

### F.arange with Symbolic Dimensions

`F.arange` requires an explicit `out_dim` parameter when used with symbolic dimensions (unlike PyTorch's `torch.arange` which infers output size):

```python
# With symbolic seq_len, out_dim is required
positions = F.arange(
    0, seq_len, step=1,
    out_dim=seq_len,     # Must match symbolic dimension
    dtype=DType.int64,
    device=device
)
```

Without `out_dim`, the graph compiler cannot infer the output shape and compilation fails.

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Dynamic batch size | Use `"batch"` string or `Dim("batch")` in shape | — |
| Variable sequence length | Use `"seq"` string or `Dim("seq")` in shape | — |
| Complex model architecture | Use `F.lazy()` context | — |
| Reusable layers | Extend `Module` base class | — |
| MAX Serve integration | Implement `PipelineModel` | [`engine-operations.md`](engine-operations.md) |
| Weight management | Use `state_dict()` / `load_state_dict()` | — |

---

## Quick Reference

- **Graph constructor**: `Graph(name="...", input_types=[TensorType(...), ...])` — always provide a name
- **Accessing inputs**: Use `graph.inputs` property (returns `Sequence[Value]`), NOT `graph.input()` or `graph[0]`
- **Input types**: Always specify explicitly with `TensorType` including `device` parameter
- **Symbolic dims**: Use strings `("batch", 128)` or `Dim("batch")` for dynamic dimensions
- **Lazy context**: Use `F.lazy()` for deferred tensor creation
- **Modules**: Extend `Module` for reusable layers with automatic weight management
- **Pipeline models**: Implement `calculate_max_seq_len()`, `execute()`, `prepare_initial_token_inputs()`, `prepare_next_token_inputs()`
- **Verification**: `graph.verify()` is called automatically - no need to call directly

---

## Best Practices

- Always use `F.lazy()` for model construction in pipelines
- Specify input types for proper graph tracing
- Use `@module_dataclass` for type-safe module definitions
- Load weights after compilation for memory efficiency
- Architecture name must match HuggingFace config `architectures` field
- Use ragged tensors for variable-length batching (avoids padding)

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `shape mismatch in graph` | Tensor dimensions don't align | Use symbolic dims consistently; check broadcasting rules |
| `TensorType device required` | Missing device parameter | Add `device=DeviceRef.GPU()` parameter to TensorType |
| `output already set` | Calling `graph.output()` multiple times | Only call `graph.output()` once per graph |
| `circular dependency in graph` | Op depends on its own output | Check graph topology; use separate ops for feedback |
| `graph.input is not callable` | Using `graph.input(type)` (wrong API) | Use `graph.inputs` property: `x = graph.inputs[0]` |
| `Graph() missing name` | Passing TensorType as first arg | Use `Graph("name", input_types=[...])` |
| `LookupError: '_CONTEXT'` | Using `Graph()` context with `Tensor.from_graph_value()` | Use `realization_context(GraphRealizationContext(graph))` |
| `F.arange` shape inference fails | Missing `out_dim` with symbolic dims | Add `out_dim=symbolic_dim` parameter |
| No `F.tril` / `F.triu` | MAX doesn't provide triangular ops | Use `F.band_part` or range comparison pattern |

---

## Version-Specific Features

### Stable APIs (v26.1+)

The following APIs are stable and consistent across versions:

| Feature | Usage |
|---------|-------|
| **Symbolic dims** | Use strings `("batch", 128)` or `Dim("batch")` |
| **Dim** | Required for expressions: `batch + 1`, `seq * 2` |
| **TensorType device** | Required parameter: `device=DeviceRef.GPU()` |
| **graph.verify()** | Called automatically during compilation - no need to call directly |
| **F.lazy()** | Stable context manager for deferred tensor creation |
| **Module/Sequential** | Stable patterns for neural network layers |

**Standard Pattern (strings - simplest):**
```python
from max.graph import Graph, TensorType, ops, DeviceRef
from max.dtype import DType

device_ref = DeviceRef.GPU()

# Strings implicitly convert to SymbolicDim
with Graph(
    "my_model",
    input_types=[
        TensorType(DType.float32, ("batch", "seq", 768), device=device_ref),
    ],
) as graph:
    x = graph.inputs[0]
```

**Standard Pattern (Dim - for expressions):**
```python
from max.graph import Graph, TensorType, ops, Dim, DeviceRef
from max.dtype import DType

batch = Dim("batch")
seq = Dim("seq")
device_ref = DeviceRef.GPU()

# Use Dim when you need math: seq + 1, batch * 2
with Graph(
    "my_model",
    input_types=[
        TensorType(DType.float32, (batch, seq, 768), device=device_ref),
    ],
) as graph:
    x = graph.inputs[0]
```

**Notes:**
- `Graph(name, input_types=[...])` is the constructor signature; access inputs via `graph.inputs`
- Strings or `Dim` can be used for symbolic dimensions (strings implicitly convert)
- `Dim` is required when you need algebraic expressions (`batch + 1`)
- `TensorType` requires a `device` parameter (has been required for ~1 year)
- `graph.verify()` is called automatically - calling it directly is unnecessary
- Lazy context (`F.lazy()`) is stable across versions
- Module and Sequential patterns are stable across versions

---

## Related Patterns

- [`engine-operations.md`](engine-operations.md) — Custom operations and architecture registration
- [`model-loading.md`](model-loading.md) — Supported architectures and weight formats
- [`perf-inference.md`](perf-inference.md) — Performance optimization

---

## References

- [MAX Graph API](https://docs.modular.com/max/graph)
- [MAX Pipeline Development](https://docs.modular.com/max/api/python/pipelines/)
