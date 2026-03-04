---
title: Model Implementation
description: Building neural network models with MAX nn module and integrating into serving pipelines
impact: HIGH
category: model
tags: [nn, module, layers, model, pipeline, weights, forward, compile]
error_patterns:
  - "Module has no attribute 'forward'"
  - "Expected TensorValue but got Tensor"
  - "Weight not found in state dict"
  - "Shape mismatch"
  - "Cannot compile module with uninitialized parameters"
  - "no matching function in call to 'forward'"
  - "load_state_dict"
  - "KeyError"
  - "missing keys in state_dict"
  - "unexpected keys in state_dict"
scenarios:
  - "Implement a new model architecture in MAX"
  - "Port a PyTorch model to MAX"
  - "Add a custom architecture to MAX Serve"
  - "Load HuggingFace weights into a MAX model"
  - "Build a transformer model with MAX nn"
  - "Define custom layers with MAX Module"
  - "Compile a model for inference"
---

# Model Implementation

**Category:** model | **Impact:** HIGH

> **Two Module Systems:** MAX provides two distinct APIs for building models:
>
> - **`max.nn.Module`** (new eager API) — Eager-style, PyTorch-like API. Recommended for new standalone models. Uses `Tensor`, `forward()`, `@F.functional`.
> - **`max.nn.Module`** (graph API) — Graph-based API. Used by all production pipeline architectures today. Uses `TensorValue`, `__call__()`, `Weight`.
>
> This pattern covers both systems and when to use each.

Comprehensive patterns for implementing neural network models in MAX, including module definition, layer composition, weight loading, compilation, and pipeline integration for serving.

---

## Core Concepts

### Choosing the Right Module System

**Use `max.nn.Module (eager API)` (eager API) when:**

- Building a new standalone model for inference
- Porting a PyTorch model to MAX
- You want eager execution with automatic graph compilation
- You don't need integration with MAX Serve pipeline (yet)

**Use `max.nn.Module` (graph API) when:**

- Adding an architecture to MAX Serve (`max serve --custom-architectures`)
- Modifying an existing production model (Llama3, Gemma3, DeepSeek, etc.)
- You need explicit graph construction with symbolic dimensions
- Working with `PipelineModel` integration

| Aspect | Eager API (`max.nn`) | Graph API (`max.nn.legacy`) |
|--------|---------------------|----------------------------|
| **Tensor type** | `Tensor` (eager) | `TensorValue` (symbolic) |
| **Define forward** | `forward()` method | `__call__()` method |
| **Weight type** | `Tensor` attributes | `Weight` objects |
| **Compilation** | `module.compile(*input_types)` | `InferenceSession.load(graph)` |
| **Device handling** | `module.to(device)` | `DeviceRef` in constructors |
| **Production use** | New — standalone models | All production pipeline models |
| **MAX Serve** | Not yet used in pipeline models | Required for `max serve` |
| **API reference** | [`max.nn`](https://docs.modular.com/max/api/python/nn/) | [`max.experimental.nn](https://docs.modular.com/max/api/python/experimental/nn/) |

**❌ WRONG:** Mixing tensor types across APIs

```python
# nocompile - Demonstrates anti-pattern
from max.nn import Module
from max.graph import TensorValue  # Graph API type

class BadModel(Module):
    def forward(self, x: TensorValue):  # WRONG: TensorValue in nn.Module
        return x
```

**✅ CORRECT:** Match tensor types to module system

```python
from max.nn import Module
from max.tensor import Tensor

class GoodModel(Module[[Tensor], Tensor]):
    def forward(self, x: Tensor) -> Tensor:
        return x
```

**Why?** `max.nn.Module (eager API).forward()` operates on eager `Tensor` objects. `max.nn.Module.__call__()` operates on symbolic `TensorValue` objects. Mixing them causes type errors at compilation or runtime.

---

## Building Models with `max.nn` (Eager API)

### Import Reference

```python
from max.nn import Module, Linear, Embedding, Sequential, ModuleList
from max.nn import module_dataclass
from max.nn.norm import RMSNorm, GemmaRMSNorm
from max.nn.rope import RotaryEmbedding
from max.tensor import Tensor
from max import functional as F
from max.dtype import DType
from max.engine import InferenceSession
from max.driver import CPU, Accelerator
```

### Module Definition

Subclass `Module` with typed generic parameters for the forward signature:

```python
from max.nn import Module, Linear
from max.tensor import Tensor
from max import functional as F

class MLP(Module[[Tensor], Tensor]):
    """Simple feedforward network."""

    def __init__(self, dim: int, hidden_dim: int):
        self.gate = Linear(dim, hidden_dim, bias=False)
        self.up = Linear(dim, hidden_dim, bias=False)
        self.down = Linear(hidden_dim, dim, bias=False)

    @F.functional
    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

**Key points:**

- Generic `Module[[InputTypes], ReturnType]` declares the forward signature
- `@F.functional` decorator is required — enables graph tracing during compilation
- `Tensor` attributes are auto-discovered as parameters via `vars(self)`
- `Module` attributes are auto-discovered as children

### @module_dataclass

For modules with simple initialization, use `@module_dataclass` instead of `__init__`:

```python
from max.nn import Module, module_dataclass
from max.tensor import Tensor
from max import functional as F

@module_dataclass
class ScaledDotProductAttention(Module[[Tensor, Tensor, Tensor], Tensor]):
    scale: float

    @F.functional
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        attn = (q @ k.T) * self.scale
        return F.softmax(attn, axis=-1) @ v
```

### Built-in Layers

**Linear** — fully connected layer:

```python
from max.nn import Linear

linear = Linear(in_dim=768, out_dim=3072, bias=True)
# weight shape: [out_dim, in_dim] (stored transposed)
# forward: x @ weight.T + bias
```

**Embedding** — token lookup table:

```python
from max.nn import Embedding

embed = Embedding(vocab_size=32000, dim=768)
# forward: F.gather(weight, indices, axis=0)
```

**RMSNorm** — root mean square normalization:

```python
from max.nn.norm import RMSNorm

norm = RMSNorm(dim=768, eps=1e-6)
# weight initialized to ones
# Uses custom Mojo kernel via F.custom("rms_norm", ...)
```

**Sequential / ModuleList** — layer containers:

```python
from max.nn import Sequential, ModuleList, Linear

# Sequential: chains forward calls
mlp = Sequential(Linear(768, 3072), Linear(3072, 768))

# ModuleList: indexed container (no auto-chaining)
layers = ModuleList([Linear(768, 768) for _ in range(12)])
# Children named "0", "1", "2", etc.
```

**RotaryEmbedding** — positional encoding:

```python
from max.nn.rope import RotaryEmbedding, positional_embedding

rope = RotaryEmbedding(
    weight=positional_embedding(dim=64, base=500000.0, max_sequence_length=8192),
)
# weight: pre-computed [max_sequence_length, dim//2, 2] cos/sin tensor
```

### The Literal[0] Pattern for Optional Bias

```python
from typing import Literal
from max.tensor import Tensor
from max import random

class Linear(Module[[Tensor], Tensor]):
    weight: Tensor
    bias: Tensor | Literal[0]  # Not None — allows clean arithmetic

    def __init__(self, in_dim: int, out_dim: int, *, bias: bool = True):
        self.weight = random.normal([out_dim, in_dim])
        self.bias = random.normal([out_dim]) if bias else 0

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias  # Works with bias=0 or Tensor
```

**Why `Literal[0]` instead of `None`?** Using `0` avoids branching in the forward pass — `x + 0` is valid arithmetic, while `x + None` would require an `if` check.

### Weight Loading

Load pre-trained weights from a state dict:

```python
import numpy as np
from max.tensor import Tensor

model = MLP(dim=768, hidden_dim=3072)

# From dict (e.g., parsed from safetensors)
state_dict = {
    "gate.weight": Tensor.from_numpy(np.load("gate_weight.npy")),
    "up.weight": Tensor.from_numpy(np.load("up_weight.npy")),
    "down.weight": Tensor.from_numpy(np.load("down_weight.npy")),
}
model.load_state_dict(state_dict, strict=True)

# Or using a lookup function (for name remapping)
model.load_state(lambda name: load_and_remap(name))
```

**Parameter discovery:**

```python
# Iterate all parameters (depth-first, qualified names)
for name, param in model.parameters:
    print(f"{name}: shape={param.shape}, dtype={param.dtype}")
# Output: gate.weight: shape=[3072, 768], dtype=float32
#         up.weight: shape=[3072, 768], dtype=float32
#         down.weight: shape=[768, 3072], dtype=float32
```

### Compilation and Execution

Compile an eager model into an optimized inference graph:

```python
from max.nn import Module, Linear
from max.tensor import Tensor
from max.graph import TensorType
from max.driver import CPU
from max.dtype import DType
from max import functional as F

class SimpleModel(Module[[Tensor], Tensor]):
    def __init__(self):
        self.linear = Linear(10, 5, bias=False)

    @F.functional
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.linear(x))

# Create and compile
model = SimpleModel()
compiled = model.compile(
    TensorType(DType.float32, ("batch", 10), device=CPU())
)

# Execute
x = Tensor.from_numpy(np.random.randn(4, 10).astype(np.float32))
result = compiled(x)
```

**❌ WRONG:** Compiling without `@F.functional`

```python
# nocompile - Demonstrates anti-pattern
class BadModel(Module[[Tensor], Tensor]):
    def forward(self, x: Tensor) -> Tensor:  # Missing @F.functional
        return self.linear(x)

model = BadModel()
model.compile(...)  # Fails: cannot trace operations
```

**✅ CORRECT:** Always use `@F.functional` on `forward()` for compilation

```python
class GoodModel(Module[[Tensor], Tensor]):
    @F.functional
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
```

**Why?** The `@F.functional` decorator enables the graph tracer to record operations during `compile()`. Without it, eager operations aren't captured into the optimized graph.

---

## Building Models with `max.nn` (Graph API)

### Import Reference

```python
# Graph module system
from max.nn import Module
from max.nn.legacy.layer import LayerList
from max.nn.linear import Linear, MLP, ColumnParallelLinear
from max.nn.norm import RMSNorm, LayerNorm
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.transformer import Transformer, TransformerBlock

# Graph construction
from max.graph import Graph, TensorType, Weight, ops, DeviceRef
from max.graph.weights import SafetensorWeights, GGUFWeights, load_weights
from max.dtype import DType

# Engine
from max.engine import InferenceSession
```

### Graph Module Definition

Graph modules use `Weight` objects and operate on symbolic `TensorValue`:

```python
from max.nn import Module
from max.graph import Weight, TensorValue, ops, DeviceRef
from max.dtype import DType

class LegacyLinear(Module):
    def __init__(self, in_dim: int, out_dim: int, dtype: DType, device: DeviceRef):
        self.weight = Weight("weight", dtype, (in_dim, out_dim), DeviceRef.CPU())

    def __call__(self, x: TensorValue) -> TensorValue:
        return x @ self.weight
```

**Key differences from new API:**

- `__call__()` instead of `forward()`
- `Weight(name, dtype, shape, device)` instead of `Tensor` attributes
- Explicit `DeviceRef` required in constructors
- Operations use `ops.*` (graph operations), not `F.*` (functional)

### Composing a Transformer

Production models compose via `Transformer` and `TransformerBlock`:

```python
from max.nn.transformer import Transformer, TransformerBlock
from max.nn.linear import Linear, MLP
from max.nn.norm import RMSNorm
from max.nn.legacy.layer import LayerList

# Create layers using factory pattern (common in production code)
def create_transformer(config):
    layers = []
    for i in range(config.num_layers):
        attention = create_attention(config)  # Architecture-specific
        mlp = MLP(config.hidden_size, config.intermediate_size, config.dtype, config.device)
        layers.append(TransformerBlock(
            attention=attention,
            mlp=mlp,
            attention_norm=RMSNorm(config.hidden_size, config.rms_norm_eps),
            mlp_norm=RMSNorm(config.hidden_size, config.rms_norm_eps),
        ))

    return Transformer(
        dim=config.hidden_size,
        n_heads=config.num_attention_heads,
        layers=LayerList(layers),
        norm=RMSNorm(config.hidden_size, config.rms_norm_eps),
        output=Linear(...),
        embedding=Embedding(...),
        kv_params=config.get_kv_params(),
        rope=RotaryEmbedding(...),
    )
```

### Graph Construction with Graph Modules

Build a computation graph by calling graph modules with symbolic inputs:

```python
from max.graph import Graph, TensorType, DeviceRef
from max.dtype import DType
from max.engine import InferenceSession

# Define input types with symbolic dimensions
input_types = [
    TensorType(DType.int64, ("total_seq_len",), device=DeviceRef.GPU()),    # tokens
    TensorType(DType.int64, ("padded_batch_plus_1",), device=DeviceRef.GPU()),  # offsets
]

# Build graph
with Graph("my_model", input_types=input_types) as graph:
    tokens, offsets = graph.inputs
    outputs = model(tokens.tensor, offsets.tensor)  # __call__ builds graph ops
    graph.output(*outputs)

# Load weights and compile
state_dict = model.state_dict()
session = InferenceSession(devices=[Accelerator()])
compiled_model = session.load(graph, weights_registry=state_dict)
```

### Weight Loading (Graph API)

Graph API models use `load_state_dict()` with weight adapter remapping:

```python
from max.graph.weights import load_weights

# Load from file
weights = load_weights(["model.safetensors"])

# Build state dict with adapter
state_dict = {}
for name, value in weights.items():
    max_name = remap_weight_name(name)  # HF name -> MAX name
    state_dict[max_name] = value.data()

# Load into model
model.load_state_dict(state_dict, weight_alignment=1)
```

---

## Pipeline Integration

### Architecture File Structure

Every MAX Serve architecture follows this structure:

```
my_architecture/
├── __init__.py          # ARCHITECTURES = [my_arch]
├── arch.py              # SupportedArchitecture registration
├── model_config.py      # Config dataclass (ArchConfig)
├── model.py             # PipelineModel subclass
├── my_model.py          # Graph model (extends Module)
└── weight_adapters.py   # Weight name mapping functions
```

### Architecture Registration

Register your architecture in `arch.py`:

```python
from max.pipelines.lib.registry import SupportedArchitecture
from max.interfaces import PipelineTask
from max.pipelines.lib import SupportedEncoding
from max.nn.kv_cache import KVCacheStrategy
from max.graph.weights import WeightsFormat

my_arch = SupportedArchitecture(
    name="MyModelForCausalLM_Legacy",  # HF name + "_Legacy" suffix when BOTH legacy and non-legacy versions exist
    example_repo_ids=["org/MyModel-7B-Instruct"],
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16": ["paged"],
        "float32": ["paged"],
    },
    pipeline_model=MyPipelineModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=MyModelConfig,
)
```

Export in `__init__.py`:

```python
from .arch import my_arch
ARCHITECTURES = [my_arch]
```

### Config Pattern (Two-Phase Initialization)

Architecture configs use a two-phase pattern:

```python
from dataclasses import dataclass
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache

@dataclass(kw_only=True)
class MyModelConfig(ArchConfigWithKVCache):
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    num_layers: int = 32
    intermediate_size: int = 14336
    rms_norm_eps: float = 1e-5
    vocab_size: int = 32000
    rope_theta: float = 500000.0

    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> "MyModelConfig":
        """Phase 1: Read from HuggingFace config.
        Note: huggingface_config is an AutoConfig object — access fields
        via attributes (e.g., hf.hidden_size), not dict keys."""
        hf = pipeline_config.model.huggingface_config
        return cls(
            hidden_size=hf.hidden_size,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=getattr(hf, "num_key_value_heads", hf.num_attention_heads),
            num_layers=hf.num_hidden_layers,
            intermediate_size=hf.intermediate_size,
            vocab_size=hf.vocab_size,
        )

    def finalize(
        self,
        huggingface_config,
        state_dict,
        return_logits,
        return_hidden_states=...,
        norm_method="rms_norm",
        attention_bias=False,
    ):
        """Phase 2: Inspect weights for encoding-dependent config.
        Note: The actual signature requires additional parameters beyond
        huggingface_config and state_dict. See llama3/model_config.py for
        the full signature:
            finalize(self, huggingface_config: AutoConfig,
                     state_dict: dict[str, WeightData],
                     return_logits: ReturnLogits,
                     return_hidden_states: ReturnHiddenStates = ...,
                     norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm",
                     attention_bias: bool = False)
        """
        # Check for quantization, stacked weights, etc.
        pass

    def get_max_seq_len(self) -> int:
        return self._max_seq_len

    def get_kv_params(self) -> KVCacheParams:
        return KVCacheParams(...)
```

**Why two phases?** `initialize()` reads HuggingFace config JSON. `finalize()` inspects the loaded weights to detect quantization format, stacked weight patterns, or other weight-dependent configuration.

### PipelineModel Implementation

```python
from max.pipelines.lib import PipelineModel, KVCacheMixin
from max.pipelines.core import TextContext
from max.engine import InferenceSession, Model
from max.graph import Graph

class MyPipelineModel(PipelineModel[TextContext], KVCacheMixin):

    def _build_graph(self, weights, adapter) -> Graph:
        """Construct the computation graph."""
        state_dict = self._get_state_dict(weights, adapter)
        config = MyModelConfig.initialize(self.pipeline_config)
        config.finalize(state_dict)

        # Create legacy model and load weights
        model = MyGraphModel(config)
        model.load_state_dict(state_dict)

        # Build graph
        with Graph("my_model", input_types=model.input_types()) as graph:
            tokens, offsets, *kv_inputs = graph.inputs
            outputs = model(tokens.tensor, offsets.tensor, ...)
            graph.output(*outputs)

        return graph

    def load_model(self, session: InferenceSession) -> Model:
        """Compile graph into executable model."""
        graph = self._build_graph(self.weights, self.adapter)
        return session.load(graph, weights_registry=self.state_dict)

    def execute(self, model_inputs):
        """Run inference."""
        return self.model.execute(**model_inputs.to_dict())
```

### Weight Adapters

Map weight names from HuggingFace/GGUF format to MAX internal naming:

```python
# Weight name mappings: source_prefix -> max_prefix
WEIGHT_MAPPING = {
    "model.": "",                    # Remove "model." prefix
    "self_attn": "self_attn",       # Keep attention naming
    "g_idx": "perm_idx",            # GPTQ index rename
}

def convert_safetensor_state_dict(state_dict, huggingface_config, pipeline_config):
    new_state_dict = {}
    for name, value in state_dict.items():
        max_name = name
        for before, after in WEIGHT_MAPPING.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()
    return new_state_dict
```

**❌ WRONG:** Not providing a weight adapter

```python
# nocompile - Demonstrates anti-pattern
# HuggingFace names: "model.layers.0.self_attn.q_proj.weight"
# MAX expects:       "layers.0.self_attn.q_proj.weight"
# Without adapter: KeyError or silent weight misalignment
```

**✅ CORRECT:** Always provide weight adapters that map source names to MAX parameter names.

**Why?** Weight names differ between formats. HuggingFace safetensors typically prefix with `"model."`, GGUF uses different naming entirely. The adapter ensures weights load into the correct parameters.

### Serving Custom Architectures

```bash
# Serve with custom architecture
max serve \
    --model org/MyModel-7B-Instruct \
    --custom-architectures /path/to/my_architecture

# The architecture's __init__.py ARCHITECTURES list is auto-discovered
```

---

## Complete Example: Transformer Block

### New API Version

```python
from max.nn import Module, Linear, ModuleList, module_dataclass
from max.nn.norm import RMSNorm
from max.nn.rope import RotaryEmbedding
from max.tensor import Tensor
from max import functional as F
from max.dtype import DType

class Attention(Module[[Tensor], Tensor]):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.q_proj = Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = Linear(n_heads * self.head_dim, dim, bias=False)

    @F.functional
    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
        # Simplified attention (production code uses flash attention)
        attn = F.softmax(q @ k.permute(0, 1, 3, 2) / (self.head_dim ** 0.5), axis=-1)
        out = (attn @ v).reshape(bsz, seq_len, -1)
        return self.o_proj(out)


class TransformerBlock(Module[[Tensor], Tensor]):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, ff_dim: int, eps: float):
        self.self_attn = Attention(dim, n_heads, n_kv_heads)
        self.mlp = MLP(dim, ff_dim)
        self.input_layernorm = RMSNorm(dim, eps=eps)
        self.post_attention_layernorm = RMSNorm(dim, eps=eps)

    @F.functional
    def forward(self, x: Tensor) -> Tensor:
        h = x + self.self_attn(self.input_layernorm(x))
        return h + self.mlp(self.post_attention_layernorm(h))
```

---

## Decision Guide

| Scenario | Approach | Key Imports |
|----------|----------|-------------|
| New standalone model | `max.nn.Module (eager API)` + `compile()` | `from max.nn import Module, Linear` |
| Add architecture to MAX Serve | `max.nn.Module` + `PipelineModel` | `from max.nn import Module` |
| Port PyTorch model | `max.nn.Module (eager API)` (closest to PyTorch) | `from max.nn import Module` |
| Modify Llama3/Gemma3/DeepSeek | Graph API (match existing code) | `from max.nn import ...` |
| Quantized GGUF inference | Graph API + `GGUFWeights` | `from max.graph.weights import GGUFWeights` |
| Quick prototype | `max.nn.Module (eager API)` + `compile()` | `from max.nn import Module` |
| Multi-GPU model | Graph API + `ColumnParallelLinear` | `from max.nn.linear import ColumnParallelLinear` |

---

## Quick Reference

- **Eager API Module** (`max.nn`): `class MyModel(Module[[Tensor], Tensor])` with `forward()` and `@F.functional`
- **Graph API Module** (`max.nn`): `class MyModel(Module)` with `__call__()` and `Weight` objects
- **Compile new API**: `compiled = model.compile(TensorType(dtype, shape, device=CPU()))`
- **Compile legacy**: `session = InferenceSession(); model = session.load(graph, weights_registry=state_dict)`
- **Load weights**: `model.load_state_dict(state_dict)` (both APIs)
- **Parameter discovery**: `for name, param in model.parameters:` (new API)
- **Move to device**: `model.to(Accelerator())` (new API) or `DeviceRef.GPU()` in constructors (legacy)
- **Layer containers**: `Sequential` (auto-chain), `ModuleList`/`LayerList` (indexed)
- **Optional bias**: Use `Literal[0]` not `None` for clean arithmetic in forward
- **Pipeline files**: arch.py + model.py + model_config.py + weight_adapters.py

---

## Version-Specific Features

### Stable (v26.1) vs Nightly (v26.2+)

| Feature | Stable (v26.1) | Nightly (v26.2+) |
|---------|----------------|------------------|
| **`max.nn.Module (eager API)`** | Eager API — recommended for new models | Same |
| **`max.nn.Module`** | Graph API — for pipeline models | Same |
| **`driver.Buffer`** | `driver.Buffer` (renamed from `Tensor`) | Same |
| **V1 layers** | Removed (`LinearV1`, `Conv2dV1`, etc.) | Same |
| **`ops.gather()`** | `axis` parameter required (no default) | Same |
| **`foreach` callback** | `fn[width: Int, element_alignment: Int](idx)` | `fn[width: Int](idx)` |
| **DeviceRef** | `DeviceRef.GPU()` / `DeviceRef.CPU()` | Same |
| **`ops.custom()`** | `ops.custom(name, device, values, out_types)` | Same |

**Migration from pre-v26.1:**

```python
# Old (pre-v26.1)
from max.driver import Tensor as DriverTensor  # removed in v26.1+
t = DriverTensor(shape=[2, 3], dtype=DType.float32)  # removed in v26.1+

# New (v26.1+)
from max.driver import Buffer
b = Buffer(shape=[2, 3], dtype=DType.float32)
```

**Custom op foreach signature:**

Stable (v26.1):

```mojo
# nocompile
# foreach callback signature snippet (v26.1 stable)
# Stable v26.1 foreach callback
fn my_kernel[width: Int, element_alignment: Int](
    idx: IndexList[rank],
) -> SIMD[dtype, width]:
    return rebind[SIMD[dtype, width]](inp[idx])
```

Nightly (v26.2+):

```mojo
# nocompile
# Nightly v26.2+ only
fn my_kernel[width: Int](
    idx: IndexList[rank],
) -> SIMD[dtype, width]:
    return rebind[SIMD[dtype, width]](inp[idx])
```

---

## Practical Learnings (from GPT-2 Implementation)

These patterns were discovered while implementing GPT-2 (124M) from scratch using both the eager and graph APIs. They represent real issues encountered during standalone model development.

### ops.layer_norm Crashes on Apple Silicon (BLOCKER) [Temporary]

**Problem:** `ops.layer_norm` / `F.layer_norm` triggers the Metal GPU device context path even during CPU-only graph compilation on Apple Silicon, hitting `"Unimplemented at device_context.mojo:1950"`. This prevents CPU execution of any model using LayerNorm on macOS with Metal GPU. This is a backend bug that will be fixed in a future MAX release.

**Workaround — ManualLayerNorm (eager API):**

```python
from max.nn import Module
from max.tensor import Tensor
from max import functional as F

class ManualLayerNorm(Module[[Tensor], Tensor]):
    """LayerNorm using basic arithmetic ops (avoids Metal GPU dispatch)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = Tensor.ones([dim])
        self.bias = Tensor.zeros([dim])

    @F.functional
    def forward(self, x: Tensor) -> Tensor:
        mu = F.mean(x, axis=-1)
        x_centered = x - mu
        var = F.mean(x_centered * x_centered, axis=-1)
        eps_t = Tensor.constant(self.eps, dtype=x.dtype, device=x.device)
        inv_std = F.rsqrt(var + eps_t)
        normed = x_centered * inv_std
        return normed * self.weight + self.bias
```

**Workaround — ManualLayerNorm (graph API):**

```python
from max.nn import Module
from max.graph import Weight, TensorValue, ops, DeviceRef
from max.dtype import DType

class ManualLayerNorm(Module):
    def __init__(self, dim: int, dtype: DType, device: DeviceRef, eps: float = 1e-5):
        self.eps = eps
        self.weight = Weight("weight", dtype, [dim], device)
        self.bias = Weight("bias", dtype, [dim], device)

    def __call__(self, x: TensorValue) -> TensorValue:
        mu = ops.mean(x, axis=-1)
        x_centered = x - mu
        var = ops.mean(x_centered * x_centered, axis=-1)
        inv_std = ops.rsqrt(var + self.eps)
        return (x_centered * inv_std) * self.weight + self.bias
```

**Note:** All other basic ops (mean, rsqrt, mul, add, matmul, softmax) compile and execute correctly on CPU with Apple Metal present. Only `layer_norm` triggers this issue.

### CPU-Only Session on Apple Silicon [Temporary]

**Problem:** The internal `_session()` function scans all available devices including Metal GPU. On Apple Silicon, the GPU device context is initialized even when only CPU execution is desired. This will be addressed when a public API for device targeting is added.

**Workaround (both APIs):**

```python
import max._realization_context as _rc_module
import max.engine.api as _engine_api
import max.nn.module as _nn_module
from max.driver import CPU, DeviceSpec, load_devices

def _cpu_only_session() -> _engine_api.InferenceSession:
    """Returns a CPU-only InferenceSession, creating one if needed."""
    if not (session := _rc_module._SESSION.get(None)):
        device_specs = [DeviceSpec.cpu()]
        devices = load_devices(device_specs)
        _rc_module._SESSION.set(
            session := _engine_api.InferenceSession(devices=devices)
        )
    return session

# Apply BEFORE any model construction
_cpu_session = _cpu_only_session()
_rc_module._session = _cpu_only_session
_nn_module._session = _cpu_only_session
```

**Important:** This is a temporary workaround that depends on internal module structure. Apply it at module load time, before constructing any models.

### Weight Layout Mismatch When Porting from HuggingFace

**Problem:** Some HuggingFace models (GPT-2, GPT-J, GPT-Neo) use Conv1D layers that store weights as `[in_features, out_features]`, but MAX's `Linear` layer expects `[out_features, in_features]`. This applies to any model using OpenAI-style Conv1D projections. If missed, the model produces garbage output with **no error**.

```python
# Weight names that need transposition (Conv1D -> Linear format)
TRANSPOSE_WEIGHTS = {
    "attn.c_attn.weight",   # Fused QKV projection
    "attn.c_proj.weight",   # Output projection
    "mlp.c_fc.weight",      # MLP up-projection
    "mlp.c_proj.weight",    # MLP down-projection
}

def load_weights(model, safetensors_path):
    raw = load_safetensors(str(safetensors_path))
    state_dict = {}
    for name, weight in raw.items():
        # Strip prefix, remap names...
        if any(name.endswith(p) for p in TRANSPOSE_WEIGHTS):
            weight = weight.T.copy()  # Conv1D -> Linear
        state_dict[name] = weight
    model.load_state_dict(state_dict, strict=False)
```

**Tip:** Bias vectors do NOT need transposition — only weight matrices.

### Tied Weights (lm_head = embedding)

Many models (GPT-2, GPT-J, etc.) tie the output projection (`lm_head.weight`) with the input embedding (`wte.weight`). In MAX, these must be loaded as separate parameters:

```python
# Weight tying: copy embedding weights to lm_head
state_dict["lm_head.weight"] = state_dict["wte.weight"].copy()
```

### Eager vs Compiled Mode Performance

**Without `compile()`:** Each `@F.functional` sub-module creates its own compilation context per forward call. For GPT-2 124M, this means ~60s per token — impractical for inference.

**With `compile()`:** The entire model is traced and compiled as a single optimized graph. GPT-2 124M runs at ~19 tok/s on M3 Max CPU after compilation.

```python
# Compile once (takes ~20-55s), then fast inference
compiled = model.compile(
    TensorType(DType.int64, ("batch", "seq_len"), device=CPU())
)
# Each call is now fast
result = compiled(input_tensor)
```

### Missing Functional Wrappers

`F.mean` is available in the eager API (used by ManualLayerNorm above), but several other operations lack `F.*` wrappers:

| Need | Graph API (`ops.*`) | Functional (`F.*`) | Workaround |
|------|--------------------|--------------------|------------|
| Mean | `ops.mean(x, axis)` | `F.mean(x, axis)` (eager) | Available — works in `@F.functional` methods |
| Sum | `ops.sum(x, axis)` | Missing | `_F_sum = F.functional(ops.sum)` |
| Triangular mask | No `ops.tril` / `ops.triu` | No `F.tril` / `F.triu` | `F.band_part` (eager) or `F.select` + range comparison (graph) |
| Ones tensor | — | No `F.ones()` | `Tensor.ones([dim])` with `default_device()` context |

### Layer Containers (ModuleList Absence in Graph API)

In the graph API, sub-modules stored in a plain Python list are **not** discovered by parameter introspection:

```python
# WRONG: Parameters not discovered
class Model(Module):
    def __init__(self):
        self.blocks = [TransformerBlock() for _ in range(12)]

# CORRECT: Use setattr for named attributes
class Model(Module):
    def __init__(self):
        for i in range(12):
            setattr(self, f"block_{i}", TransformerBlock())

    def __call__(self, x):
        for i in range(12):
            x = getattr(self, f"block_{i}")(x)
        return x
```

**Note:** The eager API provides `Sequential` and `ModuleList` which handle this correctly. Use `Sequential` when blocks chain outputs, or `ModuleList` when you need indexed access.

### Tensor vs Buffer Output Inconsistency

- **Eager mode** (`model(tensor)`): Returns `Tensor` objects — use `np.array(tensor)` or `np.from_dlpack(tensor)`
- **Compiled mode** (`compiled.execute(buffer)`): Returns `Buffer` objects — use `result[0].to(CPU()).to_numpy()`

```python
# Eager output
logits = model(x)  # Returns Tensor
np_logits = np.from_dlpack(logits)  # NumPy 2.x compatible

# Compiled output
results = compiled.execute(buffer)  # Returns list[Buffer]
np_logits = results[0].to(CPU()).to_numpy()  # Must transfer to CPU first
```

### Manual Graph Construction (Advanced)

When `model.compile()` fails (e.g., due to GPU context issues), use manual graph construction:

```python
from max.graph import Graph, TensorType
from max.nn._realization_context import (
    GraphRealizationContext,
    realization_context,
    as_weight,  # Converts Tensor parameters to graph Weights
)
from max.tensor import Tensor
from max.driver import CPU, Buffer
from max.engine import InferenceSession
from max.dtype import DType

# 1. Create graph with input types
graph = Graph("my_model", input_types=[
    TensorType(DType.int64, ("batch", "seq_len"), device=CPU())
])

# 2. Trace model within realization context
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

> **[Temporary]** This pattern uses internal APIs (`_realization_context`, `_mapped_parameters`, `as_weight`) and may change between versions. Use `model.compile()` when possible.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Module has no attribute 'forward'` | Using graph Module (has `__call__`, not `forward`) | Use `max.nn.Module` eager API or override `__call__` for graph API |
| `Expected TensorValue but got Tensor` | Passing eager Tensor to graph Module | Use `TensorValue` from graph inputs with graph modules |
| `Weight not found in state dict` | Weight name mismatch | Check weight adapter mapping; use `strict=False` to debug |
| `missing keys in state_dict` | Weight adapter not remapping all names | Print `model.state_dict().keys()` and compare to loaded names |
| `Shape mismatch loading weights` | Transposed or wrong-dim weights | Check if weight needs `.T`; verify HF config matches |
| `Cannot compile module` | Missing `@F.functional` on `forward()` | Add `@F.functional` decorator |
| `no matching function in call to 'foreach'` | Wrong callback signature for version | Check stable vs nightly `foreach` signature |
| `KeyError` in weight loading | HF names not remapped to MAX names | Add weight adapter with proper name mapping |
| `Unimplemented at device_context.mojo:1950` | `ops.layer_norm` dispatching to Metal GPU | Use ManualLayerNorm workaround (see above) |
| `LookupError: '_CONTEXT'` | Using `Graph()` context manager without `realization_context` | Use `realization_context(GraphRealizationContext(graph))` |
| Garbage output, no error | Conv1D weight transposition missed | Transpose `[in, out]` weights to `[out, in]` for Linear |
| ~60s per token | Eager mode without `compile()` | Use `model.compile(input_types)` for compiled inference |

---

## Related Patterns

- [`graph-construction.md`](graph-construction.md) — Graph building, symbolic dimensions, TensorType specs
- [`engine-operations.md`](engine-operations.md) — Custom Mojo ops, architecture registration, inference sessions
- [`model-loading.md`](model-loading.md) — Supported architectures, quantization formats, HuggingFace tokens
- [`engine-weights.md`](engine-weights.md) — Weight formats, adapters, quantization encoding
- [`engine-lora.md`](engine-lora.md) — LoRA adapter serving for fine-tuned models

---

## References

- [MAX Develop Guide](https://docs.modular.com/max/develop/) — Official developer documentation
- [MAX nn API Reference](https://docs.modular.com/max/api/python/nn/) — nn module API docs
- [MAX Graph API Reference](https://docs.modular.com/max/api/python/graph/) — Graph construction API
- [MAX Custom Architectures](https://docs.modular.com/max/develop/custom-ops/) — Custom ops and architectures guide
- [MAX GitHub](https://github.com/modular/modular/tree/main/max) — Source code reference
