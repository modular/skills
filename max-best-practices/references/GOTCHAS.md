# MAX - Common Mistakes & Gotchas

Quick reference for common MAX pitfalls. Each entry shows the wrong approach, the correct approach, and explains why.

**See also:** [ERROR_INDEX.md](ERROR_INDEX.md) for error message lookup, pattern files for full context.

---

## Table of Contents

- [Version Alignment](#version-alignment)
- [MAX Serve Configuration](#max-serve-configuration)
- [KV Cache](#kv-cache)
- [Multi-GPU](#multi-gpu)
- [Graph API](#graph-api)
- [Model Implementation](#model-implementation)
- [Custom Operations](#custom-operations)

---

## Version Alignment

### Mojo/MAX Version Mismatch

**❌ WRONG:** Mixing nightly Mojo with stable MAX (or vice versa)
```bash
# nocompile - Demonstrates anti-pattern
$ mojo --version
mojo 26.2.0.dev2026020305 (nightly)

$ pip show max | grep Version
Version: 26.1.0.0  # MISMATCH - stable!
```

**✅ CORRECT:** Ensure versions align
```bash
# Both should be same channel (stable or nightly)
$ mojo --version
mojo 26.1.0.0 (stable)

$ pip show max | grep Version
Version: 26.1.0.0  # MATCH
```

**Why?** Kernel compilation fails with cryptic errors when versions don't match. The `foreach` callback signature changed between versions.

**Symptoms:**
```
error: no matching function in call to 'foreach'
note: callee parameter 'func' has 'fn[width: Int, element_alignment: Int]...' type,
      but value has type 'fn[width: Int]...'
```

---

## MAX Serve Configuration

### Wrong Batch Size Semantics

**❌ WRONG:** Assuming batch size is aggregate (pre-26.2)
```bash
# nocompile - Old behavior
max serve --max-batch-size=256  # Thought this was total across GPUs
```

**✅ CORRECT:** Understand per-replica semantics (26.2+)
```bash
# 26.2+: batch size is PER REPLICA with data parallel
# 4 GPUs with batch-size=64 = 64 per GPU, 256 aggregate
max serve --max-batch-size=64 --devices gpu:0,1,2,3
```

**Why?** Starting in 26.2, batch size is per-replica when using data parallelism. Plan GPU memory accordingly.

---

### Missing KV Cache Configuration

**❌ WRONG:** Default KV cache settings
```bash
max serve --model meta-llama/Llama-3.1-8B
# Uses default cache settings - may OOM or underutilize memory
```

**✅ CORRECT:** Explicit KV cache configuration
```bash
max serve --model meta-llama/Llama-3.1-8B \
  --cache-strategy paged \
  --kv-cache-page-size 256 \
  --max-batch-total-tokens 32768
```

**Why?** Default settings don't optimize for your hardware. Explicit configuration prevents OOM and maximizes throughput.

---

### Deprecated CLI Flags

**❌ WRONG:** Using deprecated flags
```bash
# nocompile - Deprecated in 26.2
max serve --max-ce-batch-size=128 \
          --prefill_chunk_size=4096 \
          --max-batch-context-length=8192
```

**✅ CORRECT:** Use current flags (26.2+)
```bash
max serve --max-batch-size=128 \
          --max-batch-input-tokens=4096 \
          --max-batch-total-tokens=8192
```

**Why?** CLI flags changed in 26.2. Old flags may be ignored or cause errors.

| Old Flag (26.1) | New Flag (26.2+) |
|-----------------|------------------|
| `--max-ce-batch-size` | `--max-batch-size` |
| `--prefill_chunk_size` | `--max-batch-input-tokens` |
| `--max-batch-context-length` | `--max-batch-total-tokens` |

---

## KV Cache

### KV Cache Page Size Not Multiple of 128

**❌ WRONG:** Arbitrary page size
```bash
max serve --kv-cache-page-size=100  # Not aligned!
```

**✅ CORRECT:** Use multiple of 128
```bash
max serve --kv-cache-page-size=256  # Aligned for tensor cores
```

**Why?** Page size must be multiple of 128 for efficient tensor core utilization. Misaligned sizes waste memory and compute.

---

### Missing Prefix Caching for Shared Prompts

**❌ WRONG:** No prefix caching with repeated system prompts
```bash
max serve --model meta-llama/Llama-3.1-8B
# Each request recomputes the same system prompt
```

**✅ CORRECT:** Enable prefix caching
```bash
max serve --model meta-llama/Llama-3.1-8B \
  --enable-prefix-caching
```

**Why?** If many requests share the same prefix (system prompt), prefix caching avoids redundant computation.

---

## Multi-GPU

### Wrong Device Specification

**❌ WRONG:** Using device count instead of device list
```bash
# nocompile - Wrong format
max serve --devices=4  # Not valid
```

**✅ CORRECT:** Specify device list
```bash
max serve --devices gpu:0,1,2,3
# Or for specific GPUs:
max serve --devices gpu:0,2  # Use GPU 0 and 2 only
```

**Why?** `--devices` expects a comma-separated list of device identifiers, not a count.

---

### Tensor Parallel Without Enough GPUs

**❌ WRONG:** Using fabricated `--tensor-parallel` flag
```bash
# nocompile - --tensor-parallel does NOT exist
max serve --model meta-llama/Llama-3.1-70B \
  --tensor-parallel=4  # Error: unrecognized flag
```

**✅ CORRECT:** Use `--devices` to specify GPUs (tensor parallelism is automatic)
```bash
# Check available GPUs first
nvidia-smi -L  # Shows 2 GPUs

# TP degree = number of GPUs listed
max serve --model meta-llama/Llama-3.1-70B \
  --devices gpu:0,1  # 2-way tensor parallelism
```

**Why?** There is no `--tensor-parallel` flag. Tensor parallelism degree is determined by the number of GPUs in `--devices`. Use `--data-parallel-degree` for data parallelism.

---

## Graph API

### Missing Device in TensorType (26.2+)

**❌ WRONG:** TensorType without device (breaks in 26.2+)
```python
# nocompile - Missing device parameter
from max.graph import Graph, TensorType

graph = Graph()
x = graph.input(TensorType(dtype=DType.float32, shape=[1, 512]))
```

**✅ CORRECT:** Include device parameter
```python
from max.graph import Graph, TensorType, DeviceRef

graph = Graph()
x = graph.input(TensorType(
    dtype=DType.float32,
    shape=[1, 512],
    device=DeviceRef.GPU()  # Required in 26.2+
))
```

**Why?** Starting in 26.2, TensorType requires explicit device specification. This enables multi-device graphs.

---

### DeviceRef Construction (26.2+)

Both `DeviceRef.from_device()` and factory methods work:
```python
# Factory methods (convenient for new code)
cpu = DeviceRef.CPU()
gpu = DeviceRef.GPU()
gpu1 = DeviceRef.GPU(1)  # Specific GPU

# from_device() also works (used in PipelineModel.__init__ and throughout codebase)
device = DeviceRef.from_device(my_device)
```

**Note:** `DeviceRef.from_device()` is NOT deprecated — it is used extensively in production code including `PipelineModel.__init__`. Factory methods are convenient shortcuts for new code.

---

### Forgetting `graph.output()`

**❌ WRONG:** No output specification
```python
# nocompile - Graph has no outputs
graph = Graph()
x = graph.input(TensorType(...))
y = ops.relu(x)
# Missing: graph.output(y)
```

**✅ CORRECT:** Specify outputs
```python
graph = Graph()
x = graph.input(TensorType(...))
y = ops.relu(x)
graph.output(y)  # Mark as output
```

**Why?** Graphs without outputs are invalid. The compiler doesn't know what to compute.

---

## Model Implementation

### ops.layer_norm Crashes on Apple Silicon

**❌ WRONG:** Using `ops.layer_norm` or `F.layer_norm` on Apple Silicon
```python
# nocompile - Crashes on macOS with Metal GPU
from max.nn.norm import LayerNorm

norm = LayerNorm(768)  # Triggers Metal GPU dispatch
# Error: Unimplemented at device_context.mojo:1950
```

**✅ CORRECT:** Use ManualLayerNorm with basic arithmetic ops (`F.mean` + `F.rsqrt` + arithmetic)

```python
# Abbreviated — see model-implementation.md for full implementation
class ManualLayerNorm(Module[[Tensor], Tensor]):
    @F.functional
    def forward(self, x: Tensor) -> Tensor:
        mu = F.mean(x, axis=-1)
        var = F.mean((x - mu) * (x - mu), axis=-1)
        return ((x - mu) * F.rsqrt(var + self.eps)) * self.weight + self.bias
```

See [model-implementation.md](../patterns/model-implementation.md#opslayer_norm-crashes-on-apple-silicon-blocker-temporary) for full eager and graph API implementations.

**Why?** `ops.layer_norm` dispatches to the Metal GPU device context path even during CPU-only compilation. All other basic ops work correctly. **[Temporary]** — will be fixed in a future MAX release.

---

### Conv1D Weight Transposition (GPT-2 / OpenAI Models)

**❌ WRONG:** Loading GPT-2 weights without transposition
```python
# nocompile - Silent correctness bug
state_dict["attn.c_attn.weight"] = raw_weights["attn.c_attn.weight"]
# Model produces garbage output — no error raised
```

**✅ CORRECT:** Transpose Conv1D weights for Linear layers
```python
if name.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
    weight = weight.T.copy()  # [in, out] -> [out, in]
state_dict[name] = weight
```

**Why?** GPT-2 Conv1D stores weights as `[in_features, out_features]`, but MAX Linear expects `[out_features, in_features]`. The mismatch produces wrong outputs with no error.

---

### Graph API: ModuleList Not Available

**❌ WRONG:** Storing sub-modules in a plain list (graph API)
```python
# nocompile - Parameters not discovered
class Model(Module):
    def __init__(self):
        self.blocks = [Block() for _ in range(12)]
```

**✅ CORRECT:** Use setattr for named attributes
```python
class Model(Module):
    def __init__(self):
        for i in range(12):
            setattr(self, f"block_{i}", Block())

    def __call__(self, x):
        for i in range(12):
            x = getattr(self, f"block_{i}")(x)
        return x
```

**Why?** Graph API Module parameter discovery uses attribute introspection. List elements are not iterated. Use `Sequential` or `ModuleList` in the eager API instead.

---

### F.arange Requires out_dim with Symbolic Shapes

**❌ WRONG:** F.arange without out_dim for dynamic shapes
```python
# nocompile - Graph compiler can't infer output shape
positions = F.arange(0, seq_len, step=1)  # Fails with symbolic seq_len
```

**✅ CORRECT:** Provide out_dim matching the symbolic dimension
```python
positions = F.arange(0, seq_len, step=1, out_dim=seq_len, dtype=DType.int64, device=device)
```

**Why?** With symbolic dimensions, the graph compiler cannot infer the output shape from the inputs alone. `out_dim` explicitly declares the symbolic output size.

---

### No F.tril/F.triu for Causal Masks

**❌ WRONG:** Expecting PyTorch-like tril/triu
```python
# nocompile - F.tril does not exist
mask = F.tril(F.ones(seq_len, seq_len))
```

**✅ CORRECT:** Build causal mask using range comparison or F.band_part
```python
# Option 1: F.band_part (eager API)
mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)
mask = F.broadcast_to(mask, shape=(seq_len, seq_len))
mask = F.band_part(mask, num_lower=None, num_upper=0, exclude=True)

# Option 2: Range comparison (graph API)
rows = F.arange(0, seq_len, out_dim=seq_len).unsqueeze(1)
cols = F.arange(0, seq_len, out_dim=seq_len).unsqueeze(0)
is_masked = F.greater(cols, rows)  # True where cols > rows (upper triangle)
# Use F.select to avoid NaN from 0 * -inf:
zero = Tensor.constant(0.0, dtype=dtype, device=device)
neg_inf = Tensor.constant(float("-inf"), dtype=dtype, device=device)
mask = F.select(is_masked, neg_inf, zero)
# Use: attn_weights = attn_weights + mask
```

**Why?** MAX does not provide `F.tril`/`F.triu`. Use `F.band_part` (eager) or range comparison (graph) as alternatives.

---

### CPU-Only Session on Apple Silicon

**❌ WRONG:** Relying on default session (includes Metal GPU)
```python
# nocompile - _session() detects Metal GPU, initializes GPU context
model = MyModel()
compiled = model.compile(input_types)  # May hit GPU context errors
```

**✅ CORRECT:** Force CPU-only session via monkey-patch

See [model-implementation.md](../patterns/model-implementation.md#cpu-only-session-on-apple-silicon-temporary) for the full workaround code.

**Why?** On Apple Silicon, the internal `_session()` function detects Metal GPU and initializes the GPU device context. Some ops (e.g., `layer_norm`) then incorrectly dispatch to GPU even for CPU tensors. **[Temporary]** — will be addressed when a public device-targeting API is added.

---

## Custom Operations

### Wrong `ops.custom()` Signature (26.2+)

**❌ WRONG:** Missing device parameter
```python
# nocompile - 26.1 API, breaks in 26.2
result = ops.custom("my_op", [input_tensor], [output_type])
```

**✅ CORRECT:** Include device parameter (26.2+)
```python
# 26.2+ API
result = ops.custom("my_op", DeviceRef.GPU(), [input_tensor], [output_type])
```

**Why?** `ops.custom()` requires device parameter in 26.2+ for multi-device support.

---

### Missing `@compiler.register` for Custom Kernels

**❌ WRONG:** Kernel not registered
```mojo
# nocompile - Kernel exists but not registered
fn my_kernel(...):
    pass
```

**✅ CORRECT:** Register the kernel
```mojo
from max.compiler import compiler

@compiler.register("my_kernel")
fn my_kernel(...):
    pass
```

**Why?** MAX Engine only sees kernels that are registered with `@compiler.register`. Unregistered kernels are invisible to the graph compiler.

---

## Quick Lookup Table

| Error/Symptom | Likely Cause | Section |
|---------------|--------------|---------|
| "no matching function in call to 'foreach'" | Version mismatch | [Version Alignment](#version-alignment) |
| OOM during serving | Missing KV cache config | [KV Cache](#kv-cache) |
| Slow multi-request throughput | Missing prefix caching | [Prefix Caching](#missing-prefix-caching-for-shared-prompts) |
| "device required" | Missing device in TensorType | [TensorType](#missing-device-in-tensortype-262) |
| Custom op not found | Missing @compiler.register | [Custom Operations](#custom-operations) |
| "not enough GPUs" | TP > available GPUs | [Tensor Parallel](#tensor-parallel-without-enough-gpus) |
| CLI flag ignored | Deprecated flag name | [Deprecated Flags](#deprecated-cli-flags) |
| "Unimplemented at device_context.mojo:1950" | ops.layer_norm on Apple Silicon | [ops.layer_norm Crashes](#opslayer_norm-crashes-on-apple-silicon) |
| Garbage output, no error | Conv1D weight transposition | [Conv1D Weight Transposition](#conv1d-weight-transposition-gpt-2--openai-models) |
| Parameters not found in graph API | Modules stored in plain list | [ModuleList Not Available](#graph-api-modulelist-not-available) |
| Graph compiler can't infer shape | Missing out_dim on F.arange | [F.arange out_dim](#farange-requires-out_dim-with-symbolic-shapes) |
| ~60s per token | Eager mode without compile() | [model-implementation.md](../patterns/model-implementation.md#eager-vs-compiled-mode-performance) |

---

## Related

- [ERROR_INDEX.md](ERROR_INDEX.md) - Full error message lookup
- [serve-configuration.md](../patterns/serve-configuration.md) - Complete serve config
- [multigpu-scaling.md](../patterns/multigpu-scaling.md) - Multi-GPU patterns
- [graph-construction.md](../patterns/graph-construction.md) - Graph API patterns
