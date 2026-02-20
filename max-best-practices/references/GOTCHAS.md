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
  --kv-cache-strategy paged \
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

**❌ WRONG:** Requesting more tensor parallel than available
```bash
# nocompile - Only 2 GPUs available
max serve --model meta-llama/Llama-3.1-70B \
  --tensor-parallel=4  # Error: not enough GPUs
```

**✅ CORRECT:** Match tensor parallel to available GPUs
```bash
# Check available GPUs first
nvidia-smi -L  # Shows 2 GPUs

max serve --model meta-llama/Llama-3.1-70B \
  --tensor-parallel=2 --devices gpu:0,1
```

**Why?** Tensor parallelism requires the specified number of GPUs. Use a smaller model or fewer TP shards if limited.

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

### Wrong DeviceRef Construction (26.2+)

**❌ WRONG:** Old DeviceRef API
```python
# nocompile - Deprecated in 26.2
device = DeviceRef.from_device(my_device)
```

**✅ CORRECT:** Use factory methods
```python
# 26.2+ API
cpu = DeviceRef.CPU()
gpu = DeviceRef.GPU()
gpu1 = DeviceRef.GPU(1)  # Specific GPU
```

**Why?** DeviceRef API simplified in 26.2. `from_device` is deprecated.

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

---

## Related

- [ERROR_INDEX.md](ERROR_INDEX.md) - Full error message lookup
- [serve-configuration.md](../patterns/serve-configuration.md) - Complete serve config
- [multigpu-scaling.md](../patterns/multigpu-scaling.md) - Multi-GPU patterns
- [graph-construction.md](../patterns/graph-construction.md) - Graph API patterns
