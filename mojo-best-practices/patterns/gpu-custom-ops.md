---
title: Custom GPU Operations for MAX Graph
description: Custom op registration, OutputTensor/InputTensor types, Python graph integration, multi-kernel orchestration
impact: HIGH
category: gpu
tags: [gpu, custom-op, max-graph, compiler-register, layout-tensor, python]
error_patterns:
  - "op not found"
  - "custom op registration"
  - "OutputTensor"
  - "InputTensor"
  - "to_layout_tensor"
  - "enqueue_function"
  - "mojopkg"
  - "rebind"
  - "DeviceBuffer"
  - "kernel changes not reflected"
scenarios:
  - "Register a custom GPU operation for MAX Graph"
  - "Convert OutputTensor/InputTensor to LayoutTensor"
  - "Call custom op from Python graph code"
  - "Chain multiple GPU kernels in a single operation"
  - "Package custom op as mojopkg"
  - "Zero output buffer before kernel launch"
  - "Dispatch between CPU and GPU implementations"
---

# Custom GPU Operations for MAX Graph

**Category:** GPU | **Impact:** HIGH

Register custom operations that integrate with the MAX Graph compiler, dispatch between CPU and GPU targets at compile time, and orchestrate multi-kernel pipelines. These patterns are the foundation for extending MAX Graph with user-defined GPU kernels.

## API Availability

> **Note:** All APIs listed below are available in the Mojo nightly toolchain (v26.2+). Code examples are documentation snippets -- adapt import paths and parameters for your use case.

| API | Import Path | Notes |
|-----|-------------|-------|
| `@compiler.register` | Available when `compiler` module is imported | Custom op registration decorator |
| `OutputTensor` | Compiler-injected type (no explicit import needed) | Mutable output tensor in custom ops |
| `InputTensor` | Compiler-injected type (no explicit import needed) | Immutable input tensor in custom ops |
| `DeviceContextPtr` | Compiler-injected type (no explicit import needed) | GPU context handle passed to ops |
| `LayoutTensor`, `Layout` | `from layout import LayoutTensor, Layout` | Type-safe tensor with compile-time layout |
| `DeviceBuffer` | `from gpu.host import DeviceBuffer` | Non-owning GPU buffer wrapper |
| `rebind` | Built-in (prelude) | Type-level reinterpret cast |
| `enqueue_function` | via `DeviceContext` | Submit GPU kernel for execution |
| `enqueue_memset` | via `DeviceContext` | Zero or fill GPU memory |

> **Import note:** `OutputTensor`, `InputTensor`, and `DeviceContextPtr` are compiler-injected types available inside `@compiler.register` structs. You do not need to import them explicitly. Import paths may vary between Mojo versions -- adapt for your toolchain.

---

## Custom Op Registration Pattern

Register a struct with `@compiler.register` to make it callable from Python graph code via `ops.custom()`. The `execute` method receives typed tensors and a device context.

```mojo
# nocompile
@compiler.register("my_op")
struct MyOp:
    @staticmethod
    fn execute[target: StaticString, ...](
        output: OutputTensor[rank=1],
        input: InputTensor[rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            gpu_impl(output, input, ctx)
        else:
            cpu_impl(output, input, ctx)
```

**Key points:**
- The string passed to `@compiler.register("my_op")` must exactly match the `name=` argument in the Python `ops.custom()` call
- The `target` parameter is a compile-time string (`"gpu"` or `"cpu"`) -- dispatch with `@parameter if`
- `execute` uses variadic generic parameters (`...`) to accept compiler-injected type information
- `OutputTensor` is mutable (write results here); `InputTensor` is immutable (read inputs from here)

---

## LayoutTensor Conversion from OutputTensor/InputTensor

`OutputTensor` and `InputTensor` are compiler-level types. To use the full `LayoutTensor` API (indexing, tiling, thread-mapped access), convert them with `to_layout_tensor()` and `rebind`.

```mojo
# nocompile
from utils import rebind
from layout import LayoutTensor

# Convert OutputTensor (mutable) to LayoutTensor
var out_lt = rebind[LayoutTensor[dtype, layout, MutAnyOrigin]](
    output.to_layout_tensor()
)

# Convert InputTensor (immutable) to LayoutTensor
var in_lt = rebind[LayoutTensor[dtype, layout, ImmutAnyOrigin]](
    input.to_layout_tensor()
)
```

**Why `rebind` is needed:**
- `to_layout_tensor()` returns a LayoutTensor with a specific origin type tied to the OutputTensor/InputTensor lifetime
- `rebind` reinterprets the origin to `MutAnyOrigin` (for outputs) or `ImmutAnyOrigin` (for inputs), allowing the tensor to be passed into kernel functions that use generic origins
- This is a zero-cost type-level cast -- no data is copied

---

## enqueue_function Kernel Launch

Launch GPU kernels from custom ops via `enqueue_function`. The exact parameter form depends on the context:

- Inside custom ops with `DeviceContextPtr`: use the form shown below
- From `DeviceContext` directly: `ctx.enqueue_function[kernel](args, grid_dim=G, block_dim=B)`

```mojo
# nocompile
# Define the kernel
fn my_kernel[layout: Layout, dtype: DType](
    out: LayoutTensor[dtype, layout, MutAnyOrigin],
    inp: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    var tid = thread_idx.x + block_idx.x * block_dim.x
    if tid < out.size():
        out[tid] = inp[tid] * 2

# Launch the kernel from a custom op
fn gpu_impl(output: OutputTensor[rank=1], input: InputTensor[rank=1], ctx: DeviceContextPtr) raises:
    var gpu_ctx = ctx.get_device_context()

    var out_lt = rebind[LayoutTensor[dtype, layout, MutAnyOrigin]](output.to_layout_tensor())
    var in_lt = rebind[LayoutTensor[dtype, layout, ImmutAnyOrigin]](input.to_layout_tensor())

    comptime kernel = my_kernel[layout, dtype]
    gpu_ctx.enqueue_function[kernel](
        out_lt,
        in_lt,
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
    )
```

**Key points:**
- `enqueue_function` takes the kernel as a compile-time parameter
- The kernel function must be fully specialized (all type parameters resolved) before passing to `enqueue_function`
- Use `comptime` to bind specialized kernel, then pass as parameter
- If you encounter compilation errors about parameter count, check the `enqueue_function` overloads available in your Mojo version

---

## Target Dispatch with CPU Fallback

Use `@parameter if` for compile-time target dispatch. The unused branch is eliminated entirely -- no runtime overhead.

```mojo
# nocompile
@compiler.register("my_op")
struct MyOp:
    @staticmethod
    fn execute[target: StaticString, ...](
        output: OutputTensor[rank=1],
        input: InputTensor[rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            # GPU path: parallel across blocks and threads
            var gpu_ctx = ctx.get_device_context()
            var out_lt = rebind[LayoutTensor[dtype, layout, MutAnyOrigin]](
                output.to_layout_tensor()
            )
            var in_lt = rebind[LayoutTensor[dtype, layout, ImmutAnyOrigin]](
                input.to_layout_tensor()
            )
            comptime kernel = my_kernel[layout, dtype]
            gpu_ctx.enqueue_function[kernel](
                out_lt, in_lt,
                grid_dim=(size + BLOCK_SIZE - 1) // BLOCK_SIZE,
                block_dim=BLOCK_SIZE,
            )
        else:
            # CPU path: sequential loop
            for i in range(output.size()):
                output[i] = input[i] * 2
```

**CPU fallback is essential for:**
- Development and debugging (easier to print values, set breakpoints)
- Platforms without GPU support
- Numerical validation against a known-correct reference

---

## Python Graph Integration

Call your custom op from Python using `ops.custom()`. The `name` argument must match the registered name exactly.

```python
from max.graph import Graph, TensorType, ops
from max.dtype import DType
from max.driver import DeviceRef

def build_graph(device: DeviceRef, size: int, dtype: DType) -> Graph:
    graph = Graph("my_graph")

    # Define graph input
    input_type = TensorType(dtype=dtype, shape=(size,), device=device)
    input_value = graph.input(input_type)

    # Call the custom op
    output = ops.custom(
        name="my_op",  # Must match @compiler.register name
        device=DeviceRef.from_device(device),
        values=[input_value],
        out_types=[TensorType(dtype=dtype, shape=(size,), device=device)],
        parameters={"input_size": size, "dtype": dtype},
    )[0].tensor

    graph.output(output)
    return graph
```

**Key details:**
- `values` is a list of graph tensors to pass as inputs to the custom op
- `out_types` specifies the shape, dtype, and device of each output tensor
- `parameters` passes compile-time constants (integers, dtypes, strings) to the Mojo op
- `ops.custom()` returns a list of results; use `[0].tensor` to extract the first output tensor
- The `device` argument controls whether the `target` parameter in Mojo is `"gpu"` or `"cpu"`

---

## mojopkg Packaging Workflow

Custom ops must be packaged as `.mojopkg` files for the MAX Graph compiler to find them at graph compilation time.

```bash
# Package the custom op directory into a .mojopkg
mojo package problems/pXX/op -o problems/pXX/op.mojopkg

# Typical project structure:
# problems/pXX/
#   op/
#     __init__.mojo       # Re-exports the op struct
#     my_op.mojo          # Contains @compiler.register struct
#   op.mojopkg            # Generated package (used at graph compile time)
#   main.py               # Python script that builds and runs the graph
```

**Important:**
- Re-package after every change to the Mojo source -- the `.mojopkg` is a compiled artifact
- If kernel behavior does not change after edits, purge caches: `rm -rf ~/.modular/.max_cache ~/.modular/.mojo_cache ~/.modular/.mogg_cache`
- The `mojopkg` file is what the MAX compiler loads, not the raw `.mojo` source files

---

## Multi-Kernel Pipeline Orchestration

Complex operations (e.g., attention, fused norm+project) often require multiple kernel launches chained together. Each kernel shares the same `DeviceContext`, and synchronization between consecutive `enqueue_function` calls is implicit -- the GPU executes them in order on the same stream.

```mojo
# nocompile
fn attention_gpu_impl(
    output: OutputTensor[rank=3],
    queries: InputTensor[rank=3],
    keys: InputTensor[rank=3],
    values: InputTensor[rank=3],
    ctx: DeviceContextPtr,
) raises:
    var gpu_ctx = ctx.get_device_context()

    # Step 1: Compute QK^T scores
    var scores_buf = DeviceBuffer[score_dtype](gpu_ctx, seq_len * seq_len)
    comptime qk_kernel = qk_matmul_kernel[layout, dtype]
    gpu_ctx.enqueue_function[qk_kernel](
        q_lt, k_lt, scores_buf,
        grid_dim=grid_qk, block_dim=BLOCK_SIZE,
    )

    # Step 2: Softmax over scores (in-place)
    comptime softmax_kernel = softmax_kernel[layout, score_dtype]
    gpu_ctx.enqueue_function[softmax_kernel](
        scores_buf, seq_len,
        grid_dim=grid_softmax, block_dim=BLOCK_SIZE,
    )

    # Step 3: Multiply scores by values
    comptime sv_kernel = score_value_kernel[layout, dtype]
    gpu_ctx.enqueue_function[sv_kernel](
        scores_buf, v_lt, out_lt,
        grid_dim=grid_sv, block_dim=BLOCK_SIZE,
    )
```

**Key points:**
- Kernels enqueued on the same `DeviceContext` execute sequentially on the GPU stream -- no explicit synchronization needed between them
- Use `DeviceBuffer` for intermediate GPU allocations between kernels
- `DeviceBuffer` with `owning=False` creates a non-owning wrapper for rank adaptation (reinterpreting a flat buffer as a different-rank tensor)

### DeviceBuffer for Rank Adaptation

When one kernel outputs a flat buffer but the next expects a 2D tensor, use a non-owning `DeviceBuffer` wrapper:

```mojo
# nocompile
# Kernel 1 outputs to a flat buffer
var flat_buf = DeviceBuffer[dtype](gpu_ctx, total_elements)

# Wrap the same memory as a different view for kernel 2 (non-owning)
var reshaped_buf = DeviceBuffer[dtype](gpu_ctx, flat_buf.ptr, rows * cols, owning=False)
```

The `owning=False` parameter means this buffer does not manage the memory lifetime -- the original buffer still owns it.

---

## Zero Output Buffer Before Kernel Launch

When a kernel uses atomic additions or accumulates partial results, the output buffer must be zeroed first. Use `enqueue_memset` before the kernel launch.

```mojo
# nocompile
fn gpu_impl(output: OutputTensor[rank=1], input: InputTensor[rank=1], ctx: DeviceContextPtr) raises:
    var gpu_ctx = ctx.get_device_context()
    var size = output.size()

    var out_lt = rebind[LayoutTensor[dtype, layout, MutAnyOrigin]](output.to_layout_tensor())

    # Zero the output buffer BEFORE kernel launch
    gpu_ctx.enqueue_memset(
        DeviceBuffer[dtype](gpu_ctx, out_lt.ptr, size, owning=False),
        0,
    )

    # Now launch the kernel (output is guaranteed zeroed)
    comptime kernel = accumulate_kernel[layout, dtype]
    gpu_ctx.enqueue_function[kernel](
        out_lt, in_lt,
        grid_dim=(size + BLOCK_SIZE - 1) // BLOCK_SIZE,
        block_dim=BLOCK_SIZE,
    )
```

**When to zero:**
- Reduction kernels where threads accumulate into shared output locations
- Scatter operations where not every output element is written
- Any kernel using atomic operations (`atomic_add`, etc.)

**When NOT to zero:**
- Elementwise kernels that write every output element exactly once
- Kernels that overwrite the entire output unconditionally

---

## Decision Guide

| Task | Pattern | Key Detail |
|------|---------|------------|
| Add a new op to MAX Graph | Custom Op Registration | `@compiler.register` + `ops.custom()` in Python |
| Use LayoutTensor inside custom op | LayoutTensor Conversion | `rebind` + `to_layout_tensor()` |
| Launch GPU kernel from custom op | enqueue_function | `enqueue_function[kernel](args, grid_dim=G, block_dim=B)` |
| Support both CPU and GPU | Target Dispatch | `@parameter if target == "gpu"` |
| Chain kernels (attention, etc.) | Multi-Kernel Pipeline | Sequential `enqueue_function` on same context |
| Intermediate buffers between kernels | DeviceBuffer | `owning=False` for non-owning views |
| Atomic/reduction output | Zero Before Launch | `enqueue_memset` before kernel |
| Deploy custom op | mojopkg Packaging | `mojo package` after every change |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `op not found` / `unknown custom op` | Name mismatch between `@compiler.register` and `ops.custom(name=...)` | Ensure strings match exactly (case-sensitive) |
| `rebind` type error | Wrong origin type in `rebind` target | Use `MutAnyOrigin` for `OutputTensor`, `ImmutAnyOrigin` for `InputTensor` |
| `enqueue_function` compilation error | Kernel not fully specialized or wrong parameter form | Use `comptime kernel = my_fn[params]` then `enqueue_function[kernel](...)` |
| Kernel produces all zeros | Output buffer not written, or wrong grid/block dims | Check `grid_dim` covers all elements; verify kernel indexing |
| Kernel results not changing after edit | Stale `.mojopkg` or compilation cache | Re-run `mojo package`; purge `~/.modular/.max_cache`, `~/.modular/.mojo_cache`, `~/.modular/.mogg_cache` |
| `DeviceBuffer` use-after-free | Owning buffer freed while non-owning view still in use | Ensure owning buffer outlives all `owning=False` views |
| Python `ops.custom` shape mismatch | `out_types` shape does not match what the kernel writes | Align `TensorType(shape=...)` with actual kernel output dimensions |
| `parameters` not received in Mojo | Parameter name/type mismatch | Ensure Python `parameters={"key": val}` keys match Mojo op parameter names |
| GPU kernel runs but wrong results | CPU fallback running instead of GPU | Verify `device=DeviceRef.from_device(device)` points to GPU in `ops.custom()` |
| `to_layout_tensor()` fails | Tensor not on expected device | Confirm tensor device matches target dispatch path |

---

## Quick Reference

- **Register:** `@compiler.register("name")` on a struct with `execute` static method
- **Convert tensors:** `rebind[LayoutTensor[..., MutAnyOrigin]](output.to_layout_tensor())`
- **Launch kernel:** `comptime k = my_fn[params]` then `enqueue_function[k](args, grid_dim=G, block_dim=B)`
- **Python call:** `ops.custom(name="name", device=dev, values=[...], out_types=[...])`
- **Package:** `mojo package path/to/op -o path/to/op.mojopkg`
- **Zero buffer:** `enqueue_memset(DeviceBuffer[dtype](ctx, ptr, size, owning=False), 0)`
- **Cache purge:** `rm -rf ~/.modular/.max_cache ~/.modular/.mojo_cache ~/.modular/.mogg_cache`

---

## Related Patterns

- [`gpu-fundamentals.md`](gpu-fundamentals.md) -- Thread hierarchy, memory model, DeviceContext management
- [`gpu-kernels.md`](gpu-kernels.md) -- Kernel fusion, producer-consumer pipelines, double-buffering
- [`gpu-memory-access.md`](gpu-memory-access.md) -- TMA, shared memory, and coalescing patterns
- [`gpu-structured-kernels.md`](gpu-structured-kernels.md) -- ScatterGather/RingBuffer/TileOp architecture
- [`gpu-warp-sync.md`](gpu-warp-sync.md) -- Barriers and async operations

---

## References

- [MAX Graph Custom Ops](https://docs.modular.com/max/graph/custom-ops)
- [MAX Kernels](https://github.com/modular/modular/tree/main/max/kernels)
