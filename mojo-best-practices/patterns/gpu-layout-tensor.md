---
title: LayoutTensor API and Patterns
description: LayoutTensor indexing, rebind pattern, tile API, shared memory allocation, cross-rank reshape
impact: CRITICAL
category: gpu
tags: [gpu, layout-tensor, rebind, tile, shared-memory, reshape]
error_patterns:
  - "cannot implicitly convert 'LayoutTensor"
  - "element_type"
  - "no matching method in call to 'load'"
  - "rebind input type"
  - "cannot implicitly convert 'SIMD"
  - "expected 'Scalar' but got 'SIMD'"
  - "IndexList"
  - "no matching method in call to 'store'"
scenarios:
  - "Fix LayoutTensor type mismatch"
  - "Create shared memory LayoutTensor"
  - "Tile a LayoutTensor"
  - "Reshape LayoutTensor across ranks"
  - "Read from LayoutTensor without rebind error"
  - "Construct LayoutTensor from DeviceBuffer"
---

# LayoutTensor API and Patterns

**Category:** gpu | **Impact:** CRITICAL

LayoutTensor provides type-safe, multi-dimensional tensor access with compile-time layout information. It is the primary abstraction for structured GPU memory access in Mojo. The single most common error when using LayoutTensor is the element type mismatch on reads -- every read returns `SIMD[dtype, symbolic_size]`, NOT `Scalar[dtype]`, and requires `rebind` to fix.

## API Availability

> **Note:** All APIs listed below are available in the Mojo nightly toolchain (v26.2+). Code examples are documentation snippets -- adapt import paths and parameters for your use case.

| API | Import Path | Notes |
|-----|-------------|-------|
| `LayoutTensor` | `from layout import LayoutTensor` | Core type-safe tensor |
| `Layout` | `from layout import Layout` | Compile-time layout descriptor |
| `IndexList` | `from utils import IndexList` | Multi-dimensional index (required for `load`/`store`) |
| `stack_allocation` | Method on `LayoutTensor` | Shared memory allocation for LayoutTensor |
| `AddressSpace` | `from gpu.memory import AddressSpace` | Address space enum (SHARED, GENERIC) |
| `DeviceBuffer` | `from gpu.host import DeviceBuffer` | Device memory buffer |
| `rebind` | Built-in (prelude) | Type reinterpretation, no runtime cost |

---

## CRITICAL: rebind Typically Required for Reads

**This is the #1 most common LayoutTensor error.** Reading an element from a LayoutTensor via indexing (e.g., `tensor[i]`) returns `SIMD[dtype, symbolic_size]`, not `Scalar[dtype]`. This happens with most layouts -- concrete and generic. The compiler cannot implicitly convert the symbolic SIMD width to a scalar.

> **Alternative:** Some LayoutTensor APIs (e.g., `load_scalar`) may return `Scalar[dtype]` directly without needing `rebind`. However, basic indexing (`tensor[i]`) always requires `rebind`, and this is by far the most common access pattern.

**The error looks like:**
```
cannot implicitly convert 'SIMD[float32, symbolic_size]' value to 'SIMD[float32, 1]'
```

**Fix:** Wrap every read with `rebind[Scalar[dtype]](...)`.

**Incorrect -- fails to compile:**
```mojo
# nocompile

fn broken_kernel(tensor: LayoutTensor[DType.float32, layout, MutAnyOrigin]):
    var tid = global_idx.x
    var val: Float32 = tensor[tid]  # ERROR: SIMD[float32, symbolic_size] != Scalar[float32]
```

**Correct -- use rebind on every read:**
```mojo
# nocompile

fn working_kernel(tensor: LayoutTensor[DType.float32, layout, MutAnyOrigin]):
    var tid = global_idx.x
    var val = rebind[Scalar[DType.float32]](tensor[tid])  # OK
```

**Writes do NOT need rebind** -- scalar values are accepted directly:
```mojo
# nocompile

fn write_example(tensor: LayoutTensor[DType.float32, layout, MutAnyOrigin]):
    var tid = global_idx.x
    tensor[tid] = Float32(42.0)  # OK -- writes accept scalars directly
```

**Accumulation pattern** -- rebind both sides of the expression:
```mojo
# nocompile

fn reduction_kernel[dtype: DType](
    shared: LayoutTensor[dtype, Layout.row_major(BLOCK_SIZE), MutAnyOrigin,
                         address_space=AddressSpace.SHARED],
):
    var tid = thread_idx.x

    # WRONG: shared[tid] + shared[tid + stride] returns symbolic SIMD
    # shared[tid] = shared[tid] + shared[tid + stride]  # ERROR

    # CORRECT: rebind both reads
    shared[tid] = rebind[Scalar[dtype]](shared[tid]) + rebind[Scalar[dtype]](shared[tid + stride])
```

**Multi-dimensional read -- same rule applies:**
```mojo
# nocompile

fn matmul_accumulate[dtype: DType](
    A: LayoutTensor[dtype, layout_a, MutAnyOrigin],
    B: LayoutTensor[dtype, layout_b, MutAnyOrigin],
    C: LayoutTensor[dtype, layout_c, MutAnyOrigin],
):
    var row = block_idx.y * TILE + thread_idx.y
    var col = block_idx.x * TILE + thread_idx.x

    var acc = Scalar[dtype](0)
    for k in range(K):
        acc += rebind[Scalar[dtype]](A[row, k]) * rebind[Scalar[dtype]](B[k, col])
    C[row, col] = acc
```

---

## Construction from DeviceBuffer

LayoutTensor wraps a DeviceBuffer with compile-time layout information.

**From a DeviceBuffer directly:**
```mojo

from gpu.host import DeviceContext, DeviceBuffer
from layout import LayoutTensor, Layout

fn example() raises:
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.float32](1024)

    # Wrap buffer with 1D layout
    var tensor = LayoutTensor[DType.float32, Layout.row_major(1024), MutAnyOrigin](buf)
```

**From a custom op output (rebind required):**
```mojo
# nocompile

# Inside a custom op implementation
var out_tensor = rebind[
    LayoutTensor[DType.float32, Layout.row_major(N), MutAnyOrigin]
](output.to_layout_tensor())
```

---

## Tile API

The `tile` method creates a sub-view into a LayoutTensor without copying data. It is the primary mechanism for tiled algorithms (matmul, convolution, reduction).

```mojo
# nocompile

fn tiled_kernel[dtype: DType, TILE_SIZE: Int](
    input: LayoutTensor[dtype, layout, MutAnyOrigin],
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    size: Int,
):
    var tile_idx = block_idx.x
    var tid = thread_idx.x

    # Create a tile view -- no data copy, just pointer + offset
    var in_tile = input.tile[TILE_SIZE](tile_idx)
    var out_tile = output.tile[TILE_SIZE](tile_idx)

    # Index within the tile as if it were its own tensor
    if tile_idx * TILE_SIZE + tid < size:
        out_tile[tid] = rebind[Scalar[dtype]](in_tile[tid]) * Scalar[dtype](2.0)
```

**2D tiling for matrix operations:**
```mojo

fn tiled_matmul[dtype: DType, BM: Int, BN: Int, BK: Int](
    A: LayoutTensor[dtype, layout_a, MutAnyOrigin],
    B: LayoutTensor[dtype, layout_b, MutAnyOrigin],
    C: LayoutTensor[dtype, layout_c, MutAnyOrigin],
):
    # Tile A along rows and K, tile B along K and columns
    var a_tile = A.tile[BM, BK](block_idx.y, k_idx)
    var b_tile = B.tile[BK, BN](k_idx, block_idx.x)
    var c_tile = C.tile[BM, BN](block_idx.y, block_idx.x)
    # ... accumulate into c_tile
```

---

## load/store with IndexList or Index

The `load` and `store` methods typically require `IndexList` or `Index()` (a factory that returns `IndexList`), NOT bare `Int` arguments. Some overloads accept `(row, col)` Ints directly, but `Index`/`IndexList` is the most portable pattern.

**Incorrect -- bare Int fails:**
```mojo
# nocompile

# ERROR: no matching method in call to 'load'
var val = tensor.load[4](0)
```

**Correct -- use IndexList or Index:**
```mojo
# nocompile

from utils import IndexList, Index

# 1D tensor: Index (simplest for 1D)
var val = tensor.aligned_load[4](Index(0))
tensor.store[4](Index(0), val)

# 1D tensor: IndexList (equivalent, more verbose)
var val2 = tensor.load[4](IndexList[1](0))
tensor.store[4](IndexList[1](0), val2)

# 2D tensor: IndexList with rank 2
var val_2d = tensor.load[4](IndexList[2](row, col))
tensor.store[4](IndexList[2](row, col), val_2d)
```

**Vectorized element-wise kernel using load/store:**
```mojo
# nocompile

fn vectorized_kernel[dtype: DType, WIDTH: Int](
    input: LayoutTensor[dtype, layout, MutAnyOrigin],
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    var vec_idx = tid * WIDTH

    if vec_idx + WIDTH <= size:
        var vec = input.load[WIDTH](IndexList[1](vec_idx))
        output.store[WIDTH](IndexList[1](vec_idx), vec * 2.0)
```

---

## Shared Memory LayoutTensor

LayoutTensor provides a clean API for shared memory allocation via `stack_allocation()`. This replaces the lower-level `stack_allocation` + manual offset math pattern.

**1D shared memory:**
```mojo
# nocompile

from layout import LayoutTensor, Layout
from gpu.memory import AddressSpace
from gpu import barrier

fn block_reduce[dtype: DType, BLOCK_SIZE: Int](
    input: LayoutTensor[dtype, layout, MutAnyOrigin],
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
):
    var shared = LayoutTensor[
        dtype, Layout.row_major(BLOCK_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED
    ].stack_allocation()

    var tid = thread_idx.x

    # Load into shared memory
    shared[tid] = rebind[Scalar[dtype]](input[block_idx.x * BLOCK_SIZE + tid])
    barrier()

    # Tree reduction in shared memory
    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            shared[tid] = rebind[Scalar[dtype]](shared[tid]) + rebind[Scalar[dtype]](shared[tid + stride])
        barrier()
        stride //= 2

    if tid == 0:
        output[block_idx.x] = rebind[Scalar[dtype]](shared[0])
```

**2D shared memory:**
```mojo
# nocompile

fn tiled_transpose[dtype: DType, TILE: Int](
    input: LayoutTensor[dtype, layout_in, MutAnyOrigin],
    output: LayoutTensor[dtype, layout_out, MutAnyOrigin],
):
    # 2D shared memory with bank conflict avoidance padding
    var shared = LayoutTensor[
        dtype, Layout.row_major(TILE, TILE + 1), MutAnyOrigin,
        address_space=AddressSpace.SHARED
    ].stack_allocation()

    var tx = thread_idx.x
    var ty = thread_idx.y

    # Load into shared memory with 2D indexing
    shared[ty, tx] = rebind[Scalar[dtype]](input[block_idx.y * TILE + ty, block_idx.x * TILE + tx])
    barrier()

    # Store transposed -- note swapped tx/ty
    output[block_idx.x * TILE + ty, block_idx.y * TILE + tx] = rebind[Scalar[dtype]](shared[tx, ty])
```

**Bank conflict avoidance:** Use `Layout.row_major(N, M+1)` to add one element of padding per row, preventing bank conflicts when threads in a warp access the same column across different rows.

---

## Cross-Rank Reshape via DeviceBuffer

When working with custom ops, you may need to reinterpret a tensor with a different rank (e.g., 1D to 2D). Direct `rebind` between LayoutTensors of different ranks typically fails because the type parameters are incompatible. The recommended workaround is to go through a non-owning DeviceBuffer.

**The problem:**
```mojo
# nocompile

# This typically FAILS -- rebind between different-rank LayoutTensors
var tensor_1d: LayoutTensor[DType.float32, Layout.row_major(1024), MutAnyOrigin] = ...
var tensor_2d = rebind[
    LayoutTensor[DType.float32, Layout.row_major(32, 32), MutAnyOrigin]
](tensor_1d)  # ERROR: rebind input type mismatch
```

**Quick approach -- construct from `.ptr` (same memory, no copy):**
```mojo
# nocompile

# Direct pointer reinterpretation -- only safe when:
# 1. Element counts match (32*32 == 1024)
# 2. Source memory is contiguous
# 3. Pointer lifetime outlives the new tensor
var tensor_2d = LayoutTensor[dtype, Layout.row_major(32, 32), MutAnyOrigin](tensor_1d.ptr)
```

**DeviceBuffer intermediary (when you need buffer lifecycle management):**
```mojo
# nocompile

from gpu.host import DeviceContext, DeviceBuffer
from layout import LayoutTensor, Layout

fn reshape_1d_to_2d[dtype: DType](
    ctx: DeviceContext,
    tensor_1d: LayoutTensor[dtype, Layout.row_major(total), MutAnyOrigin],
    rows: Int, cols: Int,
) raises -> LayoutTensor[dtype, Layout.row_major(rows, cols), MutAnyOrigin]:
    # Step 1: Create non-owning DeviceBuffer wrapping the 1D tensor's memory
    # CRITICAL: buf must be a named variable -- RValue cannot bind to ref parameter
    var buf = DeviceBuffer[dtype](ctx, tensor_1d.ptr, rows * cols, owning=False)

    # Step 2: Construct new LayoutTensor with different rank from the buffer
    var tensor_2d = LayoutTensor[dtype, Layout.row_major(rows, cols), MutAnyOrigin](buf)

    return tensor_2d
```

**CRITICAL: DeviceBuffer must be a named variable.** The following fails because an RValue (temporary) cannot bind to a reference parameter:

```mojo
# nocompile

# WRONG -- temporary RValue cannot bind to DeviceBuffer ref param
var tensor_2d = LayoutTensor[dtype, Layout.row_major(rows, cols), MutAnyOrigin](
    DeviceBuffer[dtype](ctx, tensor_1d.ptr, rows * cols, owning=False)  # ERROR
)

# CORRECT -- named variable works
var buf = DeviceBuffer[dtype](ctx, tensor_1d.ptr, rows * cols, owning=False)
var tensor_2d = LayoutTensor[dtype, Layout.row_major(rows, cols), MutAnyOrigin](buf)
```

**2D to 3D reshape (same pattern):**
```mojo
# nocompile

fn reshape_2d_to_3d[dtype: DType](
    ctx: DeviceContext,
    tensor_2d: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    batch: Int, rows: Int, cols: Int,
) raises -> LayoutTensor[dtype, Layout.row_major(batch, rows, cols), MutAnyOrigin]:
    var buf = DeviceBuffer[dtype](ctx, tensor_2d.ptr, batch * rows * cols, owning=False)
    var tensor_3d = LayoutTensor[dtype, Layout.row_major(batch, rows, cols), MutAnyOrigin](buf)
    return tensor_3d
```

---

## Common Errors

| Error Message | Cause | Fix |
|---------------|-------|-----|
| `cannot implicitly convert 'SIMD[dtype, symbolic_size]'` | Reading LayoutTensor without rebind | `rebind[Scalar[dtype]](tensor[idx])` |
| `expected 'Scalar' but got 'SIMD'` | Same as above, different wording | Same fix: wrap read with `rebind` |
| `no matching method in call to 'load'` | Passing bare `Int` instead of `IndexList`/`Index` | `tensor.load[width](Index(idx))` or `tensor.load[width](IndexList[rank](idx))` |
| `no matching method in call to 'store'` | Same issue with store | `tensor.store[width](Index(idx), val)` or `tensor.store[width](IndexList[rank](idx), val)` |
| `rebind input type` / `rebind` mismatch | Trying to rebind between different-rank LayoutTensors | Use DeviceBuffer intermediary (see Cross-Rank Reshape) |
| `cannot implicitly convert 'LayoutTensor'` | Layout or origin mismatch between expected and actual | Check layout dimensions, dtype, and origin match exactly |
| `element_type` mismatch | dtype of LayoutTensor does not match operation | Ensure consistent dtype across all tensors in expression |
| RValue cannot bind to ref parameter | DeviceBuffer created as temporary | Assign DeviceBuffer to a named `var` first |

---

## Decision Guide

| Scenario | Approach | Key Detail |
|----------|----------|------------|
| Reading any element from LayoutTensor | Use `rebind[Scalar[dtype]]()` | Applies to ALL layouts, concrete and generic |
| Writing an element | Direct assignment: `tensor[idx] = val` | No rebind needed for writes |
| Vectorized read/write | `load`/`store` with `Index` or `IndexList` | Never pass bare Int to load/store |
| Shared memory (1D) | `LayoutTensor[..., address_space=SHARED].stack_allocation()` | Use `Layout.row_major(N)` |
| Shared memory (2D) | Same, with `Layout.row_major(N, M+1)` | `+1` padding avoids bank conflicts |
| Tiled algorithm | `tensor.tile[TILE_SIZE](tile_idx)` | Returns a view, no data copy |
| Reshape 1D to 2D | Construct from `.ptr` or go through non-owning `DeviceBuffer` | `rebind` fails across ranks; `.ptr` is simplest |
| Custom op output | `rebind[LayoutTensor[...]](output.to_layout_tensor())` | Match dtype and layout exactly |
| Accumulation in shared memory | `rebind` both operands | `shared[i] = rebind[...](shared[i]) + rebind[...](shared[i+s])` |

---

## Related Patterns

- [`gpu-fundamentals.md`](gpu-fundamentals.md) -- Thread hierarchy, memory model, basic kernel patterns
- [`gpu-memory-access.md`](gpu-memory-access.md) -- TMA loading, swizzling, bank conflict avoidance
- [`gpu-synchronization.md`](gpu-synchronization.md) -- Barrier and synchronization patterns
- [`gpu-tensor-cores.md`](gpu-tensor-cores.md) -- Tensor core programming with LayoutTensor
- [`memory-safety.md`](memory-safety.md) -- Safe pointer patterns and rebind caveats
