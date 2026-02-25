---
title: GPU Memory Access Patterns
description: TMA hardware loading, prefetch patterns, shared memory swizzling, and dynamic data caching
impact: HIGH
category: gpu
tags: [gpu, tma, memory, prefetch, swizzle, sm90, sm100]
error_patterns:
  - "TMA error"
  - "uncoalesced access"
  - "bank conflict"
  - "swizzle"
  - "misaligned address"
  - "out of bounds"
  - "cannot implicitly convert 'LayoutTensor"
  - "no matching method in call to 'load'"
  - "rebind input type"
  - "does not match result type"
scenarios:
  - "Load 2D tiles with TMA"
  - "Eliminate bank conflicts with swizzling"
  - "Implement double-buffered prefetch"
  - "Optimize memory coalescing"
  - "Fix LayoutTensor element type error"
  - "Reshape LayoutTensor between ranks"
  - "Use async copy for latency hiding"
  - "Load SIMD vectors from LayoutTensor"
consolidates:
  - gpu-tma-loading.md
  - gpu-tma-prefetch.md
  - gpu-shared-memory-swizzle.md
  - gpu-dynamic-data-caching.md
---

# GPU Memory Access Patterns

**Category:** gpu | **Impact:** HIGH

Optimal memory access patterns are essential for GPU performance. TMA (Tensor Memory Accelerator) provides hardware-accelerated 2D tile loads (2-3x faster than manual). Swizzling eliminates bank conflicts. Prefetching hides memory latency. Understanding these patterns enables peak bandwidth utilization.

## API Availability

> **Note:** All APIs listed below are available in the Mojo nightly toolchain (v26.2+). Code examples are documentation snippets — adapt import paths and parameters for your use case.

| API | Import Path | Notes |
|-----|-------------|-------|
| `thread_idx`, `block_idx` | `from gpu import thread_idx, block_idx` | Core GPU primitives |
| `stack_allocation` | `from memory import stack_allocation` | Shared memory allocation |
| `shared_memory` | `from gpu.memory import shared_memory` | Shared memory allocation (alternative API) |
| `AddressSpace` | `from gpu.memory import AddressSpace` | Address space enum (SHARED, GENERIC, etc.) |
| `LayoutTensor`, `Layout` | `from layout import LayoutTensor, Layout` | Type-safe tensor with compile-time layout |
| `create_tma_descriptor` | `from gpu.host.nvidia.tma import create_tma_descriptor` | TMA descriptor creation (SM90+) |
| `TMATensorTile`, `PipelineState` | `from layout.tma_async import ...` | TMA async pipeline |
| `make_swizzle` | `from layout.swizzle import make_swizzle` | Swizzle pattern creation |
| `elect_one_sync` | `from gpu.primitives.cluster import elect_one_sync` | Cluster primitives (SM90+) |

---

## Core Concepts

### TMA Hardware Loading (SM90+)

TMA handles address calculation, bounds checking, and coalescing in hardware:

| Feature | Manual Loading | TMA Loading |
|---------|---------------|-------------|
| Address calculation | Per-thread | Hardware |
| Bounds checking | Manual | Hardware |
| Coalescing | Manual | Hardware |
| Performance | Baseline | 2-3x faster |
| Availability | All GPUs | SM90+ only |

### Shared Memory Bank Conflicts

Shared memory has 32 banks (4 bytes each). Simultaneous access to the same bank causes serialization:

| Access Pattern | Transactions | Performance |
|----------------|--------------|-------------|
| Adjacent (stride 1) | 1 | Best |
| Stride 32 | 32 | Worst (serialized) |
| Random | Up to 32 | Poor |
| Swizzled | 1 | Best |

---

## Common Patterns

### TMA Hardware Loading

**When:** Loading 2D tiles on SM90/SM100 GPUs


**Do:**
```mojo
# nocompile

from gpu.host.nvidia.tma import create_tma_descriptor

# Create TMA descriptor (once, before kernel launch)
var tma_desc = create_tma_descriptor[
    dtype=DType.float16,
    tile_shape=(BM, BK),
    swizzle_mode=swizzle,
](global_ptr, global_shape, global_strides)

# In kernel: async load via TMA
fn load_tile_tma(
    tma_op: TMAOp,
    smem_tile: SMemTile,
    barrier: SharedMemBarrier,
    coords: Tuple[UInt, UInt],
):
    # Single instruction loads entire 2D tile!
    tma_op.async_load(smem_tile, barrier, coords)
```

**Don't:**
```mojo
# nocompile

fn load_tile_manual(
    global_ptr: UnsafePointer[Float16],
    smem_ptr: UnsafePointer[Float16],
    tile_row: Int, tile_col: Int,
    global_stride: Int
):
    var tid = thread_idx.x
    var row = tid // TILE_WIDTH
    var col = tid % TILE_WIDTH

    # Manual bounds check
    if tile_row + row < M and tile_col + col < K:
        var global_idx = (tile_row + row) * global_stride + tile_col + col
        var smem_idx = row * TILE_WIDTH + col
        smem_ptr[smem_idx] = global_ptr[global_idx]  # Slow, error-prone
```

### TMA Multicast (Clusters)

**When:** Multiple blocks need the same data


```mojo
# nocompile

from gpu.primitives.cluster import elect_one_sync

# Only one thread per warp issues TMA
if elect_one_sync():
    # Multicast to multiple blocks in cluster
    tma_op.async_multicast_load[cta_group](
        dest=smem_tile,
        barrier=barrier,
        coords=(k_coord, row_coord),
        multicast_mask=multicast_mask,
    )
```

### TMA Prefetch Pipeline

**When:** Memory-bound kernels with large K dimension


```mojo
from layout.tma_async import TMATensorTile, PipelineState
from gpu.sync import SharedMemBarrier

fn kernel_with_prefetch[num_pipeline_stages: Int](
    a_tma_op: TMATensorTile[...],
    b_tma_op: TMATensorTile[...],
):
    # 1. Prefetch TMA descriptors into L2 (once, at kernel start)
    if thread_idx.x == 0:
        a_tma_op.prefetch_descriptor()
        b_tma_op.prefetch_descriptor()

    # 2. Initialize multi-stage pipeline
    var full_mbar = stack_allocation[num_pipeline_stages, SharedMemBarrier,
                                     address_space=AddressSpace.SHARED]()
    # ... init barriers ...

    # 3. Prime the pipeline: prefetch first stages
    @parameter
    for stage in range(min(num_pipeline_stages, num_k_iters)):
        full_mbar[stage].expect_bytes(tile_bytes)
        a_tma_op.async_copy(a_smem[stage], full_mbar[stage], coords)
        b_tma_op.async_copy(b_smem[stage], full_mbar[stage], coords)

    # 4. Main loop: consume current, prefetch future
    for k_iter in range(num_k_iters):
        var stage = k_iter % num_pipeline_stages

        # Wait for current tile
        full_mbar[stage].wait(phase)

        # Compute on current tile
        wgmma_op.mma(a_smem[stage], b_smem[stage], accum)

        # Start prefetching future tile
        var future_k = k_iter + num_pipeline_stages
        if future_k < num_k_iters:
            full_mbar[stage].expect_bytes(tile_bytes)
            a_tma_op.async_copy(a_smem[stage], full_mbar[stage], future_coords)
            b_tma_op.async_copy(b_smem[stage], full_mbar[stage], future_coords)
```

### Pipeline Depth Selection

```mojo
# nocompile

fn select_pipeline_depth(M: Int, N: Int, K: Int, BK: Int) -> Int:
    """Heuristic for optimal pipeline depth."""
    var output_block_size = M * N
    var num_k_iters = ceildiv(K, BK)

    # Small output blocks can afford deeper pipelines
    if output_block_size <= 64 * 48:
        return min(12, num_k_iters)
    elif output_block_size <= 64 * 64:
        return min(8, num_k_iters)
    elif output_block_size <= 128 * 128:
        return min(6, num_k_iters)
    else:
        return min(4, num_k_iters)
```

| Stages | Latency Hidden | Shared Memory | Use Case |
|--------|---------------|---------------|----------|
| 2 | 1 iteration | 2x tile | Compute-bound |
| 3-4 | 2-3 iterations | 3-4x tile | Balanced |
| 6-8 | 5-7 iterations | 6-8x tile | Memory-bound |

### Shared Memory Swizzling

**When:** Matrix tiles in shared memory with column access


```mojo
# nocompile

from layout.swizzle import make_swizzle

# Create swizzle pattern
comptime swizzle = make_swizzle[
    8,                              # Swizzle bits
    swizzle_bytes // size_of[dtype](),
    simd_width_of[dtype](),
]()

fn load_with_swizzle(
    smem: UnsafePointer[Float16],
    row: Int, col: Int,
) -> Float16:
    # Apply swizzle to compute actual address
    var linear_idx = row * COLS + col
    var swizzled_idx = swizzle(linear_idx)
    return smem[swizzled_idx]
```

**K-major tile layouts for tensor cores:**

```mojo
# nocompile

# From MAX kernels - optimized for tensor core access
comptime a_smem_layout = tile_layout_k_major[
    a_type,
    BM,           # Tile M dimension
    BK,           # Tile K dimension
    swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
]()
```

**Swizzle modes:**

| Mode | Swizzle Bytes | Use Case |
|------|---------------|----------|
| 32B | 32 | Small tiles, FP32 |
| 64B | 64 | Medium tiles, FP16/BF16 |
| 128B | 128 | Large tiles, maximum throughput |

**Understanding swizzle:**
```text
Without swizzle (32 banks, 4 bytes each):
Thread 0: bank 0, Thread 1: bank 1, ..., Thread 31: bank 31
Thread 32: bank 0 (CONFLICT!), Thread 33: bank 1 (CONFLICT!), ...

With 128B swizzle:
Row 0: banks 0-31 (normal)
Row 1: banks 4-31, 0-3 (rotated by 4)
Row 2: banks 8-31, 0-7 (rotated by 8)
...
No conflicts when accessing columns!
```

### Dynamic Data Caching Anti-Pattern

**When:** GPU weight caching with per-iteration dynamic data

**Problem:** Weight caching uses CPU pointer addresses as keys. If heap reuses addresses, cache returns stale GPU data.

**Don't:**
```c
// WRONG: Caching dynamic per-iteration data
void gpu_operation(MPSContext* ctx, const float* dynamic_params) {
    // dynamic_params may reuse same address from freed buffer!
    id<MTLBuffer> paramsBuf = get_cached_weight_buffer(ctx, dynamic_params, size);
    // GPU uses stale data from previous iteration!
}
```

**Do:**
```c
// CORRECT: Create fresh buffers for dynamic data
void gpu_operation(MPSContext* ctx, const float* dynamic_params) {
    // Fresh buffer every time - data changes each iteration!
    id<MTLBuffer> paramsBuf = [ctx->device newBufferWithBytes:dynamic_params
                                                        length:size
                                                       options:MTLResourceStorageModeShared];
    // GPU uses correct current data
}
```

**What to cache vs. not cache:**

| Cache (persistent) | Don't Cache (dynamic) |
|-------------------|----------------------|
| Static weights | Per-iteration parameters |
| Pre-computed constants | Modulation values |
| Fixed position embeddings | Dynamic embeddings |
| Lookup tables | Activations |

**Debugging symptoms:**
- First iteration correct, subsequent incorrect
- Individual operations match in isolation
- Full pipeline fails across iterations

---

## LayoutTensor for Safe Tensor Access


`LayoutTensor` provides type-safe multi-dimensional GPU access with automatic layout handling.

**Don't (raw pointer arithmetic):**
```mojo
# nocompile

fn matmul_kernel(A: UnsafePointer[Float32], M: Int, N: Int, K: Int):
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x
    # Error-prone manual indexing
    sum += A[row * K + k] * B[k * N + col]  # Easy to mess up!
```

**Do (LayoutTensor):**
```mojo
# nocompile

from layout import LayoutTensor, Layout

fn matmul_kernel[dtype: DType, M: Int, N: Int, K: Int](
    A: LayoutTensor[dtype, Layout.row_major(M, K), MutAnyOrigin],
    B: LayoutTensor[dtype, Layout.row_major(K, N), MutAnyOrigin],
    C: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
):
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x
    var sum = Scalar[dtype](0)
    for k in range(K):
        sum += A[row, k] * B[k, col]  # Clean and correct!
    C[row, col] = sum
```

**Creating LayoutTensors:**
```mojo
# nocompile

# Shared memory tile
var shared_tile = LayoutTensor[
    DType.float32,
    Layout.row_major(16, 16),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()

# Vectorized loads
var vec = tensor.load[width=4](row, col)
tensor.store(row, col, vec * 2.0)

# Tiling (zero-copy views)
var A_tile = A.tile[TILE_M, K](block_idx.y, 0)
```

---

## LayoutTensor Patterns

### Element Type Mismatch and `rebind`

**Error:**
```
cannot implicitly convert 'LayoutTensor[...].element_type' value to 'Float32'
```

When `layout: Layout` is a compile-time parameter, indexing a LayoutTensor returns `SIMD[dtype, symbolic_size]` where the element size is a symbolic expression — not `Scalar[dtype]`. This prevents implicit conversion in expressions like multiplication across different layouts.

**When it happens (cross-layout operations):**
```mojo
# nocompile
# FAILS: Different layouts produce incompatible symbolic element types
var a: LayoutTensor[dtype, layout_a, ...]  # From one buffer
var b: LayoutTensor[dtype, layout_b, ...]  # From different buffer
var result = a[i] * b[j]  # ERROR: element types don't match
```

**When it doesn't happen (same-layout operations):**
```mojo
# nocompile
# WORKS: Same layout → same element type → no conversion needed
result_tensor[idx] = source_tensor[idx]
```

**Fix 1 — `rebind[Scalar[dtype]]` cast (most common):**
```mojo
# nocompile
var scalar_val = rebind[Scalar[dtype]](tensor[idx])
# Now scalar_val is Scalar[dtype] and works in all expressions
```

**Fix 2 — Use `stack_allocation` (UnsafePointer) for shared memory:**
```mojo
# nocompile
# UnsafePointer indexing always returns clean Scalar types
var shared = stack_allocation[SIZE, Scalar[dtype], address_space=AddressSpace.SHARED]()
shared[tid] = rebind[Scalar[dtype]](input_tensor[global_idx])
# shared[tid] is Scalar[dtype] — no conversion issues
```

**Fix 3 — Same-layout operations need no conversion:**
```mojo
# nocompile
# When source and dest have the same layout, element types match
output[row, col] = input[row, col]  # Works directly
```

### `IndexList[rank]` for `load` / `store`

**Error:**
```
no matching method in call to 'load'
```

LayoutTensor's `load[width]` and `store[width]` require `IndexList[rank]` coordinates, not bare integers.

**Don't:**
```mojo
# nocompile
# FAILS: Bare integer not accepted
var val = tensor.load[width](idx)
tensor.store[width](idx, value)
```

**Do:**
```mojo
# nocompile
# WORKS: Wrap in IndexList matching tensor rank
var val = tensor.load[width](IndexList[1](idx))
tensor.store[width](IndexList[1](idx), value)

# For 2D tensors:
var val_2d = tensor.load[width](IndexList[2](row, col))
```

### Tile Views with `.tile[size](idx)`

`LayoutTensor.tile` creates a zero-copy tile view — used in tiled matmul, async copy, and convolution patterns.

```mojo
# nocompile
# Create tile views for tiled matrix multiply
var A_tile = A.tile[TILE_SIZE](tile_idx)       # 1D tiling
var B_tile = B.tile[TILE_M, TILE_K](row, col)  # 2D tiling

# Use with async copy
copy_dram_to_sram_async[thread_layout=Layout.row_major(TILE)](
    shared_tile, global_tensor.tile[TILE](block_idx.x)
)
```

### Cross-Rank Reshape via DeviceBuffer

**Error:**
```
rebind input type '...(array<1>)...' does not match '...(array<2>)...'
```

`rebind` fails when source and target LayoutTensors have different ranks (e.g., 1D→2D). Reshape through a non-owning DeviceBuffer instead.

**Don't:**
```mojo
# FAILS: Rank mismatch (1D → 2D)
var reshaped = rebind[LayoutTensor[dtype, layout_2d, ...]](tensor_1d)
```

**Do:**
```mojo
# nocompile
from gpu.host import DeviceBuffer

# Create non-owning buffer, then construct new LayoutTensor
var buf = DeviceBuffer[dtype](ctx, tensor.ptr, size, owning=False)
var reshaped = LayoutTensor[dtype, new_layout, MutAnyOrigin](buf)
```

**Note:** `buf` must be a named variable (not an RValue) — the LayoutTensor borrows from it.

### Shared Memory Allocation (Summary)

Two patterns for shared memory, each with tradeoffs:

**Pattern 1: `stack_allocation` (UnsafePointer) — clean scalar types**
```mojo
# nocompile
from memory import stack_allocation
from gpu.memory import AddressSpace

var shared = stack_allocation[SIZE, Scalar[dtype], address_space=AddressSpace.SHARED]()
# shared[i] returns Scalar[dtype] — no rebind needed
# Best for: reductions, simple 1D buffers, cross-layout operations
```

**Pattern 2: `LayoutTensor.stack_allocation()` — type-safe 2D access**
```mojo
# nocompile
from layout import LayoutTensor, Layout
from gpu.memory import AddressSpace

var shared = LayoutTensor[
    dtype, Layout.row_major(ROWS, COLS), MutAnyOrigin,
    address_space=AddressSpace.SHARED,
].stack_allocation()
# shared[row, col] — multi-dimensional indexing
# Best for: tiled matmul, 2D convolution, structured access
```

### Async Copy (non-SM90)

For GPUs without TMA (SM80, Apple Silicon), use `copy_dram_to_sram_async` to hide global memory latency behind compute.

```mojo
# nocompile
from layout.layout_tensor import copy_dram_to_sram_async
from gpu.memory import async_copy_wait_all
from gpu import barrier

# Async copy from global to shared memory
copy_dram_to_sram_async[thread_layout=Layout.row_major(TILE)](
    shared_tile,                          # Destination (shared memory LayoutTensor)
    global_tensor.tile[TILE](block_idx.x) # Source (global memory tile view)
)
async_copy_wait_all()  # Wait for all async copies to complete
barrier()              # Ensure all threads see the data

# Now compute on shared_tile
```

**See also:** [`gpu-warp-sync.md`](gpu-warp-sync.md) for SM90+ TMA async pipeline patterns.

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| 2D tile loads (SM90+) | Use TMA hardware loading | - |
| Memory-bound kernel | Add prefetch pipeline (4-8 stages) | - |
| Column access in shared memory | Use swizzled layout | - |
| Multiple blocks need same data | Use TMA multicast | [`gpu-kernels.md`](gpu-kernels.md) |
| Dynamic per-iteration data | Never cache by pointer address | - |
| SM80 and earlier | Use async_copy pipeline | [`gpu-warp-sync.md`](gpu-warp-sync.md) |

---

## Quick Reference

### TMA Descriptor Creation

```mojo
# nocompile

# 2D TMA for matrix tiles
var a_tma_desc = tma_create_descriptor_2d[a_type, a_tma_shape, a_swizzle](
    a_ptr,
    StaticTuple[Int32, 2](K, M),  # Global shape
    StaticTuple[Int64, 1](K * size_of[a_type]()),  # Stride
)

# 3D TMA for batched operations
tma_op.async_copy_3d(
    dest=smem_tile_3d,
    barrier=barrier,
    coords=(batch_idx, k_coord, row_coord),
)
```

### Bank Conflict Padding Trick

```mojo
# nocompile

from gpu.memory import AddressSpace
from memory import stack_allocation

# BAD: Bank conflicts on column access
var tile = stack_allocation[32 * 32, Float32, address_space=AddressSpace.SHARED]()
tile[row * 32 + col]

# GOOD: Add padding to avoid conflicts
var tile = stack_allocation[32 * 33, Float32, address_space=AddressSpace.SHARED]()  # 33 instead of 32
tile[row * 33 + col]  # Shifted access pattern
```

#### Shared Memory Configuration by Architecture

Different GPU architectures have different shared memory capacities and configuration options:

| Architecture | Shared Mem/SM | Configurable | Max per Block | Notes |
|---|---|---|---|---|
| SM80 (A100) | 164 KB | L1/Shared split | 163 KB | `shared_mem_bytes` in `enqueue_function` |
| SM89 (L40S) | 100 KB | L1/Shared split | 99 KB | `shared_mem_bytes` in `enqueue_function` |
| SM90 (H100) | 228 KB | L1/Shared split | 227 KB | `shared_mem_bytes` in `enqueue_function` |
| CDNA3 (MI300X) | 64 KB LDS | Fixed | 64 KB | Not dynamically configurable |
| Apple M3 | 32 KB | Fixed | 32 KB | Compile-time `stack_allocation` only |

**Requesting dynamic shared memory (NVIDIA):**

```mojo
# nocompile

# Request extra shared memory beyond the kernel's static allocation
ctx.enqueue_function[my_kernel](
    args...,
    grid_dim=grid, block_dim=block,
    shared_mem_bytes=49152,  # 48 KB of dynamic shared memory
)
```

> **Apple Silicon:** Shared memory size must be known at compile time. Use `stack_allocation` with a `comptime` size constant. See the Apple Metal section in [`gpu-fundamentals.md`](gpu-fundamentals.md) for the `_alloc_shared` pattern that handles both NVIDIA (dynamic) and Apple (static) allocation.

### Performance Impact

| Optimization | Without | With | Improvement |
|--------------|---------|------|-------------|
| TMA vs manual loading | Baseline | 2-3x | 100-200% |
| Swizzle vs linear | Bank conflicts | Conflict-free | 2-4x |
| Prefetch pipeline | ~60% BW | ~85% BW | ~40% |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **TMA APIs** | Available (v26.1+) | `from layout.tma_async import TMATensorTile` |
| **Memory allocation** | `from memory import alloc; alloc[T](n)` | v26.1+ |
| **Shared memory** | `stack_allocation[N, T, address_space=AddressSpace.SHARED]()` | Stable |
| **LayoutTensor shared** | `LayoutTensor[..., address_space=AddressSpace.SHARED]` | Available (v26.1+) |
| **Swizzle patterns** | `from layout.swizzle import make_swizzle` | Available (v26.1+) |
| **TensorMapSwizzle** | `from gpu.host.nvidia.tma import TensorMapSwizzle` | Available (v26.1+) |
| **Async copy** | `cp_async_*` intrinsics | Stable |

### Key Notes

- **Hardware requirement**: TMA requires SM90+ (Hopper architecture). On older GPUs, use cp.async or manual loads.
- Use `LayoutTensor[..., address_space=AddressSpace.SHARED]` for type-safe multi-dimensional shared memory access.
- Use `stack_allocation` with `AddressSpace.SHARED` for simple flat shared memory allocation.

**Basic shared memory pattern (v26.1+):**
```mojo
# nocompile
from gpu.memory import AddressSpace
from memory import stack_allocation
from gpu import barrier

# Allocate shared memory tile using stack_allocation
var smem = stack_allocation[TILE_SIZE, Float32, address_space=AddressSpace.SHARED]()

# Manual load with coalescing
var tid = gpu.thread_idx()
smem[tid] = global_ptr[tid + block_offset]
barrier()  # Sync before use
```

**LayoutTensor shared memory pattern (v26.1+):**
```mojo
# nocompile
from layout import LayoutTensor, Layout
from gpu.memory import AddressSpace

# Type-safe shared memory with LayoutTensor
var tile = LayoutTensor[
    DType.float32,
    Layout.row_major(TILE_M, TILE_K),
    MutAnyOrigin,
    address_space=AddressSpace.SHARED,
].stack_allocation()

# Clean multi-dimensional indexing
tile[row, col] = global_tensor[global_row, global_col]
barrier()
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `TMA descriptor invalid` | Wrong tensor dimensions or swizzle mode | Verify tensor shape matches descriptor; use correct swizzle for access pattern |
| `shared memory bank conflict` | Multiple threads access same bank | Apply swizzle pattern: `XOR` bits or 128B interleaving |
| `uncoalesced global memory access` | Non-contiguous thread access | Ensure thread 0-31 access consecutive 128B region |
| `TMA out of bounds` | Copy region exceeds tensor | Check bounds; TMA copies exact rectangle, no automatic clamping |
| `prefetch stall` | Prefetch not issued early enough | Issue prefetch 2-3 iterations ahead; overlap with compute |
| `async copy incomplete` | Missing wait after cp.async | Call `cp_async_wait_all()` before using shared memory data |

---

## Related Patterns

- [`gpu-fundamentals.md`](gpu-fundamentals.md) — Memory hierarchy and coalescing basics
- [`gpu-warp-sync.md`](gpu-warp-sync.md) — Mbarrier patterns for TMA
- [`gpu-tensor-cores.md`](gpu-tensor-cores.md) — Memory layouts for tensor cores
- [`gpu-kernels.md`](gpu-kernels.md) — Double buffering and pipelines
- [`gpu-structured-kernels.md`](gpu-structured-kernels.md) — ScatterGather abstraction for data movement

---

## References

- [MAX Kernels](https://github.com/modular/modular/tree/main/max/kernels)
