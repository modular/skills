---
title: "CuTe DSL to Mojo Porting Guide"
description: Map NVIDIA CuTe DSL layout algebra, TiledCopy, and TiledMMA to Mojo LayoutTensor, TMA, and WGMMA equivalents
impact: HIGH
category: gpu
tags: [cute, cutlass, porting, layout, tensor-cores, tma, wgmma]
error_patterns:
  - "CuTe equivalent"
  - "CUTLASS to Mojo"
  - "layout algebra"
  - "TiledCopy"
  - "TiledMMA"
scenarios:
  - "Port CuTe DSL kernel to Mojo"
  - "Find Mojo equivalent of CuTe Layout"
  - "Convert CUTLASS GEMM to Mojo structured kernel"
  - "Map CuTe TiledCopy to Mojo TMA"
  - "Map CuTe TiledMMA to Mojo WGMMA"
---

# CuTe DSL to Mojo Porting Guide

**Category:** GPU | **Impact:** HIGH

Maps NVIDIA's CuTe DSL (part of CUTLASS 3.x) concepts to their Mojo equivalents. CuTe and Mojo's layout system share the same mathematical foundation — layout algebra — making porting conceptually straightforward.

> **Version:** These patterns target Mojo nightly (v26.2+). All APIs shown are available in the public nightly toolchain.

---

## Concept Mapping Overview

| CuTe Concept | Mojo Equivalent | Key Difference |
|--------------|-----------------|----------------|
| `Layout<Shape, Stride>` | `Layout(shape, stride)` | Mojo uses `IntTuple`; CuTe uses variadic templates |
| `Tensor<Engine, Layout>` | `LayoutTensor[dtype, layout, origin, address_space]` | Mojo encodes address space + dtype in type |
| `make_layout(shape, stride)` | `Layout(IntTuple(shape), IntTuple(stride))` | Same semantics, different syntax |
| `composition(A, B)` | `composition(a, b)` | Freestanding function: `from layout import composition` |
| `complement(layout, size)` | `complement(layout, size)` | Freestanding function: `from layout.layout import complement` |
| `logical_product(A, B)` | `logical_product(A, B)` | Available in `layout.layout` |
| `blocked_product(A, B)` | `blocked_product(A, B)` | Available in `layout.layout` |
| `TiledCopy` | `TMATensorTile` / `ScatterGather` | TMA replaces cp.async; ScatterGather for AMD |
| `TiledMMA` | `TensorCoreAsync` (SM90) / TCGen05 (SM100) | Wraps WGMMA/TCGen05 instructions |
| `local_partition` | `LayoutTensor.distribute[thread_layout]()` | Thread-level partitioning |
| `local_tile` | `LayoutTensor.tile[tile_shape]()` | Tile-level partitioning |
| `Swizzle<B,M,S>` | `Swizzle(bits=B, base=M, shift=S)` | Runtime args, not compile-time params |
| `Pipeline` | `RingBuffer` + `PipelineBarrier` | Context manager based |
| `PipelineState` | `PipelineState` | Same concept, Mojo struct |

---

## Layout Algebra

The mathematical foundation is identical in CuTe and Mojo: a layout maps logical coordinates to physical offsets using shape and stride tuples.

### CuTe: Layout Creation

```cpp
// CuTe: Layout with shape and stride
auto layout_2d = make_layout(
    make_shape(Int<64>{}, Int<32>{}),    // Shape: (64, 32)
    make_stride(Int<32>{}, Int<1>{})     // Stride: (32, 1) = row-major
);

// CuTe: Column-major
auto col_major = make_layout(
    make_shape(Int<64>{}, Int<32>{}),
    make_stride(Int<1>{}, Int<64>{})     // Stride: (1, 64) = col-major
);

// CuTe: Hierarchical layout
auto hierarchical = make_layout(
    make_shape(make_shape(4, 8), make_shape(2, 16)),
    make_stride(make_stride(16, 2), make_stride(1, 64))
);
```

### Mojo: Layout Creation

```mojo
from layout import Layout, IntTuple

# Mojo: Layout with shape and stride using IntTuple
var layout_2d = Layout(
    IntTuple(64, 32),   # Shape: (64, 32)
    IntTuple(32, 1),    # Stride: (32, 1) = row-major
)

# Mojo: Column-major
var col_major = Layout(
    IntTuple(64, 32),
    IntTuple(1, 64),    # Stride: (1, 64) = col-major
)

# Mojo: Hierarchical layout (nested IntTuples)
var hierarchical = Layout(
    IntTuple(IntTuple(4, 8), IntTuple(2, 16)),
    IntTuple(IntTuple(16, 2), IntTuple(1, 64)),
)

# Compile-time layout (most common in kernels)
comptime my_layout = Layout(IntTuple(64, 32), IntTuple(32, 1))
```

### Layout Operations

| CuTe | Mojo | Description |
|------|------|-------------|
| `size(layout)` | `layout.size()` | Total number of elements |
| `rank(layout)` | `layout.rank()` | Number of dimensions |
| `depth(layout)` | No direct equivalent | CuTe nesting depth has no Mojo API |
| `shape(layout)` | `layout.shape` | Shape tuple |
| `stride(layout)` | `layout.stride` | Stride tuple |
| `cosize(layout)` | `layout.cosize()` | Domain size (max index + 1) |
| `layout(coord)` | `layout(coord)` | Map coordinate → offset |
| `composition(A, B)` | `composition(a, b)` | Compose two layouts (freestanding function) |
| `complement(A, size)` | `complement(a, size)` | Freestanding function: `from layout.layout import complement` |
| `product(A, B)` | `logical_product(a, b)` | Logical product |
| `blocked_product(A, B)` | `blocked_product(a, b)` | Blocked product |
| `coalesce(layout)` | `coalesce(layout)` | Merge contiguous modes |
| `slice(layout, coord)` | `layout.slice(coord)` | Fix dimension, get sub-layout |

---

## Tensor = Pointer + Layout

### CuTe: Tensor Creation

```cpp
// CuTe: Create tensor from pointer and layout
auto gA = make_tensor(
    make_gmem_ptr(A_ptr),                    // Global memory pointer
    make_layout(make_shape(M, K), make_stride(K, 1))  // Row-major (M, K)
);

// CuTe: Shared memory tensor
auto sA = make_tensor(
    make_smem_ptr(smem_ptr),
    make_layout(make_shape(BM, BK), make_stride(BK, 1))
);

// CuTe: Register tensor
auto rA = make_tensor<float>(make_layout(make_shape(4, 2)));
```

### Mojo: LayoutTensor Creation

```mojo
# nocompile
from layout import Layout, LayoutTensor
from memory import UnsafePointer

# Mojo: Global memory tensor
var gA = LayoutTensor[
    DType.float16,
    Layout(IntTuple(M, K), IntTuple(K, 1)),  # Row-major
](global_ptr)

# Mojo: Shared memory tensor (address space encoded in type)
var sA = LayoutTensor[
    DType.float16,
    Layout(IntTuple(BM, BK), IntTuple(BK, 1)),
    MutAnyOrigin,
    address_space=AddressSpace.SHARED,
](smem_ptr)

# Mojo: Register tensor (local address space)
var rA_ptr = stack_allocation[8, DType.float32, address_space=AddressSpace.LOCAL]()
var rA = LayoutTensor[
    DType.float32,
    Layout(IntTuple(4, 2), IntTuple(2, 1)),
    MutAnyOrigin,
    address_space=AddressSpace.LOCAL,
](rA_ptr)
```

### Key Difference: Address Space in Type

CuTe uses `gmem_ptr`, `smem_ptr`, `rmem_ptr` wrappers. Mojo encodes the address space directly in the `LayoutTensor` type parameter:

```mojo
# nocompile
# Global memory (DRAM)
LayoutTensor[dtype, layout, origin]  # address_space defaults to GENERIC

# Shared memory (SMEM / LDS)
LayoutTensor[dtype, layout, origin, address_space=AddressSpace.SHARED]

# Registers (local)
LayoutTensor[dtype, layout, origin, address_space=AddressSpace.LOCAL]
```

Type aliases simplify this:

```mojo
from linalg.structuring import SMemTile, RegTile

# SMemTile = LayoutTensor[..., address_space=AddressSpace.SHARED]
# RegTile  = LayoutTensor[..., address_space=AddressSpace.LOCAL]
```

---

## Thread Partitioning

### CuTe: local_partition

```cpp
// CuTe: Partition tensor among threads in a thread block
auto tAgA = local_partition(gA, tiled_copy, thread_idx, Step<_1, _0>{});
// Each thread gets a view of its portion of gA
```

### Mojo: LayoutTensor.distribute

```mojo
# nocompile
# Mojo: distribute tensor elements among threads
comptime thread_layout = Layout(IntTuple(32, 4), IntTuple(4, 1))  # 128 threads

# Each thread gets its slice of the tensor based on thread_layout
var my_slice = tensor.distribute[thread_layout]()
```

### CuTe: local_tile

```cpp
// CuTe: Tile a tensor into sub-tiles
auto block_gA = local_tile(gA, make_shape(BM, BK), block_coord);
```

### Mojo: LayoutTensor.tile

```mojo
# nocompile
# Mojo: Partition tensor into tiles
var tiled = tensor.tile[BM, BK]()
var block_tile = tiled[block_m, block_k]  # Get specific tile
```

---

## Swizzle: Bank-Conflict-Free Shared Memory

Both CuTe and Mojo use the same `Swizzle<B, M, S>` parameterization to avoid shared memory bank conflicts.

### CuTe: Swizzle

```cpp
// CuTe: Swizzle for 128-byte access pattern
using SmemSwizzle = Swizzle<3, 3, 3>;  // B=3, M=3, S=3
auto swizzled_layout = composition(SmemSwizzle{}, base_layout);
```

### Mojo: Swizzle

```mojo
from layout.swizzle import Swizzle, make_swizzle

# Mojo: Same parameterization, but uses runtime constructor args
comptime swizzle = Swizzle(bits=3, base=3, shift=3)  # bits=3, base=3, shift=3

# Or use the tile_layout_k_major helper which applies swizzle automatically
from layout.tensor_core_async import tile_layout_k_major
from gpu.host.nvidia.tma import TensorMapSwizzle

comptime smem_layout = tile_layout_k_major[
    DType.float16, 128, 64, TensorMapSwizzle.SWIZZLE_128B
]()
# This creates a k-major layout with 128B swizzle — ready for WGMMA
```

### TensorMapSwizzle Options

| CuTe Swizzle | Mojo TensorMapSwizzle | Access Width |
|--------------|----------------------|--------------|
| `Swizzle<0,4,3>` | `SWIZZLE_32B` | 32 bytes |
| `Swizzle<1,3,3>` | `SWIZZLE_64B` | 64 bytes |
| `Swizzle<2,3,3>` or `Swizzle<3,3,3>` | `SWIZZLE_128B` | 128 bytes |
| No swizzle | `SWIZZLE_NONE` | — |

---

## TiledCopy → TMA / ScatterGather

CuTe's `TiledCopy` abstracts cooperative data movement. In Mojo, this maps to different mechanisms depending on the hardware.

### CuTe: TiledCopy with TMA

```cpp
// CuTe: Create TMA-based tiled copy
auto tma_load = make_tma_copy(
    SM90_TMA_LOAD{},
    gA,                     // Source tensor (global)
    sA_layout,              // Destination layout (shared)
    make_shape(BM, BK),     // Tile shape
    size<0>(cluster_shape)  // Multicast size
);

// Execute the copy
cute::copy(tma_load, tAgA(_, _, k_tile), tAsA(_, _, stage));
```

### Mojo: TMATensorTile (NVIDIA SM90+)

```mojo
# nocompile
from layout.tma_async import TMATensorTile, SharedMemBarrier

# Mojo: TMA descriptor wraps the TMA hardware unit
var tma_a = TMATensorTile[DType.float16, tile_layout, desc_layout](
    tensor_ptr=a_global_ptr,
    tensor_shape=(M, K),
    swizzle=TensorMapSwizzle.SWIZZLE_128B,
)

# Load tile asynchronously: global → shared
fn producer_load_tile(
    tma: TMATensorTile[...],
    dst: SMemTile[...],
    barrier: UnsafePointer[SharedMemBarrier, address_space=AddressSpace.SHARED],
    coords: Tuple[UInt, UInt],
):
    barrier[].arrive_expect_tx(tile_bytes)
    tma.async_load(dst, barrier, coords)
```

### Mojo: ScatterGather (AMD)

```mojo
# nocompile
from linalg.structuring import ScatterGatherAmd

# AMD: ScatterGather for DRAM ↔ register data movement (no shared memory intermediate)
var scatter_gather = ScatterGatherAmd[thread_layout](global_tensor)

# Copy: global memory → registers (direct, no shared memory)
scatter_gather.copy(dst_reg_tile, src_gmem_tile)

# Copy: registers → global memory
scatter_gather.copy(dst_gmem_tile, src_reg_tile)
```

### CuTe cp.async → Mojo async_copy (SM80)

```mojo
# nocompile
from gpu.memory import async_copy, async_copy_commit_group, async_copy_wait_group

# Mojo: cp.async equivalent (for SM80 / Ampere)
async_copy(dst_smem_ptr, src_gmem_ptr, bytes)
async_copy_commit_group()        # Commit outstanding copies
async_copy_wait_group(0)         # Wait for all committed groups
```

---

## TiledMMA → TensorCoreAsync / TCGen05

CuTe's `TiledMMA` describes cooperative matrix-multiply-accumulate. In Mojo, this maps to hardware-specific wrappers.

### CuTe: TiledMMA

```cpp
// CuTe: Define MMA operation
using MMA_Atom = SM90_64x128x16_F32F16F16_SS;  // SM90 WGMMA
using TiledMMA = TiledMMA<
    MMA_Atom,
    Layout<Shape<_2, _1, _1>>,  // Thread layout
    Tile<_64, _128, _16>        // Tile shape
>;

// Partition tensors for MMA
auto tCsA = tiled_mma.partition_A(sA);
auto tCsB = tiled_mma.partition_B(sB);
auto tCrC = tiled_mma.partition_C(accum);

// Execute MMA
cute::gemm(tiled_mma, tCsA(_, _, k), tCsB(_, _, k), tCrC);
```

### Mojo: TensorCoreAsync (SM90 WGMMA)

```mojo
# nocompile
from layout.tensor_core_async import TensorCoreAsync, warpgroup_fence, tile_layout_k_major

# Mojo: Configure WGMMA operation
comptime WgmmaOp = TensorCoreAsync[
    DType.float32,    # Accumulator type (c_type)
    DType.float16,    # A matrix type (a_type)
    DType.float16,    # B matrix type (b_type)
    mma_shape=Index(64, 128, 16),  # MMA instruction shape (M, N, K)
    a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
    transpose_b=True,
]

# Execute MMA: shared memory (A, B) → registers (accumulator)
fn compute_tiles(
    a_tile: SMemTile[DType.float16, a_smem_layout, ...],
    b_tile: SMemTile[DType.float16, b_smem_layout, ...],
    accum: RegTile[DType.float32, accum_layout, ...],
):
    warpgroup_fence()
    WgmmaOp.mma(accum, a_tile, b_tile)
    warpgroup_fence()
```

### CuTe MMA Atoms → Mojo

| CuTe MMA Atom | Mojo Equivalent | GPU |
|----------------|-----------------|-----|
| `SM90_64x128x16_F32F16F16_SS` | `TensorCoreAsync[f32, f16, f16, Index(64, 128, 16)]` | H100 |
| `SM90_64x64x16_F32F16F16_SS` | `TensorCoreAsync[f32, f16, f16, Index(64, 64, 16)]` | H100 |
| `SM90_64x128x32_F32E4M3E4M3_SS` | `TensorCoreAsync[f32, f8e4m3, f8e4m3, Index(64, 128, 32)]` | H100 |
| SM100 `tcgen05.mma` | TCGen05 MMA API | B200 |

---

## Pipeline → RingBuffer

CuTe uses `PipelineTmaAsync` for multi-stage pipelining. Mojo uses `RingBuffer` with context managers.

### CuTe: Pipeline

```cpp
// CuTe: Multi-stage pipeline
using MainloopPipeline = PipelineTmaAsync<NUM_STAGES>;
auto pipeline = MainloopPipeline(smem, params);
PipelineState smem_pipe_read;
PipelineState smem_pipe_write;

// Producer
pipeline.producer_acquire(smem_pipe_write);
cute::copy(tma_load_a, ..., smem_pipe_write.index());
pipeline.producer_commit(smem_pipe_write);
++smem_pipe_write;

// Consumer
pipeline.consumer_wait(smem_pipe_read);
cute::gemm(tiled_mma, sA(_, _, smem_pipe_read.index()), ...);
pipeline.consumer_release(smem_pipe_read);
++smem_pipe_read;
```

### Mojo: RingBuffer with Context Managers

```mojo
# nocompile
# Mojo: Ring buffer manages pipeline stages automatically
var ring_buffer = RingBuffer[...](full_barriers, empty_barriers, a_tiles, b_tiles)

# Producer — context manager handles acquire/commit
with ring_buffer.producer() as producer:
    for k in range(num_k_tiles):
        with producer.get_tiles() as tiles:
            # tiles.barrier, tiles.a_tile_array, tiles.b_tile_array
            tma_a.async_load(tiles.a_tile_array[0], tiles.barrier, coords)
            # __exit__ automatically signals tile is ready

# Consumer — context manager handles wait/release
with ring_buffer.consumer() as consumer:
    for k in range(num_k_tiles):
        with consumer.get_tiles() as tiles:
            # tiles.a_tile, tiles.b_tile available after wait
            WgmmaOp.mma(accum, tiles.a_tile, tiles.b_tile)
            # __exit__ automatically releases the stage
```

### Pipeline Mapping

| CuTe Pipeline | Mojo RingBuffer |
|---------------|-----------------|
| `PipelineTmaAsync<STAGES>` | `RingBuffer[num_stages, ...]` |
| `pipeline.producer_acquire(state)` | `with producer.get_tiles() as tiles:` (implicit) |
| `pipeline.producer_commit(state)` | Context manager `__exit__` (implicit) |
| `pipeline.consumer_wait(state)` | `with consumer.get_tiles() as tiles:` (implicit) |
| `pipeline.consumer_release(state)` | Context manager `__exit__` (implicit) |
| `PipelineState` | Managed internally by `RingBuffer` |
| `pipeline.producer_tail(state)` | Handled in `RingBufferProducer.__exit__()` |

---

## Complete Example: CuTe GEMM → Mojo GEMM

### CuTe GEMM (Simplified)

```cpp
// Simplified CuTe GEMM kernel structure
template <class TiledMMA, class TiledCopy, int STAGES>
__global__ void gemm_kernel(float* C, half* A, half* B, int M, int N, int K) {
    // 1. Create tensors from pointers + layouts
    auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, 1));
    auto gB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, 1));
    auto gC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, 1));

    // 2. Tile for this thread block
    auto bA = local_tile(gA, make_shape(BM, BK), make_coord(blockIdx.x, _));
    auto bB = local_tile(gB, make_shape(BN, BK), make_coord(blockIdx.y, _));

    // 3. Shared memory tiles
    __shared__ half sA_data[STAGES][BM * BK];
    __shared__ half sB_data[STAGES][BN * BK];
    auto sA = make_tensor(make_smem_ptr(sA_data), sA_layout);
    auto sB = make_tensor(make_smem_ptr(sB_data), sB_layout);

    // 4. Partition for this thread's MMA
    auto tCrC = tiled_mma.partition_C(accum);
    clear(tCrC);

    // 5. Main loop: load + compute with pipeline
    for (int k = 0; k < K_tiles; k++) {
        // Producer: TMA load
        cute::copy(tma_a, bA(_, _, k), sA(_, _, stage));
        cute::copy(tma_b, bB(_, _, k), sB(_, _, stage));
        // Consumer: WGMMA
        cute::gemm(tiled_mma, sA(_, _, stage), sB(_, _, stage), tCrC);
    }

    // 6. Epilogue: store results
    cute::copy(tCrC, bC);
}
```

### Mojo GEMM (Using Structured Kernel Architecture)

```mojo
from gpu import thread_idx, block_idx, WARP_SIZE
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from layout import Layout, LayoutTensor, IntTuple
from layout.tensor_core_async import TensorCoreAsync, warpgroup_fence, tile_layout_k_major
from layout.tma_async import TMATensorTile, SharedMemBarrier
from gpu.host.nvidia.tma import TensorMapSwizzle
from linalg.structuring import (
    SMemTile, RegTile, PipelineBarrier,
    NVIDIASharedMemoryManager as SharedMemoryManager,
)

struct GemmKernel[
    BM: Int, BN: Int, BK: Int,             # Block tile shape
    wgmma_m: Int, wgmma_n: Int, wgmma_k: Int,  # MMA shape
    num_stages: Int,                         # Pipeline stages
]:
    # Layouts
    comptime a_smem_layout = tile_layout_k_major[
        DType.float16, BM, BK, TensorMapSwizzle.SWIZZLE_128B
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        DType.float16, BN, BK, TensorMapSwizzle.SWIZZLE_128B
    ]()

    # MMA operation
    comptime WgmmaOp = TensorCoreAsync[
        DType.float32, DType.float16, DType.float16,
        mma_shape=Index(wgmma_m, wgmma_n, wgmma_k),
        a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
        b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
        transpose_b=True,
    ]

    @staticmethod
    fn kernel(
        tma_a: TMATensorTile[...],
        tma_b: TMATensorTile[...],
        c_ptr: UnsafePointer[Float32],
        M: Int, N: Int, K: Int,
    ):
        # 1. Set up shared memory manager
        var smm = SharedMemoryManager()
        var a_tiles = smm.build[ATileArray]()    # Multi-stage A tiles
        var b_tiles = smm.build[BTileArray]()    # Multi-stage B tiles
        var full_mbar = smm.build[PipelineBarrier]()
        var empty_mbar = smm.build[PipelineBarrier]()

        # 2. Tile coordinates for this block
        var block_m = block_idx.x
        var block_n = block_idx.y
        var num_k_tiles = (K + BK - 1) // BK

        # 3. Warp group roles
        var wg_id = thread_idx.x // 128

        if wg_id == 0:
            # === PRODUCER (ScatterGather role) ===
            warpgroup_reg_dealloc[40]()
            for k in range(num_k_tiles):
                var stage = k % num_stages
                # Arrive at barrier with expected bytes
                full_mbar[stage].arrive_expect_tx(tile_bytes)
                # TMA async load: global → shared
                tma_a.async_load(a_tiles[stage], full_mbar[stage], (block_m, k))
                tma_b.async_load(b_tiles[stage], full_mbar[stage], (block_n, k))

        else:
            # === CONSUMER (TileOp role) ===
            warpgroup_reg_alloc[232]()
            var accum = RegTile[DType.float32, accum_layout]()
            accum.zero()

            for k in range(num_k_tiles):
                var stage = k % num_stages
                full_mbar[stage].wait()  # Wait for producer

                # Tensor core MMA
                warpgroup_fence()
                Self.WgmmaOp.mma(accum, a_tiles[stage], b_tiles[stage])
                warpgroup_fence()

                empty_mbar[stage].arrive()  # Signal tiles consumed

            # 4. Epilogue: store accumulators to global memory
            # ... (via shared memory + TMA store or direct register store)
```

---

## Key Architectural Differences

| Aspect | CuTe/CUTLASS | Mojo Structured Kernels |
|--------|-------------|------------------------|
| **Abstraction style** | Template metaprogramming | `comptime` parameters + structs |
| **Memory management** | Manual `__shared__` allocation | `SharedMemoryManager` bump allocator |
| **Pipeline** | `PipelineTmaAsync` + manual state | `RingBuffer` with context managers |
| **Error handling** | Compile-time static asserts | `comptime assert` + Mojo error handling |
| **Multi-backend** | CUTLASS has separate CUDA/HIP files | Same Mojo code compiles for NVIDIA/AMD via `@parameter` |
| **Swizzle** | Composed into layout | Applied via `tile_layout_k_major` helper or `Swizzle` struct |
| **Warp specialization** | Manual `if (warp_id == ...)` | Same pattern, but with `warpgroup_reg_alloc/dealloc` |
| **Address space** | Pointer wrappers (`smem_ptr`) | Type parameter (`address_space=AddressSpace.SHARED`) |

---

## Version-Specific Features

### Nightly (v26.2+)

All APIs in this pattern require the Mojo nightly toolchain (v26.2+). This includes:

- `LayoutTensor`, `Layout`, `IntTuple` from the `layout` module
- Layout algebra operations: `composition`, `complement`, `logical_product`, `blocked_product`, `coalesce`
- `TensorCoreAsync` (SM90 WGMMA wrapper) from `layout.tensor_core_async`
- `TMATensorTile` and `SharedMemBarrier` from `layout.tma_async`
- `RingBuffer` and `PipelineBarrier` from `linalg.structuring`
- `ScatterGatherAmd` from `linalg.structuring`
- `Swizzle` and `make_swizzle` from `layout.swizzle`
- `tile_layout_k_major` helper from `layout.tensor_core_async`
- TCGen05 MMA APIs for SM100 (Blackwell) from `gpu.compute.arch.tcgen05`

### Stable (v26.1)

Core layout algebra (`Layout`, `IntTuple`, `composition`, `complement`) and `LayoutTensor` are available in v26.1 stable. SM90 WGMMA patterns via `TensorCoreAsync` and TMA via `TMATensorTile` are also available in v26.1. SM100 UMMA/TCGen05 patterns are nightly-only (v26.2+). The `RingBuffer` context manager API and `ScatterGatherAmd` are available in v26.1. If porting CuTe code on stable, the basic layout algebra and SM90 tensor core patterns will work; SM100-specific features require nightly.

---

## Related Patterns

- [`gpu-porting-cuda.md`](gpu-porting-cuda.md) — Basic CUDA → Mojo porting (start here)
- [`gpu-porting-rocm.md`](gpu-porting-rocm.md) — ROCm/HIP → Mojo porting
- [`gpu-structured-kernels.md`](gpu-structured-kernels.md) — Full ScatterGather/RingBuffer/TileOp architecture
- [`gpu-tensor-cores.md`](gpu-tensor-cores.md) — SM90/SM100 tensor core details
- [`gpu-memory-access.md`](gpu-memory-access.md) — TMA, swizzle, prefetch patterns
