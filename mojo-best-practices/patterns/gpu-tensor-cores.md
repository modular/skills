---
title: Tensor Core Programming for SM90 and SM100
description: WGMMA (SM90), UMMA/TCGEN05 (SM100), tensor memory, and descriptor patterns for maximum tensor core throughput
impact: CRITICAL
category: gpu
maturity: beta
tags: [gpu, tensor-core, wgmma, umma, tcgen05, sm90, sm100, mma, tmem]
error_patterns:
  - "WGMMA error"
  - "descriptor invalid"
  - "MMA shape mismatch"
  - "tensor core"
  - "UMMA"
  - "TCGEN05"
scenarios:
  - "Use tensor cores for matrix multiply"
  - "Write WGMMA kernel for H100"
  - "Implement SM100 UMMA pattern"
  - "Create TMA descriptors for WGMMA"
consolidates:
  - gpu-tensor-core-sm90-sm100.md
  - gpu-wgmma-descriptors.md
  - gpu-sm100-tcgen05.md
  - gpu-layout-tensor.md
---

# Tensor Core Programming for SM90 and SM100

**Category:** gpu | **Impact:** CRITICAL

Tensor cores provide 10-100x compute throughput for matrix operations. SM90 (Hopper) uses WGMMA with register accumulators; SM100 (Blackwell) introduces UMMA with dedicated Tensor Memory (TMEM). Proper usage requires correct descriptors, synchronization, and memory layouts.

## API Availability

> **Note:** All APIs listed below are available in the Mojo nightly toolchain (v26.2+). Code examples are documentation snippets — adapt import paths and parameters for your use case. See also [`gpu-porting-cuda.md`](gpu-porting-cuda.md) and [`gpu-porting-cute.md`](gpu-porting-cute.md) for porting guides.

| API | Import Path | Notes |
|-----|-------------|-------|
| `stack_allocation[N, T, address_space=AddressSpace.SHARED]()` | `from memory import stack_allocation` | Shared memory allocation |
| `AddressSpace` | `from gpu.memory import AddressSpace` | Address space enum (SHARED, GENERIC, etc.) |
| `warpgroup_reg_alloc`, `warpgroup_reg_dealloc` | `from gpu.intrinsics import ...` | SM90+ warp group register management |
| `warpgroup_fence` | `from layout.tensor_core_async import warpgroup_fence` | SM90+ warp group fence (defined in `tensor_core_async`, not in `gpu.intrinsics`) |
| `WGMMADescriptor` | `from gpu.compute.mma import WGMMADescriptor` | WGMMA descriptor creation |
| `wgmma_async`, `wgmma_fence_aligned` | `from gpu.compute.mma import ...` | WGMMA async operations |
| `wgmma_commit_group_sync`, `wgmma_wait_group_sync` | `from gpu.compute.mma import ...` | WGMMA synchronization |
| `tcgen05_alloc`, `tcgen05_ld`, etc. | `from gpu.compute.arch.tcgen05 import ...` | SM100 TMEM allocation/loading |
| `UMMAKind`, `UMMAInsDescriptor`, `mma` | `from gpu.compute.arch.mma_nvidia_sm100 import ...` | SM100 UMMA operations |
| `LayoutTensor`, `Layout` | `from layout import LayoutTensor, Layout` | Type-safe tensor with compile-time layout |
| `TensorCoreAsync` | `from layout.tensor_core_async import TensorCoreAsync` | High-level tensor core wrapper |
| `TensorMapSwizzle` | `from gpu.host.nvidia.tma import TensorMapSwizzle` | TMA swizzle modes |

---

## Core Concepts

### Architecture Comparison

| Aspect | SM90 WGMMA | SM100 UMMA/TCGEN05 |
|--------|------------|---------------------|
| **Accumulator** | Registers | Tensor Memory (TMEM) |
| **Warp grouping** | 4 warps (128 threads) | Flexible CTA groups |
| **Register pressure** | High (need reg alloc) | Low (uses TMEM) |
| **Synchronization** | warpgroup_fence | mma_arrive barrier |
| **Data source** | Shared memory | Shared memory |

### MMA Shape Options

**SM90 WGMMA Shapes (M x N x K):**

| M | N | K | Data Type |
|---|---|---|-----------|
| 64 | 8-256 (multiples of 8) | 8 | TF32 |
| 64 | 8-256 (multiples of 8) | 16 | BF16/F16 |
| 64 | 8-256 (multiples of 8) | 32 | FP8/INT8 |

**SM100 UMMA Shapes:**

| UMMAKind | cta_group=1 | cta_group=2 | K |
|----------|-------------|-------------|---|
| KIND_TF32 | M: 64,128; N: 8-256 | M: 128,256; N: 32-256 | 8 |
| KIND_F16 | M: 64,128; N: 8-256 | M: 128,256; N: 32-256 | 16 |
| KIND_F8F6F4 | M: 64,128; N: 8-256 | M: 128,256; N: 32-256 | 32 |

---

## Common Patterns

### SM90 WGMMA Pattern

**When:** H100/Hopper GPU matrix multiplication


```mojo
# nocompile

from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from layout.tensor_core_async import warpgroup_fence
from gpu.compute.mma import (
    WGMMADescriptor, wgmma_async, wgmma_fence_aligned,
    wgmma_commit_group_sync, wgmma_wait_group_sync,
)

fn correct_wgmma_kernel[
    WMMA_M: Int, WMMA_N: Int, WMMA_K: Int,
](
    a_smem: LayoutTensor[DType.bfloat16, ..., address_space=AddressSpace.SHARED],
    b_smem: LayoutTensor[DType.bfloat16, ..., address_space=AddressSpace.SHARED],
):
    # Initialize accumulator registers
    comptime num_output_regs = WMMA_M * WMMA_N // 128
    var c_reg = SIMD[DType.float32, num_output_regs](0)

    for k_i in range(K // WMMA_K):
        barrier()  # Ensure shared memory is populated

        # Create descriptors from shared memory
        var mat_a_desc = _lhs_descriptor(a_smem)
        var mat_b_desc = _rhs_descriptor[transposed=False](b_smem)

        # Step 1: FENCE - ensures shared memory writes are visible
        wgmma_fence_aligned()

        # Step 2: Issue async MMA operation
        c_reg = wgmma_async[
            WMMA_M, WMMA_N, WMMA_K,
            a_type=DType.bfloat16, b_type=DType.bfloat16,
        ](mat_a_desc, mat_b_desc, c_reg)

        # Step 3: COMMIT - submits the operation group
        wgmma_commit_group_sync()

        # Step 4: WAIT - blocks until computation completes
        wgmma_wait_group_sync()

    # Now safe to use c_reg results
```

### WGMMA Descriptor Structure


The `WGMMADescriptor` encodes shared memory operand information in a 64-bit value:

| Bit field | Size | Description |
|-----------|------|-------------|
| 0-13 | 14 | Base address in shared memory |
| 16-29 | 14 | LBO: leading dimension byte offset |
| 32-45 | 14 | SBO: stride dimension byte offset |
| 49-51 | 3 | Matrix base offset |
| 62-63 | 2 | Swizzle mode: 0=none, 1=128B, 2=64B, 3=32B |

```mojo
# nocompile

# Create descriptor with explicit offsets
var desc = WGMMADescriptor[DType.bfloat16].create[
    stride_byte_offset=SBO,
    leading_byte_offset=LBO,
    swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
](smem_ptr)

# Offset descriptor for different tiles
var desc_next_tile = desc + tile_byte_offset
```

### SM100 TCGEN05 Pattern

**When:** B100/B200/Blackwell GPU matrix multiplication


```mojo
from gpu.compute.arch.tcgen05 import (
    tcgen05_alloc, tcgen05_dealloc, tcgen05_ld, tcgen05_load_wait,
    tcgen05_release_allocation_lock,
)
from gpu.compute.arch.mma_nvidia_sm100 import (
    UMMAKind, UMMAInsDescriptor, MMASmemDescriptor, mma, mma_arrive,
)

fn sm100_tcgen05_mma[
    MMA_M: Int = 128,
    MMA_N: Int = 128,
    cta_group: Int = 2,
](
    a_smem: LayoutTensor[DType.bfloat16, ..., address_space=AddressSpace.SHARED],
    b_smem: LayoutTensor[DType.bfloat16, ..., address_space=AddressSpace.SHARED],
    mbar_ptr: UnsafePointer[Int64, address_space=AddressSpace.SHARED],
):
    # Step 1: Allocate TMEM (once per kernel, by MMA warp)
    var tmem_addr_storage = SMemArray[UInt32, 1]()
    tcgen05_alloc[cta_group](tmem_addr_storage.ptr, 512)  # 512 columns max
    syncwarp()
    var tmem_addr = tmem_addr_storage.ptr[0]

    # Step 2: Create instruction descriptor
    comptime idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        DType.float32,      # d_type (accumulator)
        DType.bfloat16,     # a_type
        DType.bfloat16,     # b_type
        Index(MMA_M, MMA_N),
        transpose_b=True,
    ]()

    # Step 3: Create shared memory descriptors
    var a_desc = MMASmemDescriptor.create[
        stride_byte_offset=SBO_A,
        leading_byte_offset=LBO_A,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
    ](a_smem.ptr)

    var b_desc = MMASmemDescriptor.create[...](b_smem.ptr)

    # Step 4: Issue TCGEN05 MMA (accumulates to TMEM)
    mma[UMMAKind.KIND_F16, cta_group, c_scale=0](  # c_scale=0 initializes
        a_desc, b_desc, tmem_addr, idesc
    )

    # Step 5: Signal completion via barrier
    mma_arrive[cta_group](mbar_ptr)

    # Step 6: Load results from TMEM when ready
    tcgen05_release_allocation_lock[cta_group]()
    # ... wait for barrier ...

    comptime TMEM_LOWER_ROW_OFFSET: UInt32 = 16 << 16
    var c_upper = tcgen05_ld[datapaths=16, bits=256, repeat=4, dtype=DType.float32](
        tmem_addr
    )
    var c_lower = tcgen05_ld[datapaths=16, bits=256, repeat=4, dtype=DType.float32](
        tmem_addr + TMEM_LOWER_ROW_OFFSET
    )
    tcgen05_load_wait()

    # Step 7: Deallocate TMEM
    tcgen05_dealloc[cta_group](tmem_addr, 512)
```

### Tensor Memory (TMEM) Lifecycle


```mojo
# nocompile

# 1. Allocate (MMA warp, once per kernel)
tcgen05_alloc[cta_group](smem_ptr, num_cols)
syncwarp()

# 2. Use in MMA operations
mma[kind, cta_group](a_desc, b_desc, tmem_addr, idesc, c_scale=...)

# 3. Release lock before waiting for epilogue
tcgen05_release_allocation_lock[cta_group]()

# 4. Load results (after MMA completion)
var result = tcgen05_ld[...](tmem_addr)
tcgen05_load_wait()

# 5. Deallocate (end of kernel)
tcgen05_dealloc[cta_group](tmem_addr, num_cols)
```

### Memory Layout for Tensor Cores

**K-major layouts for optimal tensor core access:**


```mojo
from layout.tensor_core_async import tile_layout_k_major, tile_layout_mn_major
from gpu.host.nvidia.tma import TensorMapSwizzle

# K-major layout for A matrix (row-major A)
comptime a_smem_layout = tile_layout_k_major[
    DType.bfloat16,
    BM=64,    # M dimension
    BK=16,    # K dimension
    swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
]()

# K-major layout for transposed B
comptime b_smem_layout = tile_layout_k_major[
    DType.bfloat16,
    BM=128,   # N dimension
    BK=16,    # K dimension
    swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
]()
```

### LayoutTensor for Safe Access

**When:** Any GPU kernel with multi-dimensional data


```mojo
# nocompile

from layout import LayoutTensor, Layout

fn matmul_kernel[
    dtype: DType, M: Int, N: Int, K: Int,
](
    A: LayoutTensor[dtype, Layout.row_major(M, K), MutAnyOrigin],
    B: LayoutTensor[dtype, Layout.row_major(K, N), MutAnyOrigin],
    C: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
):
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x

    # Type-safe indexing - layout handled automatically
    var sum = Scalar[dtype](0)
    for k in range(K):
        sum += A[row, k] * B[k, col]  # Clean and correct!
    C[row, col] = sum

# Creating LayoutTensors
var shared_tile = LayoutTensor[
    DType.float32,
    Layout.row_major(16, 16),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()

# Tiling and subtensors (zero-copy)
var A_tile = A.tile[TILE_M, K](block_idx.y, 0)
var B_tile = B.tile[K, TILE_N](0, block_idx.x)
```

### TensorCoreAsync Abstraction

**When:** Cleaner WGMMA code with automatic descriptor handling


```mojo
# nocompile

from layout.tensor_core_async import TensorCoreAsync, warpgroup_fence

comptime WgmmaOp = TensorCoreAsync[
    c_type=DType.float32,
    a_type=DType.bfloat16,
    b_type=DType.bfloat16,
    mma_shape=Index(64, 128, 16),
    a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
    transpose_b=True,
]

fn wgmma_with_abstraction(
    a_smem: LayoutTensor[..., address_space=AddressSpace.SHARED],
    b_smem: LayoutTensor[..., address_space=AddressSpace.SHARED],
    c_reg: LayoutTensor[..., address_space=AddressSpace.LOCAL],
):
    # Fence before wgmma batch
    warpgroup_fence(c_reg)

    # Arrive signals readiness
    WgmmaOp.arrive()

    # Issue WGMMA
    WgmmaOp.wgmma[num_warp_groups=1](a_smem, b_smem, c_reg)

    # Commit and wait
    WgmmaOp.commit_group()
    warpgroup_fence(c_reg)
    WgmmaOp.wait_group()
```

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| H100/Hopper GEMM | Use WGMMA with proper fence/commit/wait | - |
| B100/B200/Blackwell GEMM | Use TCGEN05 with TMEM | - |
| Custom tensor layouts | Use LayoutTensor with proper layouts | - |
| High-level abstraction | Use TensorCoreAsync struct | - |
| Mixed precision (FP8) | Use appropriate UMMAKind | - |
| SM80 and earlier | Use WMMA or standard MMA | - |

---

## Quick Reference

### SM90 WGMMA Protocol

1. `wgmma_fence_aligned()` - Ensure shared memory visible
2. `wgmma_async[...]()` - Issue async MMA
3. `wgmma_commit_group_sync()` - Submit operation group
4. `wgmma_wait_group_sync()` - Wait for completion

### SM100 TCGEN05 Protocol

1. `tcgen05_alloc[cta_group]()` - Allocate TMEM
2. `mma[kind, cta_group]()` - Issue MMA to TMEM
3. `mma_arrive[cta_group]()` - Signal completion
4. `tcgen05_release_allocation_lock()` - Release lock
5. `tcgen05_ld[...]()` + `tcgen05_load_wait()` - Load results
6. `tcgen05_dealloc()` - Free TMEM

### SM90 to SM100 Migration

| SM90 (WGMMA) | SM100 (TCGEN05) |
|--------------|-----------------|
| Register accumulators | TMEM accumulators |
| `wgmma_fence_aligned()` | Not needed |
| `wgmma_async()` | `mma[kind, cta_group]()` |
| `wgmma_commit_group_sync()` | `mma_arrive[cta_group]()` |
| `WGMMADescriptor` | `MMASmemDescriptor` |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `WGMMA shape mismatch` | Wrong M/N/K dimensions for data type | Check MMA Shape Options table: BF16 needs K=16, FP8 needs K=32 |
| `descriptor invalid` | TMA descriptor not properly initialized | Ensure descriptor created with correct swizzle mode and base address |
| `illegal memory access in wgmma` | Shared memory not properly aligned | Align shared memory to 16 bytes; use swizzle patterns for conflict-free access |
| `warpgroup_reg_alloc failed` | Insufficient registers for accumulator | Reduce tile size or N dimension; use fewer concurrent MMA operations |
| `TCGEN05 TMEM allocation failed` | TMEM capacity exceeded on SM100 | Reduce accumulator size; verify TMEM fits within 256KB per SM |
| `mbarrier sync timeout` | Missing or incorrect barrier synchronization | Ensure `wgmma_commit_group_sync` called before `wgmma_wait_group_sync` |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **SM90 WGMMA** | Available (v26.1+) | Stable, `from gpu.compute.mma import ...` |
| **SM100 UMMA/TCGEN05** | Available (v26.2+) | `from gpu.compute.arch.tcgen05 import ...` |
| **Constants** | `alias` or `comptime` | Both work in v26.1+ |
| **LayoutTensor** | Available (v26.1+) | `from layout import LayoutTensor, Layout` |
| **TensorCoreAsync** | Available (v26.1+) | `from layout.tensor_core_async import TensorCoreAsync` |

**Example (v26.1+):**
```mojo
comptime WMMA_M = 64
comptime WMMA_N = 128
comptime WMMA_K = 16

# Tensor core constants for H100
comptime WARP_GROUP_SIZE = 128  # 4 warps
```

**Notes:**
- Both `alias` and `comptime` work for compile-time constants in v26.1+
- SM90 WGMMA patterns (H100/Hopper) are stable in v26.1+
- SM100 UMMA/TCGEN05 patterns (Blackwell) added in v26.2+ nightly
- LayoutTensor and TensorCoreAsync wrappers are available in v26.1+

---

## Related Patterns

- [`gpu-memory-access.md`](gpu-memory-access.md) — TMA loading and shared memory swizzling
- [`gpu-warp-sync.md`](gpu-warp-sync.md) — Mbarrier patterns for async operations
- [`gpu-kernels.md`](gpu-kernels.md) — Cluster programming and warp specialization
- [`gpu-amd.md`](gpu-amd.md) — AMD MFMA tensor core patterns
- [`gpu-structured-kernels.md`](gpu-structured-kernels.md) — TileOp abstraction for tensor core operations

---

## References

- [MAX Kernels](https://github.com/modular/modular/tree/main/max/kernels)
