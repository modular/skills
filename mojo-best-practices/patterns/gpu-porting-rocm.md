---
title: "ROCm/HIP to Mojo Porting Guide"
description: Side-by-side ROCm/HIP→Mojo porting guide covering MFMA tensor cores, wavefront operations, LDS, and AMD scheduling
impact: HIGH
category: gpu
tags: [rocm, hip, amd, porting, mfma, mi300, mi250, cdna]
error_patterns:
  - "port HIP kernel"
  - "convert ROCm to Mojo"
  - "HIP equivalent"
  - "MFMA in Mojo"
  - "AMD GPU kernel"
scenarios:
  - "Port a HIP kernel to Mojo"
  - "Find Mojo equivalent of HIP API"
  - "Write MFMA kernel in Mojo"
  - "Convert AMD scheduling to Mojo"
  - "Port ROCm warp operations to Mojo"
---

# ROCm/HIP to Mojo Porting Guide

**Category:** GPU | **Impact:** HIGH

Complete guide for porting ROCm/HIP kernels to Mojo, covering the HIP basics, AMD MFMA tensor cores, wavefront operations, LDS patterns, and AMD-specific scheduling.

> **Version:** These patterns target Mojo nightly (v26.2+). All APIs shown are available in the public nightly toolchain.

---

## Quick Reference: HIP → Mojo Mapping

### Thread Hierarchy

| HIP | Mojo | Notes |
|-----|------|-------|
| `threadIdx.x` | `thread_idx.x` | Identical semantics |
| `blockIdx.x` | `block_idx.x` | Identical semantics |
| `blockDim.x` | `block_dim.x` | Identical semantics |
| `hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x` | `global_idx.x` | Built-in |

### Memory

| HIP | Mojo | Notes |
|-----|------|-------|
| `__shared__ float smem[N]` | `stack_allocation[N, DType.float32, address_space=AddressSpace.SHARED]()` | LDS on AMD |
| `hipMalloc(&ptr, size)` | `ctx.enqueue_create_buffer[dtype](count)` | Unified API |
| `hipMemcpy(dst, src, size, kind)` | `ctx.enqueue_copy(dst, src)` | Direction auto-detected |
| `hipDeviceSynchronize()` | `ctx.synchronize()` | Unified API |

### Synchronization

| HIP | Mojo | Notes |
|-----|------|-------|
| `__syncthreads()` | `barrier()` | Maps to `s_barrier` on AMD |
| `__threadfence()` | `gpu.sync.fence()` | Memory fence |

### Kernel Launch

| HIP | Mojo |
|-----|------|
| `hipLaunchKernelGGL(kernel, grid, block, smem, stream, args...)` | `ctx.enqueue_function[kernel](args, grid_dim=grid, block_dim=block)` |
| `kernel<<<grid, block, smem, stream>>>(args)` | Same as above — HIP's CUDA-like syntax maps identically |

> **Key insight:** Most HIP→Mojo porting is identical to CUDA→Mojo porting since HIP mirrors the CUDA API. See [`gpu-porting-cuda.md`](gpu-porting-cuda.md) for basics. This guide focuses on **AMD-specific** patterns.

---

## AMD-Specific: Wavefront vs Warp

AMD GPUs use wavefronts instead of warps. The wavefront size is 64 on CDNA (MI100/MI200/MI300) and can be 32 or 64 on RDNA.

### Key Difference

| Concept | NVIDIA | AMD CDNA | Impact |
|---------|--------|----------|--------|
| SIMT width | 32 (warp) | 64 (wavefront) | Reduction needs 6 iterations, not 5 |
| `WARP_SIZE` | 32 | 64 | Compile-time constant in Mojo |
| Warp shuffle | `__shfl_down_sync` | `__shfl_down` (no mask needed) | Same Mojo API |
| Register file | 256 per thread | 256 per thread (SGPR + VGPR) | AMD has SGPR for uniform values |

### Mojo: Cross-Platform Warp Size

```mojo
# nocompile
from gpu import WARP_SIZE  # 32 on NVIDIA, 64 on AMD

# Write code that works on both:
fn warp_reduce_sum(var val: Float32) -> Float32:
    # Automatically handles warp size 32 or 64
    var offset = WARP_SIZE // 2
    while offset > 0:
        val += shfl_down[DType.float32, 1](val, offset)
        offset >>= 1
    return val
```

### HIP: Wavefront Operations

```cpp
// HIP: wavefront reduction (wavefront_size = 64)
float warp_reduce(float val) {
    for (int offset = 32; offset > 0; offset >>= 1)
        val += __shfl_down(val, offset);
    return val;
}
```

### Mojo: Same Code Works

```mojo
# nocompile
from gpu import WARP_SIZE, lane_id
from gpu.warp import shfl_down

fn warp_reduce(var val: Float32) -> Float32:
    var offset = WARP_SIZE // 2
    while offset > 0:
        val += shfl_down[DType.float32, 1](val, offset)
        offset >>= 1
    return val
```

---

## AMD MFMA Tensor Cores

AMD's Matrix Fused Multiply-Add (MFMA) instructions are accessed through `TensorCore` and `TiledTensorCore` in Mojo.

### HIP: MFMA via Built-in

```cpp
// HIP: MFMA 32x32x8 with fp16 inputs, fp32 accumulator
__attribute__((amdgpu_flat_work_group_size(64, 64)))
__global__ void mfma_kernel(...) {
    // Load A and B fragments into registers
    // ...

    // Execute MFMA instruction
    float result[16] = __builtin_amdgcn_mfma_f32_32x32x8f16(a_frag, b_frag, c_frag, 0, 0, 0);
}
```

### Mojo: TensorCore (AMD MFMA Wrapper)

```mojo
# nocompile
from layout.tensor_core import TensorCore, TiledTensorCore, num_matrix_reg

# Configure MFMA operation
comptime mma = TensorCore[
    DType.float32,     # Accumulator type
    DType.float16,     # A input type
    DType.float16,     # B input type
    mma_m=32,          # M dimension
    mma_n=32,          # N dimension
    mma_k=8,           # K dimension
]

# Or use TiledTensorCore for multi-tile MMA
comptime tiled_mma = TiledTensorCore[
    DType.float32,     # Accumulator type
    DType.bfloat16,    # Input type
    mma_m=32,          # Base MMA M
    mma_n=32,          # Base MMA N
    mma_k=16,          # Base MMA K (double-rate for bf16)
    tile_m=128,        # Tile M (4x base)
    tile_n=128,        # Tile N (4x base)
]

# Execute MMA
fn compute_tile(
    a_reg: RegTile[DType.float16, ...],
    b_reg: RegTile[DType.float16, ...],
    accum: RegTile[DType.float32, ...],
):
    mma.mma(accum, a_reg, b_reg)
```

### MFMA Shape Selection

| Shape | Types | Regs/Thread | Best For | Mojo MMA Parameters |
|-------|-------|-------------|----------|---------------------|
| 32x32x8 | f16→f32 | 16 | General GEMM | `mma_m=32, mma_n=32, mma_k=8` |
| 16x16x16 | f16→f32 | 4 | Batched small | `mma_m=16, mma_n=16, mma_k=16` |
| 32x32x16 | bf16→f32 | 16 | Double-rate | `mma_m=32, mma_n=32, mma_k=16` |
| 16x16x32 | bf16→f32 | 4 | Double-rate small | `mma_m=16, mma_n=16, mma_k=32` |
| 32x32x64 | fp8→f32 | 16 | FP8 (MI300+) | `mma_m=32, mma_n=32, mma_k=64` |

> **Decision guide:** Use 32x32 shapes for maximum throughput. Use 16x16 shapes when register pressure is high or for small matrix dimensions.

---

## AMD Scheduling: s_waitcnt and Barriers

AMD GPUs require explicit scheduling control for optimal performance. This is fundamentally different from NVIDIA where the hardware scheduler handles most ordering.

### HIP: Manual Scheduling

```cpp
// HIP: AMD scheduling intrinsics
__builtin_amdgcn_sched_barrier(0);  // Full scheduling barrier
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);  // MFMA barrier
__builtin_amdgcn_sched_group_barrier(0x020, 4, 0);  // DS read barrier
__builtin_amdgcn_s_waitcnt(0xf71f);  // Wait for memory ops
```

### Mojo: AMD Scheduling API

```mojo
# nocompile
from gpu.sync import (
    schedule_barrier,
    schedule_group_barrier,
    AMDScheduleBarrierMask,
    s_waitcnt,
)

# Full scheduling barrier — prevents reordering across this point
schedule_barrier()

# Group barrier — wait for specific instruction types
# AMDScheduleBarrierMask controls which instructions to wait for:
schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, Int32(group))   # Wait for 1 MFMA
schedule_group_barrier(AMDScheduleBarrierMask.VALU, 6, Int32(group))   # Wait for 6 VALU ops
schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 4, Int32(group))  # Wait for 4 LDS reads
schedule_group_barrier(AMDScheduleBarrierMask.TRANS, 1, Int32(group))  # Wait for 1 trans

# s_waitcnt — low-level wait counter
s_waitcnt(vmcnt=0, lgkmcnt=0)  # Wait for all VM and LDS/GDS ops
```

### AMDScheduleBarrierMask Values

| Mask | Instruction Type | Description |
|------|-----------------|-------------|
| `AMDScheduleBarrierMask.MFMA` | Matrix FMA | Tensor core operations |
| `AMDScheduleBarrierMask.VALU` | Vector ALU | Regular compute |
| `AMDScheduleBarrierMask.DS_READ` | LDS read | Shared memory loads |
| `AMDScheduleBarrierMask.DS_WRITE` | LDS write | Shared memory stores |
| `AMDScheduleBarrierMask.VMEM_READ` | Global read | DRAM loads |
| `AMDScheduleBarrierMask.VMEM_WRITE` | Global write | DRAM stores |
| `AMDScheduleBarrierMask.TRANS` | Transcendental | Math ops (exp, log, etc.) |

### Scheduling Pattern: Interleave MFMA with Memory

```mojo
# nocompile
# Pattern: interleave MFMA with LDS reads for latency hiding
fn interleaved_compute(
    a_tiles: SMemTile[...],
    b_tiles: SMemTile[...],
    accum: RegTile[...],
):
    for k in range(num_k_tiles):
        # Start MFMA on current tile
        schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, Int32(0))
        mma.mma(accum, a_reg, b_reg)

        # Interleave: load next tile from LDS while MFMA executes
        schedule_group_barrier(AMDScheduleBarrierMask.VALU, 6, Int32(0))
        # Load next a_reg, b_reg from shared memory

        schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, Int32(0))
```

---

## LDS (Local Data Share) = Shared Memory

AMD's LDS is functionally equivalent to NVIDIA's shared memory. The same Mojo `AddressSpace.SHARED` works for both.

### HIP: LDS Usage

```cpp
__shared__ float lds_data[1024];  // LDS allocation
lds_data[threadIdx.x] = global_data[gid];
__syncthreads();
float val = lds_data[threadIdx.x ^ 1];  // XOR for bank-conflict-free access
```

### Mojo: Shared Memory (Same Code for AMD and NVIDIA)

```mojo
from memory import stack_allocation
from gpu import thread_idx, global_idx
from gpu.sync import barrier

fn lds_kernel(global_data: UnsafePointer[Float32]):
    # stack_allocation maps to LDS on AMD, SMEM on NVIDIA
    var lds = stack_allocation[
        1024, DType.float32, address_space=AddressSpace.SHARED
    ]()

    lds[thread_idx.x] = global_data[global_idx.x]
    barrier()  # Maps to s_barrier on AMD
    var val = lds[thread_idx.x ^ 1]
```

### LDS Capacity by GPU

| GPU | LDS per CU | Maximum per Block |
|-----|-----------|-------------------|
| MI100 | 64 KB | 64 KB |
| MI200 (MI250X) | 64 KB | 64 KB |
| MI300X | 64 KB | 64 KB |
| MI355X | 64 KB | 64 KB |

---

## AMD ScatterGather: Direct DRAM→Register

AMD GPUs can load directly from global memory to registers using buffer resources, bypassing shared memory. This is unique to AMD — NVIDIA requires going through shared memory for coalesced loads.

### Mojo: ScatterGatherAmd

```mojo
# nocompile
from linalg.structuring import ScatterGatherAmd

# Thread layout describes how threads cooperate on the copy
comptime thread_layout = Layout(IntTuple(4, 64), IntTuple(64, 1))

# Create scatter-gather from a global memory tensor
var sg = ScatterGatherAmd[thread_layout](global_tensor)

# Copy: DRAM → registers (direct, no LDS intermediate)
sg.copy(dst_reg_tile, src_gmem_tile)

# Copy: registers → DRAM (direct store)
sg.copy(dst_gmem_tile, src_reg_tile)
```

### AMD Buffer Resources

```mojo
# nocompile
from gpu.intrinsics import AMDBufferResource
from layout._utils import make_amd_buffer_resource

# AMDBufferResource encapsulates the AMD buffer descriptor for efficient memory access
var buffer = make_amd_buffer_resource(tensor)
# Used internally by ScatterGatherAmd for bounds-checked, coalesced memory access
```

---

## AMD vs NVIDIA: Data Movement Patterns

| Pattern | NVIDIA | AMD | Mojo Abstraction |
|---------|--------|-----|------------------|
| Global → Shared | TMA (SM90+) or cp.async (SM80) | Buffer load → LDS write | `TMATensorTile` / `ScatterGatherAmd` |
| Global → Register | Through shared memory | Direct via buffer resources | `ScatterGatherAmd` |
| Shared → Register | Direct load | Direct load (LDS) | `LayoutTensor[..., SHARED]` indexing |
| Register → Shared | Direct store | Direct store (LDS) | `LayoutTensor[..., SHARED]` store |
| Shared → Global | TMA store or direct | Buffer store | `ScatterGatherAmd` / direct store |

---

## Complete AMD GEMM Structure

```mojo
from gpu import thread_idx, block_idx, WARP_SIZE, lane_id
from gpu.sync import barrier, schedule_barrier, schedule_group_barrier, AMDScheduleBarrierMask
from layout import Layout, LayoutTensor
from layout.tensor_core import TensorCore, TiledTensorCore
from linalg.structuring import (
    ScatterGatherAmd, SMemTile, RegTile, SharedMemoryManager,
)

struct AMDGemmKernel[
    BM: Int, BN: Int, BK: Int,          # Block tile shape
    mma_m: Int, mma_n: Int, mma_k: Int, # MFMA shape
]:
    comptime mma = TensorCore[
        DType.float32, DType.bfloat16, DType.bfloat16,
        mma_m=mma_m, mma_n=mma_n, mma_k=mma_k,
    ]

    @staticmethod
    fn kernel(
        a_ptr: UnsafePointer[BFloat16],
        b_ptr: UnsafePointer[BFloat16],
        c_ptr: UnsafePointer[Float32],
        M: Int, N: Int, K: Int,
    ):
        # 1. Create global memory LayoutTensors
        var a_global = LayoutTensor[DType.bfloat16, a_global_layout](a_ptr)
        var b_global = LayoutTensor[DType.bfloat16, b_global_layout](b_ptr)

        # 2. Allocate shared memory (LDS)
        var smm = SharedMemoryManager()
        var a_smem = smm.build[SMemTile[DType.bfloat16, a_smem_layout]]()
        var b_smem = smm.build[SMemTile[DType.bfloat16, b_smem_layout]]()

        # 3. Set up ScatterGather for DRAM access
        var sg_a = ScatterGatherAmd[thread_layout](a_global)
        var sg_b = ScatterGatherAmd[thread_layout](b_global)

        # 4. Initialize accumulators in registers
        var accum = RegTile[DType.float32, accum_layout]()
        accum.zero()

        # 5. Main loop
        for k_tile in range(K // BK):
            # Load A and B tiles to shared memory
            sg_a.copy(a_smem, a_global_tile)
            sg_b.copy(b_smem, b_global_tile)
            barrier()

            # Compute: load from LDS → registers → MFMA
            for k_inner in range(BK // mma_k):
                # Load register tiles from shared memory
                var a_reg = RegTile[DType.bfloat16, a_reg_layout]()
                var b_reg = RegTile[DType.bfloat16, b_reg_layout]()
                # ... load a_reg, b_reg from a_smem, b_smem ...

                # Interleave scheduling for latency hiding
                schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, Int32(0))

                # MFMA: tensor core multiply-accumulate
                Self.mma.mma(accum, a_reg, b_reg)

            barrier()

        # 6. Store results: registers → global memory
        var sg_c = ScatterGatherAmd[thread_layout](c_global)
        sg_c.copy(c_global, accum)
```

---

## Cross-Platform Kernels

Mojo's `@parameter if` enables writing kernels that work on both NVIDIA and AMD:

```mojo
# nocompile
from sys import is_nvidia_gpu, is_amd_gpu

fn cross_platform_gemm_tile(
    a_smem: SMemTile[...],
    b_smem: SMemTile[...],
    accum: RegTile[...],
):
    @parameter
    if is_nvidia_gpu():
        # Use WGMMA for NVIDIA
        warpgroup_fence()
        WgmmaOp.mma(accum, a_smem, b_smem)
        warpgroup_fence()
    elif is_amd_gpu():
        # Use MFMA for AMD
        schedule_barrier()
        mfma_op.mma(accum, a_reg, b_reg)
        schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, Int32(0))
```

---

## Common Gotchas When Porting from HIP

### 1. Wavefront Size

```mojo
# nocompile
# HIP: often hardcoded to 64
# Mojo: use WARP_SIZE which is correct on any platform
from gpu import WARP_SIZE

var warp_count = block_dim.x // WARP_SIZE  # Works on both NVIDIA and AMD
```

### 2. No Warp Mask Parameter

```mojo
# nocompile
# HIP: __shfl_down(val, offset) — no mask needed (full wave)
# Mojo: same — no mask parameter
var result = shfl_down[DType.float32, 1](val, offset)
```

### 3. AMD Scheduling is Performance-Critical

```mojo
# nocompile
# On NVIDIA: hardware scheduler handles instruction ordering
# On AMD: explicit scheduling barriers are REQUIRED for good performance
# Always add scheduling hints in performance-critical AMD kernels

schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 1, Int32(0))
# Without this, MFMA and VALU may serialize instead of overlapping
```

### 4. LDS Bank Conflicts

```mojo
# AMD LDS has 32 banks (same as NVIDIA shared memory)
# Use swizzle patterns for bank-conflict-free access
# The Swizzle abstraction works identically on both platforms
```

### 5. Buffer Resources for Bounds Checking

```mojo
# AMD buffer resources provide hardware bounds checking
# ScatterGatherAmd uses this automatically
# This is more efficient than software bounds checking:
#   if (idx < N) data[idx] = val;  // Software check
# vs:
#   sg.copy(dst, src);  // Hardware bounds check via buffer resource
```

---

## Import Cheat Sheet (AMD-Specific)

```mojo
# nocompile
# === Core GPU (cross-platform) ===
from gpu import thread_idx, block_idx, block_dim, global_idx, WARP_SIZE, lane_id
from gpu.host import DeviceContext
from gpu.sync import barrier

# === AMD Scheduling ===
from gpu.sync import (
    schedule_barrier,
    schedule_group_barrier,
    AMDScheduleBarrierMask,
    s_waitcnt,
)

# === AMD Tensor Cores (MFMA) ===
from layout.tensor_core import TensorCore, TiledTensorCore, num_matrix_reg

# === AMD Memory (ScatterGather) ===
from linalg.structuring import ScatterGatherAmd, IteratorScatterGatherAmd
from gpu.intrinsics import AMDBufferResource

# === Shared Memory / Layout (cross-platform) ===
from layout import Layout, LayoutTensor
from linalg.structuring import SMemTile, RegTile, SharedMemoryManager
from memory import stack_allocation
```

---

## Related Patterns

- [`gpu-porting-cuda.md`](gpu-porting-cuda.md) — CUDA → Mojo porting (start here for basics)
- [`gpu-porting-cute.md`](gpu-porting-cute.md) — CuTe DSL → Mojo mapping
- [`gpu-amd.md`](gpu-amd.md) — AMD-specific details (MFMA shapes, scheduling, waitcnt)
- [`gpu-structured-kernels.md`](gpu-structured-kernels.md) — ScatterGather/RingBuffer/TileOp architecture
- [`gpu-fundamentals.md`](gpu-fundamentals.md) — Mojo GPU programming fundamentals
