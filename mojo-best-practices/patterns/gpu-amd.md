---
title: AMD GPU Programming
description: MFMA shapes, scheduling barriers, and waitcnt for AMD CDNA GPUs
impact: MEDIUM
category: gpu
tags: [gpu, amd, mfma, scheduling, waitcnt, mi300]
error_patterns:
  - "MFMA error"
  - "ROCm"
  - "HIP error"
  - "wavefront"
  - "s_waitcnt"
  - "MI300"
scenarios:
  - "Write kernel for AMD MI300X"
  - "Select optimal MFMA shape"
  - "Use s_waitcnt correctly"
  - "Port CUDA kernel to AMD"
consolidates:
  - gpu-amd-mfma-shapes.md
  - gpu-amd-scheduling.md
  - gpu-amd-waitcnt.md
---

# AMD GPU Programming

**Category:** GPU | **Impact:** MEDIUM

AMD CDNA GPUs (MI100, MI200, MI300X) require specific patterns for optimal performance: correct MFMA shape selection, explicit scheduling barriers, and fine-grained memory synchronization with s_waitcnt.

## API Availability

> **Note:** All APIs shown in this pattern are available in the **Mojo nightly toolchain** (v26.2+). AMD-specific instructions require AMD CDNA hardware (MI100/MI200/MI300).

| API | Import | Notes |
|-----|--------|-------|
| `thread_idx`, `block_idx` | `from gpu import thread_idx, block_idx` | Core GPU primitives (cross-platform) |
| `stack_allocation` | `from memory import stack_allocation` | Shared memory (LDS on AMD) |
| `AddressSpace` | `from gpu.memory import AddressSpace` | Address space enum (SHARED maps to LDS) |
| `LayoutTensor` | `from layout import LayoutTensor` | Type-safe tensor with address space |
| `schedule_group_barrier` | `from gpu.sync import schedule_group_barrier` | AMD scheduling barriers |
| `schedule_barrier` | `from gpu.sync import schedule_barrier` | AMD scheduling barrier (simpler form) |
| `AMDScheduleBarrierMask` | `from gpu.sync import AMDScheduleBarrierMask` | AMD scheduling mask constants |
| `barrier` | `from gpu.sync import barrier` | Block-level barrier (maps to `s_barrier` on AMD) |
| `s_waitcnt` | `from gpu.sync import s_waitcnt` | AMD memory synchronization intrinsic |
| `TensorCore` | `from layout.tensor_core import TensorCore` | AMD MFMA wrapper |
| `ScatterGatherAmd` | `from linalg.structuring import ScatterGatherAmd` | AMD DRAM↔register data movement |

> **Porting from ROCm/HIP?** See [`gpu-porting-rocm.md`](gpu-porting-rocm.md) for side-by-side HIP→Mojo examples.

---

## Core Concepts

### AMD MFMA Shapes

Matrix Fused Multiply-Add (MFMA) instructions come in multiple shapes. Choosing the wrong shape can halve throughput.

**Available Shapes:**

| Shape | Data Type | Regs/Thread | Best For |
|-------|-----------|-------------|----------|
| 16x16x4 | float32 | 4 | FP32 (only option) |
| 16x16x16 | bf16/f16 | 4 | Small matrices, odd dimensions |
| 32x32x8 | bf16/f16 | 16 | Large matrices, peak throughput |
| 32x32x16 | bf16/f16 | 16 | Double-rate MFMA |
| 16x16x128 | fp8 | 16 | FP8, flexible dimensions |
| 32x32x64 | fp8 | 32 | FP8, peak throughput |

**Optimal Shape Selection:**

```mojo
# nocompile

fn select_mfma_shape[
    dtype: DType,
    M: Int,
    N: Int,
](out result: IndexList[3]):
    """Select optimal MFMA shape based on problem dimensions."""
    @parameter
    if dtype.is_half_float():  # bfloat16 or float16
        @parameter
        if M >= 32 and N >= 32 and M % 32 == 0 and N % 32 == 0:
            return IndexList[3](32, 32, 8)  # Higher throughput
        else:
            return IndexList[3](16, 16, 16)  # Better for small/odd sizes
    elif dtype == DType.float32:
        return IndexList[3](16, 16, 4)  # Only option for FP32
    elif dtype.is_float8():
        @parameter
        if M >= 32 and N >= 32:
            return IndexList[3](32, 32, 64)
        else:
            return IndexList[3](16, 16, 128)
```

**Register Calculation (64-thread wavefront):**

```mojo
fn num_matrix_reg[MMA_M: Int, MMA_N: Int]() -> Int:
    """Calculate registers per thread for accumulator."""
    return (MMA_M * MMA_N) // 64  # WARP_SIZE = 64 on AMD

# Examples:
# 16x16: 16 * 16 / 64 = 4 registers
# 32x32: 32 * 32 / 64 = 16 registers
```

---

## Scheduling Barriers

AMD GPUs require explicit scheduling hints to overlap MFMA operations with memory loads. The `schedule_group_barrier` and `AMDScheduleBarrierMask` APIs are public in the `gpu.sync` module.

**Without Scheduling (Slow):**

```mojo
# nocompile

fn amd_matmul_kernel(...):
    # Compiler may serialize all loads, then all MFMAs
    for k in range(K_TILES):
        var a_tile = load_from_lds(A_smem, k)
        var b_tile = load_from_lds(B_smem, k)
        acc = mfma(a_tile, b_tile, acc)  # MFMA waits for loads
```

**With Scheduling Barriers (Fast):**

```mojo
# nocompile

from gpu.sync import schedule_group_barrier, AMDScheduleBarrierMask

fn amd_matmul_kernel(...):
    @parameter
    if is_amd_gpu():
        for k in range(K_TILES):
            # Allow 1 DS_READ to be scheduled
            schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 1, 0)
            # Allow 2 MFMA instructions to be scheduled
            schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 2, 0)

            var a_tile = load_from_lds(A_smem, k)
            var b_tile = load_from_lds(B_smem, k)
            acc = mfma(a_tile, b_tile, acc)
```

**AMDScheduleBarrierMask Options:**

```mojo
# nocompile

struct AMDScheduleBarrierMask:
    comptime NONE = Self(0)           # Full barrier
    comptime MFMA = Self(1 << 3)      # Matrix multiply
    comptime VALU = Self(1 << 4)      # Vector ALU operations
    comptime VMEM_READ = Self(1 << 5) # Vector memory reads
    comptime DS_READ = Self(1 << 8)   # LDS reads
    comptime DS_WRITE = Self(1 << 9)  # LDS writes

    # Combine with |
    fn __or__(self, other: Self) -> Self:
        return Self(Int(self) | Int(other))
```

---

## Memory Synchronization with s_waitcnt

> **Note:** The `s_waitcnt` intrinsic is available in the Mojo nightly toolchain (v26.2+).

AMD uses `s_waitcnt` for fine-grained memory synchronization. Wait for "N operations remaining" rather than "all complete".

**Wait Count Parameters:**

```mojo
# nocompile

fn s_waitcnt[
    *,
    vmcnt: UInt32 = MAX_VM_CNT,    # Global memory ops (max 63)
    lgkmcnt: UInt32 = MAX_LGKM_CNT, # LDS ops (max 15)
]():
    """Wait until operation counters reach specified values.

    vmcnt=0: Wait for all global loads
    lgkmcnt=0: Wait for all LDS operations
    """
```

**Partial Synchronization for Pipelining:**

```mojo
# nocompile

fn pipelined_amd_kernel(...):
    @parameter
    if is_amd_gpu():
        # Issue 4 LDS loads
        ds_read(a0, ptr + 0)
        ds_read(a1, ptr + 1)
        ds_read(a2, ptr + 2)
        ds_read(a3, ptr + 3)

        # Wait for first 2 loads only (2 remaining = first 2 done)
        s_waitcnt[lgkmcnt=2]()

        # Use first 2 values while last 2 still loading
        compute(a0, a1)

        # Wait for remaining loads
        s_waitcnt[lgkmcnt=0]()
        compute(a2, a3)
```

**Counter Semantics:**

```mojo
# nocompile

# Counter = "operations remaining", NOT "operations completed"
# lgkmcnt=0: wait until 0 remaining = all done
# lgkmcnt=2: wait until 2 remaining = older ones done

# Example: 6 LDS reads issued
ds_read(a)  # lgkmcnt = 1
ds_read(b)  # lgkmcnt = 2
ds_read(c)  # lgkmcnt = 3
ds_read(d)  # lgkmcnt = 4
ds_read(e)  # lgkmcnt = 5
ds_read(f)  # lgkmcnt = 6

s_waitcnt[lgkmcnt=2]()  # Wait for a,b,c,d (4 done, 2 remaining)
```

---

## MI300X Configuration

```mojo
# nocompile

fn mi300x_gemm_config[
    dtype: DType,
    M: Int, N: Int, K: Int,
]() -> MatmulConfig:
    """Optimal GEMM configuration for MI300X.

    MI300X specs:
    - 304 compute units
    - 256 VGPRs per thread
    - 64 KB LDS per CU
    - 5.3 TB/s HBM3 bandwidth
    """
    @parameter
    if dtype.is_half_float():
        return MatmulConfig[
            block_tile_shape = Index(128, 128, 64),
            warp_tile_shape = Index(64, 64, 64),
            mma_shape = IndexList[3](32, 32, 8),
        ]()
    elif dtype.is_float8():
        return MatmulConfig[
            block_tile_shape = Index(128, 128, 128),
            warp_tile_shape = Index(64, 64, 128),
            mma_shape = IndexList[3](32, 32, 64),
        ]()
    else:  # FP32
        return MatmulConfig[
            block_tile_shape = Index(64, 64, 16),
            warp_tile_shape = Index(32, 32, 16),
            mma_shape = IndexList[3](16, 16, 4),
        ]()
```

---

## Decision Guide

| Condition | Shape | Reason |
|-----------|-------|--------|
| BF16, M%32==0, N%32==0 | 32x32x8 | Peak throughput |
| BF16, M<32 or N<32 | 16x16x16 | Divisibility |
| FP32 | 16x16x4 | Only option |
| FP8, M%32==0 | 32x32x64 | Peak FP8 throughput |
| High register pressure | 16x16x* | Lower reg/thread |

---

## Quick Reference

- **32x32x8**: Use for large BF16/FP16 matrices with dimensions divisible by 32
- **16x16x16**: Use for small matrices or non-divisible dimensions
- **Scheduling**: Interleave 1-2 DS_READ per 2-4 MFMA
- **s_waitcnt**: Start with lgkmcnt=0, gradually increase for overlap
- **Profiling**: Use `rocprof --stats` and `omniperf` to verify

---

## ROCm Setup and Environment

### Installation and Verification

```bash
# Check ROCm installation
rocm-smi --showproductname

# Verify GPU detection
rocminfo | grep "gfx"
# Output: gfx90a (MI100/MI210), gfx942 (MI300X)

# Check ROCm version
cat /opt/rocm/.info/version
# Example: 6.0.0

# Set environment for compilation
export ROCM_PATH=/opt/rocm
export HIP_PLATFORM=amd
export HIP_VISIBLE_DEVICES=0,1,2,3  # Select GPUs
```

### Mojo GPU Target for AMD

```bash
# Target AMD GPU architecture
mojo build --target-accelerator=amd:gfx90a kernel.mojo   # MI100/MI210
mojo build --target-accelerator=amd:gfx942 kernel.mojo   # MI300X
mojo build --target-accelerator=amd:gfx950 kernel.mojo   # MI355X (CDNA4)
```

**Architecture Reference:**

| GPU | Architecture | GFX Version | Compute Units | VRAM |
|-----|--------------|-------------|---------------|------|
| MI100 | CDNA | gfx908 | 120 | 32GB HBM2 |
| MI210 | CDNA2 | gfx90a | 104 | 64GB HBM2e |
| MI250/MI250X | CDNA2 | gfx90a | 220 (2 dies) | 128GB HBM2e |
| MI300X | CDNA3 | gfx942 | 304 | 192GB HBM3 |
| MI355X | CDNA4 | gfx950 | TBD | HBM3e |

---

## MI100/MI200 Series Patterns

### MI100 Specific Configuration

```mojo
# nocompile

fn mi100_gemm_config[dtype: DType]() -> MatmulConfig:
    """Optimal GEMM configuration for MI100.

    MI100 specs:
    - 120 compute units
    - 256 VGPRs per thread
    - 64 KB LDS per CU
    - 1.23 TB/s HBM2 bandwidth
    - Peak: 11.5 TF FP32, 23 TF FP16/BF16
    """
    @parameter
    if dtype.is_half_float():
        return MatmulConfig[
            block_tile_shape = Index(64, 64, 32),
            warp_tile_shape = Index(32, 32, 32),
            mma_shape = IndexList[3](16, 16, 16),
        ]()
    else:  # FP32
        return MatmulConfig[
            block_tile_shape = Index(64, 64, 16),
            warp_tile_shape = Index(32, 32, 16),
            mma_shape = IndexList[3](16, 16, 4),
        ]()
```

### MI200 (MI210/MI250) Specific Configuration

```mojo
# nocompile

fn mi200_gemm_config[dtype: DType]() -> MatmulConfig:
    """Optimal GEMM configuration for MI200 series.

    MI210 specs:
    - 104 compute units
    - 512 VGPRs per thread (increased from MI100)
    - 64 KB LDS per CU
    - 1.6 TB/s HBM2e bandwidth
    - Peak: 22.6 TF FP32, 45.3 TF FP16/BF16, 90.5 TF INT8

    MI250X (dual-die) specs:
    - 220 compute units total (110 per GCD)
    - 128GB HBM2e (64GB per GCD)
    - 3.2 TB/s combined bandwidth
    """
    @parameter
    if dtype.is_half_float():
        return MatmulConfig[
            block_tile_shape = Index(128, 128, 32),
            warp_tile_shape = Index(64, 64, 32),
            mma_shape = IndexList[3](16, 16, 16),
        ]()
    else:  # FP32
        return MatmulConfig[
            block_tile_shape = Index(64, 64, 16),
            warp_tile_shape = Index(32, 32, 16),
            mma_shape = IndexList[3](16, 16, 4),
        ]()
```

### MI355X (CDNA4) Specific Configuration

MI355X (`gfx950`) introduces new scheduling patterns and the 8-wave ping-pong architecture. See [`gpu-structured-kernels.md`](gpu-structured-kernels.md) for the full 8-wave ping-pong pattern.

```mojo
# nocompile

fn mi355x_gemm_config[dtype: DType]() -> MatmulConfig:
    """Optimal GEMM configuration for MI355X (CDNA4).

    MI355X uses 8-wave ping-pong pattern for maximum register utilization.
    See gpu-structured-kernels.md for architectural details.
    """
    @parameter
    if dtype.is_half_float():
        return MatmulConfig[
            block_tile_shape = Index(128, 128, 64),
            warp_tile_shape = Index(64, 64, 64),
            mma_shape = IndexList[3](32, 32, 16),
        ]()
    elif dtype.is_float8():
        return MatmulConfig[
            block_tile_shape = Index(128, 128, 128),
            warp_tile_shape = Index(64, 64, 128),
            mma_shape = IndexList[3](32, 32, 64),
        ]()
```

```bash
# Run tests on MI355X
./bazelw test --config=remote-mi355 //max/kernels/test/gpu/...
```

### Wavefront (Warp) Size Differences

AMD uses 64-thread wavefronts (vs NVIDIA's 32-thread warps):

```mojo
fn get_wavefront_size() -> Int:
    """AMD wavefront size."""
    return 64  # Always 64 on AMD CDNA

fn num_matrix_reg[MMA_M: Int, MMA_N: Int]() -> Int:
    """Calculate registers per thread for accumulator on AMD."""
    return (MMA_M * MMA_N) // 64  # WAVEFRONT_SIZE = 64

# Examples (AMD vs NVIDIA):
# 16x16 on AMD: 16 * 16 / 64 = 4 registers
# 16x16 on NVIDIA: 16 * 16 / 32 = 8 registers (different!)
# 32x32 on AMD: 32 * 32 / 64 = 16 registers
```

---

## Profiling with AMD Tools

### rocprof Basics

```bash
# Basic profiling
rocprof --stats ./my_kernel
# Output: results.stats.csv with kernel timings

# Hardware counters
rocprof --timestamp on -i input.txt ./my_kernel
# input.txt contains counter names

# Trace mode
rocprof --hip-trace --hsa-trace ./my_kernel
# Output: results.json for Chrome tracing
```

**Counter Configuration (input.txt):**
```text
pmc: GRBM_COUNT, GRBM_GUI_ACTIVE
pmc: SQ_WAVES, SQ_INSTS_VALU
pmc: TCC_HIT, TCC_MISS
pmc: TA_BUSY, TCP_TOTAL_CACHE_ACCESSES
```

### rocprof Hardware Counters Reference

| Counter | Description | Optimization Target |
|---------|-------------|---------------------|
| `SQ_WAVES` | Wavefronts launched | Occupancy |
| `SQ_INSTS_VALU` | Vector ALU instructions | Compute utilization |
| `TCC_HIT` / `TCC_MISS` | L2 cache hit/miss | Memory efficiency |
| `TCP_TOTAL_CACHE_ACCESSES` | L1 cache accesses | Cache behavior |
| `GRBM_GUI_ACTIVE` | GPU active cycles | Overall utilization |
| `TA_BUSY` | Texture unit busy | Memory bandwidth |

### Omniperf Deep Analysis

```bash
# Collect omniperf profile
omniperf profile -n my_run -- ./my_kernel

# Analyze results
omniperf analyze -p workloads/my_run/MI300X

# Web UI
omniperf analyze -p workloads/my_run/MI300X --gui
```

**Key Omniperf Metrics:**

| Section | Metrics | Interpretation |
|---------|---------|----------------|
| Speed-of-Light | % of peak | Compare to roofline |
| Compute | VALU utilization | Should be >50% for compute-bound |
| Memory | HBM bandwidth % | Should be >70% for memory-bound |
| Cache | L1/L2 hit rates | Higher is better |
| Wavefront | Occupancy | 50%+ for latency hiding |

---

## AMD-Specific Synchronization

### s_waitcnt Detailed Usage

```mojo
# nocompile

fn s_waitcnt[
    *,
    vmcnt: UInt32 = MAX_VM_CNT,     # Global memory ops (max 63)
    lgkmcnt: UInt32 = MAX_LGKM_CNT,  # LDS + GDS + Scalar ops (max 15)
    expcnt: UInt32 = MAX_EXP_CNT,    # Export/GDS ops (max 7)
]():
    """AMD s_waitcnt intrinsic for memory synchronization.

    Counter semantics: wait until counter <= specified value
    - vmcnt=0: Wait for ALL global memory loads/stores
    - lgkmcnt=0: Wait for ALL LDS operations
    - expcnt=0: Wait for ALL exports

    Counter starts at 0, increments with each issued op, decrements on completion.
    """
```

**Pipelining Pattern with s_waitcnt:**

```mojo
# nocompile

fn software_pipelined_kernel(...):
    """Double-buffered kernel with s_waitcnt pipelining."""
    @parameter
    if is_amd_gpu():
        # Stage 0: Load first tile
        ds_read(tile_a[0], A_smem + 0 * TILE_K)
        ds_read(tile_b[0], B_smem + 0 * TILE_K)

        for k in range(1, K_TILES):
            # Stage 1: Load next tile (pipelined)
            ds_read(tile_a[k % 2], A_smem + k * TILE_K)
            ds_read(tile_b[k % 2], B_smem + k * TILE_K)

            # Wait for previous tile only (lgkmcnt=2 means 2 pending)
            s_waitcnt[lgkmcnt=2]()

            # Stage 2: Compute with previous tile
            acc = mfma(tile_a[(k-1) % 2], tile_b[(k-1) % 2], acc)

        # Final iteration
        s_waitcnt[lgkmcnt=0]()
        acc = mfma(tile_a[(K_TILES-1) % 2], tile_b[(K_TILES-1) % 2], acc)
```

### s_barrier for Block Synchronization

```mojo
# nocompile

fn amd_block_sync():
    """Block-level synchronization on AMD."""
    @parameter
    if is_amd_gpu():
        # s_barrier synchronizes all wavefronts in workgroup
        s_barrier()
```

---

## Common AMD Issues and Solutions

### Issue: Low Occupancy

**Symptoms:** Low GPU utilization despite high parallelism

```mojo
# Problem: Too many VGPRs per thread
fn bad_kernel(...):
    # 128 Float32 variables = 128 VGPRs
    var temp0: Float32 = 0.0
    var temp1: Float32 = 0.0
    # ... 126 more variables
    # Result: Only 2 wavefronts per SIMD (256/128 = 2)

# Solution: Reduce register pressure
fn good_kernel(...):
    # Reuse registers, use LDS (shared memory on AMD)
    var tile = stack_allocation[64, Float32, address_space=AddressSpace.SHARED]()
    var acc: Float32 = 0.0
    for i in range(64):
        acc += tile[i]
```

**Register budget by architecture:**

| Architecture | VGPRs/Thread | Max Occupancy |
|--------------|--------------|---------------|
| MI100 | 256 | 8 waves/SIMD |
| MI200 | 512 | 8 waves/SIMD |
| MI300X | 512 | 8 waves/SIMD |

### Issue: LDS Bank Conflicts

**Symptoms:** High LDS stall cycles

```mojo
# nocompile

from gpu.memory import AddressSpace
from memory import stack_allocation

# Problem: 32 consecutive threads access same bank
fn bad_lds_access():
    var smem = stack_allocation[1024, Float32, address_space=AddressSpace.SHARED]()
    var tid = thread_idx.x
    var val = smem[tid * 32]  # Bank conflict!

# Solution: Add padding or interleave access
fn good_lds_access():
    # Padded layout (1024 + 32 = 1056) avoids bank conflicts
    var smem = stack_allocation[1056, Float32, address_space=AddressSpace.SHARED]()
    var tid = thread_idx.x
    var row = tid // 32
    var col = tid % 32
    var val = smem[row * 33 + col]  # +1 padding per row
```

### Issue: Suboptimal MFMA Utilization

```mojo
# nocompile

# Problem: Wrong MFMA shape for matrix dimensions
fn bad_mfma_selection(M: Int, N: Int):
    # Using 32x32 for small matrix
    if M == 8 and N == 8:
        mma_shape = IndexList[3](32, 32, 8)  # BAD: Waste compute

# Solution: Select appropriate shape
fn good_mfma_selection(M: Int, N: Int) -> IndexList[3]:
    if M < 32 or N < 32:
        return IndexList[3](16, 16, 16)  # Better for small
    else:
        return IndexList[3](32, 32, 8)   # Better for large
```

---

## Cross-Platform Patterns

### AMD/NVIDIA Compatibility

```mojo
from sys import is_amd_gpu, is_nvidia_gpu

fn cross_platform_kernel(...):
    """Kernel that works on both AMD and NVIDIA."""
    @parameter
    if is_amd_gpu():
        # AMD: 64-thread wavefronts
        comptime WARP_SIZE = 64
        # Use AMD scheduling hints
        schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 2, 0)
    else:
        # NVIDIA: 32-thread warps
        comptime WARP_SIZE = 32
        # NVIDIA doesn't need explicit scheduling

    var lane_id = thread_idx.x % WARP_SIZE
    # ... rest of kernel
```

### Memory Coalescing Differences

```mojo
# nocompile

fn platform_aware_coalescing(...):
    """Coalescing rules differ slightly between vendors."""
    @parameter
    if is_amd_gpu():
        # AMD: 64-byte cache line, 64-thread wavefront
        # Optimal: 64 threads read consecutive 4-byte elements = 256 bytes
        comptime VECTOR_WIDTH = 4  # Each thread reads 4 elements
    else:
        # NVIDIA: 128-byte cache line, 32-thread warp
        # Optimal: 32 threads read consecutive 4-byte elements = 128 bytes
        comptime VECTOR_WIDTH = 4

    var tid = thread_idx.x
    var base_idx = block_idx.x * block_dim.x * VECTOR_WIDTH
    var vec_idx = base_idx + tid * VECTOR_WIDTH

    var vec = ptr.load[width=VECTOR_WIDTH](vec_idx)
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `HIP error 999` | Invalid device function | Check gfx target matches GPU |
| `HSA_STATUS_ERROR_OUT_OF_RESOURCES` | Too many registers | Reduce VGPR usage |
| `s_waitcnt ignored` | Missing fence after waitcnt | Add `s_barrier` if needed |
| `LDS allocation failed` | Exceeded 64KB per CU | Reduce shared memory or tile size |
| `Invalid wavefront count` | Block size not multiple of 64 | Use block sizes: 64, 128, 192, 256 |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **Constants** | `alias` or `comptime` | Both work in v26.1+ |
| **MFMA shapes** | Stable | Unchanged across versions |
| **Schedule barriers** | PUBLIC (`gpu.sync`) | `schedule_group_barrier`, `AMDScheduleBarrierMask` |

**Example (v26.1+):**
```mojo
comptime WARP_SIZE = 64  # AMD wavefront size
comptime MMA_M = 32
comptime MMA_N = 32
comptime MMA_K = 8

fn select_mfma_shape():
    # Shape selection logic same across versions
    pass
```

**Notes:**
- Both `alias` and `comptime` work for compile-time constants in v26.1+
- AMD MFMA instruction patterns are stable across versions
- Schedule barriers (`schedule_group_barrier`, `AMDScheduleBarrierMask`) are PUBLIC via `gpu.sync`
- `s_waitcnt` available in nightly (v26.2+)
- MI300X support and MFMA shapes remain stable

---

## Related Patterns

- [`gpu-fundamentals.md`](gpu-fundamentals.md) — Thread hierarchy basics
- [`gpu-tensor-cores.md`](gpu-tensor-cores.md) — NVIDIA tensor core patterns
- [`gpu-structured-kernels.md`](gpu-structured-kernels.md) — 8-wave ping-pong pattern for AMD CDNA4

---

## References

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD CDNA Architecture Whitepaper](https://www.amd.com/en/technologies/cdna)
- [Omniperf User Guide](https://rocm.docs.amd.com/projects/omniperf/)
- MAX Kernels AMD implementations
