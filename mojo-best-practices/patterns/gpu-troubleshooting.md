---
title: GPU Troubleshooting
description: Systematic diagnosis of GPU build failures, performance issues, and integration problems
impact: HIGH
category: gpu
tags: [gpu, troubleshooting, profiling, debugging, performance, occupancy]
error_patterns:
  - "kernel launch failed"
  - "register spill"
  - "occupancy"
  - "bank conflict"
  - "uncoalesced"
  - "address space"
  - "cannot implicitly convert"
  - "@compiler.register"
  - "does not support operation"
  - "kernel changes not reflected"
scenarios:
  - "Debug GPU kernel compilation failure"
  - "Diagnose poor GPU kernel performance"
  - "Profile GPU kernel bandwidth efficiency"
  - "Fix MAX Graph custom op integration"
  - "Troubleshoot platform-specific GPU issues"
  - "Resolve kernel caching problems"
consolidates: []
---

# GPU Troubleshooting

**Category:** gpu | **Impact:** HIGH

Systematic diagnosis of GPU build failures, runtime errors, performance bottlenecks, and integration problems. This guide covers operational GPU issues — for numerical correctness and accuracy debugging, see [`debug-debugging.md`](debug-debugging.md).

## API Availability

> **Note:** All APIs referenced below are available in the Mojo nightly toolchain (v26.2+). Code examples are documentation snippets — adapt import paths and parameters for your use case.

| API | Import Path | Notes |
|-----|-------------|-------|
| `LayoutTensor`, `Layout` | `from layout import LayoutTensor, Layout` | Type-safe tensor with compile-time layout |
| `AddressSpace` | `from gpu.memory import AddressSpace` | Address space enum (SHARED, GENERIC) |
| `rebind` | Built-in (prelude) | Type reinterpretation, no runtime cost |
| `stack_allocation` | `from memory import stack_allocation` | Shared memory (compile-time size) |
| `external_memory` | `from gpu.memory import external_memory` | Dynamic shared memory (NVIDIA/AMD only) |
| `barrier` | `from gpu import barrier` | Block synchronization |
| `perf_counter_ns` | `from time import perf_counter_ns` | Host-side timing (returns `UInt`) |
| `is_apple_gpu` | `from sys import is_apple_gpu` | Platform detection |

---

## Build/Compilation Failures

### Type Mismatches: LayoutTensor Element Type

Reading from a `LayoutTensor` via indexing returns `SIMD[dtype, symbolic_size]`, not `Scalar[dtype]`. This is the #1 compilation error.

**Error:**
```
cannot implicitly convert 'SIMD[float32, symbolic_size]' value to 'SIMD[float32, 1]'
```

**Fix:** Wrap every read with `rebind`:

```mojo
# nocompile

# WRONG — fails to compile
var val: Float32 = tensor[tid]

# CORRECT — rebind on every read
var val = rebind[Scalar[DType.float32]](tensor[tid])
```

For accumulation, rebind both sides:

```mojo
# nocompile

# CORRECT — rebind both reads in an expression
shared[tid] = rebind[Scalar[dtype]](shared[tid]) + rebind[Scalar[dtype]](shared[tid + stride])
```

See [`gpu-layout-tensor.md`](gpu-layout-tensor.md) for complete rebind patterns.

### Address Space Errors

**Error:**
```
cannot implicitly convert value of type '...AddressSpace.GENERIC...' to '...AddressSpace.SHARED...'
```

**Fix:** Match the address space to the allocation method — shared memory requires `AddressSpace.SHARED`, global memory uses `AddressSpace.GENERIC` (default):

```mojo
# nocompile
from gpu.memory import AddressSpace
from memory import stack_allocation

# Shared memory: must specify AddressSpace.SHARED
var smem = stack_allocation[256, Float32, address_space=AddressSpace.SHARED]()
```

### Missing `block_size` on Block Collectives

All `gpu.primitives.block.*` functions require `[block_size=N]`. Without it, kernels may silently produce wrong results or **fail entirely on Apple Silicon**.

```mojo
# nocompile
from gpu.primitives.block import sum as block_sum

comptime TPB = 256

# WRONG — missing block_size, may fail on Apple Silicon
var total = block_sum(my_val)

# CORRECT — always specify block_size
var total = block_sum[block_size=TPB](my_val)
```

See [`gpu-block-collectives.md`](gpu-block-collectives.md) for all collective operations.

### Common Import Path Mistakes

| Wrong Import | Correct Import | Notes |
|-------------|---------------|-------|
| `from gpu.id import thread_idx` | `from gpu import thread_idx` | `gpu.id` removed in v26.1+ |
| `from gpu.id import block_idx` | `from gpu import block_idx` | Same |
| `from gpu.host import DeviceContext, Dim` | `from gpu.host import DeviceContext, Dim` | Unchanged |
| `from gpu import AddressSpace` | `from gpu.memory import AddressSpace` | AddressSpace is in `gpu.memory` |

---

## Performance Diagnosis

### Bank Conflicts

**Symptom:** Poor shared memory throughput despite correct results. Shared memory bandwidth far below theoretical peak.

**Cause:** Multiple threads in the same warp access different addresses that map to the same memory bank. Shared memory has 32 banks (NVIDIA) — addresses are interleaved across banks at 4-byte granularity.

**Diagnosis:** If your shared memory tile width is a multiple of 32, adjacent rows map to the same bank.

**Fix:** Add +1 padding to the row size:

```mojo
# nocompile

# BANK CONFLICTS — 32 columns maps perfectly to 32 banks
comptime TILE = 32
var smem = LayoutTensor[dtype, Layout.row_major(TILE, TILE),
                        MutAnyOrigin, address_space=AddressSpace.SHARED]
    .stack_allocation()

# NO BANK CONFLICTS — 33 columns staggers bank mapping
comptime TILE = 32
comptime PADDED = TILE + 1  # Add 1 to break bank alignment
var smem = LayoutTensor[dtype, Layout.row_major(TILE, PADDED),
                        MutAnyOrigin, address_space=AddressSpace.SHARED]
    .stack_allocation()
```

### Low Occupancy

**Symptom:** GPU utilization is low. Profiler shows few active warps per SM.

**Cause:** Each SM has a fixed register file (typically 65536 registers on NVIDIA). Occupancy depends on register usage per thread and threads per block.

**Formula:**
```
max_blocks_per_SM = min(
    register_file_size / (registers_per_thread * threads_per_block),
    max_blocks_per_SM_limit,
    shared_mem_per_SM / shared_mem_per_block
)

occupancy = (active_warps / max_warps_per_SM) * 100%
```

**Fix strategies:**
- Reduce register pressure: use smaller tile sizes, fewer local variables
- Reduce shared memory per block: use smaller tiles, shared memory across stages
- Adjust block size: sometimes smaller blocks allow more concurrent blocks per SM

### Uncoalesced Memory Access

**Symptom:** Memory bandwidth far below peak. Profiler reports excessive memory transactions.

**Cause:** Adjacent threads in a warp access non-adjacent memory addresses. GPUs load memory in 128-byte (32-word) cache lines — scattered accesses waste bandwidth.

**Fix:** Ensure adjacent threads (consecutive `thread_idx.x`) access adjacent memory addresses:

```mojo
# nocompile

# UNCOALESCED — threads stride across rows (column-major access of row-major data)
var val = rebind[Scalar[dtype]](data[thread_idx.x * N + col])

# COALESCED — adjacent threads access adjacent elements
var val = rebind[Scalar[dtype]](data[row * N + thread_idx.x])
```

### Register Spills

**Symptom:** Compiler warning: `register spill to local memory`. Performance degrades due to local memory (DRAM) traffic.

**Cause:** Kernel uses more registers per thread than the hardware limit. Excess values "spill" to slow local memory.

**Fix:**
- Reduce tile size (fewer values per thread)
- Split a large kernel into multiple smaller kernels
- Reduce local arrays — use shared memory instead of register arrays
- Minimize live variables — recompute rather than storing intermediates

---

## Profiling Methodology

### Timing GPU Kernels

Use `perf_counter_ns()` for host-side timing. Warm up with 5+ iterations, then average over `num_iters`:

```mojo
# nocompile
from time import perf_counter_ns

for _ in range(5):  # Warmup
    ctx.enqueue_function[my_kernel, my_kernel](args..., grid_dim=grid, block_dim=block)
ctx.synchronize()

var start = perf_counter_ns()
for _ in range(num_iters):
    ctx.enqueue_function[my_kernel, my_kernel](args..., grid_dim=grid, block_dim=block)
ctx.synchronize()
var elapsed_ns = Int(perf_counter_ns() - start) // num_iters
```

### Bandwidth Efficiency

Calculate effective bandwidth and compare against peak:

```
actual_GB_s = total_bytes / (elapsed_ns / 1e9) / 1e9
efficiency = actual_GB_s / peak_GB_s * 100
```

**Peak Memory Bandwidth:**

| GPU | Peak BW (GB/s) | Typical Efficiency Target |
|-----|---------------|--------------------------|
| H100 SXM | 3350 | > 70% for memory-bound kernels |
| H100 PCIe | 2000 | > 70% |
| A100 80GB | 2000 | > 70% |
| L40S | 864 | > 60% |
| MI300X | 5300 | > 60% |

### Roofline Model

Determine whether your kernel is compute-bound or memory-bound:

```
operational_intensity = FLOPs / bytes_transferred
ridge_point = peak_FLOPS / peak_bandwidth

operational_intensity < ridge_point → memory-bound (optimize coalescing, tiling)
operational_intensity > ridge_point → compute-bound (use tensor cores, increase tiles)
```

---

## Integration Issues

### `@compiler.register` Pitfalls

Custom ops registered with `@compiler.register` have specific requirements that differ from standalone kernels.

**Double-generic `enqueue_function` requirement:**

```mojo
# nocompile

# WRONG — fails in custom op context
ctx.enqueue_function[my_kernel](args..., grid_dim=grid, block_dim=block)

# CORRECT — pass kernel function twice
ctx.enqueue_function[my_kernel, my_kernel](args..., grid_dim=grid, block_dim=block)
```

See [`gpu-kernels.md`](gpu-kernels.md) for the full custom op registration pattern.

### `to_layout_tensor()` and `rebind`

`InputTensor` and `OutputTensor` from the graph system convert to `LayoutTensor` via `to_layout_tensor()`. Use `rebind` when you need a specific known layout:

```mojo
# nocompile
from max.tensor import InputTensor, OutputTensor

@compiler.register("my_op")
struct MyOp:
    @staticmethod
    fn execute[dtype: DType, rank: Int](
        output: OutputTensor[rank=rank],
        input: InputTensor[rank=rank],
        ctx: DeviceContextPtr,
    ):
        var out = output.to_layout_tensor()
        var inp = input.to_layout_tensor()

        # If you know the exact layout at compile time:
        var typed = rebind[LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin]](out)
```

### Kernel Caching — Stale Kernels

**Symptom:** Kernel changes are not reflected at runtime. Old behavior persists despite code edits.

**Cause:** Mojo and MAX cache compiled kernels. Stale cache entries serve the old kernel.

**Fix:** Clear all caches:

```bash
# Clear all Mojo/MAX kernel caches
rm -rf .max_cache/ .mojo_cache/ .mogg_cache/

# Or use the convenience aliases (if available)
clear-mojo-cache
clear-max-cache
clear-cache
```

Always clear caches when:
- Kernel behavior doesn't match source code
- Switching between GPU targets (e.g., NVIDIA → Apple Metal)
- Updating the Mojo toolchain version

---

## Platform-Specific Issues

| Issue | NVIDIA | AMD | Apple Metal |
|-------|--------|-----|-------------|
| **Warp/wavefront/SIMD width** | 32 threads | 64 threads | 32 SIMD lanes |
| **Block size alignment** | Multiple of 32 | Multiple of 64 | Multiple of 32 |
| **Shared memory allocation** | `external_memory` (dynamic) | `external_memory` (dynamic) | `stack_allocation` (compile-time fixed) |
| **`@__copy_capture`** | Not required (implicit) | Not required (implicit) | **REQUIRED** on all nested functions |
| **`print()` in kernels** | Not supported | Not supported | Not supported |
| **Max threads per block** | 1024 | 1024 | 1024 |
| **Shared memory per block** | 48KB default (configurable to 100KB+) | 64KB LDS | 32KB threadgroup |

### Apple Metal: `@__copy_capture` Required

Functions passed to `elementwise` or `foreach` **must** use `@__copy_capture` to list captured variables. Without this, kernels compile but execute with zeros/garbage on Metal.

```mojo
# nocompile

var out_ptr = device_buffer.unsafe_ptr()

# WRONG — captured variable not on GPU
@parameter
fn bad_func(idx: Int) capturing:
    out_ptr[idx] = 42.0  # Reads zero on Metal

# CORRECT — explicitly capture all variables
@parameter
@__copy_capture(out_ptr)
fn good_func(idx: Int):
    out_ptr[idx] = 42.0  # Works on Metal
```

### Apple Metal: Shared Memory

Apple Metal requires compile-time known shared memory sizes. `external_memory` (dynamic allocation) fails on Metal — use `is_apple_gpu()` + `stack_allocation`. See the Apple Metal section in [`gpu-fundamentals.md`](gpu-fundamentals.md) for the full conditional compilation pattern.

### AMD: Wavefront Size

AMD GPUs use 64-thread wavefronts. Block sizes must be multiples of 64:

```mojo
# nocompile

# NVIDIA: warp size = 32
comptime NVIDIA_BLOCK = 256  # 8 warps

# AMD: wavefront size = 64
comptime AMD_BLOCK = 256     # 4 wavefronts (must be multiple of 64)
```

See [`gpu-amd.md`](gpu-amd.md) for AMD-specific MFMA shapes, scheduling barriers, and `s_waitcnt` patterns.

---

## Quick Reference: Symptom → Diagnosis → Fix

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `cannot implicitly convert 'SIMD[..., symbolic_size]'` | LayoutTensor read without `rebind` | `rebind[Scalar[dtype]](tensor[i])` — see [`gpu-layout-tensor.md`](gpu-layout-tensor.md) |
| `cannot implicitly convert '...GENERIC...' to '...SHARED...'` | Wrong address space on pointer/tensor | Match `AddressSpace` to allocation method |
| `kernel launch failed` | Invalid grid/block dimensions | Ensure block size ≤ 1024, grid size > 0 |
| `GPU OOM` / `out of memory` | Exceeding GPU VRAM | Reduce batch size, use smaller tensors |
| `does not support operation: print` | `print()` in GPU kernel | Remove print, use output buffer for debug values |
| `register spill to local memory` | Too many registers per thread | Reduce tile size, split kernel, recompute intermediates |
| `Invalid wavefront count` (AMD) | Block size not multiple of 64 | Use block sizes: 64, 128, 192, 256 |
| Metal GPU returns zeros/garbage | Missing `@__copy_capture` on nested functions | Add `@__copy_capture(var1, var2, ...)` to ALL functions in the kernel call chain |
| `external_memory` fails on Apple | Metal requires compile-time shared memory | Use `is_apple_gpu()` + `stack_allocation` — see [`gpu-fundamentals.md`](gpu-fundamentals.md) |
| Kernel changes not reflected | Stale kernel cache | `rm -rf .max_cache/ .mojo_cache/ .mogg_cache/` |
| `no matching method in call to 'enqueue_function'` | Missing double-generic in custom op | `ctx.enqueue_function[kernel, kernel](...)` — see [`gpu-kernels.md`](gpu-kernels.md) |
| Low GPU utilization | Low occupancy | Reduce registers/thread or shared memory/block |
| Memory bandwidth far below peak | Uncoalesced access pattern | Ensure adjacent threads access adjacent addresses |
| Shared memory throughput low | Bank conflicts | Add +1 padding to shared memory row width |
| Block collectives wrong on Apple | Missing `block_size` parameter | Always pass `[block_size=N]` — see [`gpu-block-collectives.md`](gpu-block-collectives.md) |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **Block collectives** | Require `block_size` param | `sum[block_size=N]`, `broadcast[block_size=N]` |
| **`@__copy_capture`** | Required on Apple Metal | Explicit capture list for GPU closures |
| **`external_memory`** | NVIDIA/AMD only | Apple uses `stack_allocation` instead |
| **`perf_counter_ns`** | Returns `UInt` | Host-side timing, stable across versions |
| **Cache paths** | `~/.modular/.max_cache`, `.mojo_cache`, `.mogg_cache` | Purge when kernel changes aren't reflected |

---

## Related Patterns

- [`debug-debugging.md`](debug-debugging.md) — Numerical accuracy and correctness debugging
- [`gpu-fundamentals.md`](gpu-fundamentals.md) — Thread hierarchy, memory model, hardware guidance
- [`gpu-kernels.md`](gpu-kernels.md) — Kernel optimization, custom ops, producer-consumer pipelines
- [`gpu-layout-tensor.md`](gpu-layout-tensor.md) — LayoutTensor API, rebind patterns, shared memory allocation
- [`gpu-block-collectives.md`](gpu-block-collectives.md) — Block-wide sum, prefix_sum, broadcast operations
- [`gpu-amd.md`](gpu-amd.md) — AMD MFMA shapes, scheduling barriers, wavefront patterns
- [`gpu-memory-access.md`](gpu-memory-access.md) — Coalesced access, shared memory tiling

---

## References

- [Mojo GPU Fundamentals](https://docs.modular.com/mojo/manual/gpu/fundamentals)
- [Mojo GPU Block and Warp](https://docs.modular.com/mojo/manual/gpu/block-and-warp)
- [NVIDIA Occupancy Calculator](https://developer.nvidia.com/cuda-occupancy-calculator)
- [AMD ROCm Profiling (Omniperf)](https://rocm.docs.amd.com/projects/omniperf/)
