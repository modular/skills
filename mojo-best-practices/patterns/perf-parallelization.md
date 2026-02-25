---
title: Multi-Core Parallelization Patterns
description: Comprehensive guide to parallelize[], work distribution, and parallel attention patterns for multi-core CPU execution
impact: CRITICAL
category: perf
tags: [parallel, multicore, parallelize, attention, work-distribution]
error_patterns:
  - "parallelize"
  - "data race"
  - "thread"
  - "concurrent"
  - "num_physical_cores"
  - "work distribution"
  - "slow performance"
scenarios:
  - "Parallelize loop across CPU cores"
  - "Distribute work evenly"
  - "Implement parallel attention"
  - "Fix data race in parallel code"
  - "Optimize multi-core performance"
  - "Achieve 1000x speedup vs Python"
consolidates:
  - perf-parallelize.md
  - perf-parallel-attention.md
---

# Multi-Core Parallelization Patterns

**Category:** perf | **Impact:** CRITICAL (10x-17,000x vs Python)

The `parallelize` function distributes work across CPU cores for embarrassingly parallel workloads. Combined with SIMD and raw pointers, this achieves up to 984x speedup vs Python and 17x vs NumPy.

---

## Core Concepts

### Performance Baseline

**Benchmark Results (1M element array addition, 16 cores, Apple M-series):**

| Implementation | Time | vs Python | vs NumPy |
|----------------|------|-----------|----------|
| Python lists | 34.463 ms | 1x | - |
| NumPy | 0.741 ms | 46x | 1x |
| Mojo List parallel+unroll | 0.099 ms | 348x | 7.5x |
| Mojo Ptr+SIMD x8 parallel | 0.043 ms | 801x | 17x |
| **Mojo Ptr+SIMD x16 parallel** | **0.035 ms** | **984x** | **21x** |

### Basic Parallelize Pattern

```mojo
from algorithm import parallelize
from sys import num_physical_cores

fn add_arrays_parallel(mut result: List[Int], a: List[Int], b: List[Int]):
    """Parallel + unrolled - the ultimate speed combination."""
    var size = len(a)
    var cores = num_physical_cores()

    @parameter
    fn worker(core_id: Int):
        var chunk = size // cores
        var start = core_id * chunk
        var end = start + chunk if core_id < cores - 1 else size

        # Manual unrolling inside each parallel chunk
        var i = start
        while i + 8 <= end:
            result[i] = a[i] + b[i]
            result[i+1] = a[i+1] + b[i+1]
            result[i+2] = a[i+2] + b[i+2]
            result[i+3] = a[i+3] + b[i+3]
            result[i+4] = a[i+4] + b[i+4]
            result[i+5] = a[i+5] + b[i+5]
            result[i+6] = a[i+6] + b[i+6]
            result[i+7] = a[i+7] + b[i+7]
            i += 8

        # Handle remainder
        while i < end:
            result[i] = a[i] + b[i]
            i += 1

    parallelize[worker](cores)
```

---

## Common Patterns

### Pattern 1: Ultimate Performance (Pointer + SIMD + Parallel)

**When:** Maximum throughput for numeric arrays

```mojo
from algorithm import parallelize
from sys import num_physical_cores
from memory import UnsafePointer
from memory import UnsafePointer
from builtin.type_aliases import MutAnyOrigin

comptime Int64Ptr = UnsafePointer[mut=True, type=Int64, origin=MutAnyOrigin]

fn add_arrays_ultimate(
    a: Int64Ptr,
    b: Int64Ptr,
    result: Int64Ptr,
    size: Int
):
    """Ultimate performance: raw pointers + SIMD x16 + parallel.

    Achieves 984x faster than Python, 21x faster than NumPy.
    """
    var cores = num_physical_cores()
    comptime WIDTH: Int = 16  # Process 16 Int64s per SIMD op

    @parameter
    fn worker(core_id: Int):
        var chunk = size // cores
        var start = core_id * chunk
        var end = start + chunk if core_id < cores - 1 else size
        var chunk_size = end - start

        var a_ptr = a + start
        var b_ptr = b + start
        var r_ptr = result + start

        # SIMD x16 loop
        var i = 0
        while i + WIDTH <= chunk_size:
            var va = a_ptr.load[width=WIDTH](i)
            var vb = b_ptr.load[width=WIDTH](i)
            r_ptr.store(i, va + vb)
            i += WIDTH

        # SIMD x8 remainder
        while i + 8 <= chunk_size:
            var va = a_ptr.load[width=8](i)
            var vb = b_ptr.load[width=8](i)
            r_ptr.store(i, va + vb)
            i += 8

        # Scalar remainder
        while i < chunk_size:
            r_ptr[i] = a_ptr[i] + b_ptr[i]
            i += 1

    parallelize[worker](cores)
```

**Architecture-specific SIMD widths:**

| Architecture | Recommended SIMD Width | Register Size |
|--------------|------------------------|---------------|
| Apple M1/M2/M3 (ARM NEON) | 8-16 for Int64 | 128-bit native |
| Intel/AMD AVX2 | 4 for Int64 | 256-bit |
| Intel/AMD AVX-512 | 8 for Int64 | 512-bit |

---

### Pattern 2: Parallel Reduction

**When:** Computing aggregates (sum, max, min) over large arrays

```mojo
from algorithm import parallelize
from sys import num_physical_cores

fn sum_parallel(data: List[Int]) -> Int:
    var size = len(data)
    var cores = num_physical_cores()
    var partial_sums = List[Int]()
    for _ in range(cores):
        partial_sums.append(0)

    @parameter
    fn sum_chunk(core_id: Int):
        var chunk = size // cores
        var start = core_id * chunk
        var end = start + chunk if core_id < cores - 1 else size
        var local_sum: Int = 0
        for i in range(start, end):
            local_sum += data[i]
        partial_sums[core_id] = local_sum

    parallelize[sum_chunk](cores)

    var total: Int = 0
    for i in range(cores):
        total += partial_sums[i]
    return total
```

---

### Pattern 3: Parallel Multi-Head Attention

**When:** CPU-based attention computation in transformers

**Impact:** 8-10% faster with proper buffer isolation

**Key Insight:** Each parallel head needs its own score buffer to avoid data races.

**Don't (shared buffer causes race conditions):**
```mojo
# BAD: All heads share the same scores buffer
var scores = alloc[Float32](seq_len * seq_len)

for h in range(num_heads):  # This loop is parallelizable...
    BLASContext.sgemm(...)  # Q[h] @ K[h]^T -> scores
    softmax(scores)         # ...but writes to shared buffer!
    BLASContext.sgemm(...)  # scores @ V[h] -> out[h]
```

**Do (per-head buffers enable safe parallelization):**
```mojo
# GOOD: Each head gets its own score buffer
var scores_base = alloc[Float32](num_heads * seq_len * seq_len)
var score_stride = seq_len * seq_len

@parameter
fn process_head(h: Int):
    var scores = scores_base + h * score_stride  # Per-head buffer
    var kv_h = h // heads_per_kv  # GQA support

    # Q @ K^T - each head writes to its own scores slice
    BLASContext.sgemm(
        CblasNoTrans, CblasTrans,
        seq_len, seq_len, head_dim,
        scale,
        q_buf + h * head_dim, q_dim,
        k_buf + kv_h * head_dim, kv_dim,
        0.0,
        scores, seq_len  # Per-head score buffer
    )

    # Softmax on per-head buffer (safe)
    for i in range(seq_len):
        var row = scores + i * seq_len
        # ... causal softmax implementation

    # scores @ V -> output (non-overlapping output slices)
    BLASContext.sgemm(
        CblasNoTrans, CblasNoTrans,
        seq_len, head_dim, seq_len,
        1.0,
        scores, seq_len,
        v_buf + kv_h * head_dim, kv_dim,
        0.0,
        attn_out + h * head_dim, q_dim
    )

# Run all heads in parallel
parallelize[process_head](num_heads, num_physical_cores())
```

**Memory Trade-off:**
- Single buffer: `seq_len * seq_len * 4` bytes (e.g., 512 * 512 * 4 = 1MB)
- Per-head buffers: `num_heads * seq_len * seq_len * 4` bytes (e.g., 32 * 512 * 512 * 4 = 32MB)

For sequence lengths up to 512 with 32 heads, the extra 31MB is acceptable for 8-10% speedup.

---

### Pattern 4: Ultimate Matrix Multiply

**When:** Performance-critical matrix operations

**Benchmark (1024x1024, Apple M-series):**
- Without optimization: 2.4 GFLOP/s
- With transpose + SIMD + parallel + unroll: **117 GFLOP/s (up to 17,000x vs Python)**

```mojo
# nocompile
from algorithm import parallelize
from sys import num_physical_cores
from memory import UnsafePointer

fn matmul_ultimate(C: Matrix, A: Matrix, B: Matrix):
    """Ultimate matmul: transpose + parallel + SIMD + unroll."""
    var M = C.rows
    var N = C.cols
    var K = A.cols
    var cores = num_physical_cores()
    var BT = transpose(B)  # Critical: enables coalesced access

    comptime TILE: Int = 128
    comptime WIDTH: Int = 8

    @parameter
    fn process_rows(core_id: Int):
        var rows_per_core = (M + cores - 1) // cores
        var row_start = core_id * rows_per_core
        var row_end = min(row_start + rows_per_core, M)

        for m0 in range(row_start, row_end, TILE):
            var m1 = min(m0 + TILE, row_end)
            for n0 in range(0, N, TILE):
                var n1 = min(n0 + TILE, N)
                for m in range(m0, m1):
                    var a_row = A.data + m * K
                    # Unroll by 4 columns
                    var n = n0
                    while n + 4 <= n1:
                        var acc0 = SIMD[DType.float64, WIDTH]()
                        var acc1 = SIMD[DType.float64, WIDTH]()
                        var acc2 = SIMD[DType.float64, WIDTH]()
                        var acc3 = SIMD[DType.float64, WIDTH]()
                        var k = 0
                        while k + WIDTH <= K:
                            var a_vec = (a_row + k).load[width=WIDTH]()
                            acc0 += a_vec * (BT.data + n*K + k).load[width=WIDTH]()
                            acc1 += a_vec * (BT.data + (n+1)*K + k).load[width=WIDTH]()
                            acc2 += a_vec * (BT.data + (n+2)*K + k).load[width=WIDTH]()
                            acc3 += a_vec * (BT.data + (n+3)*K + k).load[width=WIDTH]()
                            k += WIDTH
                        C.set(m, n, acc0.reduce_add())
                        C.set(m, n+1, acc1.reduce_add())
                        C.set(m, n+2, acc2.reduce_add())
                        C.set(m, n+3, acc3.reduce_add())
                        n += 4

    parallelize[process_rows](cores)
```

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Large arrays (>10K elements) | parallelize + SIMD | [`perf-vectorization.md`](perf-vectorization.md) |
| Reduction (sum, max, min) | Partial sums per core | - |
| Multi-head attention | Per-head buffers | - |
| Matrix multiply | Transpose B + tile + parallel | [`perf-optimization.md`](perf-optimization.md) |
| Small arrays (<10K) | SIMD only (parallelize overhead exceeds benefit) | - |

---

## Performance Scaling

| Cores | Typical Speedup |
|-------|-----------------|
| 4 | 3-4x |
| 8 | 6-8x |
| 16 | 10-14x |
| 32+ | 15-25x |

---

## Quick Reference

- **Get core count**: `num_physical_cores()`
- **Parallelize pattern**: `parallelize[worker_fn](num_workers)`
- **Chunk calculation**: `var chunk = size // cores`
- **Last chunk handling**: `var end = start + chunk if core_id < cores - 1 else size`
- **Combine with SIMD**: Use SIMD inside each worker for maximum performance
- **Avoid data races**: Each worker writes to non-overlapping regions

---

## When NOT to Parallelize

- **Small datasets** (< 10,000 elements) - overhead exceeds benefit
- **Strong data dependencies** between elements
- **Heavy memory contention** - use SIMD instead
- **I/O bound** operations

### Minimum Workload Guidance

Thread creation and synchronization have fixed overhead (~1-10 microseconds per thread spawn). For parallelization to be beneficial, the work per thread must exceed this overhead.

**Recommended minimums:**

| Operation Type | Minimum Elements | Work per Element |
|----------------|------------------|------------------|
| Simple math (add, mul) | 100,000+ | ~1-2 ops |
| Complex math (exp, sin) | 10,000+ | ~10-50 ops |
| Memory-intensive | 50,000+ | Load + store |
| Matrix operations | 1,000+ rows | Full row computation |

**Adaptive parallelization pattern:**

```mojo
# nocompile
from algorithm import parallelize
from sys import num_physical_cores

fn process_adaptive(data: List[Float64], size: Int):
    comptime MIN_PARALLEL_SIZE: Int = 100_000

    if size < MIN_PARALLEL_SIZE:
        # Small workload: use SIMD only, no parallel overhead
        process_simd_only(data, size)
    else:
        # Large workload: parallel + SIMD
        var cores = num_physical_cores()

        @parameter
        fn worker(core_id: Int):
            var chunk = size // cores
            var start = core_id * chunk
            var end = start + chunk if core_id < cores - 1 else size
            process_simd_chunk(data, start, end)

        parallelize[worker](cores)
```

**Benchmarking parallelization benefit:**

```mojo
# nocompile
from time import perf_counter_ns

fn benchmark_parallel_benefit(size: Int):
    # Measure single-threaded
    var start = perf_counter_ns()
    process_serial(data, size)
    var serial_time = perf_counter_ns() - start

    # Measure parallel
    start = perf_counter_ns()
    process_parallel(data, size)
    var parallel_time = perf_counter_ns() - start

    var speedup = Float64(serial_time) / Float64(parallel_time)
    print("Size:", size, "Speedup:", speedup)

    # If speedup < 1.5x, parallelization overhead dominates
    if speedup < 1.5:
        print("WARNING: Parallelization not beneficial at this size")
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `data race` | Multiple threads writing to same memory | Use atomic operations or partition data |
| `slow parallel performance` | Work too small for overhead | Increase chunk size or use SIMD instead |
| `inconsistent results` | Race condition in reduction | Use proper reduction pattern with barriers |
| `load imbalance` | Uneven work distribution | Use dynamic scheduling or balanced chunks |
| `out of memory` | Too many parallel allocations | Reuse buffers, reduce parallelism |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **parallelize** | `from algorithm import parallelize` | Stable |
| **num_physical_cores** | `from sys import num_physical_cores` | Stable |
| **Constants** | `alias` or `comptime` | Both work in v26.1+ |
| **Heap allocation** | `from memory import alloc; alloc[T](n)` | v26.1+ |

**Example (v26.1+):**
```mojo
from algorithm import parallelize
from sys import num_physical_cores

comptime CHUNK_SIZE = 64

fn parallel_work[func: fn(Int) -> None](size: Int):
    var cores = num_physical_cores()

    @parameter
    fn worker(core_id: Int):
        var chunk = size // cores
        var start = core_id * chunk
        var end = start + chunk if core_id < cores - 1 else size
        for i in range(start, end):
            func(i)

    parallelize[worker](cores)
```

**Notes:**
- `parallelize[]` API is stable across versions
- `num_physical_cores()` is stable across versions
- Both `alias` and `comptime` work for compile-time constants in v26.1+
- Work distribution and chunking patterns are stable

### Alternative: sync_parallelize

Modular's internal kernels use `sync_parallelize` from `algorithm.sync` for some workloads:

```mojo
# nocompile
from algorithm.sync import sync_parallelize

@parameter
fn task_func(task_id: Int):
    # process chunk for task_id
    pass

sync_parallelize[task_func](num_tasks)
```

This is functionally similar to `parallelize` but may have different synchronization semantics. Check Modular's latest kernel implementations for recommended patterns when migrating existing code.

---

## Related Patterns

- [`perf-vectorization.md`](perf-vectorization.md) — SIMD patterns to combine with parallel
- [`perf-optimization.md`](perf-optimization.md) — Memory layout for parallel efficiency
- [`perf-optimization.md`](perf-optimization.md) — General optimization strategies

---

## References

- [Mojo Algorithm Module](https://docs.modular.com/mojo/std/algorithm/)
- [Mojo sys Module](https://docs.modular.com/mojo/std/sys/)
