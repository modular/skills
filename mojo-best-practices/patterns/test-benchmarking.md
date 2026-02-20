---
title: Mojo Benchmarking Patterns
description: Performance benchmarking patterns including QuickBench, proper timing methodology, and performance testing best practices
impact: HIGH
category: test
tags: [benchmark, performance, timing, quickbench, profiling, gpu, bandwidth]
error_patterns:
  - "benchmark"
  - "QuickBench"
  - "perf_counter"
  - "timing"
  - "throughput"
scenarios:
  - "Benchmark function performance"
  - "Compare algorithm implementations"
  - "Measure SIMD speedup"
  - "Create performance regression tests"
  - "Measure throughput metrics"
  - "Benchmark GPU kernel bandwidth"
  - "Calculate memory bandwidth efficiency"
consolidates:
  - benchmark-patterns.md
  - quickbench-patterns.md
  - perf-testing.md
---

# Mojo Benchmarking Patterns

**Category:** test | **Impact:** HIGH

Proper benchmarking requires careful methodology: warmup iterations to stabilize caches, multiple samples for statistical significance, and techniques to prevent compiler optimization from eliminating measured code.

---

## Core Concepts

### Basic Benchmark Structure

Use proper benchmarking methodology with warmup iterations, multiple samples, and appropriate timing. Note: `perf_counter_ns()` returns `UInt`.

**Pattern:**

```mojo
from time import perf_counter_ns

fn operation_to_benchmark() -> Int:
    var sum = 0
    for i in range(1000):
        sum += i
    return sum

fn benchmark_operation() -> Float64:
    """Benchmark with warmup and multiple iterations."""
    comptime WARMUP_ITERS: Int = 5
    comptime BENCH_ITERS: Int = 100

    # Warmup - don't measure these
    for _ in range(WARMUP_ITERS):
        _ = operation_to_benchmark()

    # Actual measurement (UInt because perf_counter_ns returns UInt)
    var total_ns: UInt = 0
    for _ in range(BENCH_ITERS):
        var start = perf_counter_ns()
        _ = operation_to_benchmark()
        var end = perf_counter_ns()
        total_ns += end - start

    # Return average in milliseconds
    return Float64(total_ns) / Float64(BENCH_ITERS) / 1_000_000.0

fn main():
    var avg_ms = benchmark_operation()
    print("Average time:", avg_ms, "ms")
```

### Best-of-N Pattern

**Pattern:**

```mojo
from time import perf_counter_ns

fn operation_to_benchmark() -> Int:
    var sum = 0
    for i in range(1000):
        sum += i
    return sum

fn benchmark_best_of_n[N: Int]() -> UInt:
    """Return the best (minimum) time from N runs."""
    comptime WARMUP: Int = 3

    # Warmup
    for _ in range(WARMUP):
        _ = operation_to_benchmark()

    var best_ns: UInt = UInt.MAX
    for _ in range(N):
        var start = perf_counter_ns()
        _ = operation_to_benchmark()
        var elapsed = perf_counter_ns() - start
        if elapsed < best_ns:
            best_ns = elapsed

    return best_ns

fn main():
    var best_ns = benchmark_best_of_n[10]()
    print("Best time:", best_ns, "ns")
```

### Preventing Compiler Optimization

Use `keep()` and `clobber_memory()` to prevent dead code elimination.

**Pattern:**

```mojo
# nocompile
from benchmark import keep, clobber_memory

fn benchmark_kernel():
    var result = expensive_computation()

    # CRITICAL: Prevent optimizer from eliminating result
    keep(result)

    # CRITICAL: Prevent optimizer from reordering memory ops
    clobber_memory()
```

### Benchmarking SIMD Operations

**Pattern:**

```mojo
from time import perf_counter_ns
from benchmark import keep

fn benchmark_simd_vs_scalar():
    comptime SIZE: Int = 1024 * 1024
    comptime ITERS: Int = 100

    var data = List[Float32](capacity=SIZE)
    for i in range(SIZE):
        data.append(Float32(i))

    # Scalar benchmark
    var scalar_start = perf_counter_ns()
    for _ in range(ITERS):
        var sum: Float32 = 0.0
        for i in range(SIZE):
            sum += data[i]
        keep(sum)
    var scalar_ns = perf_counter_ns() - scalar_start  # UInt type

    # SIMD benchmark
    var simd_start = perf_counter_ns()
    for _ in range(ITERS):
        var ptr = data.unsafe_ptr()
        var sum = SIMD[DType.float32, 8](0.0)
        for i in range(0, SIZE, 8):
            sum += ptr.offset(i).load[width=8]()
        keep(sum.reduce_add())
    var simd_ns = perf_counter_ns() - simd_start  # UInt type

    print("Scalar:", scalar_ns / ITERS, "ns")
    print("SIMD:", simd_ns / ITERS, "ns")
    print("Speedup:", Float64(scalar_ns) / Float64(simd_ns), "x")
```

---

## QuickBench Structured Benchmarking

### Basic QuickBench Usage

The `QuickBench` API provides structured benchmarking with throughput metrics and proper methodology.

**Incorrect (naive timing):**

```mojo
# nocompile
# BAD: Susceptible to optimizer, no warmup, single sample
from time import now

def bad_benchmark():
    var start = now()
    var result = my_function()  # Might be optimized away!
    var elapsed = now() - start
    print("Time:", elapsed)
```

**Correct (QuickBench with proper methodology):**

```mojo
from benchmark import QuickBench, BenchId, BenchMetric, ThroughputMeasure
from benchmark import keep, clobber_memory

fn my_function(x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return x * x + x

def main():
    var qb = QuickBench()

    var input = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)

    qb.run(
        my_function,
        input,
        bench_id=BenchId("my_function"),
        measures=[
            ThroughputMeasure(BenchMetric.elements, 4),  # 4 elements per call
        ],
    )

    qb.dump_report()
```

### Using the `run[]` Function

**Pattern:**

```mojo
# nocompile
from benchmark import run

fn my_kernel():
    # Work to benchmark
    var data = compute_something()
    keep(data)
    clobber_memory()

def main():
    var report = run[func2=my_kernel](
        min_runtime_secs=0.1,   # Run for at least 100ms
        max_runtime_secs=1.0,   # Stop after 1s
        max_iters=10000,        # Or after 10k iterations
    )

    # Access results
    print("Mean time:", report.mean(), "s")
    print("Mean time:", report.mean("ms"), "ms")
    print("Mean time:", report.mean("us"), "us")
    print("Iterations:", report.iters())

    # Full report
    print(report.as_string())
```

### Multiple Benchmark Comparison

**Pattern:**

```mojo
from benchmark import QuickBench, BenchId, BenchMetric, ThroughputMeasure
import math

@always_inline
fn exp_simd(x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return math.exp(x)

@always_inline
fn tanh_simd(x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return math.tanh(x)

@always_inline
fn manual_exp(x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    # Custom implementation
    return 1.0 + x + x*x/2.0 + x*x*x/6.0

def main():
    var qb = QuickBench()
    var input = SIMD[DType.float32, 4](0.5)

    # Benchmark multiple implementations
    qb.run(exp_simd, input, bench_id=BenchId("math.exp"))
    qb.run(tanh_simd, input, bench_id=BenchId("math.tanh"))
    qb.run(manual_exp, input, bench_id=BenchId("manual_exp"))

    qb.dump_report()  # Prints comparison table
```

### Throughput Metrics

**Pattern:**

```mojo
# nocompile
# Define what you're measuring
measures = [
    ThroughputMeasure(BenchMetric.elements, 1024),    # 1024 elements
    ThroughputMeasure(BenchMetric.bytes, 1024 * 4),   # 4KB
    ThroughputMeasure(BenchMetric.flops, 1024 * 2),   # 2 FLOPS per element
]
```

| Metric | Use Case |
|--------|----------|
| `BenchMetric.elements` | Array processing |
| `BenchMetric.bytes` | Memory bandwidth |
| `BenchMetric.flops` | Compute throughput |

---

## GPU Kernel Benchmarking

When benchmarking GPU kernels, raw execution time is less useful than bandwidth efficiency — the percentage of theoretical peak memory bandwidth achieved.

### Bandwidth Efficiency Formula

```
actual_GB_s = bytes_moved / (time_ns / 1e9) / 1e9
efficiency = actual_GB_s / peak_GB_s * 100
```

### Peak Memory Bandwidth Reference (as of Feb 2026)

| GPU | Architecture | Peak BW (GB/s) |
|---|---|---|
| H100 SXM | SM90 | 3350 |
| H100 PCIe | SM90 | 2000 |
| A100 80GB | SM80 | 2000 |
| L40S | SM89 | 864 |
| MI300X | CDNA3 | 5300 |
| Apple M3 Max | Apple GPU | ~400 |

### Efficiency Verdicts

| Efficiency | Assessment | Action |
|---|---|---|
| >60% | Excellent | Memory-bound kernel well-optimized |
| 30-60% | Good | Check coalescing and bank conflicts |
| 10-30% | Needs work | Profile for uncoalesced access, bank conflicts |
| <10% | Compute-bound or broken | Check if compute-bound (expected) or severely uncoalesced |

### Amdahl's Law Warning

A 2.67x kernel speedup sounds impressive, but if the kernel is only 2.3% of total pipeline time, the end-to-end improvement is ~1.5%. Always measure the kernel's contribution to total runtime before optimizing.

### A/B Benchmarking Methodology

- **Warmup:** 5 iterations (discard) — stabilizes GPU clocks and caches
- **Measurement:** 100 iterations
- **Report:** mean, median, p99, and min (min = best achievable)
- Use `keep()` from `benchmark` module to prevent dead code elimination

### GPU Bandwidth Benchmark Pattern

```mojo
# nocompile

from time import perf_counter_ns
from gpu.host import DeviceContext

fn benchmark_kernel_bw[
    kernel_fn: fn(...) capturing -> None,
](
    ctx: DeviceContext,
    bytes_moved: Int,
    peak_bw_gb_s: Float64,
) raises -> Float64:
    """Benchmark GPU kernel and return bandwidth efficiency percentage."""
    comptime WARMUP: Int = 5
    comptime ITERS: Int = 100

    # Warmup
    for _ in range(WARMUP):
        kernel_fn(...)
        ctx.synchronize()

    # Measure
    var total_ns: UInt = 0
    for _ in range(ITERS):
        var start = perf_counter_ns()
        kernel_fn(...)
        ctx.synchronize()
        total_ns += perf_counter_ns() - start

    var avg_ns = Float64(total_ns) / Float64(ITERS)
    var actual_gb_s = Float64(bytes_moved) / avg_ns  # bytes/ns = GB/s
    var efficiency = actual_gb_s / peak_bw_gb_s * 100.0
    return efficiency
```

**Key difference from CPU benchmarks:** Always call `ctx.synchronize()` after each kernel launch to ensure timing captures actual GPU execution, not just launch overhead.

---

## Decision Guide

| Scenario | Approach |
|----------|----------|
| Quick one-off timing | Basic benchmark with warmup |
| Structured comparison | QuickBench with multiple benchmarks |
| Algorithm comparison | Best-of-N pattern |
| Throughput measurement | QuickBench with ThroughputMeasure |
| CI regression testing | Benchmark report to JSON |
| GPU kernel bandwidth | Bandwidth efficiency benchmark |

---

## Benchmark Best Practices

| Practice | Rationale |
|----------|-----------|
| **Warmup iterations (3-10)** | Stabilize caches, JIT, memory |
| **Multiple samples (10-100)** | Statistical significance |
| **Report best/avg/worst** | Understand variance |
| **Use perf_counter_ns()** | Nanosecond precision |
| **Prevent dead code elimination** | `keep()` or return results |
| **Isolate the operation** | Exclude setup/teardown from timing |

---

## Common Benchmark Mistakes

| Mistake | Problem | Solution |
|---------|---------|----------|
| No warmup | Cold cache effects | Add 3-10 warmup iterations |
| Single measurement | High variance | Take 10+ samples |
| Including setup in timing | Misleading results | Time only the operation |
| Compiler optimizing away code | 0ns measurements | Use `keep()` or return values |

---

## When to Apply

### Use Benchmarks for:
- Performance-critical code optimization
- Comparing algorithm implementations
- Regression testing for performance
- Profiling hot paths
- Measuring SIMD vs scalar speedup

### Don't use Benchmarks for:
- Correctness testing (use TestSuite)
- One-time timing measurements
- Integration/system tests

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **Constants** | `alias` or `comptime` | Both work in v26.1+ |
| **perf_counter_ns** | Returns `UInt` | Stable |
| **QuickBench API** | Available | Stable |
| **keep/clobber_memory** | Available | Stable |

**Example (v26.1+):**
```mojo
# nocompile
from time import perf_counter_ns
from benchmark import keep

comptime WARMUP_ITERS = 5
comptime BENCH_ITERS = 100

fn bench():
    for _ in range(WARMUP_ITERS):
        _ = operation()  # Replace with your function
    var total: UInt = 0
    for _ in range(BENCH_ITERS):
        var start = perf_counter_ns()
        var result = operation()  # Replace with your function
        total += perf_counter_ns() - start
        keep(result)
```

---

## Related Patterns

- [`test-testing.md`](test-testing.md) — Unit testing and test organization
- [`perf-vectorization.md`](perf-vectorization.md) — SIMD code to benchmark
- [`perf-parallelization.md`](perf-parallelization.md) — Parallel code benchmarking

---

## References

- [Mojo Benchmark Module](https://github.com/modular/modular/blob/main/mojo/stdlib/std/benchmark/)
- [QuickBench API](https://github.com/modular/modular/blob/main/mojo/stdlib/std/benchmark/quick_bench.mojo)
