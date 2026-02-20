---
title: Memory Optimization Patterns
description: Comprehensive guide to memory alignment, data layout, prefetching, stack vs heap allocation, multiple accumulators, and tiled processing
impact: HIGH
category: perf
tags: [memory, alignment, layout, prefetch, cache, tiling, accumulators]
error_patterns:
  - "cache miss"
  - "alignment"
  - "unaligned"
  - "memory bandwidth"
  - "false sharing"
  - "cache line"
  - "slow memory access"
  - "prefetch"
scenarios:
  - "Optimize memory access pattern"
  - "Align data for SIMD"
  - "Use prefetching"
  - "Choose stack vs heap"
  - "Implement tiled processing"
  - "Reduce cache misses"
  - "Use multiple accumulators"
consolidates:
  - perf-memory-alignment.md
  - perf-memory-layout.md
  - perf-memory-prefetch.md
  - memory-borrow-vs-copy.md
  - perf-multiple-accumulators.md
  - perf-tiled-processing.md
---

# Memory Optimization Patterns

**Category:** perf | **Impact:** HIGH

Memory access patterns are often the primary bottleneck in high-performance code. This pattern covers alignment for SIMD, data layout optimization, prefetching, stack vs heap decisions, multiple accumulators for instruction-level parallelism, and tiled processing for cache efficiency.

---

## Core Concepts

### Memory Hierarchy Impact

| Level | Latency | Size | Optimization |
|-------|---------|------|--------------|
| L1 Cache | ~1 ns | 32-64 KB | Sequential access, prefetch |
| L2 Cache | ~4 ns | 256-512 KB | Tiling, blocking |
| L3 Cache | ~10 ns | 4-32 MB | Data layout |
| Main Memory | ~100 ns | GBs | Minimize access |

### Cache Line Alignment

Modern CPUs fetch memory in 64-byte cache lines. Proper alignment ensures:
- Single cache line fetch for SIMD vectors
- No false sharing in parallel code
- Optimal prefetcher behavior

```mojo
# nocompile
@align(64)  # Align to cache line (64 bytes)
struct AlignedBuffer:
    var data: SIMD[DType.float32, 16]

    fn __init__(out self):
        self.data = SIMD[DType.float32, 16]()
```

---

## Common Patterns

### Pattern 1: Struct Alignment for SIMD

**When:** Structs containing SIMD vectors or performance-critical data

**Common alignment values:**

| Alignment | Use Case |
|-----------|----------|
| 16 bytes | SSE vectors (128-bit) |
| 32 bytes | AVX vectors (256-bit) |
| 64 bytes | AVX-512 vectors, cache lines |
| 128 bytes | Some GPU requirements |

**Do:**
```mojo
# nocompile
@align(32)
struct GoodBuffer:
    var data: SIMD[DType.float32, 8]  # 32 bytes, properly aligned
    var header: Int8

fn process(b: GoodBuffer):
    # Aligned loads are fast and safe
    pass
```

**Don't:**
```mojo
struct BadBuffer:
    var header: Int8
    var data: SIMD[DType.float32, 8]  # Likely unaligned

fn process(b: BadBuffer):
    # Unaligned loads are slower or may crash on some architectures
    pass
```

**Prevent false sharing in parallel code:**
```mojo
# nocompile
@align(64)  # Cache line alignment
struct CacheAlignedCounter:
    var count: Int
    # Padding to fill cache line, prevent false sharing
    var _padding: SIMD[DType.int64, 7]

    fn __init__(out self):
        self.count = 0
        self._padding = SIMD[DType.int64, 7]()

    fn increment(mut self):
        self.count += 1
```

---

### Pattern 2: Data Layout (AoS vs SoA)

**When:** Batch operations on large datasets

**Benchmark (particle simulation):**
- AoS (Array of Structs): Poor cache utilization for single-field access
- SoA (Struct of Arrays): 2-10x faster for batch field operations

**Don't (AoS with strided access):**
```mojo
# nocompile
struct ParticleAoS:
    var x: Float64
    var y: Float64
    var z: Float64
    var vx: Float64
    var vy: Float64
    var vz: Float64
    var mass: Float64

fn sum_x_aos(particles: List[ParticleAoS]) -> Float64:
    var total: Float64 = 0.0
    for p in particles:
        # Accessing x values jumps 56 bytes between particles
        # Cache lines (64 bytes) mostly wasted on unused fields
        total += p[].x
    return total
```

**Do (SoA for batch operations):**
```mojo
struct ParticlesSoA:
    var x: List[Float64]
    var y: List[Float64]
    var z: List[Float64]
    var vx: List[Float64]
    var vy: List[Float64]
    var vz: List[Float64]
    var mass: List[Float64]

fn sum_x_soa(particles: ParticlesSoA) -> Float64:
    var total: Float64 = 0.0
    # Sequential memory access - perfect cache utilization
    for i in range(len(particles.x)):
        total += particles.x[i]
    return total
```

**Layout decision guide:**

| Layout | Best For | Cache Behavior |
|--------|----------|----------------|
| AoS | Single particle operations, infrequent access | All fields of one item cached together |
| SoA | Batch operations on single fields | Sequential access, perfect prefetching |
| Hybrid | Mixed access patterns | Balance between both |

---

### Pattern 3: Matrix Transpose for Coalesced Access

**When:** Matrix multiplication where B is accessed column-wise

**Benchmark (1024x1024, Apple M-series):**
- Without transpose: 2.4 GFLOP/s
- With transpose + SIMD: 105 GFLOP/s (**44x faster**)
- With transpose + SIMD + parallel + unroll: **117 GFLOP/s**

```mojo
# nocompile
# SLOW: Column-wise access to B causes cache misses
fn matmul_slow(C: Matrix, A: Matrix, B: Matrix):
    for m in range(M):
        for n in range(N):
            var acc: Float64 = 0.0
            for k in range(K):
                acc += A.get(m, k) * B.get(k, n)  # B[k,n] is strided!
            C.set(m, n, acc)

# FAST: Transpose B, then both accesses are row-wise
fn transpose(B: Matrix) -> Matrix:
    var BT = Matrix(B.cols, B.rows)
    for i in range(B.rows):
        for j in range(B.cols):
            BT.set(j, i, B.get(i, j))
    return BT^

fn matmul_fast(C: Matrix, A: Matrix, B: Matrix):
    var BT = transpose(B)  # One-time O(n^2) cost
    for m in range(M):
        var a_row = A.data + m * K
        for n in range(N):
            var bt_row = BT.data + n * K  # Now coalesced!
            # SIMD dot product - both accesses sequential
            var acc = SIMD[DType.float64, 8]()
            var k = 0
            while k + 8 <= K:
                acc += (a_row + k).load[width=8]() * (bt_row + k).load[width=8]()
                k += 8
            C.set(m, n, acc.reduce_add())
```

---

### Pattern 4: Memory Prefetching

**When:** Large sequential or strided array traversals

**Benchmark (100M element traversal):**
- Without prefetch: 8.5 ms
- With prefetch: 6.2 ms (**27% faster**)

```mojo
# nocompile
from memory import prefetch, PrefetchOptions

fn sum_with_prefetch(data: UnsafePointer[Float64], size: Int) -> Float64:
    comptime WIDTH: Int = 8
    comptime PREFETCH_DISTANCE: Int = 256  # Elements ahead to prefetch

    var acc = SIMD[DType.float64, WIDTH]()
    var i = 0

    while i + WIDTH <= size:
        # Prefetch data that will be needed in future iterations
        if i + PREFETCH_DISTANCE < size:
            prefetch[PrefetchOptions().for_read().high_locality()](
                data.offset(i + PREFETCH_DISTANCE)
            )

        acc += data.load[width=WIDTH](i)
        i += WIDTH

    return acc.reduce_add()
```

**Prefetch options:**
```mojo
# nocompile
# Read prefetch (most common)
prefetch[PrefetchOptions().for_read().high_locality()](ptr)

# Write prefetch (when you'll write to memory soon)
prefetch[PrefetchOptions().for_write().high_locality()](ptr)

# Low locality (data used once, don't pollute cache)
prefetch[PrefetchOptions().for_read().low_locality()](ptr)
```

**Prefetch distance guidelines:**

| Access Pattern | Recommended Distance | Reasoning |
|----------------|---------------------|-----------|
| Sequential read | 256-512 elements | ~2-4 cache lines ahead |
| Strided access | 128-256 elements | Account for stride |
| Random access | Don't prefetch | Unpredictable, may hurt |

---

### Pattern 5: Multiple Accumulators for ILP

**When:** Reduction operations with independent iterations

Modern CPUs can execute multiple independent instructions simultaneously. Using a single accumulator creates a dependency chain that limits throughput.

**Benchmark (100M element sum):**
- Single accumulator: 12.0 ms
- 4 accumulators: 8.2 ms (**1.5x faster**)
- 8 accumulators: 7.0 ms (**1.7x faster**)

**Don't (single accumulator creates dependency chain):**
```mojo
fn sum_single_acc(data: UnsafePointer[Float64], size: Int) -> Float64:
    comptime WIDTH: Int = 8
    var acc = SIMD[DType.float64, WIDTH]()

    var i = 0
    while i + WIDTH <= size:
        acc += data.load[width=WIDTH](i)  # Must wait for previous acc update
        i += WIDTH

    return acc.reduce_add()
```

**Do (multiple accumulators enable parallel execution):**
```mojo
fn sum_multi_acc(data: UnsafePointer[Float64], size: Int) -> Float64:
    comptime WIDTH: Int = 8

    # 8 independent accumulators for maximum ILP
    var acc0 = SIMD[DType.float64, WIDTH]()
    var acc1 = SIMD[DType.float64, WIDTH]()
    var acc2 = SIMD[DType.float64, WIDTH]()
    var acc3 = SIMD[DType.float64, WIDTH]()
    var acc4 = SIMD[DType.float64, WIDTH]()
    var acc5 = SIMD[DType.float64, WIDTH]()
    var acc6 = SIMD[DType.float64, WIDTH]()
    var acc7 = SIMD[DType.float64, WIDTH]()

    var i = 0
    # Process 8 SIMD vectors per iteration (64 elements)
    while i + WIDTH * 8 <= size:
        acc0 += data.load[width=WIDTH](i)
        acc1 += data.load[width=WIDTH](i + WIDTH)
        acc2 += data.load[width=WIDTH](i + WIDTH * 2)
        acc3 += data.load[width=WIDTH](i + WIDTH * 3)
        acc4 += data.load[width=WIDTH](i + WIDTH * 4)
        acc5 += data.load[width=WIDTH](i + WIDTH * 5)
        acc6 += data.load[width=WIDTH](i + WIDTH * 6)
        acc7 += data.load[width=WIDTH](i + WIDTH * 7)
        i += WIDTH * 8

    # Combine accumulators
    acc0 = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7

    # Handle remaining elements
    while i + WIDTH <= size:
        acc0 += data.load[width=WIDTH](i)
        i += WIDTH

    var sum = acc0.reduce_add()
    while i < size:
        sum += data[i]
        i += 1

    return sum
```

**Accumulator count guidelines:**

| CPU Type | Recommended Accumulators | Reasoning |
|----------|-------------------------|-----------|
| Apple M-series | 4-8 | Wide execution units |
| Intel/AMD x86 | 4-8 | Multiple ALUs per core |
| Memory-bound ops | 2-4 | Diminishing returns |

---

### Pattern 6: Tiled Processing for Large Data

**When:** Large image/tensor processing that exceeds cache or GPU memory

**Impact:** 75% memory reduction for large image processing (1024x1024+)

```mojo
# nocompile
@fieldwise_init
struct TiledConfig(Copyable, Movable):
    var tile_size: Int      # Tile size (e.g., 512 pixels)
    var overlap: Int        # Overlap between tiles (e.g., 64 pixels)
    var enabled: Bool

fn compute_blend_weight(y: Int, x: Int, h: Int, w: Int, overlap: Int) -> Float32:
    """Linear falloff weight at tile edges."""
    var wt: Float32 = 1.0

    # Top edge
    if y < overlap:
        wt *= Float32(y) / Float32(overlap)
    # Bottom edge
    if y >= h - overlap:
        wt *= Float32(h - 1 - y) / Float32(overlap)
    # Left edge
    if x < overlap:
        wt *= Float32(x) / Float32(overlap)
    # Right edge
    if x >= w - overlap:
        wt *= Float32(w - 1 - x) / Float32(overlap)

    return wt

fn process_tiled(input: Float32Ptr, h: Int, w: Int, config: TiledConfig) -> Float32Ptr:
    """Process input in overlapping tiles."""
    var stride = config.tile_size - config.overlap
    var output = alloc[Float32](h * w)
    var weight_sum = alloc[Float32](h * w)

    # Initialize accumulators
    for i in range(h * w):
        output[i] = 0.0
        weight_sum[i] = 0.0

    # Process tiles
    var y = 0
    while y < h:
        var tile_h = min(config.tile_size, h - y)
        var x = 0
        while x < w:
            var tile_w = min(config.tile_size, w - x)

            # Extract and process tile
            var tile = extract_tile(input, y, x, tile_h, tile_w, w)
            var processed = process_single_tile(tile, tile_h, tile_w)

            # Accumulate with blending
            for ty in range(tile_h):
                for tx in range(tile_w):
                    var wt = compute_blend_weight(ty, tx, tile_h, tile_w, config.overlap)
                    var oy = y + ty
                    var ox = x + tx
                    var idx = oy * w + ox
                    output[idx] += wt * processed[ty * tile_w + tx]
                    weight_sum[idx] += wt

            x += stride
        y += stride

    # Normalize
    for i in range(h * w):
        if weight_sum[i] > 0:
            output[i] /= weight_sum[i]

    weight_sum.free()
    return output
```

**Recommended tiling settings:**

| Resolution | tile_size | overlap | Memory |
|------------|-----------|---------|--------|
| 512x512 | No tiling needed | - | - |
| 1024x1024 | 512 | 64 | ~128MB |
| 2048x2048 | 512 | 64 | ~128MB |

---

### Pattern 7: Prefer Borrowing Over Copying

**When:** Passing large data structures to functions

**Impact:** 10-100x faster for large data structures

```mojo
fn analyze(data: List[Float64]) -> Float64:
    # data is borrowed immutably - no copy occurs
    # This is the default behavior with 'read' convention
    var sum: Float64 = 0.0
    for item in data:
        sum += item
    return sum / len(data)

fn main():
    var measurements = List[Float64]()
    for i in range(1000000):
        measurements.append(Float64(i))

    var avg = analyze(measurements)  # Borrowed, not copied
    print(avg)
    print(measurements[0])  # Still valid - we only borrowed
```

**When copying is appropriate:**
- Small types that fit in registers (Int, Float64, Bool)
- When you need an independent copy to modify
- Types conforming to `TrivialRegisterType` trait

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| SIMD data structures | @align(32) or @align(64) | - |
| Batch field operations | SoA layout | - |
| Matrix multiply | Transpose B for coalesced access | - |
| Large sequential scans | Prefetch 256-512 elements ahead | - |
| Reduction operations | Multiple accumulators (4-8) | - |
| Large images (1024+) | Tiled processing with overlap | - |
| Parallel counters | Cache-line aligned to prevent false sharing | - |

---

## Quick Reference

- **Cache line size**: 64 bytes (typical)
- **AVX alignment**: 32 bytes
- **AVX-512 alignment**: 64 bytes
- **Prefetch distance**: 256-512 elements for sequential
- **Accumulator count**: 4-8 for modern CPUs
- **Tile overlap**: At least 1/8 of tile size

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `unaligned access crash` | Pointer not aligned for operation | Use `@align(N)` on struct; use aligned allocation |
| `cache thrashing` | Stride matches cache line size | Change array layout; add padding to break stride |
| `prefetch no effect` | Prefetch too late or wrong distance | Prefetch 256-512 elements ahead for sequential access |
| `false sharing` | Different threads modify same cache line | Pad data to 64-byte boundaries between threads |
| `stack overflow` | Large stack allocation | Use heap for large buffers; reduce local array sizes |
| `memory bandwidth saturated` | Too many concurrent memory streams | Reduce active streams; improve data locality |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **Heap allocation** | `from memory import alloc; alloc[T](n)` | v26.1+ |
| **Constants** | `alias` or `comptime` | Both work in v26.1+ |
| **Prefetch** | `prefetch[PrefetchOptions()]` | Stable |
| **Struct alignment** | `@align(64)` | v26.2+ nightly |

**Example (v26.1+):**
```mojo
from memory import prefetch, PrefetchOptions, alloc

comptime SIMD_WIDTH = 8
comptime CACHE_LINE = 64

fn process_data():
    var buffer = alloc[Float32](1024)
    # ... processing
    buffer.free()
```

**Notes:**
- Both `alias` and `comptime` work for compile-time constants in v26.1+
- `@align(N)` decorator is v26.2+ nightly
- Prefetching APIs (`prefetch`, `PrefetchOptions`) are stable
- `alloc()` is available in v26.1+ (not nightly-only)
- Tiled processing patterns are stable across versions

---

## Related Patterns

- [`perf-vectorization.md`](perf-vectorization.md) — SIMD patterns that benefit from alignment
- [`perf-parallelization.md`](perf-parallelization.md) — Multi-core patterns with memory considerations
- [`perf-optimization.md`](perf-optimization.md) — Buffer management and allocation strategies

---

## References

- [Mojo Memory](https://docs.modular.com/mojo/std/memory/)
- [Mojo Decorators](https://docs.modular.com/mojo/manual/decorators/)
- [Mojo Value Semantics](https://docs.modular.com/mojo/manual/values/)
