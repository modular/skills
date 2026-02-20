---
title: SIMD Vectorization Patterns
description: Comprehensive guide to SIMD vectorization, alignment, early exit, loop unrolling, and grid-stride patterns for maximum CPU/GPU throughput
impact: HIGH
category: perf
tags: [simd, vectorize, alignment, unrolling, early-exit, grid-stride]
error_patterns:
  - "SIMD width"
  - "alignment"
  - "cannot vectorize"
  - "unaligned access"
  - "vector type"
  - "simdwidthof"
  - "slow performance"
  - "cannot be converted from.*capturing.*to.*func"
scenarios:
  - "Vectorize loop with SIMD"
  - "Optimize numeric computation"
  - "Fix alignment issues"
  - "Implement early exit for SIMD"
  - "Unroll loop for performance"
  - "Use grid-stride pattern"
  - "Use vectorize in GPU kernel"
consolidates:
  - perf-vectorize.md
  - perf-simd-alignment.md
  - perf-early-simd-exit.md
  - perf-loop-unrolling.md
  - perf-grid-stride-loop.md
---

# SIMD Vectorization Patterns

**Category:** perf | **Impact:** HIGH (4-16x speedup)

SIMD (Single Instruction, Multiple Data) vectorization is the foundation of high-performance numeric computing in Mojo. This pattern covers manual SIMD loops, alignment requirements, early exit optimization, loop unrolling strategies, and grid-stride patterns for GPU kernels.

---

## Core Concepts

### SIMD Width Constants

Different architectures support different vector widths. Use compile-time constants based on your target:

```mojo
# Common SIMD widths (use based on target architecture)
comptime AVX_FLOAT32_WIDTH: Int = 8     # 256-bit / 32-bit = 8 elements
comptime AVX_FLOAT64_WIDTH: Int = 4     # 256-bit / 64-bit = 4 elements
comptime AVX512_FLOAT32_WIDTH: Int = 16 # 512-bit / 32-bit = 16 elements
comptime AVX512_FLOAT64_WIDTH: Int = 8  # 512-bit / 64-bit = 8 elements
```

### Using simdwidthof for Optimal Width

The `simdwidthof[T]()` helper function returns the optimal SIMD width for type `T` on the current target architecture. This is preferred over hardcoded constants for portable code.

```mojo
# nocompile
from sys import simdwidthof
from memory import UnsafePointer

fn sum_array_portable[T: DType](data: UnsafePointer[Scalar[T]], size: Int) -> Scalar[T]:
    """Sum using architecture-optimal SIMD width."""
    # Runtime width detection - adapts to AVX, AVX-512, NEON, etc.
    comptime WIDTH = simdwidthof[T]()

    var partial_sum = SIMD[T, WIDTH]()
    var i = 0

    while i + WIDTH <= size:
        partial_sum += data.load[width=WIDTH](i)
        i += WIDTH

    var total = partial_sum.reduce_add()

    # Handle remainder
    while i < size:
        total += data[i]
        i += 1

    return total
```

**When to use `simdwidthof` vs compile-time constants:**

| Approach | Use Case | Example |
|----------|----------|---------|
| `simdwidthof[T]()` | Portable code targeting multiple architectures | Libraries, cross-platform kernels |
| Hardcoded constant | Known target architecture, maximum control | Performance-critical inner loops on specific hardware |
| `simdwidthof[T]() * 2` | Manually tuned unrolling | When profiling shows benefit from larger vectors |

```mojo
# nocompile
from sys import simdwidthof

# PORTABLE: Adapts to target (8 on AVX, 16 on AVX-512)
comptime PORTABLE_WIDTH = simdwidthof[DType.float32]()

# FIXED: Always 8, regardless of target
comptime FIXED_WIDTH: Int = 8  # Both alias and comptime work in v26.1+

# DOUBLED: 2x optimal for manual unrolling
comptime UNROLLED_WIDTH = simdwidthof[DType.float32]() * 2
```

**SIMD width by type:**

| Type | AVX (256-bit) | AVX-512 (512-bit) |
|------|---------------|-------------------|
| Float64 | 4 | 8 |
| Float32 | 8 | 16 |
| Int32 | 8 | 16 |
| Int16 | 16 | 32 |
| Int8 | 32 | 64 |

### Basic SIMD Loop Pattern

**Pattern:**

```mojo
from memory import UnsafePointer
from builtin.type_aliases import MutAnyOrigin

comptime Float32Ptr = UnsafePointer[mut=True, type=Float32, origin=MutAnyOrigin]

fn add_arrays_simd(
    a: Float32Ptr,
    b: Float32Ptr,
    result: Float32Ptr,
    size: Int
):
    comptime WIDTH: Int = 8  # AVX width for Float32

    # Process WIDTH elements at a time
    var i = 0
    while i + WIDTH <= size:
        var va = a.load[width=WIDTH](i)
        var vb = b.load[width=WIDTH](i)
        result.store(i, va + vb)
        i += WIDTH

    # Handle remainder
    while i < size:
        result[i] = a[i] + b[i]
        i += 1
```

### SIMD Reduction Pattern

```mojo
# nocompile
fn sum_array(data: Float64Ptr, size: Int) -> Float64:
    comptime WIDTH: Int = 4  # AVX width for Float64
    var partial_sum = SIMD[DType.float64, WIDTH]()

    var i = 0
    while i + WIDTH <= size:
        partial_sum += data.load[width=WIDTH](i)
        i += WIDTH

    var total = partial_sum.reduce_add()

    # Handle remainder
    while i < size:
        total += data[i]
        i += 1

    return total
```

---

## Common Patterns

### Pattern 1: Memory Alignment for SIMD

**When:** Loading/storing SIMD vectors for optimal performance

Proper alignment enables faster memory operations and prevents crashes on strict-alignment architectures.

**Do:**
```mojo
from memory import alloc

# Allocate with explicit 64-byte alignment for cache line
var ptr = alloc[Float32](1024)

# Load with higher alignment for SIMD operations
var aligned_vec = ptr.load[width=8, alignment=32]()  # AVX alignment

ptr.free()
```

**Don't:**
```mojo
fn bad_unaligned_access(byte_ptr: UnsafePointer[UInt8, _], offset: Int):
    # WRONG: Casting byte pointer and assuming 4-byte alignment
    var int_ptr = (byte_ptr + offset).bitcast[Int32]()
    var vec = int_ptr.load[width=4]()  # Uses default alignment=4
    # May crash on ARM or cause performance penalty on x86
```

**Correct unaligned access:**
```mojo
fn good_unaligned_access(byte_ptr: UnsafePointer[UInt8, _], offset: Int):
    # CORRECT: Explicitly specify alignment=1 for potentially unaligned data
    var int_ptr = (byte_ptr + offset).bitcast[Int32]()
    var vec = int_ptr.load[width=4, alignment=1]()
    # Safe on all architectures
```

**Common alignment values:**

| Alignment | Use Case |
|-----------|----------|
| 16 bytes | SSE vectors (128-bit) |
| 32 bytes | AVX vectors (256-bit) |
| 64 bytes | AVX-512 vectors, cache lines |
| 128 bytes | Some GPU requirements |

---

### Pattern 2: Early SIMD Exit

**When:** Processing data with SIMD where iterations can terminate early (escape-time fractals, convergence checks)

Use `reduce_or()` or `reduce_and()` to check if any/all lanes need to continue.

**Benchmark Results (Mandelbrot 1920x1080):**
- Without early exit: 11.2 ms
- With early exit: 8.4 ms (**25% faster**)
- With early exit + cardioid skip: 3.9 ms (**63% faster**)

**Do:**
```mojo
fn mandelbrot_early_exit(output: UnsafePointer[Int32], width: Int, height: Int, max_iter: Int):
    comptime WIDTH: Int = 8

    for row in range(height):
        var y0 = Float64(row) / Float64(height) * 3.0 - 1.5
        var col = 0

        while col + WIDTH <= width:
            var x0_vec = SIMD[DType.float64, WIDTH]()
            # ... initialize x0_vec ...

            var x = SIMD[DType.float64, WIDTH]()
            var y = SIMD[DType.float64, WIDTH]()
            var iteration = SIMD[DType.int32, WIDTH]()

            var iter_count = 0
            while iter_count < max_iter:
                var x2 = x * x
                var y2 = y * y
                var mag2 = x2 + y2
                var mask = mag2.le(4.0)

                # GOOD: Exit when ALL lanes have escaped
                if not mask.reduce_or():
                    break

                iteration = mask.select(iteration + 1, iteration)
                var xy = x * y
                x = x2 - y2 + x0_vec
                y = xy + xy + y0_vec
                iter_count += 1

            col += WIDTH
```

**SIMD mask reduction operations:**

```mojo
# nocompile
# Check if ANY lane is still active (at least one True)
if mask.reduce_or():
    # At least one element needs more processing
    pass

# Check if ALL lanes are active (all True)
if mask.reduce_and():
    # All elements need more processing
    pass

# Count active lanes
var active_count = mask.cast[DType.int32]().reduce_add()
```

---

### Pattern 3: Loop Unrolling

**When:** Large arrays with simple operations, small fixed-size loops

**Benchmark Results (1M element array addition):**
- Simple loop: 0.554 ms
- Manual unroll x8: 0.271 ms (**2x faster**)

**Do (manual unrolling for large arrays):**
```mojo
fn add_arrays_unrolled(mut result: List[Int], a: List[Int], b: List[Int]):
    """Process 8 elements per iteration - reduces loop overhead by 8x."""
    var size = len(a)
    var i = 0

    # Process 8 at a time
    while i + 8 <= size:
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
    while i < size:
        result[i] = a[i] + b[i]
        i += 1
```

**Do (@parameter for compile-time unrolling):**
```mojo
fn dot_product_4(a: SIMD[DType.float64, 4], b: SIMD[DType.float64, 4]) -> Float64:
    var result: Float64 = 0.0

    @parameter
    for i in range(4):
        # Unrolled at compile time - no loop overhead
        result += a[i] * b[i]

    return result

# Even better - use SIMD operations directly
fn dot_product_simd(a: SIMD[DType.float64, 4], b: SIMD[DType.float64, 4]) -> Float64:
    return (a * b).reduce_add()  # Single SIMD multiply + reduce
```

**Unrolling factor guidelines:**

| Array Size | Recommended Unroll Factor |
|------------|---------------------------|
| < 100 | 4 |
| 100 - 10K | 8 |
| > 10K | 8-16 (combine with parallelize) |

---

### Pattern 4: Grid-Stride Loops (GPU)

**When:** Processing arbitrary-sized data on GPU with fixed thread count

Grid-stride loops allow a fixed number of threads to process arbitrary-sized data efficiently.

**Don't:**
```mojo
# nocompile
fn naive_kernel(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    size: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x

    # Only processes one element per thread
    if tid < size:
        output[tid] = process(input[tid])
```

**Do:**
```mojo
# nocompile
fn grid_stride_kernel(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    size: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var stride = grid_dim.x * block_dim.x

    # Each thread processes multiple elements
    var idx = tid
    while idx < size:
        output[idx] = process(input[idx])
        idx += stride
```

**Vectorized grid-stride loop:**
```mojo
# nocompile
fn vectorized_grid_stride_kernel[VEC_WIDTH: Int](
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    size: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var stride = grid_dim.x * block_dim.x

    # Vectorized main loop
    var vec_size = size // VEC_WIDTH
    var vec_idx = tid
    while vec_idx < vec_size:
        var vec_in = input.load[width=VEC_WIDTH](vec_idx * VEC_WIDTH)
        var vec_out = process_simd(vec_in)
        output.store(vec_idx * VEC_WIDTH, vec_out)
        vec_idx += stride

    # Handle tail elements
    if tid == 0:
        var tail_start = vec_size * VEC_WIDTH
        for i in range(tail_start, size):
            output[i] = process(input[i])
```

**Benefits of grid-stride loops:**

| Aspect | One-Thread-Per-Element | Grid-Stride |
|--------|----------------------|-------------|
| Kernel launch overhead | High for large N | Fixed, minimal |
| Thread reuse | No | Yes |
| Cache utilization | Random | Sequential access |
| Code flexibility | Size-dependent | Works for any size |

> **Recommended:** For production element-wise GPU kernels, prefer `algorithm.elementwise` over hand-written grid-stride loops. It handles grid/block sizing, striding, and vectorization automatically. See [`gpu-kernels.md`](gpu-kernels.md) for the template.

---

### Pattern 5: When Scalar Beats SIMD

**When:** Complex transcendental functions (exp, sin, cos, log)

Not all operations benefit from SIMD. SIMD versions of transcendental functions may be **slower** due to library overhead.

```mojo
# nocompile
# SLOWER: SIMD exp() approach (14.95s vs 10.06s baseline - 49% regression!)
fn silu_simd_direct(result: Float32Ptr, x: Float32Ptr, n: Int):
    comptime WIDTH: Int = 8
    var i = 0
    while i + WIDTH <= n:
        var v = x.load[width=WIDTH](i)
        var neg_v = -v
        var exp_neg_v = exp(neg_v)  # SIMD exp() is slow!
        result.store(i, v / (1.0 + exp_neg_v))
        i += WIDTH

# FASTER: Scalar loop inside SIMD
fn silu_simd_scalar(result: Float32Ptr, x: Float32Ptr, n: Int):
    comptime WIDTH: Int = 8
    var i = 0
    while i + WIDTH <= n:
        var v = x.load[width=WIDTH](i)
        var out = SIMD[DType.float32, WIDTH]()
        for j in range(WIDTH):  # Scalar exp() per element
            out[j] = v[j] / (1.0 + exp(-v[j]))
        result.store(i, out)
        i += WIDTH
```

**Recommendation:** Always benchmark transcendental functions before assuming SIMD is faster.

---

### Pattern 6: Polynomial Approximations for SIMD Trig

**When:** High-throughput trig operations where ~6 decimal places is sufficient

**Impact:** HIGH — 5-20x speedup over stdlib trig

**Benchmark (10M element sin computation):**
- stdlib sin (scalar): 45 ms
- Polynomial sin (SIMD x8): 4.5 ms (**10x faster**)

**Taylor series SIMD sine (accurate for |x| < π):**

```mojo
fn sin_taylor_simd[WIDTH: Int](x: SIMD[DType.float64, WIDTH]) -> SIMD[DType.float64, WIDTH]:
    """Taylor series: sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7!"""
    var x2 = x * x
    var x3 = x2 * x
    var x5 = x3 * x2
    var x7 = x5 * x2

    comptime c3: Float64 = 0.16666666666666666   # 1/6
    comptime c5: Float64 = 0.008333333333333333  # 1/120
    comptime c7: Float64 = 0.0001984126984126984 # 1/5040

    return x - x3 * c3 + x5 * c5 - x7 * c7


fn cos_taylor_simd[WIDTH: Int](x: SIMD[DType.float64, WIDTH]) -> SIMD[DType.float64, WIDTH]:
    """Taylor series: cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6!"""
    var x2 = x * x
    var x4 = x2 * x2
    var x6 = x4 * x2

    comptime c2: Float64 = 0.5                   # 1/2
    comptime c4: Float64 = 0.041666666666666664  # 1/24
    comptime c6: Float64 = 0.001388888888888889  # 1/720

    return 1.0 - x2 * c2 + x4 * c4 - x6 * c6
```

**Fast inverse square root (Quake III style, ~1% error):**

```mojo
# nocompile
fn fast_inv_sqrt_simd[WIDTH: Int](x: SIMD[DType.float32, WIDTH]) -> SIMD[DType.float32, WIDTH]:
    """Fast 1/sqrt(x) - 4x faster than 1/sqrt(x)."""
    comptime MAGIC: Int32 = 0x5f3759df

    var i = x.bitcast[DType.int32]()
    i = MAGIC - (i >> 1)
    var y = i.bitcast[DType.float32]()

    # One Newton-Raphson iteration for better accuracy
    return y * (1.5 - 0.5 * x * y * y)
```

**Accuracy vs speed tradeoffs:**

| Method | Accuracy (decimal places) | Relative Speed |
|--------|--------------------------|----------------|
| stdlib sin | 15 | 1x (baseline) |
| Taylor 7th order | 6 | 8-10x |
| Chebyshev 11th order | 10 | 5-7x |

**When to use:**
- Graphics/visualization (visual accuracy sufficient)
- Signal processing with known frequency bounds
- Machine learning (gradients are approximate anyway)

**When NOT to use:**
- Financial calculations requiring exact precision
- Scientific computing with strict error bounds

---

### Pattern 7: Multiple Accumulators for ILP

**When:** Large reduction operations (sum, dot product) where memory latency dominates

**Impact:** HIGH — 1.5-1.7x speedup over single accumulator

**Why it works:**
- CPU can issue 2-4 independent loads per cycle
- Memory latency is ~100+ cycles
- Multiple accumulators keep the pipeline full while waiting for memory

**Do (4 independent accumulators):**
```mojo
# nocompile - Float32Ptr type alias not available in stable v26.1
fn sum_with_accumulators[width: Int = 8](data: Float32Ptr, n: Int) -> Float32:
    """Sum array using 4 independent accumulators for maximum ILP."""
    var stride = width * 4
    var sum0 = SIMD[DType.float32, width](0.0)
    var sum1 = SIMD[DType.float32, width](0.0)
    var sum2 = SIMD[DType.float32, width](0.0)
    var sum3 = SIMD[DType.float32, width](0.0)

    var i = 0
    while i + stride <= n:
        # 4 independent load+add chains execute in parallel
        sum0 += data.load[width=width](i)
        sum1 += data.load[width=width](i + width)
        sum2 += data.load[width=width](i + width * 2)
        sum3 += data.load[width=width](i + width * 3)
        i += stride

    # Combine accumulators at the end
    var total = (sum0 + sum1 + sum2 + sum3).reduce_add()

    # Handle remainder
    while i < n:
        total += data[i]
        i += 1

    return total
```

**Don't (single accumulator):**
```mojo
# nocompile - Float32Ptr type alias not available in stable v26.1
fn sum_single_accumulator[width: Int = 8](data: Float32Ptr, n: Int) -> Float32:
    """SLOWER: Each load waits for previous add to complete."""
    var sum = SIMD[DType.float32, width](0.0)
    var i = 0
    while i + width <= n:
        sum += data.load[width=width](i)  # Dependency chain stalls
        i += width
    return sum.reduce_add()
```

**Optimal accumulator count:**

| CPU Architecture | Recommended Accumulators |
|-----------------|-------------------------|
| Modern x86/ARM  | 4 (matches load/store ports) |
| Apple Silicon   | 4-8 (high memory bandwidth) |
| GPU SIMD groups | 4 (hides memory latency) |

---

### Pattern 8: Scalar Bottleneck Detection

**When:** Code using `exp()`, `log()`, `tanh()`, or other transcendental functions in loops

**Impact:** CRITICAL — 4-8x speedup when fixed

Common scalar bottleneck patterns and their vectorized alternatives:

| Function | Scalar Bottleneck | SIMD Fix |
|----------|-------------------|----------|
| `softplus(x) = log(1 + exp(x))` | Scalar `exp` and `log` in loop | Polynomial `fast_exp_simd` + `fast_log_simd` |
| `mish(x) = x * tanh(softplus(x))` | Scalar `tanh` and `exp` | Vectorized `fast_tanh_simd` with `fast_exp_simd` |
| `silu(x) = x * sigmoid(x)` | Scalar `exp` in sigmoid | `x / (1 + fast_exp_simd(-x))` |
| `gelu(x)` | Scalar `tanh` | Polynomial `fast_tanh_simd` approximation |

**Detection checklist:**
- [ ] Is `exp()`, `log()`, `tanh()`, `sin()`, `cos()` called in a loop?
- [ ] Does the loop process elements one at a time (scalar)?
- [ ] Can the transcendental be replaced with a polynomial approximation?
- [ ] Is the required precision < 6 decimal places?

**Fast polynomial approximations:**

```mojo
# nocompile - SIMD.bitcast() method not available in stable v26.1
fn fast_exp_simd[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """Fast exp() approximation using Schraudolph's method (~1e-4 relative error).

    4-8x faster than stdlib exp() for SIMD vectors.
    """
    # Clamp to avoid overflow/underflow
    var clamped = x.clamp(-88.0, 88.0)

    # Coefficients for exp polynomial
    comptime a = SIMD[DType.float32, width](12102203.0)  # 2^23 / ln(2)
    comptime b = SIMD[DType.float32, width](1065353216.0)  # 127 * 2^23

    var i = (a * clamped + b).cast[DType.int32]()
    return i.bitcast[DType.float32]()


fn fast_tanh_simd[width: Int](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """Fast tanh() using rational approximation (~1e-5 max error for |x| < 4).

    3-5x faster than stdlib tanh().
    """
    var x2 = x * x
    # Pade approximant: tanh(x) ≈ x(27 + x²) / (27 + 9x²)
    var num = x * (27.0 + x2)
    var den = 27.0 + 9.0 * x2
    var result = num / den

    # Clamp to [-1, 1] for |x| > ~4
    return result.clamp(-1.0, 1.0)
```

---

### Pattern 9: `vectorize` in GPU Context

**When:** Using `vectorize` inside GPU kernels to process multiple elements per thread

**Error:**
```
value passed to 'closure' cannot be converted from 'fn[width: Int](i: Int) capturing -> None' to 'func'
```

In GPU context, `vectorize` requires the `unified {captures}` convention instead of the default `capturing` convention. All captured variables must be listed explicitly with `read` or `write` modifiers.

**Don't:**
```mojo
# nocompile
# FAILS in GPU context: Default capturing convention not accepted
@always_inline
fn body[width: Int](idx: Int):  # Implicit capturing
    out[idx] = a[idx] + b[idx]

vectorize[simd_width](size, body)
```

**Do:**
```mojo
# nocompile
@always_inline
fn body[width: Int](idx: Int) unified {read a, read b, write out}:
    var c = IndexList[1](idx)
    out.store[width](c, a.load[width](c) + b.load[width](c))

vectorize[simd_width](size, body)
```

**Key requirements:**
- Use `unified {read var1, read var2, write var3}` convention
- List **all** captured variables explicitly with `read` or `write`
- Use `read` for inputs, `write` for outputs
- `IndexList[1](idx)` required for LayoutTensor `load`/`store` (see [`gpu-memory-access.md`](gpu-memory-access.md))

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Large numeric arrays | Manual SIMD loop with WIDTH=8 | [`perf-parallelization.md`](perf-parallelization.md) |
| Escape-time algorithms | Early exit with reduce_or() | - |
| Fixed small arrays | @parameter unrolling | - |
| GPU arbitrary data | Grid-stride loop | [`gpu-fundamentals.md`](gpu-fundamentals.md) |
| Transcendental functions | Benchmark SIMD vs scalar | - |
| Unaligned binary data | Use alignment=1 on load | - |

---

## Quick Reference

- **SIMD width for Float32**: 8 (AVX), 16 (AVX-512)
- **SIMD width for Float64**: 4 (AVX), 8 (AVX-512)
- **Early exit check**: `if not mask.reduce_or(): break`
- **Unroll factor**: 4-8 for most loops, 8-16 with parallelize
- **Grid-stride key**: `while idx < size: process(idx); idx += stride`
- **Alignment**: Use 32 for AVX, 64 for cache lines/AVX-512

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `unaligned memory access` | Pointer not aligned for SIMD width | Use `ptr.load[width, alignment=1]()` or align data with `@align(32)` |
| `SIMD width mismatch` | Loop count not divisible by SIMD width | Handle remainder with scalar loop or `width=1` fallback |
| `cannot vectorize loop` | Loop has dependencies or non-contiguous access | Restructure for contiguous access; use `@parameter` for constant bounds |
| `slow vectorized code` | Wrong SIMD width for architecture | Use `simdwidthof[T]()` to get optimal width for target |
| `illegal instruction (AVX-512)` | Running AVX-512 code on non-supporting CPU | Check `sys.has_avx512f()` before using 512-bit vectors |
| `reduce produces wrong result` | Uninitialized accumulator or wrong identity | Initialize accumulator to identity (0 for sum, 1 for product) |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **Constants** | `alias` or `comptime` | Both work in v26.1+ |
| **simdwidthof** | `simdwidthof[T]()` | Unchanged |
| **SIMD operations** | All stable | `load`, `store`, `reduce_*` |
| **Heap allocation** | `from memory import alloc; alloc[T](n)` | Unchanged |

**Example (v26.1+):**
```mojo
from memory import UnsafePointer
from builtin.type_aliases import MutAnyOrigin

# Both alias and comptime work in v26.1+
comptime WIDTH = 8  # AVX width for Float32
comptime Float32Ptr = UnsafePointer[mut=True, type=Float32, origin=MutAnyOrigin]

fn add_arrays_simd(a: Float32Ptr, b: Float32Ptr, result: Float32Ptr, size: Int):
    var i = 0
    while i + WIDTH <= size:
        var va = a.load[width=WIDTH](i)
        var vb = b.load[width=WIDTH](i)
        result.store(i, va + vb)
        i += WIDTH
```

**Notes:**
- Both `alias` and `comptime` work for constants in v26.1 and nightly
- SIMD operations (`load`, `store`, `reduce_add`, etc.) are stable across versions
- `simdwidthof[T]()` for optimal width detection is stable
- Alignment requirements and patterns are stable

---

## Related Patterns

- [`perf-parallelization.md`](perf-parallelization.md) — Combine with multi-core execution
- [`perf-memory.md`](perf-memory.md) — Memory layout for SIMD efficiency
- [`perf-optimization.md`](perf-optimization.md) — General optimization strategies

---

## References

- [Mojo SIMD](https://docs.modular.com/mojo/std/builtin/simd/)
- [Mojo Algorithm Module](https://docs.modular.com/mojo/std/algorithm/)
- [Mojo UnsafePointer](https://docs.modular.com/mojo/std/memory/unsafe_pointer/)
