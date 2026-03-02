---
title: SIMD Types and Vectorization
description: SIMD type patterns for high-performance numerical operations including register-passable types, vectorization, and alignment
impact: HIGH
category: type
tags: [types, simd, vectorization, register-passable, performance, alignment]
error_patterns:
  - "SIMD width mismatch"
  - "cannot vectorize"
  - "alignment error"
  - "invalid SIMD operation"
  - "width must be"
scenarios:
  - "Vectorize numeric loop"
  - "Create SIMD-friendly data structure"
  - "Fix SIMD alignment issue"
  - "Use register-passable types"
consolidates:
  - type-simd-vectorization.md
  - type-register-passable.md
  - perf-simd-alignment.md
  - perf-vectorize.md
---

# SIMD Types and Vectorization

**Category:** type | **Impact:** HIGH

SIMD (Single Instruction Multiple Data) types enable 4-16x speedups by processing multiple values in parallel using hardware vector instructions (SSE, AVX, AVX-512, NEON). Combined with register-passable types and proper alignment, SIMD unlocks maximum CPU performance.

---

## Core Concepts

### SIMD Type Fundamentals

SIMD types process multiple values in a single instruction, utilizing hardware vector units.

**Pattern:**

```mojo
from memory import UnsafePointer

# Type alias for cleaner code
comptime Float32Ptr = UnsafePointer[mut=True, type=Float32, origin=MutAnyOrigin]

fn add_arrays_simd(
    a: Float32Ptr,
    b: Float32Ptr,
    result: Float32Ptr,  # Note: 'out' is reserved keyword
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

# For fixed-size operations, use SIMD directly
fn dot_product_4(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> Float32:
    return (a * b).reduce_add()  # Single SIMD multiply + horizontal add
```

> **Critical Notes:**
> - `out` is a reserved keyword - use `result`, `dst`, or `output` for output parameters
> - `MutAnyOrigin` and `ImmutAnyOrigin` are prelude symbols -- no import needed
> - Use `comptime` for type aliases (`alias` is deprecated)

### Register-Passable Types

Types that fit in CPU registers should conform to the `TrivialRegisterPassable` trait to avoid pointer indirection, achieving 2-5x faster operations.

**Pattern:**

```mojo
# nocompile
# Incorrect: Pointer indirection for small types
struct Point:
    var x: Float64
    var y: Float64

    fn __init__(out self, x: Float64, y: Float64):
        self.x = x
        self.y = y

# Without TrivialRegisterPassable, Point is passed by pointer
# Every access requires a memory load
fn distance(a: Point, b: Point) -> Float64:
    var dx = a.x - b.x  # Load from memory
    var dy = a.y - b.y  # Load from memory
    return sqrt(dx*dx + dy*dy)
```

```mojo
# nocompile
# Correct: Use TrivialRegisterPassable trait for register-passable types
@fieldwise_init
struct Point(TrivialRegisterPassable, Copyable):
    var x: Float64
    var y: Float64

# Now Point is passed directly in registers
# No memory indirection needed
fn distance(a: Point, b: Point) -> Float64:
    var dx = a.x - b.x  # Direct register access
    var dy = a.y - b.y  # Direct register access
    return sqrt(dx*dx + dy*dy)
```

**TrivialRegisterPassable trait:**
- Use `TrivialRegisterPassable` trait for types with no lifecycle requirements
- Combine with `@fieldwise_init` and `Copyable` trait
- These can be copied with memcpy
- **Note:** `@register_passable("trivial")` is deprecated in v26.2; use the trait instead

**stdlib example (from Int):**

```mojo
# nocompile
struct Int(
    TrivialRegisterPassable,  # v26.2+: Use trait instead of decorator
    Absable, Boolable, Comparable, Hashable,
    ImplicitlyCopyable, Intable, ...
):
    var _mlir_value: __mlir_type.index

    # Static constants
    comptime BITWIDTH: Int = bit_width_of[DType.int]()
    comptime MAX = Int(Scalar[DType.int].MAX)
    comptime MIN = Int(Scalar[DType.int].MIN)

    # Trivial flags (compiler infers these for trivial types)
    comptime __del__is_trivial: Bool = True
    comptime __moveinit__is_trivial: Bool = True
    comptime __copyinit__is_trivial: Bool = True
```

---

## Common Patterns

### SIMD Width Selection

Choose SIMD width based on data type and target architecture.

**When:** Implementing vectorized algorithms

**Do:**
```mojo
# Conservative default - works well across architectures
comptime SIMD_WIDTH: Int = 8

# SIMD width by type and architecture
# | Type | ARM NEON | AVX2 | AVX-512 | Portable Default |
# |------|----------|------|---------|------------------|
# | Float64/Int64 | 2 (native) | 4 | 8 | 8 |
# | Float32/Int32 | 4 (native) | 8 | 16 | 8 |
# | Int8 | 16 (native) | 32 | 64 | 32 |

comptime AVX_FLOAT32_WIDTH: Int = 8    # 256-bit / 32-bit = 8 elements
comptime AVX_FLOAT64_WIDTH: Int = 4    # 256-bit / 64-bit = 4 elements
comptime AVX512_FLOAT32_WIDTH: Int = 16  # 512-bit / 32-bit = 16 elements
comptime AVX512_FLOAT64_WIDTH: Int = 8   # 512-bit / 64-bit = 8 elements
```

**Don't:**
```mojo
# Wider widths may cause regressions - always benchmark
comptime SIMD_WIDTH: Int = 16  # May cause alignment issues
```

**Why wider widths can fail:**
1. Hardware register width limits (128-bit NEON = 4 x Float32)
2. Wider logical widths require multiple register operations
3. Non-aligned memory access patterns cause severe penalties
4. Compiler optimizations may break down with wider widths

### SIMD Reduction Pattern

Use SIMD for parallel accumulation with final horizontal reduction.

**When:** Summing arrays, computing dot products

**Do:**
```mojo
from memory import UnsafePointer

comptime Float64Ptr = UnsafePointer[mut=True, type=Float64, origin=MutAnyOrigin]

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

fn dot_product_simd(
    a: Float64Ptr,
    b: Float64Ptr,
    size: Int
) -> Float64:
    """SIMD dot product - core kernel for optimized matmul."""
    comptime WIDTH: Int = 8  # Use 8 for Apple Silicon, 4 for AVX
    var acc = SIMD[DType.float64, WIDTH]()

    var k = 0
    while k + WIDTH <= size:
        acc += a.load[width=WIDTH](k) * b.load[width=WIDTH](k)
        k += WIDTH

    var result = acc.reduce_add()
    while k < size:
        result += a[k] * b[k]
        k += 1

    return result
```

### SIMD Alignment for Load/Store

Proper alignment enables faster memory operations and prevents crashes on strict-alignment architectures.

**When:** Loading/storing SIMD vectors from memory

**Do:**
```mojo
from memory import alloc

fn process_aligned_data():
    # Allocate with explicit 64-byte alignment for cache line
    var ptr = alloc[Float32](1024)

    # Load with default alignment (align_of[Float32]() = 4)
    var vec = ptr.load[width=8]()  # Uses default alignment

    # For SIMD operations on aligned memory, specify higher alignment
    var aligned_vec = ptr.load[width=8, alignment=32]()  # AVX alignment

    ptr.free()

# Unaligned access (when alignment not guaranteed)
fn process_unaligned_data(ptr: UnsafePointer[UInt8, _], offset: Int):
    # Loading Int32 from byte array at arbitrary offset
    # Cannot guarantee 4-byte alignment, so use alignment=1
    var int_ptr = (ptr + offset).bitcast[Int32]()
    var value = int_ptr.load[width=4, alignment=1]()
```

**Don't:**
```mojo
fn bad_unaligned_access(byte_ptr: UnsafePointer[UInt8, _], offset: Int):
    # WRONG: Casting byte pointer and assuming 4-byte alignment
    var int_ptr = (byte_ptr + offset).bitcast[Int32]()
    var vec = int_ptr.load[width=4]()  # Uses default alignment=4
    # May crash on ARM or cause performance penalty on x86
```

**Common alignment values:**

```mojo
comptime SSE_ALIGNMENT: Int = 16    # 128-bit vectors
comptime AVX_ALIGNMENT: Int = 32    # 256-bit vectors
comptime AVX512_ALIGNMENT: Int = 64 # 512-bit vectors, cache lines
comptime GPU_ALIGNMENT: Int = 128   # Some GPU requirements
```

### When Scalar Loops Beat SIMD

Not all operations benefit from SIMD. Complex transcendental functions may be slower.

**When:** Using exp(), sin(), cos(), log() in SIMD loops

**Do:**
```mojo
# nocompile
# FASTER: Scalar loop inside SIMD (benchmark first!)
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

**Don't (assume SIMD is always faster):**
```mojo
# nocompile
# SLOWER: SIMD exp() (14.95s vs 10.06s baseline - 49% regression!)
fn silu_simd_direct(result: Float32Ptr, x: Float32Ptr, n: Int):
    comptime WIDTH: Int = 8
    var i = 0
    while i + WIDTH <= n:
        var v = x.load[width=WIDTH](i)
        var neg_v = -v
        var exp_neg_v = exp(neg_v)  # SIMD exp() is slow!
        result.store(i, v / (1.0 + exp_neg_v))
        i += WIDTH
```

**Why scalar exp() wins:**
1. SIMD `exp()` has significant library overhead
2. Scalar `exp()` is highly optimized in libm
3. The scalar loop still benefits from SIMD load/store
4. Modern CPUs can pipeline scalar operations efficiently

### Precision Tradeoffs

Lower precision types fit more elements per SIMD register, increasing throughput.

**Throughput by precision (256-bit AVX):**

| Precision | SIMD Width | Throughput | Time (100M elements) |
|-----------|------------|------------|---------------------|
| Float64 | 4 | 1x | 8.0 ms |
| Float32 | 8 | 2x | 4.1 ms |
| Float16 | 16 | 4x | 2.3 ms |

**When Float32 is sufficient:**
```mojo
fn mandelbrot_f32(output: UnsafePointer[Int32], width: Int, height: Int):
    """Float32 sufficient for pixel-level accuracy."""
    comptime WIDTH: Int = 16  # 16 Float32s per SIMD!
    # 2x faster than Float64 version
```

**Mixed precision (compute low, accumulate high):**
```mojo
fn dot_product_mixed(a: UnsafePointer[Float32], b: UnsafePointer[Float32], size: Int) -> Float64:
    """Compute in Float32, accumulate in Float64 to avoid precision loss."""
    comptime WIDTH: Int = 8
    var acc = SIMD[DType.float64, 4]()  # Accumulate in Float64

    var i = 0
    while i + WIDTH <= size:
        var a_vec = a.load[width=WIDTH](i)
        var b_vec = b.load[width=WIDTH](i)
        var product = a_vec * b_vec

        # Widen to Float64 for accumulation
        var lo = product.slice[4, offset=0]().cast[DType.float64]()
        var hi = product.slice[4, offset=4]().cast[DType.float64]()
        acc += lo + hi
        i += WIDTH

    return acc.reduce_add()
```

**Precision comparison:**

| Type | Significant Digits | Use Case |
|------|-------------------|----------|
| Float64 | 15-17 | Scientific, financial |
| Float32 | 6-9 | Graphics, ML, physics |
| Float16 | 3-4 | Deep learning, images |

**When to use Float32:**
- Graphics (pixels are integers anyway)
- ML inference (models train in mixed precision)
- Audio/image processing
- Simulations where 6 decimal places suffice

**When to keep Float64:**
- Financial calculations
- Long-running simulations (error accumulation)
- Iterative algorithms (catastrophic cancellation risk)

---

## SIMD Comparison Methods

SIMD comparison operators (`>`, `<`, etc.) only work for `Scalar` types, not SIMD vectors. For SIMD vectors, use the comparison methods.

> **Warning:** The dunder methods (`__gt__`, `__lt__`, etc.) route to scalar operators and fail for SIMD width > 1. Always use the named methods (`.gt()`, `.lt()`, etc.) for element-wise SIMD comparison.

### Comparison Methods (Required for SIMD Vectors)

```mojo
fn filter_values(data: SIMD[DType.float32, 8], threshold: Float32) -> SIMD[DType.bool, 8]:
    # WRONG: Operators don't work for SIMD vectors
    # var mask = data > threshold  # ERROR: no matching operator

    # CORRECT: Use comparison methods
    var mask = data.gt(threshold)  # Greater than
    return mask

fn all_comparisons(a: SIMD[DType.float32, 8], b: SIMD[DType.float32, 8]):
    # Available comparison methods
    var greater = a.gt(b)      # a > b (element-wise)
    var less = a.lt(b)         # a < b
    var greater_eq = a.ge(b)   # a >= b
    var less_eq = a.le(b)      # a <= b
    var equal = a.eq(b)        # a == b
    var not_equal = a.ne(b)    # a != b

    # Use with select for conditional operations
    var result = greater.select(a, b)  # Where a > b, use a; else use b
```

### Scalar vs SIMD Comparison

```mojo
# Scalar (width=1): Operators work
fn scalar_compare(a: Float32, b: Float32) -> Bool:
    return a > b  # OK - returns Bool

# SIMD vector: Must use methods
fn simd_compare(a: SIMD[DType.float32, 8], b: SIMD[DType.float32, 8]) -> SIMD[DType.bool, 8]:
    return a.gt(b)  # Returns SIMD[DType.bool, 8]
```

### Reduction After Comparison

```mojo
# nocompile
fn any_greater(a: SIMD[DType.float32, 8], threshold: Float32) -> Bool:
    var mask = a.gt(threshold)
    return mask.reduce_or()  # True if ANY element > threshold

fn all_greater(a: SIMD[DType.float32, 8], threshold: Float32) -> Bool:
    var mask = a.gt(threshold)
    return mask.reduce_and()  # True if ALL elements > threshold

fn count_greater(a: SIMD[DType.float32, 8], threshold: Float32) -> Int:
    var mask = a.gt(threshold)
    return mask.cast[DType.int32]().reduce_add()  # Count of elements > threshold
```

---

## SIMD Type Conversion Methods

SIMD types provide `.cast[]` for value conversion. For bit reinterpretation, use the free function `bitcast` from the `memory` module (SIMD has no `.bitcast[]` method).

### `.cast[TargetDType]()` - Value Conversion

Converts values to the target type, preserving the numeric value where possible.

```mojo
fn cast_examples():
    # Float to Int: Truncates toward zero (not rounding)
    var floats = SIMD[DType.float32, 4](1.9, -2.7, 3.1, -0.5)
    var ints = floats.cast[DType.int32]()  # [1, -2, 3, 0]

    # Int to Float: May lose precision for large values
    var large_int = SIMD[DType.int64, 2](9007199254740993, 9007199254740994)
    var as_float = large_int.cast[DType.float64]()  # Both become 9007199254740992.0
    # (Float64 has 53 bits of mantissa, can't represent all Int64 values exactly)

    # Signed to Unsigned: Reinterprets negative values
    var signed = SIMD[DType.int32, 4](-1, -128, 127, 0)
    var unsigned = signed.cast[DType.uint32]()  # [4294967295, 4294967168, 127, 0]

    # Unsigned to Signed: May overflow
    var big_unsigned = SIMD[DType.uint32, 2](4294967295, 2147483648)
    var as_signed = big_unsigned.cast[DType.int32]()  # [-1, -2147483648]

    # Width changes: Truncation or zero/sign extension
    var int32_vals = SIMD[DType.int32, 4](256, -1, 65536, 127)
    var as_int8 = int32_vals.cast[DType.int8]()  # [0, -1, 0, 127] (truncated)
    var as_int64 = int32_vals.cast[DType.int64]()  # [256, -1, 65536, 127] (sign-extended)
```

**Special Float Values:**

```mojo
# nocompile
fn special_float_handling():
    var special = SIMD[DType.float32, 4](
        Float32.MAX,           # Largest finite float
        math.inf[DType.float32](),  # Positive infinity
        math.nan[DType.float32](),  # NaN
        -math.inf[DType.float32]()  # Negative infinity
    )

    # Float to Int with special values: Implementation-defined behavior
    # Typically: Inf -> INT_MAX/MIN, NaN -> 0 or undefined
    var as_int = special.cast[DType.int32]()  # Results vary by platform

    # Safe pattern: Check for special values first
    from utils.numerics import isfinite
    var is_finite_mask = isfinite(special)
    var safe_vals = is_finite_mask.select(special, SIMD[DType.float32, 4](0))
    var safe_ints = safe_vals.cast[DType.int32]()
```

### `bitcast[TargetDType](value)` - Bit Reinterpretation

Reinterprets the raw bits without conversion using the free function `from memory import bitcast`. **Requires same total bit-width.**

> **Note:** `bitcast` is a free function, not a method on SIMD. Import it with `from memory import bitcast`. `UnsafePointer.bitcast` IS a real method -- only SIMD bitcasting uses the free function.

```mojo
from memory import bitcast

fn bitcast_examples():
    # View float bits as int (IEEE 754 inspection)
    var f = SIMD[DType.float32, 4](1.0, -1.0, 0.0, 2.0)
    var bits = bitcast[DType.uint32](f)
    # 1.0f = 0x3F800000, -1.0f = 0xBF800000, 0.0f = 0x00000000, 2.0f = 0x40000000

    # Construct float from bits (useful for special constants)
    var inf_bits = SIMD[DType.uint32, 4](0x7F800000)  # IEEE 754 +Inf
    var inf_float = bitcast[DType.float32](inf_bits)  # [inf, inf, inf, inf]

    # Extract sign, exponent, mantissa
    var val = SIMD[DType.float32, 1](3.14159)
    var raw = bitcast[DType.uint32](val)
    var sign = (raw >> 31) & 1           # Sign bit
    var exponent = (raw >> 23) & 0xFF    # Biased exponent
    var mantissa = raw & 0x7FFFFF        # Mantissa (fractional part)

    # Signed/unsigned reinterpretation (same bits, different interpretation)
    var signed_val = SIMD[DType.int32, 2](-1, -2147483648)
    var as_unsigned = bitcast[DType.uint32](signed_val)  # [4294967295, 2147483648]
```

**Bit-width Requirement:**

```mojo
from memory import bitcast

fn bitcast_width_rules():
    var float32_vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)

    # VALID: Same bit-width per element
    var as_int32 = bitcast[DType.int32](float32_vec)    # 32-bit -> 32-bit
    var as_uint32 = bitcast[DType.uint32](float32_vec)  # 32-bit -> 32-bit

    # INVALID: Different total bit-width (compile error)
    # var as_int64 = bitcast[DType.int64](float32_vec)  # ERROR unless width adjusted
```

### Width Preservation

`.cast[]` preserves SIMD width. `bitcast` preserves total bit-width (element count may change if target dtype has different size).

```mojo
from memory import bitcast

fn width_preservation():
    var vec8 = SIMD[DType.float32, 8](1.0)

    var cast_result = vec8.cast[DType.int32]()            # SIMD[DType.int32, 8]
    var bitcast_result = bitcast[DType.uint32](vec8)      # SIMD[DType.uint32, 8]

    # Width is preserved regardless of target type
    var vec4 = SIMD[DType.float64, 4](1.0)
    var cast_f32 = vec4.cast[DType.float32]()  # SIMD[DType.float32, 4] (not 8!)
```

### Performance Characteristics

```mojo
from memory import bitcast

# bitcast[] - Zero cost (no instructions generated)
# Just tells compiler to interpret bits differently
fn fast_sign_flip(x: SIMD[DType.float32, 8]) -> SIMD[DType.float32, 8]:
    var bits = bitcast[DType.uint32](x)
    var flipped = bits ^ 0x80000000  # XOR sign bit
    return bitcast[DType.float32](flipped)  # Often faster than -x

# .cast[] - May have CPU cost depending on types
# Float <-> Int: Requires conversion instruction (cvt*)
# Int widening: May require sign/zero extension
# Int narrowing: Usually just truncation (fast)
fn conversion_costs():
    var floats = SIMD[DType.float32, 8](1.0)

    # These generate actual conversion instructions
    var to_int = floats.cast[DType.int32]()      # cvttps2dq on x86
    var to_double = floats.cast[DType.float64]() # vcvtps2pd on x86

    # These are typically free (just register reinterpretation)
    var to_uint = bitcast[DType.uint32](floats)  # No instruction
```

### Common Use Cases

```mojo
from memory import bitcast

# Fast absolute value via bitcast
fn fast_abs(x: SIMD[DType.float32, 8]) -> SIMD[DType.float32, 8]:
    var bits = bitcast[DType.uint32](x)
    var cleared = bits & 0x7FFFFFFF  # Clear sign bit
    return bitcast[DType.float32](cleared)

# Quantization (float to int8 for ML inference)
fn quantize(x: SIMD[DType.float32, 8], scale: Float32) -> SIMD[DType.int8, 8]:
    var scaled = x * scale
    var clamped = scaled.clamp(-128.0, 127.0)
    return clamped.cast[DType.int8]()

# Mixed precision accumulation
fn accumulate_f16_to_f32(
    data: UnsafePointer[Float16], n: Int
) -> Float32:
    var acc = SIMD[DType.float32, 8](0)
    var i = 0
    while i + 8 <= n:
        var f16_vec = data.load[width=8](i)
        acc += f16_vec.cast[DType.float32]()  # Widen for precision
        i += 8
    return acc.reduce_add()
```

### Type Reinterpretation Methods

| Method | Purpose | Changes Bits? |
|--------|---------|---------------|
| `.cast[target]()` | Value conversion (e.g., float→int) | Yes |
| `bitcast[target](val)` | Bit reinterpretation (same bits, different type) | No |
| `rebind[target](val)` | Parametric type rebinding (same representation) | No |

```mojo
var f = SIMD[DType.float32, 4](1.5)
var i = f.cast[DType.int32]()        # Value conversion: [1, 1, 1, 1]
var b = bitcast[DType.int32](f)      # Bit reinterpretation: [0x3FC00000, ...]
```

---

## Math Functions Import

Math functions like `exp`, `sqrt`, `log`, `tanh`, `sin`, `cos` are NOT methods on SIMD types. They must be imported from the `math` module.

```mojo
from math import exp, sqrt, log, tanh, sin, cos

fn compute_silu(x: SIMD[DType.float32, 8]) -> SIMD[DType.float32, 8]:
    # WRONG: No method on SIMD
    # return x.sigmoid() * x  # ERROR: no such method

    # CORRECT: Import and call functions
    return x / (1.0 + exp(-x))

fn compute_softmax_denom(x: SIMD[DType.float32, 8]) -> Float32:
    var exp_x = exp(x)  # Imported function works on SIMD
    return exp_x.reduce_add()

fn compute_distances(x: SIMD[DType.float32, 8], y: SIMD[DType.float32, 8]) -> SIMD[DType.float32, 8]:
    var diff_sq = (x - y) * (x - y)
    return sqrt(diff_sq)  # sqrt from math module
```

**Available math functions:**
- Exponential: `exp`, `exp2`, `expm1`
- Logarithmic: `log`, `log2`, `log10`, `log1p`
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Power/Root: `sqrt`, `rsqrt`, `cbrt`, `pow`
- Rounding: `floor`, `ceil`, `round`, `trunc`
- Other: `abs`, `fma`, `copysign`

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Small fixed-size types (2-4 words) | Use `TrivialRegisterPassable` trait | [`struct-design.md`](struct-design.md) |
| Types with heap allocations | Do NOT use `@register_passable` | [`memory-ownership.md`](memory-ownership.md) |
| Array processing | Use SIMD with WIDTH=8 default | [`perf-parallelization.md`](perf-parallelization.md) |
| Unknown memory alignment | Use `alignment=1` in load/store | [`memory-safety.md`](memory-safety.md) |
| Transcendental functions | Benchmark scalar vs SIMD | - |
| Matrix operations | Use SIMD dot product with transposed B | [`ffi.md`](ffi.md) |

---

## Quick Reference

- **SIMD default width**: Use 8 as portable default for Float32/Float64
- **TrivialRegisterPassable trait**: Use for small types (2-4 machine words) with no heap (replaces deprecated `@register_passable("trivial")`)
- **@register_passable**: Deprecated in v26.2; use `TrivialRegisterPassable` trait instead
- **Alignment**: Use `alignment=1` when alignment not guaranteed
- **Cache line**: 64 bytes - align for AVX-512 and cache efficiency
- **Transcendentals**: Always benchmark exp/sin/cos/log - scalar may be faster
- **Reduction**: Use `reduce_add()` for horizontal sum of SIMD vector

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `SIMD width mismatch` | Mixing vectors of different widths | Ensure all operations use same width; cast explicitly if needed |
| `cannot convert SIMD to scalar` | Missing extraction | Use `vec[0]` or `reduce_add()` to get scalar from SIMD |
| `DType not supported for SIMD` | Using invalid dtype | Check supported types: float16/32/64, int8/16/32/64, bool |
| `unaligned SIMD load` | Pointer not aligned for width | Use `ptr.load[width, alignment=1]()` for unaligned access |
| `reduce_* returns wrong type` | Horizontal ops change type | `reduce_add()` returns scalar; use appropriate type handling |
| `SIMD broadcast failed` | Wrong scalar conversion | `SIMD[DType.float32, N](val)` only splats when `val` is a `Scalar`. Use `SIMD[DType.float32, N](Scalar[DType.float32](val))` for explicit splatting |

---

## Version-Specific Features

### v26.1+ (Stable) and v26.2+ (Nightly)

| Feature | Status | Notes |
|---------|--------|-------|
| **Constants** | `alias` or `comptime` | Both work; compiler warns on `alias` in v26.1+ |
| **Register-passable** | `@register_passable("trivial")` deprecated in v26.2 | Use `TrivialRegisterPassable` trait instead |
| **Type aliases** | `comptime Float32Ptr = ...` | Use `comptime` (`alias` is deprecated) |
| **Heap allocation** | `from memory import alloc; alloc[T](n)` | v26.1+ |

**Example (current best practice):**
```mojo
# nocompile
# Constants and type aliases (comptime preferred, alias still valid)
comptime SIMD_WIDTH: Int = 8
comptime Float32Ptr = UnsafePointer[Float32]

# Register-passable types (trait-based, preferred approach)
@fieldwise_init
struct Point(TrivialRegisterPassable, Copyable):
    var x: Float64
    var y: Float64

# Heap allocation
from memory import alloc
var buffer = alloc[Float32](1024)
```

**Legacy (still works in v26.1):**
```mojo
# @register_passable("trivial") is deprecated as of v26.2
# Will be removed in a future release
@register_passable("trivial")
struct LegacyPoint:
    var x: Float64
    var y: Float64
```

**Notes:**
- Use `comptime` for compile-time constants (`alias` is deprecated in nightly)
- `@register_passable("trivial")` is deprecated in v26.2; use `TrivialRegisterPassable` trait instead
- `alloc()` is available in v26.1+ (not nightly-only)

---

## Related Patterns

- [`type-system.md`](type-system.md) — Type annotations and numeric precision
- [`type-traits.md`](type-traits.md) — Trait bounds for SIMD-compatible types
- [`memory-ownership.md`](memory-ownership.md) — Why some types can't be register-passable
- [`perf-parallelization.md`](perf-parallelization.md) — Combining SIMD with parallelism

---

## References

- [Mojo Standard Library - Algorithm](https://docs.modular.com/mojo/std/algorithm/)
- [Mojo Decorators - register_passable](https://docs.modular.com/mojo/manual/decorators/register-passable)
- [Mojo UnsafePointer docs](https://docs.modular.com/mojo/std/memory/unsafe_pointer/)
