---
title: Debugging Patterns
description: Systematic debugging of numerical accuracy issues and GPU numerical correctness
impact: HIGH
category: debug
tags: [debugging, numerical, accuracy, gpu, correctness, floating-point, precision]
error_patterns:
  - "numerical accuracy"
  - "floating-point"
  - "NaN"
  - "inf"
  - "precision"
  - "divergence"
  - "wrong output"
  - "results differ"
  - "incorrect result"
scenarios:
  - "Debug numerical accuracy issues"
  - "Compare CPU vs GPU results"
  - "Port C code to Mojo"
  - "Fix floating-point precision"
  - "Debug GPU kernel correctness"
  - "Investigate output differences"
consolidates:
  - debug-numerical-accuracy.md
  - debug-gpu-numerical-correctness.md
---

# Debugging Patterns

**Category:** debug | **Impact:** HIGH

Systematic approaches for debugging numerical accuracy issues, especially when porting code or comparing CPU vs GPU implementations. Numerical divergence can cause completely wrong outputs (noisy images, wrong predictions) while still producing "valid" results that compile and run without errors.

---

## Core Concepts

### Key Insight: Divergence is Usually Upstream

When outputs differ, the root cause is almost never where you first notice it. Work backwards systematically:

```
Output differs -> Final computation -> Intermediate layers -> Input processing
     ^                                                              ^
  You notice                                                   Root cause
  it here                                                      is here
```

### Acceptable vs Unacceptable Differences

| Difference | Cause | Acceptable? |
|------------|-------|-------------|
| < 0.01% | FP rounding | Yes |
| 1-5% | Accumulation order, -ffast-math | Usually yes |
| 5-20% | Algorithm difference | Investigate |
| > 20% | Bug in implementation | No, fix it |

**Note:** C compilers with `-ffast-math` can reorder floating-point operations, causing legitimate 1-5% differences that are NOT bugs.

---

## Numerical Accuracy in C-to-Mojo Ports

When porting numerical C code to Mojo, small differences can compound through deep computation graphs (neural networks, signal processing, physics simulations) and produce dramatically different results.

### Common Root Causes (in order of frequency)

#### 1. Wrong Data Keys When Loading External Data

**Problem:** External files (model weights, config files) use different key names than expected.

**Don't:**
```mojo
# nocompile
# Looking for non-existent keys, silently defaults to zeros
var mean = data.find("latent_mean")  # Returns UNKNOWN
if mean.dtype != DataType.UNKNOWN:
    config.mean = data.get_f32(mean)
else:
    # Silently uses zeros - CORRUPTS ALL DOWNSTREAM COMPUTATION
    config.mean = alloc[Float32](128)
    for i in range(128):
        config.mean[i] = 0.0
```

**Do:**
```mojo
# nocompile
# Use exact keys from the data file
var mean = data.find("running_mean")  # Actual key in file
if mean.dtype != DataType.UNKNOWN:
    config.mean = data.get_f32(mean)
```

**How to debug:** Use Python to inspect actual keys in the data file.

#### 2. Magic Constants Mismatch

**Problem:** Different implementations use different magic constants (padding tokens, special values, etc.).

**Don't:**
```mojo
var pad_value = 0  # Common default, but wrong for many implementations
```

**Do:**
```mojo
var pad_value = 151643  # Check reference code for actual value!
```

**Impact:** Wrong constants can cause completely different computation paths and outputs.

#### 3. Float64 vs Float32 Literals

**Problem:** Mojo's `1.0` defaults to Float64, causing precision differences.

**Don't:**
```mojo
fn softmax_row(row: Float32Ptr, cols: Int):
    var sum_val: Float32 = 0.0
    # ... compute sum ...
    var inv_sum = 1.0 / sum_val  # Float64 division!
    for i in range(cols):
        row[i] = row[i] * inv_sum  # Implicit cast back to Float32
```

**Do:**
```mojo
fn softmax_row(row: Float32Ptr, cols: Int):
    var sum_val: Float32 = 0.0
    # ... compute sum ...
    var inv_sum: Float32 = Float32(1.0) / sum_val  # Float32 throughout
    for i in range(cols):
        row[i] = row[i] * inv_sum
```

#### 4. Matrix Dimension/Layout Mismatches

**Problem:** BLAS expects specific layouts (row-major vs column-major).

**Don't:**
```mojo
# nocompile
# Transposed dimensions for BLAS sgemm
BLASContext.sgemm(
    CblasNoTrans, CblasNoTrans,
    N, M, K,  # Swapped M and N!
    1.0, A, K, B, N, 0.0, C, N
)
```

**Do:**
```mojo
# nocompile
# C = A[M,K] @ B[K,N] requires M,N,K order
BLASContext.sgemm(
    CblasNoTrans, CblasNoTrans,
    M, N, K,
    1.0, A, K, B, N, 0.0, C, N
)
```

---

## Systematic Debugging Process

### Step 1: Add Comparison Points

Add debug prints at key computation stages in BOTH implementations:

```mojo
# nocompile
# Mojo
print("[DBG] output at pos 0:", output[0], output[1], output[2])
print("[DBG] output at pos 508:", output[508 * dim], output[508 * dim + 1])
```

```c
// C
fprintf(stderr, "[DBG-C] output at pos 0: %f %f %f\n", output[0], output[1], output[2]);
fprintf(stderr, "[DBG-C] output at pos 508: %f %f %f\n", output[508*dim], output[508*dim+1]);
```

### Step 2: Binary Search for Divergence Point

1. Start at output - if different, go to middle of pipeline
2. If middle matches, problem is downstream; if different, go upstream
3. Repeat until you find the first point of divergence

```
Final output differs (10% error)
    | Check intermediate stage
Stage 2 matches at position 0
Stage 2 differs at position 508 (boundary region!)
    | Check earlier stage
Stage 1 differs at position 508
    | Check input processing
Inputs match, but boundary handling differs!
    | ROOT CAUSE FOUND
```

### Step 3: Compare Specific Positions

Focus on boundary conditions and edge cases:
- Position 0 (first element)
- Padded positions (often different!)
- Last valid position before padding
- Positions with extreme values (min/max)

```mojo
# nocompile
# Print stats to quickly identify divergence
var min_val: Float32 = tensor[0]
var max_val: Float32 = tensor[0]
var sum_val: Float32 = 0.0
for i in range(size):
    min_val = min(min_val, tensor[i])
    max_val = max(max_val, tensor[i])
    sum_val += tensor[i]
print("[DBG] min:", min_val, "max:", max_val, "mean:", sum_val / size)
```

---

## GPU Numerical Correctness

When GPU output differs from CPU output, use this systematic approach to identify the root cause.

### Symptoms of GPU Numerical Issues

1. **Correlation < 1.0** between GPU and CPU outputs
2. **Different value ranges** (e.g., GPU values are larger/smaller)
3. **Errors accumulate** over layers or iterations
4. **Visual differences** in output images

### GPU Debugging Methodology

#### Step 1: Isolate with Single-Step Test

Reduce to minimum complexity to isolate the issue:

```bash
# Run with minimal configuration, fixed seed for reproducibility
./program --minimal --seed 42 -o gpu_output.bin
FORCE_CPU=1 ./program --minimal --seed 42 -o cpu_output.bin

# Compare outputs (example using Python)
python3 -c "
import numpy as np
gpu = np.fromfile('gpu_output.bin', dtype=np.float32)
cpu = np.fromfile('cpu_output.bin', dtype=np.float32)
corr = np.corrcoef(gpu.flatten(), cpu.flatten())[0, 1]
diff = np.abs(gpu - cpu)
print(f'Correlation: {corr:.4f}, Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}')
"
```

#### Step 2: Check Intermediate Values

Add debug output to compare intermediate values:

```mojo
# nocompile
# After each major operation, print stats
print("  v stats: min=", v.min(), " max=", v.max(), " mean=", v.mean())
```

If values diverge early, the bug is in early operations.
If values diverge later, the bug is in later operations or accumulation.

#### Step 3: Common GPU Root Causes

**Precision Issues:**
- BF16 to F16 conversion causes exponent overflow/underflow
- Mixed precision accumulation errors
- Fix: Use F32 weights or native BF16 shaders

**Synchronization Issues:**
- Reading GPU buffer before operations complete
- Fix: Ensure `gpu_batch_end()` or sync before reading

**Memory Layout Issues:**
- Row-major vs column-major confusion
- Transpose errors in matrix operations
- Fix: Verify matrix dimensions and layout

**Numerical Stability:**
- Attention softmax overflow
- Normalization issues
- Fix: Use numerically stable algorithms

#### Step 4: Binary Search for Bug Location

If you can't pinpoint the issue:

1. Run half the layers on GPU, half on CPU
2. If error persists, bug is in first half
3. Continue bisecting until you find the offending layer/operation

### Example: Precision Loss During Type Conversion

**Symptom:** GPU output has smaller value range than CPU

**Root cause:** Type conversion causes underflow/overflow (e.g., BF16 to F16)

**Fix:** Use higher precision types or avoid lossy conversions:

```mojo
# Before (lossy):
gpu_compute_lowprec(x, weights_bf16, ...)

# After (correct):
gpu_compute(x, weights_f32, ...)
```

---

## Verification Checklist

### For C-to-Mojo Ports

- [ ] Data file keys match exactly
- [ ] Magic constants match reference implementation
- [ ] Float32 literals used consistently (not Float64)
- [ ] Matrix dimensions and layouts correct
- [ ] Boundary conditions handled identically

### For GPU Correctness

- [ ] Minimal test output matches CPU (correlation ~1.0)
- [ ] Extended computation doesn't diverge
- [ ] Different random seeds produce valid output
- [ ] Larger problem sizes don't introduce new errors
- [ ] Output is correct (visual inspection if applicable)

---

## Decision Guide

| Symptom | Check First | Common Fix |
|---------|-------------|------------|
| Output completely wrong | Data loading keys | Use correct key names |
| Gradual drift over layers | Float precision | Use explicit Float32 |
| Boundary artifacts | Padding handling | Match reference exactly |
| GPU values out of range | Type conversion | Avoid BF16->F16 |
| Random test failures | Synchronization | Add proper GPU sync |

---

## Quick Reference

- **Upstream rule**: The root cause is almost always earlier in the pipeline than where you notice the problem
- **Binary search**: Halve the search space with each debug iteration
- **Print stats**: min/max/mean quickly reveals divergence
- **Position matters**: Check boundaries and edge cases first
- **Precision**: Mojo `1.0` is Float64; use `Float32(1.0)` for consistency
- **GPU sync**: Always sync before reading GPU buffers

---

## Cleanup After Debugging

Always remove debug prints after finding the issue:

```mojo
# Remove these after debugging!
# print("[DBG] output:", out[0], out[1], out[2])
```

Or use a debug flag:

```mojo
# nocompile
comptime DEBUG: Bool = False

if DEBUG:
    print("[DBG] output:", out[0], out[1], out[2])
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `debug_assert not triggered` | Release build strips asserts | Use `comptime assert` for compile-time checks that persist |
| `print not working in GPU` | printf not supported in GPU kernels | Use device-side debug: copy to host then print |
| `NaN in computation` | Division by zero or invalid operation | Add `debug_assert(not isnan(x))` checks at key points |
| `memory corruption` | Out-of-bounds access | Enable bounds checking; use Span instead of raw pointers |
| `different results debug vs release` | Uninitialized variables | Initialize all variables; use `var x: T = default_value` |
| `breakpoint not hit` | Optimized away or wrong location | Use `@no_inline` on function; add `debug_break()` call |
| `kernel changes not reflected` | Stale compilation cache | Purge caches: `rm -rf ~/.modular/.max_cache ~/.modular/.mojo_cache ~/.modular/.mogg_cache` |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **Debug flag** | `alias` or `comptime` | Both work in v26.1+ |
| **debug_assert** | `debug_assert(cond, msg)` | Stable |
| **comptime assert** | `comptime assert cond, "msg"` | Stable |
| **print** | `print(...)` | Stable |

**Example (v26.1+):**
```mojo
# nocompile
comptime DEBUG = True
comptime TRACE_LEVEL = 2

fn debug_computation(x: Float32) -> Float32:
    if DEBUG:
        print("[DBG] input:", x)

    debug_assert(not isnan(x), "NaN input detected")

    var result = x * 2.0

    if DEBUG:
        print("[DBG] output:", result)

    return result
```

**Notes:**
- Both `alias` and `comptime` work for compile-time constants in v26.1+
- Debugging methodology is stable across versions
- `debug_assert` is stripped in release builds (stable behavior)
- `comptime assert` for compile-time checks is stable
- GPU debugging (copy to host then print) is stable across versions

---

## Related Patterns

- [`error-handling.md`](error-handling.md) — Error handling and debug_assert usage
- [`gpu-fundamentals.md`](gpu-fundamentals.md) — GPU programming basics

---

## References

- [Mojo Manual](https://docs.modular.com/mojo/manual/)
