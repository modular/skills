---
title: Block-Level Collective Operations
description: Block-wide sum, prefix_sum, broadcast operations that replace manual shared memory + barrier patterns
impact: HIGH
category: gpu
tags: [gpu, block, collectives, sum, prefix-sum, broadcast, reduction]
error_patterns:
  - "block_size parameter"
  - "Apple Silicon"
  - "shared memory"
  - "barrier"
  - "block reduction"
  - "prefix sum"
scenarios:
  - "Block-wide reduction without shared memory"
  - "Parallel stream compaction"
  - "Block normalize pattern"
---

# Block-Level Collective Operations

**Category:** gpu | **Impact:** HIGH

Block-level collectives (`gpu.primitives.block`) replace 15+ lines of manual shared memory allocation, barrier synchronization, and tree reduction with single function calls. They handle cross-warp coordination internally, eliminating an entire class of synchronization bugs.

> **GPU-kernel-only:** All `gpu.primitives.block.*` functions must be called from within a GPU kernel function (passed to `enqueue_function`). They cannot be called from host code -- doing so will cause compilation or runtime errors.

## API Availability

| API | Import Path | Availability | Notes |
|-----|-------------|--------------|-------|
| `sum` | `from gpu.primitives.block import sum` | PUBLIC | Block-wide sum reduction |
| `max` | `from gpu.primitives.block import max` | PUBLIC | Block-wide maximum |
| `min` | `from gpu.primitives.block import min` | PUBLIC | Block-wide minimum |
| `prefix_sum` | `from gpu.primitives.block import prefix_sum` | PUBLIC | Block-wide inclusive scan |
| `broadcast` | `from gpu.primitives.block import broadcast` | PUBLIC | Share value from one thread to all |

> **Note:** All block APIs listed above are available in the Mojo nightly toolchain (v26.2+). Code examples below are documentation snippets -- adapt import paths and parameters for your use case.

---

## CRITICAL: block_size Parameter Required

All `gpu.primitives.block.*` functions require the `[block_size=N]` compile-time parameter. This specifies the number of threads per block and is used internally to size shared memory and coordinate cross-warp reductions.

**Without `block_size`, kernels may silently produce wrong results or fail entirely on Apple Silicon, even if they appear to work on NVIDIA.**

```mojo
# nocompile
comptime TPB = 256  # Threads per block

# WRONG - missing block_size parameter
var total = sum(my_val)                    # May fail on Apple Silicon
var scan = prefix_sum(my_val)              # May fail on Apple Silicon
var shared = broadcast(my_val, 0)          # May fail on Apple Silicon

# CORRECT - always specify block_size
var total = sum[block_size=TPB](my_val)
var scan = prefix_sum[block_size=TPB](my_val)
var shared = broadcast[block_size=TPB](my_val, 0)
```

> **Why this matters:** On NVIDIA, the runtime can sometimes infer block dimensions. On Apple Silicon (Metal backend), the block size must be known at compile time for correct shared memory allocation. Always specify `block_size` for portable, correct code.

---

## Core Patterns

### block.sum -- Block-Wide Reduction

**When:** Reducing values across all threads in a block (sum, max, min)

**Do:**
```mojo
# nocompile
from gpu.primitives.block import sum as block_sum

comptime TPB = 256

# One line replaces 15+ lines of manual tree reduction
var total = block_sum[block_size=TPB](my_val)
# By default (broadcast=True), the result is broadcast to ALL threads.
# No separate block.broadcast call is needed for sum/max/min.
```

**Don't (manual equivalent it replaces):**
```mojo
# nocompile
from gpu import barrier, WARP_SIZE, lane_id, warp_id, thread_idx, block_dim
from gpu.memory import AddressSpace, stack_allocation
from gpu.primitives.warp import reduce, shuffle_down

comptime TPB = 256

fn manual_block_sum(val: Scalar[DType.float32]) -> Scalar[DType.float32]:
    """Manual block reduction -- block.sum replaces all of this."""

    # Phase 1: Warp-level reduction
    @parameter
    fn _add[dtype: DType, width: Int](
        x: SIMD[dtype, width], y: SIMD[dtype, width]
    ) capturing -> SIMD[dtype, width]:
        return x + y

    var warp_result = reduce[shuffle_down, _add](val)

    # Phase 2: Store warp results to shared memory
    var shared = stack_allocation[
        TPB // WARP_SIZE, DType.float32,
        address_space=AddressSpace.SHARED,
    ]()

    if lane_id() == 0:
        shared[warp_id()] = warp_result

    barrier()

    # Phase 3: First warp reduces across warp results
    var final_result = Scalar[DType.float32](0)
    if thread_idx.x < (block_dim.x // UInt(WARP_SIZE)):
        final_result = shared[thread_idx.x]

    if warp_id() == 0:
        final_result = reduce[shuffle_down, _add](final_result)

    return final_result
```

### block.prefix_sum -- Stream Compaction / Histogram

**When:** Each thread needs the cumulative sum of values from all preceding threads (inclusive scan)

```mojo
# nocompile
from gpu.primitives.block import prefix_sum as block_prefix_sum
from gpu import thread_idx

comptime TPB = 256

# Each thread flags whether it matches a condition
var is_target: Scalar[DType.int32] = 1 if data[i] > threshold else 0

# Inclusive prefix sum: first match gets write_pos=1, second gets 2, etc.
var write_pos = block_prefix_sum[block_size=TPB](is_target)

# Scatter matching elements to contiguous output positions
if is_target == 1:
    output[Int(write_pos) - 1] = data[i]  # -1 because inclusive starts at 1
```

**Use cases:**
- Stream compaction (filter elements matching a predicate)
- Histogram bin assignment
- Dynamic work distribution within a block
- Building scatter/gather indices

### block.broadcast -- Share Value from One Thread

**When:** One thread computes a value and all threads in the block need it

```mojo
# nocompile
from gpu.primitives.block import broadcast as block_broadcast

comptime TPB = 256

# Broadcast value from thread 0 to all threads in the block
var shared_val = block_broadcast[block_size=TPB](my_val, 0)

# Can broadcast from any thread, not just thread 0
var from_thread_5 = block_broadcast[block_size=TPB](my_val, 5)
```

**This replaces the manual pattern:**
```mojo
# nocompile
# Manual broadcast (don't do this when block.broadcast is available)
var shared = stack_allocation[1, DType.float32, address_space=AddressSpace.SHARED]()
if thread_idx.x == 0:
    shared[0] = my_val
barrier()
var shared_val = shared[0]
```

---

## Combined Patterns

### Block Normalize

**When:** Normalizing values by the block-wide mean (e.g., softmax denominator, layer norm)

```mojo
# nocompile
from gpu.primitives.block import sum as block_sum
from gpu import thread_idx

comptime TPB = 256

# Step 1: Block-wide sum (broadcast=True by default, ALL threads get result)
var total = block_sum[block_size=TPB](val)

# Step 2: Compute mean (total is already available in all threads)
var mean = total / Scalar[dtype](size)

# Step 3: All threads normalize their value
output[i] = val / mean
```

### Dot Product

**When:** Computing dot product of two vectors loaded across threads in a block

```mojo
# nocompile
from gpu.primitives.block import sum as block_sum
from gpu import thread_idx

comptime TPB = 256

# Each thread computes one element-wise product
var product = rebind[Scalar[dtype]](a[i]) * rebind[Scalar[dtype]](b[i])

# Block-wide sum of products
var total = block_sum[block_size=TPB](product)

# Thread 0 writes the result
if thread_idx.x == 0:
    output[0] = total
```

### Histogram Extraction (Stream Compaction)

**When:** Extracting elements matching a condition into a dense output array

```mojo
# nocompile
from gpu.primitives.block import prefix_sum as block_prefix_sum, sum as block_sum
from gpu import thread_idx

comptime TPB = 256

# Step 1: Flag matching elements
var is_match: Scalar[DType.int32] = 1 if data[i] == target_value else 0

# Step 2: Prefix sum gives each match a unique write position
var write_pos = block_prefix_sum[block_size=TPB](is_match)

# Step 3: Get total count (sum of flags)
var total_matches = block_sum[block_size=TPB](is_match)

# Step 4: Scatter matching elements
if is_match == 1:
    output[block_offset + Int(write_pos) - 1] = data[i]

# Thread 0 can store the count for this block
if thread_idx.x == 0:
    match_counts[block_idx.x] = total_matches
```

### Block-Wide Max with Broadcast

**When:** Finding the maximum value and sharing it with all threads (e.g., softmax numerically stable pass)

```mojo
# nocompile
from gpu.primitives.block import max as block_max
from gpu import thread_idx

comptime TPB = 256

# Each thread has a local value
var local_max = my_val

# Block-wide max (broadcast=True by default, ALL threads get the result)
var global_max = block_max[block_size=TPB](local_max)

# All threads can now use the block-wide max directly
output[i] = exp(my_val - global_max)
```

---

## Warp vs Block Scope Distinction

The same API names exist at both the warp and block levels. Choosing the right scope matters for correctness and performance.

| Aspect | `gpu.primitives.warp` | `gpu.primitives.block` |
|--------|----------------------|----------------------|
| **Scope** | 32 threads (one warp) | All threads in block |
| **Sync mechanism** | Implicit (lockstep) | Handles barriers internally |
| **Parameters** | No `block_size` needed | **Requires `[block_size=N]`** |
| **Shared memory** | None (register-only) | Allocated internally |
| **Performance** | ~10-20 cycles | ~50-100 cycles (cross-warp coordination) |
| **Available ops** | `sum`, `max`, `min`, `prefix_sum`, `broadcast` | `sum`, `max`, `min`, `prefix_sum`, `broadcast` |

```mojo
# nocompile
# Warp-level (32 threads, no block_size needed)
from gpu.primitives.warp import sum as warp_sum
var warp_total = warp_sum(val)

# Block-level (all threads, block_size required)
from gpu.primitives.block import sum as block_sum
var block_total = block_sum[block_size=TPB](val)
```

> **Aliasing tip:** Always import with aliases (`as block_sum`, `as warp_sum`) to avoid confusion and name collisions when using both in the same kernel.

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Sum/max/min across entire block | `block.sum[block_size=N]`, `block.max`, `block.min` | - |
| Stream compaction (filter elements) | `block.prefix_sum` for write positions, then scatter | - |
| Share computed value to all threads | `block.broadcast[block_size=N](val, thread_id)` | - |
| Normalize by block mean/sum | `block.sum` + `block.broadcast` combo | - |
| Dot product within a block | Element-wise multiply + `block.sum` | - |
| Reduction within a single warp only | Use `warp.sum` instead (no `block_size` needed) | [`gpu-warp-sync.md`](gpu-warp-sync.md) |
| Custom reduction function | Manual warp reduce + shared memory pattern | [`gpu-warp-sync.md`](gpu-warp-sync.md) |
| Cross-block reduction | Atomic operations or multi-pass kernel | [`gpu-memory-access.md`](gpu-memory-access.md) |

---

## Quick Reference

### Import Cheat Sheet

```mojo
# Block-level collectives (always specify block_size)
from gpu.primitives.block import sum as block_sum
from gpu.primitives.block import max as block_max
from gpu.primitives.block import min as block_min
from gpu.primitives.block import prefix_sum as block_prefix_sum
from gpu.primitives.block import broadcast as block_broadcast

# Common companion imports
from gpu import thread_idx, block_idx, block_dim, barrier
```

### One-Liner Reference

```mojo
# nocompile
comptime TPB = 256

var total    = block_sum[block_size=TPB](val)           # Block-wide sum
var maximum  = block_max[block_size=TPB](val)           # Block-wide max
var minimum  = block_min[block_size=TPB](val)           # Block-wide min
var scan     = block_prefix_sum[block_size=TPB](val)    # Inclusive prefix sum
var shared   = block_broadcast[block_size=TPB](val, 0)  # Broadcast from thread 0
```

---

## Common Errors

| Error / Symptom | Cause | Fix |
|-----------------|-------|-----|
| Wrong results on Apple Silicon but correct on NVIDIA | Missing `block_size` parameter | Add `[block_size=TPB]` to all block collective calls |
| Compile error about `block_size` | Omitted required compile-time parameter | Specify `[block_size=N]` where N matches your kernel's threads per block |
| `prefix_sum` output off by one | Confusing inclusive vs exclusive scan | `block.prefix_sum` is inclusive (first match = 1); subtract 1 for 0-based index |
| Only thread 0 has correct reduction result | Using `broadcast=False` explicitly or using warp-level ops (which return to lane 0 only) | Block `sum`/`max`/`min` default to `broadcast=True` (all threads get result). Warp-level `warp.sum`/`warp.max`/`warp.min` return result to lane 0 only -- use `warp.broadcast()` or `shuffle_xor` butterfly for all-lane results |
| Name collision between warp and block ops | Importing both `warp.sum` and `block.sum` | Use aliases: `from gpu.primitives.block import sum as block_sum` |
| Deadlock or hang | Conditional block collective (not all threads participate) | All threads in block must call block collectives; use predication for values, not control flow |
| Performance regression vs warp-only reduction | Using block collectives when data fits in one warp | Use `warp.sum` etc. for 32-thread reductions; block collectives add cross-warp overhead |

---

## Related Patterns

- [`gpu-warp-sync.md`](gpu-warp-sync.md) -- Warp-level shuffle, reduction, prefix sum, and broadcast
- [`gpu-fundamentals.md`](gpu-fundamentals.md) -- Thread hierarchy, block/grid dimensions
- [`gpu-warp-sync.md`](gpu-warp-sync.md) -- Barrier and sync patterns
- [`gpu-memory-access.md`](gpu-memory-access.md) -- Shared memory and coalesced access
- [`gpu-kernels.md`](gpu-kernels.md) -- Kernel launch and structured kernel patterns

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **`gpu.primitives.block.sum`** | Stable | Block-wide sum with `[block_size=N]` |
| **`gpu.primitives.block.max`** | Stable | Block-wide maximum with `[block_size=N]` |
| **`gpu.primitives.block.min`** | Stable | Block-wide minimum with `[block_size=N]` |
| **`gpu.primitives.block.prefix_sum`** | Stable | Inclusive scan with `[block_size=N]` |
| **`gpu.primitives.block.broadcast`** | Stable | Broadcast from any thread with `[block_size=N]` |

---

## References

- [Mojo GPU Block and Warp](https://docs.modular.com/mojo/manual/gpu/block-and-warp/)
- [MAX Kernels](https://github.com/modular/modular/tree/main/max/kernels)
