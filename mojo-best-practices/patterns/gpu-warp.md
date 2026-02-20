---
title: Warp Primitives and Reduction Patterns
description: Warp shuffle operations, warp specialization, row reduction, and block reduction patterns
impact: HIGH
category: gpu
tags: [gpu, warp, shuffle, reduction, specialization]
error_patterns:
  - "warp divergence"
  - "shuffle error"
  - "invalid lane"
  - "reduction"
  - "vote mask"
  - "block_size required"
  - "shuffle_xor"
scenarios:
  - "Implement warp-level reduction"
  - "Use shuffle for fast communication"
  - "Specialize warps for producer/consumer"
  - "Fix warp divergence issue"
  - "Use block collectives for reduction"
  - "Implement butterfly all-reduce"
  - "Normalize values across a block"
  - "Use shuffle_xor for pair communication"
consolidates:
  - gpu-warp-primitives.md
  - gpu-warp-specialization.md
  - gpu-row-reduction.md
  - gpu-block-reduction.md
---

# Warp Primitives and Reduction Patterns

**Category:** gpu | **Impact:** HIGH

Warp shuffle operations enable direct register-to-register communication (5-10x faster than shared memory). Combined with warp specialization and efficient reduction patterns, these primitives are essential for high-performance GPU kernels.

## API Availability

| API | Import Path | Availability | Notes |
|-----|-------------|--------------|-------|
| `lane_id`, `warp_id` | `from gpu import lane_id, warp_id` | PUBLIC | Re-exported from `gpu.primitives.id` |
| `WARP_SIZE` | `from gpu import WARP_SIZE` | PUBLIC | Re-exported from `gpu.globals` |
| `shuffle_idx`, `shuffle_up`, `shuffle_down`, `shuffle_xor` | `from gpu.primitives.warp import ...` | PUBLIC | Warp shuffle operations |
| `vote[ret_type](val: Bool)` | `from gpu.primitives.warp import vote` | PUBLIC | Creates warp-wide bitmask from boolean votes |
| `sum`, `max`, `min`, `prefix_sum` | `from gpu.primitives.warp import ...` | PUBLIC | Warp-wide reduction builtins |
| `broadcast` | `from gpu.primitives.warp import broadcast` | PUBLIC | Broadcast lane 0 value to all lanes |
| `lane_group_sum`, `lane_group_max`, `lane_group_min` | `from gpu.primitives.warp import ...` | PUBLIC | Lane-group reductions (partial warp) |
| `lane_group_reduce`, `reduce` | `from gpu.primitives.warp import ...` | PUBLIC | Generic reduction with custom functions |
| `elect_one_sync` | `from gpu.primitives.cluster import elect_one_sync` | PUBLIC | Cluster-level (NOT warp), SM90+ |
| `stack_allocation[N, T, address_space=AddressSpace.SHARED]()` | `from gpu.memory import ...` | PUBLIC | Shared memory allocation |

> **Note:** All warp APIs listed above are available in the Mojo nightly toolchain (v26.2+). Code examples below are documentation snippets — adapt import paths and parameters for your use case.

---

## Core Concepts

### Warp Basics

| Architecture | Warp Size | Name |
|--------------|-----------|------|
| NVIDIA | 32 threads | Warp |
| AMD | 64 threads | Wavefront |

```mojo
# nocompile

from gpu import lane_id, warp_id, WARP_SIZE

# Get lane ID (0 to WARP_SIZE-1)
var lid = lane_id()

# Get warp ID within the block
var wid = warp_id()

# Get warp size (32 on NVIDIA, 64 on AMD)
var wsize = WARP_SIZE
```

### Shuffle Operations

Shuffles allow threads to read values from other threads' registers directly:

```mojo
# nocompile

from gpu.primitives.warp import shuffle_idx, shuffle_up, shuffle_down, shuffle_xor

var my_val = Scalar[DType.float32](42.0)

# shuffle_idx: Read value from any lane
var val_from_lane_5 = shuffle_idx(my_val, 5)

# shuffle_up: Read from lane (current - offset)
var val_from_below = shuffle_up(my_val, UInt32(2))  # lane 5 reads from lane 3

# shuffle_down: Read from lane (current + offset)
var val_from_above = shuffle_down(my_val, UInt32(2))  # lane 3 reads from lane 5

# shuffle_xor: Read from lane (current XOR offset)
var val_xor = shuffle_xor(my_val, UInt32(1))  # Adjacent exchange (butterfly)
```

**Shuffle function signatures:**
```mojo
# nocompile

# Without mask (uses full warp mask):
fn shuffle_idx[dtype: DType, simd_width: Int](val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[dtype, simd_width]
fn shuffle_down[dtype: DType, simd_width: Int](val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[dtype, simd_width]
fn shuffle_up[dtype: DType, simd_width: Int](val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[dtype, simd_width]
fn shuffle_xor[dtype: DType, simd_width: Int](val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[dtype, simd_width]

# With explicit mask:
fn shuffle_idx[dtype: DType, simd_width: Int](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[dtype, simd_width]
```

| Operation | Shuffle Type | Use Case |
|-----------|--------------|----------|
| Broadcast | `shuffle_idx(val, lane)` | Share result from one lane |
| Reduce | `shuffle_down` | Tree reductions |
| Scan | `shuffle_up` | Prefix sums |
| All-reduce | `shuffle_xor` | Everyone gets result |

### Vote Operation (Ballot)

The `vote` function creates a warp-wide bitmask from per-thread boolean predicates. This is the Mojo equivalent of CUDA's `__ballot_sync`:

```mojo
# nocompile

from gpu.primitives.warp import vote

# vote: Get bitmask of lanes where predicate is true
var my_predicate = some_condition()
var mask = vote[DType.uint32](my_predicate)  # Returns Scalar[DType.uint32] bitmask

# Count how many lanes have predicate true (use popcount on the mask)
var count = pop_count(mask)

# Check if all lanes voted true
var all_true = (mask == UInt32(0xFFFFFFFF))

# Check if any lane voted true
var any_true = (mask != UInt32(0))
```

**Vote function signature:**
```mojo
# nocompile

fn vote[ret_type: DType](val: Bool) -> Scalar[ret_type]
# ret_type must be DType.uint32 (NVIDIA) or DType.uint32/DType.uint64 (AMD)
```

| Operation | How to do it | Notes |
|-----------|--------------|-------|
| Ballot (get vote mask) | `vote[DType.uint32](predicate)` | Returns bitmask of lanes voting True |
| All voted true | `vote[DType.uint32](pred) == 0xFFFFFFFF` | Compare with full mask |
| Any voted true | `vote[DType.uint32](pred) != 0` | Check for any set bit |
| Count true votes | `pop_count(vote[DType.uint32](pred))` | Population count on the mask |

> **Note:** `vote` is only supported on NVIDIA and AMD GPUs. NVIDIA only supports `DType.uint32` return type. AMD supports both `DType.uint32` and `DType.uint64`.

---

## Common Patterns

### Warp Reduction (Shuffle)

**When:** Reducing values within a warp (sum, max, min)

**Do:**
```mojo

from gpu.primitives.warp import shuffle_down

fn warp_sum_fast(val: Scalar[DType.float32]) -> Scalar[DType.float32]:
    """Fast: Uses register shuffles for warp communication."""
    var result = val

    # Tree reduction using shuffle_down
    result += shuffle_down(result, UInt32(16))  # Lanes 0-15 get partial sum
    result += shuffle_down(result, UInt32(8))
    result += shuffle_down(result, UInt32(4))
    result += shuffle_down(result, UInt32(2))
    result += shuffle_down(result, UInt32(1))

    # Result is in lane 0
    return result
```

**Or use the built-in warp reduction:**
```mojo
# nocompile

from gpu.primitives.warp import sum as warp_sum, max as warp_max, min as warp_min

# These return the reduction result as a scalar
var total = warp_sum(val)      # Warp-wide sum
var max_val = warp_max(val)    # Warp-wide maximum
var min_val = warp_min(val)    # Warp-wide minimum
```

**Don't:**
```mojo
# nocompile

from gpu.memory import AddressSpace, stack_allocation

fn warp_sum_slow(val: Scalar[DType.float32]) -> Scalar[DType.float32]:
    """Slow: Uses shared memory for warp communication."""
    # Even with correct API, this is slow compared to shuffles
    var shared = stack_allocation[32, DType.float32, address_space=AddressSpace.SHARED]()
    var lid = thread_idx.x % 32

    # Write to shared memory (slow)
    shared[lid] = val
    barrier()

    # Read and accumulate (many memory accesses)
    var sum: Scalar[DType.float32] = 0.0
    if lid == 0:
        for i in range(32):
            sum += shared[i]

    barrier()
    return sum
```

### Butterfly Sum (All-Reduce)

**When:** All lanes need the final result

```mojo

from gpu.primitives.warp import shuffle_xor

fn butterfly_sum(val: Scalar[DType.float32]) -> Scalar[DType.float32]:
    """Butterfly reduction - ALL lanes get the total sum."""
    var result = val

    # XOR with increasing powers of 2
    result += shuffle_xor(result, UInt32(1))   # Exchange with neighbor
    result += shuffle_xor(result, UInt32(2))   # Exchange pairs
    result += shuffle_xor(result, UInt32(4))   # Exchange quads
    result += shuffle_xor(result, UInt32(8))   # Exchange octets
    result += shuffle_xor(result, UInt32(16))  # Exchange halves

    # All lanes now have the total sum
    return result
```

**Or use the built-in lane_group_sum_and_broadcast:**
```mojo
# nocompile

from gpu import WARP_SIZE
from gpu.primitives.warp import lane_group_sum_and_broadcast

# All lanes get the total sum
var total = lane_group_sum_and_broadcast[num_lanes=WARP_SIZE](val)
```

### Butterfly Operations (`shuffle_xor`)

`shuffle_xor` enables butterfly communication where each lane exchanges data with a partner determined by XOR of its lane ID and an offset. Unlike `shuffle_down` (result only in lane 0), butterfly patterns give **all lanes the final result**.

**Pair swap — adjacent pairs exchange values:**
```mojo
# nocompile
from gpu.primitives.warp import shuffle_xor

# Lane 0↔1, 2↔3, 4↔5, ... swap values
var partner_val = shuffle_xor(my_val, UInt32(1))
```

**Butterfly all-reduce — every lane gets the reduction result:**
```mojo
# nocompile
# All lanes end up with the global max (no shared memory needed)
var val = my_value
for offset in List(1, 2, 4, 8, 16):
    val = max(val, shuffle_xor(val, offset))
# val is now the max across all 32 lanes — in EVERY lane
```

**Butterfly prefix sum — conditional add based on lane position:**
```mojo
# nocompile
var val = my_value
var lid = lane_id()
for offset in List(1, 2, 4, 8, 16):
    var other = shuffle_xor(val, offset)
    if lid >= offset:
        val += other
# Each lane now holds the prefix sum up to its position
```

**Warp partition via `shuffle_xor`:**
```mojo
# nocompile
# Split warp into groups of 4, each group exchanges within itself
var partner = shuffle_xor(my_val, UInt32(3))  # XOR with 0b11 → pairs within groups of 4
```

**When to use `shuffle_xor` vs `shuffle_down`:**

| Use Case | Operation | Result Location |
|----------|-----------|-----------------|
| Reduction to lane 0 | `shuffle_down` | Lane 0 only |
| All-reduce (all lanes need result) | `shuffle_xor` butterfly | All lanes |
| Broadcast from lane 0 | `broadcast` or `shuffle_idx` | All lanes |
| Adjacent pair exchange | `shuffle_xor(val, 1)` | Both partners |

### Stream Compaction via prefix_sum

**When:** Filtering elements matching a predicate and packing them contiguously

```mojo
# nocompile

from gpu.primitives.warp import prefix_sum

# Flag matching elements, prefix_sum for write positions, scatter
var is_match = UInt32(1) if my_val < pivot else UInt32(0)
var write_pos = prefix_sum(is_match)  # Inclusive: first match gets 1

if is_match == 1:
    output[Int(write_pos) - 1] = my_val  # -1 because inclusive prefix sum
```

**Also available at block scope:** See [`gpu-block-collectives.md`](gpu-block-collectives.md) for `block.prefix_sum`.

### Lane Group Reductions (Partial Warp)

**When:** Reducing within a subgroup of the warp (e.g., first 16 lanes)

```mojo
# nocompile

from gpu.primitives.warp import lane_group_sum, lane_group_max, lane_group_min

# Reduce within groups of 16 lanes
var partial_sum = lane_group_sum[num_lanes=16](val)
var partial_max = lane_group_max[num_lanes=16](val)
var partial_min = lane_group_min[num_lanes=16](val)
```

### Generic Warp Reduction

**When:** You need a custom reduction function

```mojo
# nocompile

from gpu.primitives.warp import reduce, shuffle_down, lane_group_reduce

# Generic reduction with custom function
@parameter
fn _reduce_add[dtype: DType, width: Int](
    x: SIMD[dtype, width], y: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    return x + y

# Full warp reduction
var result = reduce[shuffle_down, _reduce_add](val)

# Lane group reduction (first 16 lanes)
var partial = lane_group_reduce[shuffle_down, _reduce_add, num_lanes=16](val)
```

### Block-Level Reduction

**When:** Reducing across all threads in a block (>32 threads)

> **Note:** Mojo also provides built-in block-level collectives via `gpu.primitives.block`:
> ```mojo
>
> from gpu.primitives.block import sum, max, min, broadcast, prefix_sum
> alias TPB = 256  # Must match your kernel's block_dim
> var block_total = sum[block_size=TPB](val)           # Block-wide sum
> var block_max = max[block_size=TPB](val)             # Block-wide maximum
> var shared_val = broadcast[block_size=TPB](val, 0)   # Broadcast from thread 0
> var scan = prefix_sum[block_size=TPB](val)           # Block-wide inclusive scan
> ```
> These handle the warp+shared memory two-phase reduction internally. **Always specify `[block_size=N]`** -- see [`gpu-block-collectives.md`](gpu-block-collectives.md). Use the manual pattern below for custom reduction functions.

```mojo
# nocompile

from gpu import barrier, WARP_SIZE, lane_id, warp_id, thread_idx, block_dim
from gpu.memory import AddressSpace, stack_allocation
from gpu.primitives.warp import reduce, shuffle_down

fn block_reduce[
    BLOCK_SIZE: Int,
    reduce_fn: fn[dtype: DType, width: Int](SIMD[dtype, width], SIMD[dtype, width]) -> SIMD[dtype, width],
    dtype: DType,
](val: Scalar[dtype], init: Scalar[dtype]) -> Scalar[dtype]:
    """Reduce across all threads in a block."""

    # Phase 1: Warp-level reduction using shuffles (fast)
    var warp_accum = reduce[shuffle_down, reduce_fn](val)

    # Phase 2: Store warp results to shared memory
    var shared = stack_allocation[
        BLOCK_SIZE // WARP_SIZE,
        dtype,
        address_space = AddressSpace.SHARED,
    ]()

    # Only lane 0 of each warp writes
    if lane_id() == 0:
        shared[warp_id()] = warp_accum

    barrier()  # Wait for all warps

    # Phase 3: First warp reduces across warp results
    var final_accum = init
    if thread_idx.x < (block_dim.x // UInt(WARP_SIZE)):
        final_accum = shared[thread_idx.x]
    else:
        final_accum = init

    if warp_id() == 0:
        final_accum = reduce[shuffle_down, reduce_fn](final_accum)

    return final_accum
```

### Prefix Sum (Scan)

**When:** Each thread needs the cumulative sum of all preceding threads

```mojo
# nocompile

from gpu.primitives.warp import prefix_sum

# Inclusive prefix sum (each thread gets sum of all threads up to and including itself)
var inclusive_scan = prefix_sum(my_val)

# Exclusive prefix sum (each thread gets sum of all threads before it)
var exclusive_scan = prefix_sum[exclusive=True](my_val)
```

### Block-Level Collectives

Block collectives from `gpu.primitives.block` replace 15+ lines of shared memory + barrier + tree reduction with a single function call. They handle the two-phase pattern (warp reduce → shared memory → warp reduce) internally.

**Available operations (always specify `[block_size=N]`):**
```mojo
# nocompile
from gpu.primitives.block import sum, max, min, broadcast, prefix_sum

comptime TPB = 256  # Must match your kernel's block_dim

# Block-wide reduction — all threads contribute
var total = sum[block_size=TPB](my_val)             # Sum across all threads in block
var maximum = max[block_size=TPB](my_val)           # Maximum across all threads
var minimum = min[block_size=TPB](my_val)           # Minimum across all threads

# Broadcast — distribute a value from one thread to all threads
var shared_val = broadcast[block_size=TPB](my_val, 0)  # Broadcast from thread 0

# Prefix sum — each thread gets cumulative sum of all preceding threads
var scan = prefix_sum[block_size=TPB](my_val)       # Inclusive scan across block
```

**Combined normalize pattern (sum → mean → broadcast → divide):**
```mojo
# nocompile
comptime TPB = 256

fn block_normalize[dtype: DType](
    output: LayoutTensor[dtype, ...],
    input: LayoutTensor[dtype, ...],
    size: Int,
):
    var global_i = block_idx.x * block_dim.x + thread_idx.x
    var val = rebind[Scalar[dtype]](input[global_i]) if global_i < size else Scalar[dtype](0)

    # 1. Sum across block
    var total = sum[block_size=TPB](val)

    # 2. Compute mean (only thread 0 has the sum)
    var mean: Scalar[dtype] = 0
    if thread_idx.x == 0:
        mean = total / Scalar[dtype](size)

    # 3. Broadcast mean to all threads
    mean = broadcast[block_size=TPB](mean, 0)

    # 4. Normalize
    if global_i < size:
        output[global_i] = val / mean
```

**Warp vs Block scope decision:**

| Consideration | Warp Ops (`gpu.primitives.warp`) | Block Ops (`gpu.primitives.block`) |
|---------------|----------------------------------|-------------------------------------|
| Scope | Single warp (32 threads) | Entire block (up to 1024 threads) |
| Shared memory | None needed | Used internally |
| Best for | Single-warp kernels, partial reductions | Cross-warp operations, full-block reductions |
| Performance | Minimal latency (register-only) | Slightly higher (shared mem + barriers) |
| Custom reduce fn | `reduce[shuffle_down, my_fn]` | Use manual pattern (below Block-Level Reduction) |

**Performance:** Block collectives produce the same result as the manual warp+shared memory pattern with identical performance characteristics, but require significantly less code.

### Row-Wise Reduction (Softmax, LayerNorm)

**When:** Each block processes one row for normalization operations

```mojo
# nocompile

fn row_reduce[
    BLOCK_SIZE: Int,
    input_fn: fn[dtype: DType, width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    reduce_fn: fn[dtype: DType, width: Int](SIMD[dtype, width], SIMD[dtype, width]) -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
](
    row_coords: IndexList[rank],
    axis: Int,
    init: Scalar[dtype],
    row_size: Int,
) -> Scalar[dtype]:
    """Reduce a single row, called by one block."""

    var tid = thread_idx.x
    var accum = init

    # Main loop: vectorized with grid-stride pattern
    for offset in range(0, row_size, BLOCK_SIZE * simd_width):
        var idx = (tid + offset) * simd_width
        if idx < row_size:
            row_coords[axis] = Int(idx)
            var val = input_fn[dtype, simd_width, rank](row_coords)
            accum = reduce_fn(val, accum)

    # Reduce SIMD vector to scalar
    var scalar_accum = accum.reduce_add()

    # Block reduction across threads
    return block_reduce[BLOCK_SIZE, reduce_fn](scalar_accum, init)
```

### Softmax Two-Pass Pattern

```mojo
# nocompile

fn softmax_kernel[dtype: DType, BLOCK_SIZE: Int](
    output: LayoutTensor[dtype, ...],
    input: LayoutTensor[dtype, ...],
    row_size: Int,
):
    var row_idx = block_idx.x

    # Pass 1: Find max value in row
    var max_val = row_reduce[BLOCK_SIZE, load_input, max_fn, dtype, 4](
        IndexList[2](row_idx, 0), axis=1,
        init=Scalar[dtype].MIN, row_size=row_size,
    )

    # Broadcast max to all threads
    var shared_max = stack_allocation[1, dtype, address_space=AddressSpace.SHARED]()
    if thread_idx.x == 0:
        shared_max[0] = max_val
    barrier()
    max_val = shared_max[0]

    # Pass 2: Compute sum of exp(x - max)
    @always_inline
    fn exp_shifted[d: DType, w: Int, r: Int](coords: IndexList[r]) -> SIMD[d, w]:
        var val = input.load[width=w](coords)
        return exp(val - max_val)

    var sum_exp = row_reduce[BLOCK_SIZE, exp_shifted, add_fn, dtype, 4](
        IndexList[2](row_idx, 0), axis=1,
        init=Scalar[dtype](0), row_size=row_size,
    )

    # Broadcast sum and normalize
    if thread_idx.x == 0:
        shared_max[0] = sum_exp
    barrier()
    sum_exp = shared_max[0]

    var inv_sum = 1.0 / sum_exp
    for col in range(thread_idx.x, row_size, block_dim.x):
        output[row_idx, col] = exp(input[row_idx, col] - max_val) * inv_sum
```

### Warp Specialization

**When:** Overlapping memory and compute for maximum throughput (SM90/SM100)

```mojo
# nocompile

from gpu import WARP_SIZE, warp_id

# Define warp roles
comptime SCHEDULER_THREADS = WARP_SIZE      # Warp 0: Coordination
comptime TMA_LOAD_THREADS = WARP_SIZE       # Warp 1: TMA loading
comptime MMA_THREADS = WARP_SIZE            # Warp 2: Tensor core compute
comptime EPILOGUE_THREADS = 4 * WARP_SIZE   # Warps 3-6: Output

comptime NUM_THREADS = (
    SCHEDULER_THREADS + TMA_LOAD_THREADS + MMA_THREADS + EPILOGUE_THREADS
)

fn specialized_matmul_kernel(a: Tensor, b: Tensor, c: Tensor):
    var wid = warp_id()

    if wid == 0:
        # Scheduler warp: coordinate tile distribution
        scheduler_loop()
    elif wid == 1:
        # Load warp: async TMA loads from global memory
        load_loop()
    elif wid == 2:
        # MMA warp: tensor core operations
        compute_loop()
    else:
        # Epilogue warps: scaling, accumulation, output
        epilogue_loop()
```

### SM90 Producer-Consumer Pattern

```mojo
# nocompile

from gpu import warp_id

# Warp group 0: Producer (TMA loading)
# Warp groups 1+: Consumers (WGMMA compute)

fn warp_group_kernel():
    var warp_group_idx = warp_id() // 4  # 4 warps per warp group

    if warp_group_idx == 0:
        # Producer: TMA loads with register deallocation
        warpgroup_reg_dealloc[num_regs]()
        producer_loop()
    else:
        # Consumer: WGMMA with extra registers
        warpgroup_reg_alloc[num_regs]()
        consumer_loop()
```

### Broadcast from Lane 0

**When:** One thread computes a value and all threads need it

```mojo
# nocompile

from gpu.primitives.warp import broadcast

# Broadcast a SIMD value from lane 0 to all lanes
var shared_val = broadcast(my_val)

# Broadcast an Int from lane 0
var shared_int = broadcast(my_int)
```

> **Note:** The single-argument `broadcast(val)` always broadcasts from lane 0. When the value comes from a LayoutTensor read, use `broadcast(rebind[Scalar[dtype]](val))` to ensure a concrete scalar type.

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Warp-level sum/max/min | Use built-in `sum`/`max`/`min` from `gpu.primitives.warp` | - |
| All lanes need result | Use `lane_group_sum_and_broadcast` or shuffle_xor butterfly | - |
| Partial warp reduction | Use `lane_group_sum[num_lanes=N]` | - |
| Custom reduction op | Use `reduce[shuffle_down, my_fn]` | - |
| Block-level reduction | Warp reduce + shared memory + warp reduce | - |
| Row-wise operations | One block per row, vectorized + block reduce | - |
| High-perf matmul (SM90+) | Use warp specialization | [`gpu-kernels.md`](gpu-kernels.md) |
| Boolean predicate across warp | Use `vote[DType.uint32](predicate)` | - |
| Prefix sum / scan | Use `prefix_sum(val)` or `prefix_sum[exclusive=True](val)` | - |

---

## Quick Reference

### Performance Comparison

| Method | Latency | Memory Access |
|--------|---------|---------------|
| Shared memory reduction | ~100 cycles | 32 reads + 32 writes |
| Shuffle reduction | ~10-20 cycles | 0 (register only) |
| Built-in collective | ~10-15 cycles | 0 (optimized) |

### Built-in Collectives

```mojo
# nocompile

from gpu.primitives.warp import sum, max, min, prefix_sum, broadcast

var total = sum(val)          # Warp-wide sum (returns scalar)
var max_val = max(val)        # Warp-wide maximum (returns scalar)
var min_val = min(val)        # Warp-wide minimum (returns scalar)
var prefix = prefix_sum(val)  # Inclusive prefix sum
var shared = broadcast(val)   # Broadcast lane 0 to all lanes
```

### Warp Specialization Benefits

| Aspect | Monolithic | Warp-Specialized |
|--------|------------|------------------|
| Memory latency | Fully exposed | Hidden by overlap |
| Register pressure | Uniform | Optimized per role |
| Warp divergence | High | None (separate paths) |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `warp shuffle returns wrong value` | Wrong lane offset or inactive threads | Verify all 32 threads participate; check lane math |
| `warp reduction incorrect` | Not all lanes have valid data | Initialize all lanes; use lane_group reductions for partial warps |
| `divergent warp execution` | Different threads take different branches | Restructure for uniform control flow; use predication |
| `shuffle across warp boundary` | Trying to shuffle beyond warp | Shuffle only works within WARP_SIZE-thread warp |
| `Unsupported return type` with vote | Wrong ret_type for vote on NVIDIA | NVIDIA only supports `DType.uint32` for `vote` |
| `UnsupportedTarget` error | Using vote/shuffle on unsupported target | `vote` only works on NVIDIA and AMD GPUs |

---

## Import Cheat Sheet

```mojo
# nocompile

# Thread/block IDs and constants (re-exported at gpu level)
from gpu import lane_id, warp_id, thread_idx, block_idx, block_dim, WARP_SIZE

# Shuffle operations
from gpu.primitives.warp import shuffle_idx, shuffle_up, shuffle_down, shuffle_xor

# Warp-wide reductions
from gpu.primitives.warp import sum, max, min, prefix_sum, broadcast

# Lane group (partial warp) reductions
from gpu.primitives.warp import (
    lane_group_sum,
    lane_group_max,
    lane_group_min,
    lane_group_sum_and_broadcast,
    lane_group_max_and_broadcast,
)

# Generic reduction
from gpu.primitives.warp import reduce, lane_group_reduce

# Vote (ballot) operation
from gpu.primitives.warp import vote

# Synchronization (barrier available from both gpu and gpu.sync)
from gpu import barrier
from gpu.sync import syncwarp  # Also available as: from gpu import syncwarp
```

---

## Related Patterns

- [`gpu-fundamentals.md`](gpu-fundamentals.md) — Thread hierarchy and basics
- [`gpu-synchronization.md`](gpu-synchronization.md) — Barrier and sync patterns
- [`gpu-kernels.md`](gpu-kernels.md) — Producer-consumer pipelines
- [`gpu-tensor-cores.md`](gpu-tensor-cores.md) — WGMMA warp group patterns

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **`gpu.primitives.warp`** | Stable | `vote`, `shuffle_idx`, `shuffle_down`, `broadcast` |
| **Warp reductions** | Stable | `sum`, `max`, `min`, `prefix_sum` |
| **SIMD warp operations** | Stable | Full SIMD width support |
| **`gpu` module** | Stable | `lane_id()`, `warp_id()`, `WARP_SIZE` (imported from `gpu`) |
| **`gpu.primitives.block`** | Stable | Block-level `sum`, `max`, `min`, `broadcast`, `prefix_sum` |

---

## References

- [Mojo GPU Block and Warp](https://docs.modular.com/mojo/manual/gpu/block-and-warp/)
- [MAX Kernels](https://github.com/modular/modular/tree/main/max/kernels)
