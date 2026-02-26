---
title: Warp Primitives, Synchronization, and Async Operations
description: Warp shuffle operations, reductions, barriers, mbarriers, async transactions, and async copy patterns
impact: HIGH
category: gpu
tags: [gpu, warp, shuffle, reduction, specialization, synchronization, barrier, mbarrier, async, tma, sm90]
error_patterns:
  - "warp divergence"
  - "shuffle error"
  - "invalid lane"
  - "reduction"
  - "vote mask"
  - "block_size required"
  - "shuffle_xor"
  - "barrier deadlock"
  - "race condition"
  - "sync error"
  - "mbarrier"
  - "illegal memory access"
  - "misaligned address"
scenarios:
  - "Implement warp-level reduction"
  - "Use shuffle for fast communication"
  - "Specialize warps for producer/consumer"
  - "Fix warp divergence issue"
  - "Use block collectives for reduction"
  - "Implement butterfly all-reduce"
  - "Normalize values across a block"
  - "Use shuffle_xor for pair communication"
  - "Synchronize threads in block"
  - "Use mbarrier for TMA operations"
  - "Fix race condition in shared memory"
  - "Implement async copy pipeline"
consolidates:
  - gpu-warp-primitives.md
  - gpu-warp-specialization.md
  - gpu-row-reduction.md
  - gpu-block-reduction.md
  - gpu-synchronization.md
  - gpu-mbarrier.md
  - gpu-async-transaction.md
  - gpu-async-copy.md
---
<!-- PATTERN QUICK REF
WHEN: Warp-level programming, thread synchronization, shared memory coordination, async TMA pipelines
KEY_TYPES: shuffle_idx, shuffle_down, shuffle_xor, shuffle_up, vote, broadcast, barrier, syncwarp, mbarrier, SharedMemBarrier, async_copy, cluster_sync
SYNTAX:
  - shuffle_down(val, UInt32(offset)) — tree reduction within warp
  - shuffle_xor(val, UInt32(offset)) — butterfly all-reduce (all lanes get result)
  - vote[DType.uint32](predicate) — warp-wide ballot bitmask
  - barrier() — block-level sync (all threads must reach)
  - syncwarp() — warp-level sync (~1 cycle)
  - mbarrier_init(mbar, count) / mbar.expect_bytes(N) — SM90+ async tracking
  - sum[block_size=N](val) — block collective from gpu.primitives.block
PITFALLS: barrier inside conditional → deadlock; missing expect_bytes → TMA hang; shuffle across warp boundary → wrong values; phase mismatch in mbarrier loops
RELATED: gpu-fundamentals, gpu-block-collectives, gpu-memory-access, gpu-kernels, gpu-tensor-cores
-->

# Warp Primitives, Synchronization, and Async Operations

**Category:** gpu | **Impact:** HIGH

Warp shuffle operations enable direct register-to-register communication (5-10x faster than shared memory). Combined with proper synchronization at warp, block, and cluster levels, plus async pipelines for latency hiding, these primitives are essential for high-performance GPU kernels.

## API Availability

> **Note:** All APIs listed below are available in the Mojo nightly toolchain (v26.2+). Code examples are documentation snippets — adapt import paths and parameters for your use case.

### Warp APIs

| API | Import Path | Notes |
|-----|-------------|-------|
| `lane_id`, `warp_id` | `from gpu import lane_id, warp_id` | Re-exported from `gpu.primitives.id` |
| `WARP_SIZE` | `from gpu import WARP_SIZE` | Re-exported from `gpu.globals` |
| `shuffle_idx`, `shuffle_up`, `shuffle_down`, `shuffle_xor` | `from gpu.primitives.warp import ...` | Warp shuffle operations |
| `vote[ret_type](val: Bool)` | `from gpu.primitives.warp import vote` | Warp-wide bitmask from boolean votes |
| `sum`, `max`, `min`, `prefix_sum` | `from gpu.primitives.warp import ...` | Warp-wide reduction builtins |
| `broadcast` | `from gpu.primitives.warp import broadcast` | Broadcast lane 0 value to all lanes |
| `lane_group_sum`, `lane_group_max`, `lane_group_min` | `from gpu.primitives.warp import ...` | Lane-group reductions (partial warp) |
| `lane_group_reduce`, `reduce` | `from gpu.primitives.warp import ...` | Generic reduction with custom functions |

### Synchronization APIs

| API | Import Path | Notes |
|-----|-------------|-------|
| `barrier` | `from gpu import barrier` (or `from gpu.sync import barrier`) | Block-level barrier (both paths work) |
| `syncwarp` | `from gpu.sync import syncwarp` (or `from gpu import syncwarp`) | Warp-level synchronization |
| `mbarrier_init`, `mbarrier_arrive`, `mbarrier_try_wait_parity_shared` | `from gpu.sync import ...` | SM90+ mbarrier primitives |
| `SharedMemBarrier` | `from gpu.sync import SharedMemBarrier` (or `from layout.tma_async import SharedMemBarrier`) | Shared memory barrier wrapper |
| `fence_mbarrier_init`, `cluster_sync`, `cluster_sync_relaxed` | `from gpu.sync import ...` | SM90+ cluster operations |

### Memory & Async APIs

| API | Import Path | Notes |
|-----|-------------|-------|
| `stack_allocation` | `from memory import stack_allocation` | Shared memory allocation |
| `AddressSpace` | `from gpu.memory import AddressSpace` | Address space enum (SHARED, GENERIC, etc.) |
| `LayoutTensor`, `Layout` | `from layout import LayoutTensor, Layout` | Type-safe tensor with compile-time layout |
| `async_copy`, `async_copy_commit_group`, `async_copy_wait_group` | `from gpu.memory import ...` | Async copy primitives |
| `elect_one_sync` | `from gpu.primitives.cluster import elect_one_sync` | Cluster-level (NOT warp), SM90+ |

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

### Synchronization Levels

| Level | Function | Scope | Latency | Use Case |
|-------|----------|-------|---------|----------|
| Warp | `syncwarp()` | 32/64 threads | ~1 cycle | Warp-level algorithms |
| Block | `barrier()` | All block threads | ~20-50 cycles | Shared memory access |
| Named | `named_barrier[N]()` | Subset of threads | ~20-50 cycles | Partial sync (SM90+) |
| Cluster | `cluster_sync()` | All cluster blocks | ~100-200 cycles | Cross-block cooperation |
| Device | `ctx.synchronize()` | All GPU work | ~microseconds | Host-device sync |

---

## Warp-Level Primitives

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

The `vote` function creates a warp-wide bitmask from per-thread boolean predicates (equivalent to CUDA's `__ballot_sync`):

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

## Warp Synchronization

### Warp Reduction (Shuffle)

**When:** Reducing values within a warp (sum, max, min)

**Do:**

```mojo

from gpu.primitives.warp import shuffle_down

fn warp_sum_fast(val: Scalar[DType.float32]) -> Scalar[DType.float32]:
    """Fast: Uses register shuffles for warp communication."""
    comptime LOG2_WARP_SIZE = log2_floor(WARP_SIZE)

    var result = val
    # Tree reduction using shuffle_down
    comptime for i in reversed(range(LOG2_WARP_SIZE)):
        result += shuffle_down(result, UInt32(1 << i))

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

### Butterfly Sum (All-Reduce)

**When:** All lanes need the final result

```mojo

from gpu.primitives.warp import shuffle_xor

fn butterfly_sum(val: Scalar[DType.float32]) -> Scalar[DType.float32]:
    """Butterfly reduction - ALL lanes get the total sum."""
    comptime LOG2_WARP_SIZE = log2_floor(WARP_SIZE)

    var result = val
    comptime for i in range(LOG2_WARP_SIZE):
        result += shuffle_xor(result, UInt32(1 << LOG2_WARP_SIZE))

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

---

## Block-Level Synchronization

### Block Barrier Pattern

**When:** Threads need to coordinate shared memory access

**Public API pattern using `stack_allocation`:**

```mojo
# nocompile

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.memory import AddressSpace
from memory import stack_allocation

fn reduce_kernel_correct(
    data: UnsafePointer[Float32],
    result: UnsafePointer[Float32],
    size: Int
):
    # Allocate shared memory using stack_allocation with AddressSpace.SHARED
    var shared = stack_allocation[256, Float32, address_space=AddressSpace.SHARED]()

    var tid = thread_idx.x
    var gid = block_idx.x * block_dim.x + tid

    # Load to shared memory
    if gid < size:
        shared[tid] = data[gid]
    else:
        shared[tid] = 0.0

    # CRITICAL: Wait for ALL threads to complete their writes
    barrier()

    # Tree reduction pattern
    var stride = block_dim.x // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        barrier()  # Sync after each reduction step
        stride //= 2

    # Thread 0 writes final result
    if tid == 0:
        result[block_idx.x] = shared[0]
```

**LayoutTensor variant** — use `LayoutTensor[..., address_space=AddressSpace.SHARED].stack_allocation()` for type-safe shared memory with the same barrier pattern.

### Avoiding Deadlocks

**When:** Any use of barriers in conditional code

**Do:**

```mojo
fn safe_kernel(data: UnsafePointer[Float32], size: Int):
    var tid = block_idx.x * block_dim.x + thread_idx.x

    # ALL threads reach barrier (even those outside bounds)
    barrier()

    # Now safely do conditional work
    if tid < size:
        # ... do work
        pass
```

**Don't:**

```mojo
fn deadlock_kernel(data: UnsafePointer[Float32], size: Int):
    var tid = block_idx.x * block_dim.x + thread_idx.x

    if tid < size:
        # DEADLOCK: Threads outside bounds never arrive!
        barrier()
        # ... do work
```

### Block-Level Reduction

**When:** Reducing across all threads in a block (>32 threads)

> **Note:** Mojo also provides built-in block-level collectives via `gpu.primitives.block`:
>
> ```mojo
>
> from gpu.primitives.block import sum, max, min, broadcast, prefix_sum
> alias TPB = 256  # Must match your kernel's block_dim
> var block_total = sum[block_size=TPB](val)           # Block-wide sum
> var block_max = max[block_size=TPB](val)             # Block-wide maximum
> var shared_val = broadcast[block_size=TPB](val, 0)   # Broadcast from thread 0
> var scan = prefix_sum[block_size=TPB](val)           # Block-wide inclusive scan
> ```
>
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
| Custom reduce fn | `reduce[shuffle_down, my_fn]` | Use manual pattern (above Block-Level Reduction) |

**Performance:** Block collectives produce the same result as the manual warp+shared memory pattern with identical performance characteristics, but require significantly less code.

### Row-Wise Reduction (Softmax, LayerNorm)

**When:** Each block processes one row for normalization operations. Pattern: one block per row, vectorized grid-stride loop + `block_reduce`. Two-pass for softmax: (1) find max, (2) compute sum of `exp(x - max)`, then normalize.

```mojo
# nocompile

fn softmax_kernel[dtype: DType, BLOCK_SIZE: Int](
    output: LayoutTensor[dtype, ...],
    input: LayoutTensor[dtype, ...],
    row_size: Int,
):
    var row_idx = block_idx.x

    # Pass 1: Find max value in row (use row_reduce with max_fn)
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

---

## Cross-Block & Async Patterns (SM90+)

### Mbarrier for Async TMA

**When:** Using TMA hardware loads on Hopper/Blackwell

Mbarriers track both thread arrivals AND expected bytes from async memory operations.

```mojo
# nocompile

from gpu.sync import mbarrier_init, mbarrier_arrive, mbarrier_try_wait_parity_shared
from gpu.memory import AddressSpace
from memory import stack_allocation

fn tma_with_mbarrier():
    var mbar = stack_allocation[1, Int64, address_space=AddressSpace.SHARED]()

    if thread_idx.x == 0:
        # Initialize with expected arrival count
        mbarrier_init(mbar, arrival_count=1)
        # Set expected bytes from TMA
        mbar[0].expect_bytes(256)  # Expect 256 bytes

        # Launch TMA operation
        tma_op.async_load(smem_ptr, mbar, coords)

    barrier()  # Sync so all threads see initialized mbarrier

    # All threads wait for TMA to complete
    mbarrier_try_wait_parity_shared(mbar, phase=0, timeout=10_000_000)

    # Now shared memory is ready
    process(smem_ptr)
```

### Async Transaction Barriers

**When:** TMA operations require exact byte counting to prevent hangs

**Do:**

```mojo
# nocompile

from layout.tma_async import SharedMemBarrier
from sys import size_of

fn correct_tma_load[dtype: DType, BM: Int, BK: Int]():
    var mbar_ptr = stack_allocation[
        1, SharedMemBarrier, address_space=AddressSpace.SHARED
    ]()

    # Calculate exact byte count from tile dimensions
    comptime expected_bytes = BM * BK * size_of[dtype]()

    if thread_idx.x == 0:
        mbar_ptr[].init(num_threads=1)
        # Set expected bytes BEFORE issuing TMA
        mbar_ptr[].expect_bytes(expected_bytes)
        # Issue TMA - barrier tracks completion automatically
        tma_op.async_copy(smem_tile, mbar_ptr[], coords)

    barrier()
    mbar_ptr[].wait(phase=0)
```

**Don't:**

```mojo
# nocompile

fn bad_tma_load():
    var mbar = stack_allocation[1, Int64, address_space=AddressSpace.SHARED]()

    if thread_idx.x == 0:
        mbarrier_init(mbar, arrival_count=1)
        # MISSING: mbar.expect_bytes(tile_size_bytes)!
        tma_op.async_copy(smem_tile, mbar, coords)

    # DEADLOCK: Barrier doesn't know how many bytes to expect
    mbar.wait(phase=0)
```

For multi-tile loads, sum all tile byte counts: `total = cta_group * (a_bytes + b_bytes) * k_group_size`.

**Common TMA hang causes:**

- Missing `expect_bytes()` - barrier never completes
- Wrong byte count - mismatch with actual TMA size
- Missing TMA issue after `expect_bytes()`
- Phase mismatch in multi-iteration loops

### Async Copy for Latency Hiding

**When:** Overlapping memory loads with computation

```mojo
# nocompile

from gpu.memory import async_copy, async_copy_commit_group, async_copy_wait_group

# NOTE: On Apple Silicon, async_copy may execute as synchronous copy internally.
# If you encounter issues, the fallback is regular LayoutTensor indexing + barrier():
#   shared[tid] = rebind[Scalar[dtype]](global[global_id])
#   barrier()

fn pipelined_kernel():
    # Stage 0: Start loading first tile
    async_copy[dtype, 16](global_ptr, shared_buf_0)
    async_copy_commit_group()

    for i in range(num_tiles - 1):
        # Start loading next tile (group 1)
        async_copy[dtype, 16](global_ptr + (i + 1) * 16, shared_buf_1)
        async_copy_commit_group()

        # Wait for current tile (group 0) to complete
        async_copy_wait_group[1]()  # Wait until 1 group remaining
        barrier()

        # Compute on current tile while next loads
        compute(shared_buf_0)

        # Swap buffers
        swap(shared_buf_0, shared_buf_1)

    # Process final tile
    async_copy_wait_group[0]()
    barrier()
    compute(shared_buf_0)
```

### Phase-Based Mbarrier Pipeline

**When:** Multi-iteration loops with async operations. Init mbarrier with `arrival_count=BLOCK_SIZE`, then in each iteration: `mbarrier_arrive(mbar)` → `mbarrier_try_wait_parity_shared(mbar, phase, timeout)` → `phase ^= 1`.

### Producer-Consumer with Mbarriers

**When:** Warp-specialized kernels with async loads. Use double-buffered mbarriers (`mbar_full[2]` + `mbar_empty[2]`): producer waits on empty, loads data, signals full; consumer waits on full, processes, signals empty. Toggle `phase ^= 1` after each full buffer cycle.

### Cluster Synchronization (SM90+)

**When:** Multi-block clusters need coordination

```mojo
from gpu.sync import fence_mbarrier_init, cluster_sync, cluster_sync_relaxed

fn clustered_kernel():
    # Initialize barriers across cluster (once at start)
    @parameter
    if CLUSTER_SIZE > 1:
        fence_mbarrier_init()
        cluster_sync_relaxed()  # Ensure all blocks see init

    # ... do work ...

    # Full cluster barrier (strong ordering)
    cluster_sync()

    # Relaxed cluster sync (weaker, faster)
    cluster_sync_relaxed()
```

---

## Warp Specialization

### Basic Warp Specialization

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

### Warp Specialization Benefits

| Aspect | Monolithic | Warp-Specialized |
|--------|------------|------------------|
| Memory latency | Fully exposed | Hidden by overlap |
| Register pressure | Uniform | Optimized per role |
| Warp divergence | High | None (separate paths) |

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
| Shared memory access | Use `barrier()` after writes | - |
| Warp-level sync | Use `syncwarp()` or shuffle ops | - |
| TMA loads (SM90+) | Use mbarrier with expect_bytes | [`gpu-memory-access.md`](gpu-memory-access.md) |
| Latency hiding | Use async_copy pipeline | [`gpu-kernels.md`](gpu-kernels.md) |
| Partial thread sync | Use named_barrier (SM90+) | - |
| Cross-block sync | Use cluster_sync (SM90+) | [`gpu-kernels.md`](gpu-kernels.md) |
| Host-device sync | Use ctx.synchronize() | [`gpu-fundamentals.md`](gpu-fundamentals.md) |

---

## Quick Reference

### Performance Comparison

| Method | Latency | Memory Access |
|--------|---------|---------------|
| Shared memory reduction | ~100 cycles | 32 reads + 32 writes |
| Shuffle reduction | ~10-20 cycles | 0 (register only) |
| Built-in collective | ~10-15 cycles | 0 (optimized) |

### Built-in Warp Collectives

```mojo
# nocompile

from gpu.primitives.warp import sum, max, min, prefix_sum, broadcast

var total = sum(val)          # Warp-wide sum (returns scalar)
var max_val = max(val)        # Warp-wide maximum (returns scalar)
var min_val = min(val)        # Warp-wide minimum (returns scalar)
var prefix = prefix_sum(val)  # Inclusive prefix sum
var shared = broadcast(val)   # Broadcast lane 0 to all lanes
```

### Mbarrier Operations

| Function | Purpose |
|----------|---------|
| `mbarrier_init(mbar, count)` | Initialize with arrival count |
| `mbarrier_arrive(mbar)` | Signal arrival (decrement count) |
| `mbar.expect_bytes(bytes)` | Set expected bytes for async ops |
| `mbarrier_try_wait_parity_shared(mbar, phase, timeout)` | Wait until phase complete (with timeout) |
| `SharedMemBarrier.wait(phase)` | Object-oriented wait |
| `mbarrier_invalidate(mbar)` | Reset for reuse |

### Barrier Rules

1. **All threads must reach the same barrier** - No barriers inside thread-dependent conditionals
2. **Same number of barriers on all paths** - Count barriers carefully
3. **No early returns before barriers** - Would cause deadlock
4. **Phase toggle for reuse** - `phase ^= 1` after each wait

### Debugging Hung Kernels

Common causes:

- Missing `expect_bytes()` before TMA
- Wrong byte count (mismatch with actual TMA size)
- Phase mismatch in multi-iteration loops
- Barrier inside conditional that not all threads reach

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
| `barrier deadlock` | Not all threads reach barrier | Ensure barrier is outside conditionals or all threads take same path |
| `mbarrier phase mismatch` | Wrong phase in multi-stage pipeline | Alternate phases (0,1,0,1) for double-buffering |
| `expect_bytes mismatch` | Wrong byte count for TMA | Calculate exact bytes: `elements * sizeof(dtype)` |
| `missing fence before barrier` | Memory operations not visible | Add `fence_mbarrier_init()` before first arrive |
| `warpgroup_fence missing` | WGMMA results not committed | Call `warpgroup_fence()` after WGMMA, before reading accumulator |
| `async copy not waited` | Using data before copy completes | Call appropriate wait: `cp_async_wait_all()` or `mbarrier_try_wait()` |

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

# SM90+ mbarrier
from gpu.sync import mbarrier_init, mbarrier_arrive, mbarrier_try_wait_parity_shared
from gpu.sync import SharedMemBarrier  # or: from layout.tma_async import SharedMemBarrier
from gpu.sync import fence_mbarrier_init, cluster_sync, cluster_sync_relaxed

# Async copy
from gpu.memory import async_copy, async_copy_commit_group, async_copy_wait_group

# Shared memory
from gpu.memory import AddressSpace
from memory import stack_allocation
```

---

## Version Notes

All APIs in this document are stable in **v26.1+**. Both `alias` and `comptime` work for compile-time constants. Mbarrier, cluster sync, and async copy APIs are available in nightly (v26.2+).

---

## Related Patterns

- [`gpu-fundamentals.md`](gpu-fundamentals.md) — Thread hierarchy and basics
- [`gpu-block-collectives.md`](gpu-block-collectives.md) — Block-level collective operations
- [`gpu-memory-access.md`](gpu-memory-access.md) — TMA loading and prefetch patterns
- [`gpu-kernels.md`](gpu-kernels.md) — Producer-consumer pipelines and double buffering
- [`gpu-tensor-cores.md`](gpu-tensor-cores.md) — WGMMA warp group patterns
- [`gpu-structured-kernels.md`](gpu-structured-kernels.md) — RingBuffer pattern for pipeline coordination

---

## References

- [Mojo GPU Block and Warp](https://docs.modular.com/mojo/manual/gpu/block-and-warp/)
- [MAX Kernels](https://github.com/modular/modular/tree/main/max/kernels)
- [MAX Kernels TMA](https://github.com/modular/modular/blob/main/max/kernels/src/layout/tma_async.mojo)
