---
title: GPU Synchronization and Async Operations
description: Synchronization primitives including barriers, mbarriers, async transactions, and async copy patterns
impact: HIGH
category: gpu
tags: [gpu, synchronization, barrier, mbarrier, async, tma, sm90]
error_patterns:
  - "barrier deadlock"
  - "race condition"
  - "sync error"
  - "mbarrier"
  - "illegal memory access"
  - "misaligned address"
scenarios:
  - "Synchronize threads in block"
  - "Use mbarrier for TMA operations"
  - "Fix race condition in shared memory"
  - "Implement async copy pipeline"
consolidates:
  - gpu-synchronization.md
  - gpu-mbarrier.md
  - gpu-async-transaction.md
  - gpu-async-copy.md
---

# GPU Synchronization and Async Operations

**Category:** gpu | **Impact:** HIGH

Proper synchronization prevents race conditions and enables efficient async pipelines. This pattern covers block/warp barriers, SM90+ mbarriers for TMA operations, async transaction counting, and async memory copy patterns for latency hiding.

## API Availability

> **Note:** All APIs listed below are available in the Mojo nightly toolchain (v26.2+). Code examples are documentation snippets — adapt import paths and parameters for your use case.

| API | Import Path | Notes |
|-----|-------------|-------|
| `barrier` | `from gpu import barrier` (or `from gpu.sync import barrier`) | Block-level barrier (both paths work) |
| `stack_allocation` | `from memory import stack_allocation` | Shared memory allocation |
| `AddressSpace` | `from gpu.memory import AddressSpace` | Address space enum (SHARED, GENERIC, etc.) |
| `LayoutTensor`, `Layout` | `from layout import LayoutTensor, Layout` | Type-safe tensor with compile-time layout |
| `syncwarp` | `from gpu.sync import syncwarp` (or `from gpu import syncwarp`) | Warp-level synchronization |
| `mbarrier_init`, `mbarrier_arrive`, `mbarrier_try_wait_parity_shared` | `from gpu.sync import ...` | SM90+ mbarrier primitives |
| `SharedMemBarrier` | `from gpu.sync import SharedMemBarrier` (or `from layout.tma_async import SharedMemBarrier`) | Shared memory barrier wrapper |
| `async_copy`, `async_copy_commit_group`, `async_copy_wait_group` | `from gpu.memory import ...` | Async copy primitives |
| `fence_mbarrier_init`, `cluster_sync`, `cluster_sync_relaxed` | `from gpu.sync import ...` | SM90+ cluster operations |

---

## Core Concepts

### Synchronization Levels

| Level | Function | Scope | Latency | Use Case |
|-------|----------|-------|---------|----------|
| Warp | `syncwarp()` | 32/64 threads | ~1 cycle | Warp-level algorithms |
| Block | `barrier()` | All block threads | ~20-50 cycles | Shared memory access |
| Named | `named_barrier[N]()` | Subset of threads | ~20-50 cycles | Partial sync (SM90+) |
| Cluster | `cluster_sync()` | All cluster blocks | ~100-200 cycles | Cross-block cooperation |
| Device | `ctx.synchronize()` | All GPU work | ~microseconds | Host-device sync |

### Block Barrier Pattern

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

**LayoutTensor pattern:**
```mojo
# nocompile

from gpu import thread_idx, block_idx, block_dim, barrier
from layout import LayoutTensor, Layout
from gpu.memory import AddressSpace

fn reduce_kernel_layout[BLOCK_SIZE: Int](
    data: LayoutTensor[DType.float32, ..., MutAnyOrigin],
    result: LayoutTensor[DType.float32, ..., MutAnyOrigin],
):
    # LayoutTensor with AddressSpace.SHARED for type-safe shared memory
    var shared = LayoutTensor[
        DType.float32,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var tid = thread_idx.x
    var gid = block_idx.x * block_dim.x + tid

    # Type-safe indexing
    shared[tid] = data[gid] if gid < data.shape[0] else 0.0
    barrier()

    # Tree reduction
    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        barrier()
        stride //= 2

    if tid == 0:
        result[block_idx.x] = shared[0]
```

---

## Common Patterns

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

### Mbarrier for Async TMA (SM90+)

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

**When:** Multi-iteration loops with async operations

```mojo
# nocompile

fn phased_pipeline():
    var mbar = stack_allocation[1, Int64, address_space=AddressSpace.SHARED]()

    if thread_idx.x == 0:
        mbarrier_init(mbar, arrival_count=BLOCK_SIZE)
    barrier()

    var phase: UInt32 = 0

    for iter in range(NUM_ITERATIONS):
        # Do phase work
        compute_phase(iter)

        # Arrive and wait for this phase
        mbarrier_arrive(mbar)
        mbarrier_try_wait_parity_shared(mbar, phase=phase, timeout=10_000_000)

        # Toggle phase for next iteration
        phase ^= 1

        # All threads synchronized at this phase
```

### Producer-Consumer with Mbarriers

**When:** Warp-specialized kernels with async loads

```mojo
# nocompile

fn producer_consumer_kernel():
    # Two mbarriers for double buffering
    var mbar_full = stack_allocation[2, Int64, address_space=AddressSpace.SHARED]()
    var mbar_empty = stack_allocation[2, Int64, address_space=AddressSpace.SHARED]()

    # Initialize barriers
    if thread_idx.x == 0:
        mbarrier_init(mbar_full[0], arrival_count=1)
        mbarrier_init(mbar_full[1], arrival_count=1)
        mbarrier_init(mbar_empty[0], arrival_count=1)
        mbarrier_init(mbar_empty[1], arrival_count=1)

        # Mark empty buffers as available
        mbarrier_arrive(mbar_empty[0])
        mbarrier_arrive(mbar_empty[1])
    barrier()

    var phase: UInt32 = 0

    for i in range(num_iterations):
        var buf_idx = i % 2

        if is_producer_warp():
            # Wait for buffer to be empty
            mbarrier_try_wait_parity_shared(mbar_empty[buf_idx], phase=phase, timeout=10_000_000)
            # Load data into buffer
            load_data(buffer[buf_idx])
            # Signal buffer is full
            mbarrier_arrive(mbar_full[buf_idx])

        if is_consumer_warp():
            # Wait for buffer to be full
            mbarrier_try_wait_parity_shared(mbar_full[buf_idx], phase=phase, timeout=10_000_000)
            # Process data
            process_data(buffer[buf_idx])
            # Signal buffer is empty
            mbarrier_arrive(mbar_empty[buf_idx])

        if buf_idx == 1:
            phase ^= 1  # Toggle phase after full cycle
```

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

### TMA Transaction Barriers (SM90+)

**When:** Async TMA copies need byte tracking


Transaction barriers track both thread arrivals AND expected bytes from TMA operations. The barrier completes only when all bytes have been transferred.

```mojo
# nocompile

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

    barrier()  # Sync so all threads see initialized mbarrier

    # Wait for TMA completion - releases when all bytes arrive
    mbar_ptr[].wait(phase=0)
```

**Multi-tile transaction counting:**
```mojo
# nocompile

# Calculate total bytes for multiple tiles in one stage
comptime a_expected_bytes = a_smem_layout.size() * size_of[a_type]()
comptime b_expected_bytes = b_smem_layout.size() * size_of[b_type]()
comptime total_expected_bytes = cta_group * (a_expected_bytes + b_expected_bytes) * k_group_size

if elect_one_sync():
    if elect_one_cta:
        tma_mbar[stage].expect_bytes(total_expected_bytes)

    for j in range(k_group_size):
        a_tma_op.async_multicast_load(a_tile[j], tma_mbar[stage], a_coords, mask)
        b_tma_op.async_multicast_load(b_tile[j], tma_mbar[stage], b_coords, mask)
```

**Common TMA hang causes:**
- Missing `expect_bytes()` - barrier never completes
- Wrong byte count - mismatch with actual TMA size
- Missing TMA issue after `expect_bytes()`
- Phase mismatch in multi-iteration loops

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Shared memory access | Use `barrier()` after writes | - |
| Warp-level reduction | Use `syncwarp()` or shuffle ops | [`gpu-warp.md`](gpu-warp.md) |
| TMA loads (SM90+) | Use mbarrier with expect_bytes | [`gpu-memory-access.md`](gpu-memory-access.md) |
| Latency hiding | Use async_copy pipeline | [`gpu-kernels.md`](gpu-kernels.md) |
| Partial thread sync | Use named_barrier (SM90+) | - |
| Cross-block sync | Use cluster_sync (SM90+) | [`gpu-kernels.md`](gpu-kernels.md) |
| Host-device sync | Use ctx.synchronize() | [`gpu-fundamentals.md`](gpu-fundamentals.md) |

---

## Quick Reference

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
| `barrier deadlock` | Not all threads reach barrier | Ensure barrier is outside conditionals or all threads take same path |
| `mbarrier phase mismatch` | Wrong phase in multi-stage pipeline | Alternate phases (0,1,0,1) for double-buffering |
| `expect_bytes mismatch` | Wrong byte count for TMA | Calculate exact bytes: `elements * sizeof(dtype)` |
| `missing fence before barrier` | Memory operations not visible | Add `fence_mbarrier_init()` before first arrive |
| `warpgroup_fence missing` | WGMMA results not committed | Call `warpgroup_fence()` after WGMMA, before reading accumulator |
| `async copy not waited` | Using data before copy completes | Call appropriate wait: `cp_async_wait_all()` or `mbarrier_try_wait()` |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **Block barrier** | `barrier()` | Stable |
| **Shared memory** | `stack_allocation[N, T, address_space=AddressSpace.SHARED]()` | Stable |
| **Compile-time constants** | `alias` or `comptime` | Both work in v26.1+ |
| **Mbarrier APIs** | Available (v26.1+) | `from gpu.sync import mbarrier_init, ...` |
| **SharedMemBarrier** | Available (v26.1+) | `from gpu.sync import SharedMemBarrier` |
| **Cluster sync APIs** | Available (v26.1+) | `from gpu.sync import cluster_sync, ...` |
| **Async copy APIs** | Available (v26.1+) | `from gpu.memory import async_copy, ...` |

**Example (v26.1+):**
```mojo
from gpu import barrier
from gpu.memory import AddressSpace
from memory import stack_allocation

fn sync_example():
    # Use stack_allocation with AddressSpace.SHARED for shared memory
    var shared = stack_allocation[256, Float32, address_space=AddressSpace.SHARED]()
    # ... load data ...
    barrier()
    # ... use data ...

    # Compile-time constant (both alias and comptime work)
    comptime TILE_BYTES = 256 * 4  # size_of[Float32]
```

**Notes:**
- Basic synchronization APIs (`barrier()`) are stable
- Both `alias` and `comptime` work for compile-time constants in v26.1+
- Use `stack_allocation` with `AddressSpace.SHARED` for shared memory allocation
- Use `LayoutTensor[..., address_space=AddressSpace.SHARED]` for type-safe access
- Mbarrier and cluster sync APIs are available in v26.1+ nightly

---

## Related Patterns

- [`gpu-fundamentals.md`](gpu-fundamentals.md) — Basic GPU programming concepts
- [`gpu-memory-access.md`](gpu-memory-access.md) — TMA loading and prefetch patterns
- [`gpu-kernels.md`](gpu-kernels.md) — Producer-consumer pipelines and double buffering
- [`gpu-warp.md`](gpu-warp.md) — Warp-level synchronization with shuffles
- [`gpu-structured-kernels.md`](gpu-structured-kernels.md) — RingBuffer pattern for pipeline coordination

---

## References

- [Mojo GPU Block and Warp](https://docs.modular.com/mojo/manual/gpu/block-and-warp)
- [MAX Kernels TMA](https://github.com/modular/modular/blob/main/max/kernels/src/layout/tma_async.mojo)
