---
title: GPU Kernel Optimization
description: Kernel fusion, producer-consumer pipelines, and double-buffering patterns
impact: HIGH
category: gpu
tags: [gpu, performance, kernel-fusion, pipeline, double-buffering]
error_patterns:
  - "kernel launch failed"
  - "grid size"
  - "block size"
  - "occupancy"
  - "register spill"
  - "no matching method in call to 'enqueue_function'"
  - "@compiler.register"
scenarios:
  - "Fuse multiple GPU operations"
  - "Implement producer-consumer pipeline"
  - "Use double-buffering for latency hiding"
  - "Optimize kernel launch configuration"
  - "Create custom GPU operation"
  - "Package Mojo kernel as mojopkg"
  - "Chain multiple GPU kernels in pipeline"
  - "Integrate custom op with Python graph"
  - "Create element-wise GPU kernel from template"
  - "Implement RMSNorm kernel"
  - "Choose block size for GPU kernel"
consolidates:
  - gpu-kernel-fusion.md
  - gpu-producer-consumer-pipeline.md
  - gpu-double-buffering.md
---

# GPU Kernel Optimization

**Category:** GPU | **Impact:** HIGH

Optimize GPU kernels by fusing operations, implementing producer-consumer pipelines, and using double-buffering to hide memory latency. These patterns provide 2-3x throughput improvements.

## API Availability

> **Note:** All APIs listed below are available in the Mojo nightly toolchain (v26.2+). Code examples are documentation snippets — adapt import paths and parameters for your use case.

| API | Import Path | Notes |
|-----|-------------|-------|
| `DeviceContext`, `Dim` | `from gpu.host import DeviceContext, Dim` | Kernel launch |
| `barrier` | `from gpu import barrier` | Block synchronization |
| `stack_allocation` | `from memory import stack_allocation` | Shared memory allocation |
| `AddressSpace` | `from gpu.memory import AddressSpace` | Address space enum (SHARED, GENERIC, etc.) |
| `LayoutTensor`, `Layout` | `from layout import LayoutTensor, Layout` | Type-safe tensor with compile-time layout |
| `SharedMemBarrier`, `MbarPtr` | `from gpu.sync import SharedMemBarrier, MbarPtr` | Pipeline barriers |
| `elect_one_sync`, `block_rank_in_cluster` | `from gpu.primitives.cluster import ...` | Cluster primitives (SM90+) |
| `cluster_sync`, `cluster_arrive`, `cluster_wait` | `from gpu.primitives.cluster import ...` | Cluster synchronization (SM90+) |
| `fence_mbarrier_init` | `from gpu.memory import fence_mbarrier_init` | Cluster barrier fence |

---

## Core Concepts

### Kernel Fusion

Fuse sequential GPU operations into single kernels to eliminate memory traffic and kernel launch overhead (~5-20μs per launch). For bandwidth-bound operations, fusion provides 2-3x speedups.

**Fused Normalization + Linear Projection (Metal Shading Language — for Apple GPU via FFI):**

```metal
kernel void fused_norm_project(
    device const float *input [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    device const float *shift [[buffer(2)]],
    device const float *weight [[buffer(3)]],
    device float *output [[buffer(4)]],
    // ...
) {
    // 1. Compute RMS normalization factor
    float rms_inv = rsqrt(compute_mean_sq(input) + eps);

    // 2. Fused: normalize -> modulate -> project
    float out_acc = 0.0f;
    for (int d = 0; d < dim; d++) {
        float norm_val = input[d] * rms_inv;
        float modulated = (1.0f + scale[d]) * norm_val + shift[d];
        out_acc += modulated * weight[d * out_dim + out_idx];
    }
    output[out_idx] = out_acc;
}
```

**Fused Gated Activation (GLU variants — Metal Shading Language):**

```metal
kernel void fused_gated_activation(
    device const float *input [[buffer(0)]],
    device const float *gate_weight [[buffer(1)]],
    device const float *up_weight [[buffer(2)]],
    device float *output [[buffer(3)]],
    // ...
) {
    // Compute gate and up projections simultaneously
    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int d = 0; d < input_dim; d++) {
        float x = input[d];
        gate_acc += x * gate_weight[d * hidden_dim + out_dim];
        up_acc += x * up_weight[d * hidden_dim + out_dim];
    }

    // SiLU activation: x / (1 + exp(-x))
    float activated_gate = gate_acc / (1.0f + exp(-gate_acc));
    output[out_dim] = activated_gate * up_acc;
}
```

---

## Producer-Consumer Pipeline

GPU kernels achieve maximum throughput by overlapping memory loads with compute. The producer warp(s) load data while consumer warp(s) compute on previously loaded data.


**Pipeline Structure:**

```mojo
# nocompile

from gpu.sync import SharedMemBarrier, MbarPtr

struct ProducerConsumerPipeline[num_stages: Int]:
    """Multi-stage pipeline with full/empty barriers."""
    var full: MbarPtr      # Producer signals, consumer waits
    var empty: MbarPtr     # Consumer signals, producer waits
    var stage: Int
    var phase: Int

    fn produce(mut self) -> PipelineStage:
        """Wait for empty slot, return stage for loading."""
        self.empty[self.stage].wait(self.phase)
        return PipelineStage(self.stage, self.full[self.stage])

    fn signal_produced(mut self, expected_bytes: Int):
        """Signal that production is complete."""
        self.full[self.stage].expect_bytes(expected_bytes)
        self.stage = (self.stage + 1) % num_stages
        if self.stage == 0:
            self.phase ^= 1

    fn consume(mut self) -> PipelineStage:
        """Wait for full slot, return stage for computing."""
        self.full[self.stage].wait(self.phase)
        return PipelineStage(self.stage, self.empty[self.stage])
```

**Context Manager Pattern:**

```mojo
# nocompile

# Clean context manager API
with pipeline.produce() as stage:
    tma_load(stage.buffer, stage.mbar)
# __exit__ signals production complete

with pipeline.consume() as stage:
    result = mma(stage.buffer)
# __exit__ signals consumption complete
```

---

## Double Buffering

Use two sets of shared memory buffers: one for current computation, one for loading next iteration's data.


**Sequential (Slow):**

```mojo
# nocompile

fn simple_tiled_matmul(...):
    var A_tile = LayoutTensor[..., AddressSpace.SHARED].stack_allocation()
    var B_tile = LayoutTensor[..., AddressSpace.SHARED].stack_allocation()

    for k_iter in range(K // TILE_K):
        # Must wait for load before compute
        copy_dram_to_sram(A_tile, A_global[..., k_iter])
        copy_dram_to_sram(B_tile, B_global[k_iter, ...])
        barrier()
        compute_tile(A_tile, B_tile, acc)
        barrier()
```

**Double Buffered (Fast):**

```mojo
# nocompile

fn double_buffered_matmul(...):
    # Two sets of buffers
    var A_buf_0 = LayoutTensor[..., AddressSpace.SHARED].stack_allocation()
    var A_buf_1 = LayoutTensor[..., AddressSpace.SHARED].stack_allocation()
    var B_buf_0 = LayoutTensor[..., AddressSpace.SHARED].stack_allocation()
    var B_buf_1 = LayoutTensor[..., AddressSpace.SHARED].stack_allocation()

    # Prologue: Load first tile
    copy_dram_to_sram(A_buf_0, A_global[..., 0])
    copy_dram_to_sram(B_buf_0, B_global[0, ...])
    barrier()

    for k_iter in range(num_k_iters):
        var use_buf_0 = (k_iter % 2) == 0
        var A_compute = A_buf_0 if use_buf_0 else A_buf_1
        var B_compute = B_buf_0 if use_buf_0 else B_buf_1
        var A_load = A_buf_1 if use_buf_0 else A_buf_0
        var B_load = B_buf_1 if use_buf_0 else B_buf_0

        # Start loading NEXT tile (async)
        var next_k = k_iter + 1
        if next_k < num_k_iters:
            copy_dram_to_sram_async(A_load, A_global[..., next_k])
            copy_dram_to_sram_async(B_load, B_global[next_k, ...])
            async_copy_commit_group()

        # Compute on CURRENT tile (while next is loading)
        compute_tile(A_compute, B_compute, acc)

        if next_k < num_k_iters:
            async_copy_wait_group[0]()
            barrier()
```

**Multi-Stage Pipeline:**

```mojo
# nocompile

fn multi_stage_pipeline[NUM_STAGES: Int](...):
    var A_stages = InlineArray[LayoutTensor[...], NUM_STAGES]()
    var B_stages = InlineArray[LayoutTensor[...], NUM_STAGES]()

    @parameter
    for s in range(NUM_STAGES):
        A_stages[s] = LayoutTensor[...].stack_allocation()
        B_stages[s] = LayoutTensor[...].stack_allocation()

    # Fill the pipeline (prologue)
    @parameter
    for s in range(min(NUM_STAGES, num_k_iters)):
        copy_dram_to_sram_async(A_stages[s], A_global[..., s])
        async_copy_commit_group()

    # Main loop: drain and refill
    for k_iter in range(num_k_iters):
        var stage = k_iter % NUM_STAGES
        async_copy_wait_group[NUM_STAGES - 1]()
        barrier()
        compute_tile(A_stages[stage], B_stages[stage], acc)

        var future_k = k_iter + NUM_STAGES
        if future_k < num_k_iters:
            copy_dram_to_sram_async(A_stages[stage], A_global[..., future_k])
            async_copy_commit_group()
```

---

## Decision Guide

| Technique | Speedup | Memory Cost | Best For |
|-----------|---------|-------------|----------|
| Kernel Fusion | 15-40% | None | Bandwidth-bound ops |
| Double Buffer | 50-100% | 2x tile size | Matrix multiply |
| 4-Stage Pipeline | 100-200% | 4x tile size | Large GEMM |

### Fusion Candidates

| Fusion | Operations | Typical Speedup |
|--------|-----------|-----------------|
| Norm+Project | 3→1 | 15-30% |
| Gated Activation | 4→1 | 10-20% |
| Norm+Gated | 6→1 | 25-40% |

### Pipeline Stage Count

| Stages | Shared Memory | Latency Hiding |
|--------|--------------|----------------|
| 1 | 1x tile | None |
| 2 | 2x tile | 1 iteration |
| 3 | 3x tile | Good |
| 4+ | 4x+ tile | Excellent |

---

## Quick Reference

- **Fuse when:** Operations always used together, not register-limited
- **Don't fuse when:** Need intermediate values for debugging
- **Double buffer when:** Memory latency is significant, shared memory budget allows
- **Pipeline stages:** 2-4 stages typical, diminishing returns beyond 4

---

## Multi-Block Cluster Programming (SM90+)


SM90 (Hopper) and SM100 (Blackwell) support thread block clusters - groups of blocks that can access each other's shared memory and coordinate via cluster-level primitives. Provides 1.3-1.5x throughput for large problems.

### Cluster Kernel Launch

```mojo
from gpu.host import DeviceContext, Dim

# Kernel with cluster metadata
@__llvm_metadata(`nvvm.cluster_dim`=StaticTuple[Int32, 3](2, 2, 1))
fn cluster_kernel():
    var cluster_rank = block_rank_in_cluster()
    var rank_m = cluster_rank // CLUSTER_N
    var rank_n = cluster_rank % CLUSTER_N
    # ... cluster-aware computation

# Launch with cluster dimensions
fn launch_clustered_kernel(ctx: DeviceContext) raises:
    ctx.enqueue_function_experimental[cluster_kernel](
        grid_dim=(8, 8),
        block_dim=(256),
        cluster_dim=Dim((2, 2, 1)),  # 2x2 cluster = 4 blocks
    )
    ctx.synchronize()
```

### Multicast TMA Loading

Share data across cluster blocks with single TMA load:

```mojo
# nocompile

from gpu.primitives.cluster import elect_one_sync, block_rank_in_cluster

fn multicast_load_pattern():
    var cluster_rank = block_rank_in_cluster()
    var rank_m = cluster_rank // CLUSTER_N

    # Calculate multicast mask for all blocks needing this data
    var dim0_mask = (1 << CLUSTER_N) - 1  # All blocks in N dimension
    var multicast_mask = dim0_mask << (rank_m * CLUSTER_N)

    if elect_one_sync():
        tma_op.async_multicast_load[cta_group](
            dest=smem_tile,
            barrier=barrier,
            coords=coords,
            multicast_mask=multicast_mask.cast[DType.uint16](),
        )
```

### Cluster Synchronization

```mojo
# nocompile

from gpu.primitives.cluster import cluster_sync, cluster_sync_relaxed, cluster_arrive, cluster_wait
from gpu.memory import fence_mbarrier_init

# Full cluster barrier
cluster_sync()

# Decomposed for overlapping work
cluster_arrive()
compute_local_work()  # Independent work while waiting
cluster_wait()

# Critical: Fence after barrier init
if elect_one_sync():
    mbar[0].init()
fence_mbarrier_init()  # Ensures visibility across cluster
cluster_sync()
```

### When Clusters Help vs Hurt

| Pattern | Memory BW | Best For |
|---------|-----------|----------|
| Independent blocks | 1x (redundant loads) | Small problems |
| Cluster multicast | N/cluster (shared) | Large GEMM, repeated tiles |

**Use clusters when:**
- Large matrix multiplications (4K+ dimensions)
- Significant data reuse across blocks
- SM90 (Hopper) or SM100 (Blackwell) GPUs

**Avoid clusters when:**
- Small problems (sync overhead dominates)
- SM80 and earlier (not supported)
- Already compute-bound kernels

---

## Warp-Level Reduction Optimization

**When:** Tree reductions in softmax, normalization, or other kernels with shared memory

**Impact:** HIGH — 8-15% speedup by removing unnecessary barriers

On GPUs, threads within a warp (CUDA) or SIMD group (Apple Metal) execute in lockstep. This means no synchronization is needed for reductions within the last 32 elements.

**Inefficient (barriers at every level):**

```mojo
# nocompile

# BAD: Barrier at every reduction level
var stride = BLOCK_SIZE // 2
while stride > 0:
    barrier()  # UNNECESSARY for stride < 32!
    if tid < stride:
        shared[tid] += shared[tid + stride]
    stride //= 2
```

**Optimized (warp-level reduction):**

```mojo
# nocompile

comptime WARP_SIZE: Int = 32  # Apple Silicon SIMD group size

# Tree reduction down to warp size (needs barriers)
var stride = BLOCK_SIZE // 2
while stride >= WARP_SIZE:
    if tid < stride:
        shared[tid] += shared[tid + stride]
    barrier()
    stride //= 2

# Warp-level reduction - NO barriers needed (SIMD lockstep)
if tid < WARP_SIZE:
    # stride = 16
    if tid < 16:
        shared[tid] += shared[tid + 16]
    # stride = 8
    if tid < 8:
        shared[tid] += shared[tid + 8]
    # stride = 4
    if tid < 4:
        shared[tid] += shared[tid + 4]
    # stride = 2
    if tid < 2:
        shared[tid] += shared[tid + 2]
    # stride = 1
    if tid < 1:
        shared[tid] += shared[tid + 1]

# Single barrier after warp reduction completes
barrier()
var result = shared[0]
```

**Key insight:** Within a warp/SIMD group, all threads execute the same instruction at the same time, so memory operations are naturally ordered without explicit synchronization.

---

## Vectorized Collaborative Loading

**When:** Loading tiles into shared memory for GEMM, attention, or other tiled algorithms

**Impact:** HIGH — 10-20% speedup with 4x fewer memory transactions

Instead of loading elements one at a time, use float4 (128-bit) vectorized loads to reduce memory transactions.

**Inefficient (scalar loads):**

```mojo
# nocompile

# BAD: Scalar element loads
var load_idx = tid
while load_idx < TILE_SIZE:
    var global_idx = tile_start + load_idx
    if global_idx < size:
        shared[load_idx] = global_ptr[global_idx]  # SCALAR
    load_idx += BLOCK_SIZE
```

**Optimized (float4 vectorized loads):**

```mojo
# nocompile

# GOOD: Float4 vectorized loads (4x fewer transactions)
var load_idx = tid * 4  # Each thread handles 4 elements
var total_float4s = TILE_SIZE // 4

while load_idx < TILE_SIZE:
    var global_idx = tile_start + load_idx

    # Check if we can do a full float4 load
    if global_idx + 3 < size:
        # Vectorized 128-bit load
        var vec = load4(global_ptr, global_idx)
        shared[load_idx + 0] = vec[0]
        shared[load_idx + 1] = vec[1]
        shared[load_idx + 2] = vec[2]
        shared[load_idx + 3] = vec[3]
    else:
        # Scalar fallback for edge cases
        @parameter
        for j in range(4):
            if global_idx + j < size:
                shared[load_idx + j] = global_ptr[global_idx + j]
            else:
                shared[load_idx + j] = 0.0

    load_idx += BLOCK_SIZE * 4
```

**Memory bandwidth comparison:**

| Load Pattern | Transactions per 128 elements | Bandwidth Utilization |
|--------------|------------------------------|----------------------|
| Scalar (32-bit) | 128 | 25% |
| Float2 (64-bit) | 64 | 50% |
| Float4 (128-bit) | 32 | 100% |

**Best practices:**
- Ensure tile sizes are multiples of 4 for optimal vectorization
- Handle edge cases with scalar fallback
- Align shared memory allocation to 16 bytes (128 bits)

---

## Custom Op Development

Register custom Mojo GPU operations for use in MAX graphs via `@compiler.register`.

### Registration Pattern

```mojo
# nocompile
from compiler import register
from max.tensor import InputTensor, OutputTensor
from runtime.asyncrt import DeviceContextPtr

@compiler.register("my_custom_op")
struct MyCustomOp:
    @staticmethod
    fn execute[
        # Type parameters from graph
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[rank=rank],
        input: InputTensor[rank=rank],
        ctx: DeviceContextPtr,
    ):
        # Convert to LayoutTensor for GPU work
        var out = output.to_layout_tensor()
        var inp = input.to_layout_tensor()

        # Dispatch to GPU or CPU
        @parameter
        if target == "gpu":
            ctx.enqueue_function[my_gpu_kernel, my_gpu_kernel](
                out, inp,
                grid_dim=grid, block_dim=block,
            )
        elif target == "cpu":
            # CPU fallback
            for i in range(inp.size()):
                out[i] = inp[i]
```

### `enqueue_function` Double-Generic Requirement

In custom op contexts, `enqueue_function` requires the kernel function passed **twice** as generic parameters.

**Don't:**
```mojo
# FAILS in custom op context:
ctx.enqueue_function[my_kernel](args..., grid_dim=grid, block_dim=block)
```

**Do:**
```mojo
# WORKS: Pass kernel function twice
ctx.enqueue_function[my_kernel, my_kernel](args..., grid_dim=grid, block_dim=block)
```

### `to_layout_tensor()` and `rebind`

`OutputTensor` and `InputTensor` from the graph system convert to LayoutTensors via `to_layout_tensor()`. When you need a specific known layout, use `rebind`:

```mojo
# nocompile
var tensor = output.to_layout_tensor()
# If you know the exact layout at compile time:
var typed = rebind[LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin]](tensor)
```

### Python Graph Integration

Register the custom op in Python using `ops.custom`:

```python
from max.graph import Graph, ops, TensorType

graph = Graph("my_model")
x = graph.input(TensorType(DType.float32, 128, 256))
result = ops.custom(
    name="my_custom_op",
    values=[x],
    out_types=[TensorType(DType.float32, 128, 256)],
)
```

### Packaging as mojopkg

Bundle custom ops into a package for reuse:

```bash
# Package the op directory
mojo package op/ -o op.mojopkg

# Use in other projects
from op import MyCustomOp
```

### Multi-Kernel Pipeline

Chain multiple kernels on the same DeviceContext stream for sequential execution without CPU sync:

```mojo
# nocompile
fn attention_pipeline(ctx: DeviceContextPtr, ...):
    # Kernel 1: Compute attention scores
    ctx.enqueue_function[scores_kernel, scores_kernel](
        scores_buf, query, key,
        grid_dim=grid1, block_dim=block1,
    )

    # Kernel 2: Softmax normalization (reuses scores_buf)
    ctx.enqueue_function[softmax_kernel, softmax_kernel](
        weights_buf, scores_buf,
        grid_dim=grid2, block_dim=block2,
    )

    # Kernel 3: Weighted value aggregation
    ctx.enqueue_function[aggregate_kernel, aggregate_kernel](
        output, weights_buf, value,
        grid_dim=grid3, block_dim=block3,
    )
    # All kernels execute in order on the same stream
    # Buffer reuse: scores_buf → weights_buf (same memory, different stage)
```

---

## Kernel Templates

Copy-paste starting points for the most common GPU kernel patterns.

### Element-wise (`algorithm.elementwise`)

Use `algorithm.elementwise` instead of hand-written grid-stride loops. It handles grid/block sizing, striding, and dispatch automatically.

```mojo
# nocompile

from algorithm import elementwise
from gpu.host import DeviceContext
from utils.numerics import get_accum_type

fn launch_silu[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
    ctx: DeviceContext,
) raises:
    """SiLU activation using algorithm.elementwise."""
    comptime accum_type = get_accum_type[dtype]()

    @parameter
    @__copy_capture(output, input)
    fn silu_op[width: Int, rank: Int, alignment: Int = 1](
        offset: IndexList[rank],
    ):
        var i = offset[0]
        var x = input.load[width=width](i).cast[accum_type]()
        var result = x / (1.0 + exp(-x))
        output.store(i, result.cast[dtype]())

    elementwise[silu_op, target="gpu"](size, ctx)
```

**When to use:** Activation functions, element-wise math, type casting, clipping. Any operation where each output element depends on exactly one input element.

> **Prefer `algorithm.elementwise` over manual grid-stride loops.** It selects block/grid dimensions automatically, handles edge cases, and is the idiomatic Mojo pattern for element-wise GPU work.

### Row-wise Reduction (RMSNorm)

One block per row. Threads within a block collaborate on reduction using `block_sum` and `block_broadcast` primitives.

```mojo
# nocompile

from gpu import thread_idx, block_idx, block_dim
from gpu.primitives.block import sum as block_sum, broadcast as block_broadcast
from utils.numerics import get_accum_type

fn rmsnorm_kernel[
    dtype: DType, BLOCK_SIZE: Int, D: Int,
](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    weight: UnsafePointer[Scalar[dtype]],
    eps: Float32,
):
    """One block per row. Threads collaborate on reduction."""
    comptime accum_type = get_accum_type[dtype]()

    var row = block_idx.x
    var tid = thread_idx.x
    var row_offset = row * D

    # Step 1: Accumulate sum of squares
    var sum_sq = Scalar[accum_type](0)
    var i = tid
    while i < D:
        var x = input[row_offset + i].cast[accum_type]()
        sum_sq += x * x
        i += BLOCK_SIZE

    # Step 2: Block-wide reduction (requires block_size parameter)
    var total = block_sum[block_size=BLOCK_SIZE](sum_sq)
    var rms_inv = block_broadcast[block_size=BLOCK_SIZE](
        rsqrt(total / Scalar[accum_type](D) + eps.cast[accum_type]()), 0
    )

    # Step 3: Normalize and scale
    i = tid
    while i < D:
        var x = input[row_offset + i].cast[accum_type]()
        var w = weight[i].cast[accum_type]()
        output[row_offset + i] = (x * rms_inv * w).cast[dtype]()
        i += BLOCK_SIZE
```

**When to use:** RMSNorm, LayerNorm, softmax, or any operation that reduces along the last dimension. Launch with `grid_dim=(num_rows,)` and `block_dim=(BLOCK_SIZE,)`.

### Tiled Matmul

For tiled matrix multiplication kernels, refer to the specialized guides:

- [`gpu-tensor-cores.md`](gpu-tensor-cores.md) — WGMMA-based matmul (SM90+) with tensor core integration
- [`gpu-structured-kernels.md`](gpu-structured-kernels.md) — ScatterGather/TileOp abstraction for composable tiled kernels
- The [Double Buffering](#double-buffering) section earlier in this file for overlap techniques

---

## Block Size Selection

| Operation Type | Recommended | Reasoning |
|---|---|---|
| Element-wise (1D) | 256 | Good occupancy, coalesced access |
| Row reduction | 128 or 256 | Match row width, single block per row |
| 2D tiled (matmul) | (16,16) or (32,32) | Square tiles maximize reuse |
| Warp-native | 32 (NVIDIA) / 64 (AMD) | Match hardware warp/wavefront |
| Register-heavy | 128 | Lower thread count reduces register pressure |

**Rules of thumb:**
- Start with 256 for most kernels — it balances occupancy and register usage
- If the kernel uses many registers (>64 per thread), drop to 128
- For reductions, match block size to the reduction dimension when possible
- Never exceed 1024 threads per block (hardware limit)

---

## Occupancy Quick Guide

**Occupancy** measures how many threads run concurrently on a Streaming Multiprocessor (SM) relative to the maximum. Higher occupancy hides memory latency better.

**Register budget (NVIDIA):** 65,536 registers per SM.

| Regs/Thread | Block Size | Regs/Block | Blocks/SM | Occupancy |
|---|---|---|---|---|
| 32 | 256 | 8,192 | 8 | Excellent |
| 64 | 256 | 16,384 | 4 | Good |
| 128 | 256 | 32,768 | 2 | Acceptable |
| 128 | 128 | 16,384 | 4 | Good (reduced block) |
| 256 | 256 | 65,536 | 1 | Poor |

**Rules of thumb:**
- Aim for >=2 blocks per SM — this allows the scheduler to hide latency by switching between blocks
- Too many registers per thread -> reduce tile size or spill to shared memory
- Too much shared memory per block -> fewer blocks per SM -> lower occupancy
- Use `--ptxas-options=-v` (NVIDIA) to see register and shared memory usage per kernel
- When in doubt, profile: theoretical occupancy != actual performance

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `kernel launch failed` | Invalid grid/block dimensions | Ensure block size ≤1024 threads; grid size ≤2^31-1 |
| `register spill to local memory` | Too many registers per thread | Reduce tile size; split kernel; use shared memory instead |
| `occupancy too low` | Insufficient parallelism | Increase block count; reduce shared memory per block |
| `race condition in fused kernel` | Missing synchronization between stages | Add `barrier()` between producer and consumer stages |
| `cluster_sync timeout` | Mismatched cluster barriers | Ensure all blocks in cluster reach sync point |
| `pipeline stall` | Unbalanced producer/consumer | Use double-buffering; overlap compute with memory |
| `kernel changes not reflected` | Stale compilation cache | Purge caches: `rm -rf ~/.modular/.max_cache ~/.modular/.mojo_cache ~/.modular/.mogg_cache` |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **Constants** | `alias` or `comptime` | Both work in v26.1+ |
| **Cluster metadata** | `@__llvm_metadata(...)` | Stable |
| **Stack allocation** | `stack_allocation[...]` | Stable |

**Example (v26.1+):**
```mojo
from gpu.host import DeviceContext, Dim
from gpu import barrier

comptime BLOCK_SIZE = 256
comptime NUM_STAGES = 4

fn kernel():
    var shared = stack_allocation[BLOCK_SIZE, Float32, address_space=AddressSpace.SHARED]()
    # ... kernel logic
    barrier()
```

**Notes:**
- Both `alias` and `comptime` work for compile-time constants in v26.1+
- Kernel fusion patterns are stable across versions
- Double-buffering and multi-stage pipeline patterns are stable
- Cluster programming (SM90+) uses `@__llvm_metadata` which is stable
- `LayoutTensor` and pipeline barriers are available in v26.1+ nightly

---

## Related Patterns

- [`gpu-fundamentals.md`](gpu-fundamentals.md) — Thread hierarchy and memory model
- [`gpu-synchronization.md`](gpu-synchronization.md) — Barriers and async operations
- [`gpu-memory-access.md`](gpu-memory-access.md) — TMA and shared memory patterns
- [`gpu-structured-kernels.md`](gpu-structured-kernels.md) — ScatterGather/RingBuffer/TileOp architecture for complex kernels

---

## References

- [MAX Kernels](https://github.com/modular/modular/tree/main/max/kernels)
