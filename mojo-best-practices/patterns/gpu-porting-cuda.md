---
title: "CUDA to Mojo Porting Guide"
description: Side-by-side CUDA→Mojo porting guide with complete examples from vector add to tensor core GEMM
impact: HIGH
category: gpu
tags: [cuda, porting, migration, gpu, kernel, tutorial]
error_patterns:
  - "port CUDA kernel"
  - "convert CUDA to Mojo"
  - "CUDA equivalent"
  - "migrate from CUDA"
  - "rewrite kernel in Mojo"
scenarios:
  - "Port a CUDA kernel to Mojo"
  - "Find the Mojo equivalent of a CUDA API"
  - "Write a GPU kernel in Mojo"
  - "Rewrite CUDA shared memory pattern in Mojo"
  - "Convert CUDA tensor core code to Mojo"
---

# CUDA to Mojo Porting Guide

**Category:** GPU | **Impact:** HIGH

Complete guide for porting CUDA kernels to Mojo with side-by-side examples, from basic vector operations to advanced tensor core GEMM with software pipelining.

> **Version:** These patterns target Mojo nightly (v26.2+). All APIs shown are available in the public nightly toolchain.

---

## Quick Reference: CUDA → Mojo Mapping

### Thread Hierarchy

| CUDA | Mojo | Import |
|------|------|--------|
| `threadIdx.x/y/z` | `thread_idx.x/y/z` | `from gpu import thread_idx` |
| `blockIdx.x/y/z` | `block_idx.x/y/z` | `from gpu import block_idx` |
| `blockDim.x/y/z` | `block_dim.x/y/z` | `from gpu import block_dim` |
| `gridDim.x/y/z` | `grid_dim.x/y/z` | `from gpu import grid_dim` |
| `blockIdx.x * blockDim.x + threadIdx.x` | `global_idx.x` | `from gpu import global_idx` |
| `threadIdx.x % 32` | `lane_id()` | `from gpu import lane_id` |
| `threadIdx.x / 32` | `warp_id()` | `from gpu import warp_id` |

### Memory

| CUDA | Mojo | Import |
|------|------|--------|
| `__shared__ float smem[N]` | `stack_allocation[N, DType.float32, address_space=AddressSpace.SHARED]()` | `from memory import stack_allocation` (preferred -- works on all GPUs) |
| `extern __shared__ float smem[]` | `external_memory[Float32, address_space=AddressSpace.SHARED]()` | `from gpu.memory import external_memory` (**NVIDIA only** -- not supported on Apple GPU/Metal) |
| `cudaMalloc(&ptr, size)` | `ctx.enqueue_create_buffer[dtype](count)` | `from gpu.host import DeviceContext` |
| `cudaMemcpy(dst, src, size, ...)` | `ctx.enqueue_copy(dst, src)` | `from gpu.host import DeviceContext` |
| `cudaMemcpy(h, d, ..., D2H)` | `with dev_buf.map_to_host() as host_buf:` (preferred) | RAII context manager, auto-cleanup |
| `cudaFree(ptr)` | Automatic (RAII) or `buffer.free()` | — |

### Synchronization

| CUDA | Mojo | Import |
|------|------|--------|
| `__syncthreads()` | `barrier()` | `from gpu.sync import barrier` |
| `__syncwarp(mask)` | `syncwarp()` | `from gpu.sync import syncwarp` |
| `cudaDeviceSynchronize()` | `ctx.synchronize()` | `from gpu.host import DeviceContext` |

### Kernel Launch

| CUDA | Mojo |
|------|------|
| `kernel<<<grid, block>>>(args)` | `ctx.enqueue_function[kernel, kernel](args, grid_dim=grid, block_dim=block)` |
| `kernel<<<grid, block, smem>>>(args)` | `ctx.enqueue_function[kernel, kernel](args, grid_dim=grid, block_dim=block, shared_mem_bytes=smem)` |

### Data Types

| CUDA | Mojo |
|------|------|
| `float` / `float*` | `Float32` / `UnsafePointer[Float32]` |
| `half` / `__half` | `Float16` |
| `__nv_bfloat16` | `BFloat16` |
| `int` | `Int` or `Int32` |
| `unsigned int` | `UInt` or `UInt32` |
| `cuComplex` / manual `real`/`imag` | `ComplexScalar[float_dtype]` | `from complex import ComplexScalar, ComplexSIMD` |

---

## Level 1: Hello World — Vector Add

The simplest possible GPU kernel, porting the classic CUDA vector add.

### CUDA

```cuda
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Host code
int main() {
    int n = 1024;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    vector_add<<<n/256, 256>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
```

### Mojo

```mojo
# nocompile
from gpu import global_idx
from gpu.host import DeviceContext
from math import ceildiv
from memory import UnsafePointer, alloc

# GPU kernel — equivalent to __global__ void vector_add(...)
fn vector_add(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    c: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    var i = global_idx.x
    if i < UInt(n):
        c[i] = a[i] + b[i]

# Host code
def main():
    comptime n = 1024
    comptime block_dim = 256

    with DeviceContext() as ctx:
        # Allocate device buffers
        var d_a = ctx.enqueue_create_buffer[DType.float32](n)
        var d_b = ctx.enqueue_create_buffer[DType.float32](n)
        var d_c = ctx.enqueue_create_buffer[DType.float32](n)

        # Initialize host data and copy to device
        from memory import alloc
        var h_a = alloc[Float32](n)
        var h_b = alloc[Float32](n)
        var h_c = alloc[Float32](n)
        for i in range(n):
            h_a[i] = Float32(i)
            h_b[i] = Float32(2.0)

        ctx.enqueue_copy(d_a, h_a)
        ctx.enqueue_copy(d_b, h_b)

        # Launch kernel: <<<grid_dim, block_dim>>>
        ctx.enqueue_function[vector_add, vector_add](
            d_a, d_b, d_c, n,
            grid_dim=(ceildiv(n, block_dim),),
            block_dim=(block_dim,),
        )

        # Copy results back
        ctx.enqueue_copy(h_c, d_c)
        ctx.synchronize()

        # Verify
        for i in range(10):
            print("c[", i, "] =", h_c[i])  # Expected: i + 2.0

        h_a.free()
        h_b.free()
        h_c.free()
```

### Key Differences

| Aspect | CUDA | Mojo |
|--------|------|------|
| Kernel marker | `__global__` | Just a regular `fn` — passed to `enqueue_function` |
| Global index | `blockIdx.x * blockDim.x + threadIdx.x` | `global_idx.x` (built-in) |
| Memory alloc | `cudaMalloc` + error checking | `ctx.enqueue_create_buffer[dtype](count)` |
| Memory copy | `cudaMemcpy(dst, src, bytes, direction)` | `ctx.enqueue_copy(dst, src)` — direction inferred |
| Kernel launch | `<<<grid, block>>>` | Named parameters: `grid_dim=..., block_dim=...` |
| Type safety | Manual `sizeof(float)` | Compile-time `DType.float32` |

### Device-to-Host Transfer: `map_to_host()` (Preferred)

The `map_to_host()` context manager provides safe RAII device-to-host transfer — no manual `alloc`/`free` required:

```mojo
# nocompile
# Instead of: var h_output = alloc[Float32](n); ctx.enqueue_copy(h_output, d_buf); h_output.free()
# Use:
ctx.synchronize()
with d_buf.map_to_host() as host_buf:
    var host_tensor = LayoutTensor[DType.float32, layout](host_buf)
    # ... use host_tensor — automatic cleanup on context exit
```

Use `enqueue_copy` when you need the raw pointer for further processing outside the `with` block.

> **Production note:** For simple element-wise kernels like `vector_add`, prefer `algorithm.elementwise` over manual `enqueue_function` — it handles grid/block sizing automatically. The manual form above is shown to illustrate the CUDA-to-Mojo mapping.

---

## Level 2: Shared Memory — Stencil / Neighbor Sum

### CUDA

```cuda
__global__ void neighbor_sum(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load to shared memory
    if (gid < n) smem[tid] = input[gid];
    else         smem[tid] = 0.0f;
    __syncthreads();

    // Compute neighbor sum
    float result = smem[tid];
    if (tid > 0)              result += smem[tid - 1];
    if (tid < blockDim.x - 1) result += smem[tid + 1];

    if (gid < n) output[gid] = result;
}
```

### Mojo

```mojo
from gpu import thread_idx, block_idx, block_dim, global_idx
from gpu.host import DeviceContext, FuncAttribute
from gpu.memory import external_memory
from gpu.sync import barrier
from math import ceildiv
from memory import UnsafePointer

fn neighbor_sum(input: UnsafePointer[Float32, MutAnyOrigin], output: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    # Dynamic shared memory (equivalent to extern __shared__)
    var smem = external_memory[
        Float32, address_space=AddressSpace.SHARED, alignment=4
    ]()

    var tid = thread_idx.x
    var gid = global_idx.x

    # Load to shared memory
    if gid < UInt(n):
        smem[tid] = input[gid]
    else:
        smem[tid] = 0.0

    barrier()  # __syncthreads()

    # Compute neighbor sum
    var result = smem[tid]
    if tid > 0:
        result += smem[tid - 1]
    if tid < block_dim.x - 1:
        result += smem[tid + 1]

    if gid < UInt(n):
        output[gid] = result

def main():
    comptime n = 1024
    comptime block_size = 256
    comptime smem_bytes = block_size * 4  # sizeof(float32) = 4

    with DeviceContext() as ctx:
        var d_in = ctx.enqueue_create_buffer[DType.float32](n)
        var d_out = ctx.enqueue_create_buffer[DType.float32](n)

        # ... initialize and copy data ...

        ctx.enqueue_function[neighbor_sum, neighbor_sum](
            d_in, d_out, n,
            grid_dim=(ceildiv(n, block_size),),
            block_dim=(block_size,),
            shared_mem_bytes=smem_bytes,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_bytes),
        )

        ctx.synchronize()
```

> **Apple GPU (Metal) note:** `external_memory` is NOT supported on Apple GPU. Use `stack_allocation` (below) as the primary shared memory pattern -- it works on all GPU backends (NVIDIA, AMD, Apple). Only use `external_memory` when you specifically need NVIDIA dynamic shared memory.

### Preferred: Static Shared Memory (All GPUs)

Use `stack_allocation` for shared memory -- it works on all GPU backends including Apple Metal:

```mojo
# nocompile
from memory import stack_allocation

fn kernel_with_static_smem(data: UnsafePointer[Float32, MutAnyOrigin]):
    # Equivalent to: __shared__ float smem[256];
    var smem = stack_allocation[
        256,
        DType.float32,
        address_space=AddressSpace.SHARED,
    ]()
    smem[thread_idx.x] = data[global_idx.x]
    barrier()
    data[global_idx.x] = smem[thread_idx.x]
```

---

## Level 3: Warp-Level Operations — Reduction

### CUDA

```cuda
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void block_reduce(float* input, float* output, int n) {
    __shared__ float warp_sums[32];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (gid < n) ? input[gid] : 0.0f;

    // Warp-level reduction
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (tid % 32 == 0)
        warp_sums[tid / 32] = val;
    __syncthreads();

    // Final reduction by first warp
    if (tid < 32) {
        val = (tid < blockDim.x / 32) ? warp_sums[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0)
            output[blockIdx.x] = val;
    }
}
```

### Mojo

```mojo
from gpu import thread_idx, block_idx, block_dim, global_idx, lane_id, WARP_SIZE
from gpu.sync import barrier, syncwarp
from gpu.primitives.warp import shuffle_down
from memory import stack_allocation, UnsafePointer

fn warp_reduce_sum(var val: Float32) -> Float32:
    @parameter
    for i in range(5):  # log2(32) = 5
        val += shuffle_down(val, UInt32(1 << (4 - i)))
    return val

fn block_reduce(input: UnsafePointer[Float32, MutAnyOrigin], output: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    var warp_sums = stack_allocation[
        32, DType.float32, address_space=AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var gid = global_idx.x

    var val: Float32 = input[gid] if gid < UInt(n) else Float32(0.0)

    # Warp-level reduction
    val = warp_reduce_sum(val)

    # Write warp results to shared memory
    if lane_id() == 0:
        warp_sums[tid // WARP_SIZE] = val
    barrier()

    # Final reduction by first warp
    if tid < WARP_SIZE:
        val = warp_sums[tid] if tid < block_dim.x // WARP_SIZE else Float32(0.0)
        val = warp_reduce_sum(val)
        if tid == 0:
            output[block_idx.x] = val
```

### Warp Primitive Mapping

| CUDA | Mojo | Import |
|------|------|--------|
| `__shfl_sync(mask, val, src)` | `shuffle_idx(val, src)` | `from gpu.primitives.warp import shuffle_idx` |
| `__shfl_down_sync(mask, val, delta)` | `shuffle_down(val, delta)` | `from gpu.primitives.warp import shuffle_down` |
| `__shfl_up_sync(mask, val, delta)` | `shuffle_up(val, delta)` | `from gpu.primitives.warp import shuffle_up` |
| `__shfl_xor_sync(mask, val, mask)` | `shuffle_xor(val, lane_mask)` | `from gpu.primitives.warp import shuffle_xor` |
| `__ballot_sync(mask, pred)` | `vote[ret_type](pred)` | `from gpu.primitives.warp import vote` |
| `__ballot_sync + __popc / __all / __any` | `vote[ret_type](pred)` | `from gpu.primitives.warp import vote` |

> **Note:** Mojo warp operations don't require explicit mask parameters — the full warp is always used. This matches the CUDA model for compute capability 7.0+. The `vote` function returns a bitmask; use popcount, all-bits, or any-bits checks on the result to replicate `__popc(__ballot_sync(...))`, `__all_sync`, or `__any_sync`.

---

## Level 4: LayoutTensor — Type-Safe Tensor Access

LayoutTensor is Mojo's key abstraction for GPU programming. It combines a pointer with a compile-time layout, providing safe multidimensional access with automatic index calculation.

### CUDA: Manual Index Calculation

```cuda
// CUDA: manual 2D indexing
__global__ void transpose(float* out, float* in, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
        out[col * M + row] = in[row * N + col];  // Manual stride math
}
```

### Mojo: LayoutTensor with Compile-Time Layouts

```mojo
# nocompile
from layout import Layout, LayoutTensor, IntTuple

# LayoutTensor encodes shape AND stride at compile time
# Layout((M, N), (N, 1)) = row-major, Layout((M, N), (1, M)) = column-major

fn transpose_tile[
    BM: Int, BN: Int
](
    # LayoutTensor carries dtype, layout, and address space as type parameters
    output: LayoutTensor[DType.float32, Layout((BM, BN), (1, BM)), MutAnyOrigin],
    input: LayoutTensor[DType.float32, Layout((BM, BN), (BN, 1)), MutAnyOrigin],
):
    var row = thread_idx.y
    var col = thread_idx.x

    # Direct multidimensional indexing — no manual stride calculation
    output[col, row] = input[row, col]
```

### Creating LayoutTensors

```mojo
# nocompile
from layout import Layout, LayoutTensor
from memory import stack_allocation, UnsafePointer

# From a raw pointer + layout (global memory)
fn from_pointer(ptr: UnsafePointer[Float32]):
    var tensor = LayoutTensor[DType.float32, Layout((64, 64), (64, 1))](ptr)
    # tensor[i, j] accesses ptr[i * 64 + j]

# In shared memory
fn in_shared_memory():
    var smem_ptr = stack_allocation[
        64 * 64, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var smem_tile = LayoutTensor[
        DType.float32,
        Layout((64, 64), (64, 1)),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ](smem_ptr)

# In registers (local address space)
fn in_registers():
    var reg_ptr = stack_allocation[
        8, DType.float32, address_space=AddressSpace.LOCAL
    ]()
    var reg_tile = LayoutTensor[
        DType.float32,
        Layout((4, 2), (2, 1)),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ](reg_ptr)
```

### LayoutTensor Operations

```mojo
# nocompile
# Slicing — extract a sub-tile
var row_0 = tensor.slice[axis=0](0)  # First row

# Tiling — partition into blocks
# tile[block_idx] gives you the sub-tile for that block
var tiled = tensor.tile[16, 16]()

# Vectorized load/store
var vec = tensor.load[width=4](row, col)  # SIMD load of 4 elements
tensor.store(row, col, vec)               # SIMD store
```

### Address Space Types

| Address Space | CUDA Equivalent | Mojo |
|--------------|----------------|------|
| Global (DRAM) | Default pointer | `AddressSpace.GENERIC` (default) |
| Shared (SMEM) | `__shared__` | `AddressSpace.SHARED` |
| Local (Registers) | Register variables | `AddressSpace.LOCAL` |

---

## Level 5: Tensor Cores — WGMMA (SM90 / H100)

### CUDA PTX: Warp Group MMA

```cuda
// CUDA: WGMMA via inline PTX (simplified)
// Real code uses CUTLASS or CuTe abstractions
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
    "{%0, %1, %2, ...}, "    // accumulator registers
    "{%32, %33, %34, ...}, " // A matrix descriptor
    "{%48, %49, %50, ...}, " // B matrix in shared memory
    : "+f"(d0), "+f"(d1), ...
    : "l"(desc_a), "l"(desc_b));
```

### Mojo: TensorCoreAsync (WGMMA Wrapper)

```mojo
# nocompile
from layout.tensor_core_async import TensorCoreAsync, warpgroup_fence, tile_layout_k_major
from layout import Layout, LayoutTensor
from gpu.host.nvidia.tma import TensorMapSwizzle

# Configure the WGMMA operation
comptime BM = 128  # Block tile M
comptime BN = 128  # Block tile N
comptime BK = 64   # Block tile K
comptime wgmma_shape = IndexList[3](64, 128, 16)  # MMA instruction shape

# Create layout for k-major tile in shared memory (with swizzle for bank-conflict-free access)
comptime a_smem_layout = tile_layout_k_major[
    DType.float16, BM, BK, TensorMapSwizzle.SWIZZLE_128B
]()
comptime b_smem_layout = tile_layout_k_major[
    DType.float16, BN, BK, TensorMapSwizzle.SWIZZLE_128B
]()

# Create the WGMMA operation
comptime WgmmaOp = TensorCoreAsync[
    DType.float32,         # Accumulator type (c_type)
    DType.float16,         # A matrix type (a_type)
    DType.float16,         # B matrix type (b_type)
    mma_shape=Index(64, 128, 16),  # MMA instruction shape (M, N, K)
    a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
    transpose_b=True,
]

# In the kernel:
fn consumer_mainloop(
    a_smem: SMemTile[DType.float16, a_smem_layout, ...],
    b_smem: SMemTile[DType.float16, b_smem_layout, ...],
    accum: RegTile[DType.float32, accum_layout, ...],
):
    # Fence before MMA
    warpgroup_fence()

    # Execute WGMMA: shared memory → registers (accumulate)
    WgmmaOp.mma(accum, a_smem, b_smem)

    # Fence after MMA
    warpgroup_fence()
```

### CUDA → Mojo Tensor Core Mapping

| CUDA | Mojo | Generation |
|------|------|-----------|
| `nvcuda::wmma::mma_sync()` | Not directly exposed — use `TensorCoreAsync` | SM70+ (Volta) |
| `mma.sync` (PTX) | Not directly exposed — use `TensorCoreAsync` | SM80+ (Ampere) |
| `wgmma.mma_async` (PTX) | `TensorCoreAsync.mma()` | SM90 (Hopper) |
| `tcgen05.mma` (PTX) | TCGen05 MMA API | SM100 (Blackwell) |

---

## Level 6: TMA — Tensor Memory Accelerator (SM90+)

### CUDA: TMA via CuTe

```cuda
// CUDA/CuTe: TMA load from global to shared memory
cute::copy(tma_load_a, tiled_mma.partition_A(gA)(_, _, k_tile),
                        tiled_mma.partition_A(sA)(_, _, smem_pipe_write));
```

### Mojo: TMATensorTile

```mojo
# nocompile
from layout.tma_async import TMATensorTile, SharedMemBarrier
from gpu.host.nvidia.tma import TensorMapSwizzle

# TMA descriptors are created on the host and passed to the kernel
# They describe the global memory tensor and how to load tiles from it

# Host-side: create TMA descriptor
var tma_a = TMATensorTile[DType.float16, tile_layout, desc_layout](
    tensor_ptr=a_device_ptr,
    tensor_shape=(M, K),
    swizzle=TensorMapSwizzle.SWIZZLE_128B,
)

# Kernel-side: async load from global → shared using TMA
fn producer_load(
    tma: TMATensorTile[DType.float16, ...],
    dst_smem: SMemTile[DType.float16, ...],
    barrier: UnsafePointer[SharedMemBarrier, address_space=AddressSpace.SHARED],
    coords: Tuple[UInt, UInt],
):
    # Arrive at barrier (sets expected transaction bytes)
    barrier[].arrive_expect_tx(tile_bytes)

    # Async TMA load: global memory → shared memory
    tma.async_load(dst_smem, barrier, coords)

    # No explicit wait needed here — consumer will wait on barrier
```

### CUDA → Mojo Async Copy Mapping

| CUDA | Mojo | Notes |
|------|------|-------|
| `cp.async` (SM80) | `async_copy()` | `from gpu.memory import async_copy` |
| `cp.async.commit_group` | `async_copy_commit_group()` | `from gpu.memory import async_copy_commit_group` |
| `cp.async.wait_group` | `async_copy_wait_group(N)` | `from gpu.memory import async_copy_wait_group` |
| TMA load (SM90) | `tma_tile.async_load(dst, barrier, coords)` | `from layout.tma_async import TMATensorTile` |
| TMA store (SM90) | `tma_tile.async_store(src, coords)` | `from layout.tma_async import TMATensorTile` |
| TMA prefetch (SM90) | `tma_tile.prefetch(coords)` | Prefetch to L2 cache |

---

## Level 7: Producer-Consumer Pipeline

The most advanced pattern: overlapping memory loads with compute using a multi-stage software pipeline.

### CUDA Pattern (Pseudocode)

```cuda
// CUDA: manual pipeline management
for (int stage = 0; stage < NUM_STAGES; stage++) {
    // Producer: async load tiles for this stage
    cp.async.load(&smem_a[stage], &gmem_a[...]);
    cp.async.load(&smem_b[stage], &gmem_b[...]);
    cp.async.commit_group();
}

for (int k = 0; k < K_tiles; k++) {
    // Wait for stage to be ready
    cp.async.wait_group<NUM_STAGES - 1>();
    __syncthreads();

    int stage = k % NUM_STAGES;
    // Consumer: compute on tiles in this stage
    wgmma(&accum, &smem_a[stage], &smem_b[stage]);

    // Producer: start loading next tiles
    cp.async.load(&smem_a[next_stage], &gmem_a[...]);
    cp.async.commit_group();
}
```

### Mojo: RingBuffer with Producer/Consumer Context Managers

```mojo
from .ring_buffer import RingBuffer, RingBufferProducer, RingBufferConsumer

# The ring buffer manages multi-stage pipeline synchronization
# Producer warp group loads tiles; consumer warp groups compute

fn matmul_kernel(
    smem: HopperMatmulSM90Kernel_SMem[...],
    # ... other params
):
    var warp_group_id = get_warp_id() // (WARPGROUP_SIZE // WARP_SIZE)

    # Create ring buffer with full/empty barriers
    var ring_buffer = RingBuffer[...](
        smem.full_mbar, smem.empty_mbar,
        smem.a_tiles, smem.b_tiles,
    )

    if warp_group_id == 0:
        # === PRODUCER WARP GROUP ===
        with ring_buffer.producer() as producer:
            for k_tile in range(num_k_tiles):
                with producer.get_tiles() as tiles:
                    # TMA load into ring buffer slot
                    tile_loader_a.load_tile(
                        tiles.a_tile_array[0], tiles.barrier, (m_idx, k_tile)
                    )
                    tile_loader_b.load_tile(
                        tiles.b_tile_array[0], tiles.barrier, (n_idx, k_tile)
                    )
    else:
        # === CONSUMER WARP GROUP(S) ===
        with ring_buffer.consumer() as consumer:
            for k_tile in range(num_k_tiles):
                with consumer.get_tiles() as tiles:
                    # WGMMA on tiles from ring buffer
                    warpgroup_fence()
                    WgmmaOp.mma(accum, tiles.a_tile, tiles.b_tile)
                    warpgroup_fence()
```

### Pipeline Stage Abstraction

| CUDA | Mojo |
|------|------|
| Manual `pipeline_state` tracking | `RingBuffer` with automatic stage cycling |
| `cp.async.commit_group()` | Implicit in `producer.get_tiles()` context exit |
| `cp.async.wait_group<N>()` | Implicit in `consumer.get_tiles()` context entry |
| Manual barrier arrive/wait | `ProducerTiles`/`ConsumerTiles` handle barrier protocol |
| `__pipeline_membar_arrive()` | `SharedMemBarrier.arrive_expect_tx()` |

---

## Level 8: Complete GEMM Structure

Putting it all together — the structure of a complete SM90 matmul kernel:

```mojo
from gpu import thread_idx, block_idx, WARP_SIZE
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from layout import Layout, LayoutTensor
from layout.tensor_core_async import TensorCoreAsync, warpgroup_fence
from layout.tma_async import TMATensorTile, SharedMemBarrier
from math import ceildiv

struct HopperGEMM[
    # Type parameters
    a_type: DType, b_type: DType, c_type: DType,
    # Tile dimensions
    BM: Int, BN: Int, BK: Int,
    # WGMMA shape
    wgmma_m: Int, wgmma_n: Int, wgmma_k: Int,
    # Pipeline configuration
    num_stages: Int,
    num_threads: Int = 256,  # 2 warp groups: 1 producer + 1 consumer
]:
    # ... shared memory layout, tile types, pipeline setup ...

    @staticmethod
    fn kernel(
        tma_a: TMATensorTile[...],
        tma_b: TMATensorTile[...],
        c_ptr: UnsafePointer[Scalar[c_type], MutAnyOrigin],
        M: Int, N: Int, K: Int,
    ):
        # 1. Allocate shared memory
        var smem = SharedMemoryManager()
        var a_tiles = smem.build[ATileArray]()
        var b_tiles = smem.build[BTileArray]()
        var barriers = smem.build[PipelineBarrier]()

        # 2. Determine this block's tile coordinates
        var block_m = block_idx.x
        var block_n = block_idx.y
        var num_k_tiles = ceildiv(K, BK)

        # 3. Split warp groups into roles
        var wg_id = thread_idx.x // 128  # 128 threads per warp group

        if wg_id == 0:
            # PRODUCER: Load tiles via TMA
            warpgroup_reg_dealloc[40]()  # Free registers for consumer
            for k in range(num_k_tiles):
                var stage = k % num_stages
                barriers[stage].arrive_expect_tx(tile_bytes)
                tma_a.async_load(a_tiles[stage], barriers[stage], (block_m, k))
                tma_b.async_load(b_tiles[stage], barriers[stage], (block_n, k))
        else:
            # CONSUMER: Compute using tensor cores
            warpgroup_reg_alloc[232]()  # Claim registers for accumulators
            var accum = RegTile[accum_type, accum_layout]()
            accum.zero()

            for k in range(num_k_tiles):
                var stage = k % num_stages
                barriers[stage].wait()  # Wait for producer to fill this stage

                warpgroup_fence()
                WgmmaOp.mma(accum, a_tiles[stage], b_tiles[stage])
                warpgroup_fence()

                barriers[stage].arrive()  # Signal tiles consumed

            # 4. Write output
            # Store accumulators to global memory via shared memory
            # (epilogue handling)
```

---

## Common Gotchas When Porting

### 1. Unsigned vs Signed Comparisons

```mojo
# CUDA: if (tid < n) — works because both are int
# Mojo: global_idx returns UInt, n is Int — must cast
if global_idx.x < UInt(n):  # Cast n to UInt for comparison
    # ...
```

### 2. No Implicit Warp Mask

```mojo
# nocompile
# CUDA: __shfl_down_sync(0xFFFFFFFF, val, offset)
# Mojo: No mask parameter — full warp is always used
# NOTE: offset must be UInt32 — use explicit cast
var result = shuffle_down(val, UInt32(offset))
```

### 3. Compile-Time vs Runtime Parameters

```mojo
# nocompile
# CUDA: template parameters are compile-time
# Mojo: use comptime for compile-time values
comptime BLOCK_SIZE = 256      # Compile-time constant
comptime SMEM_SIZE = BLOCK_SIZE * 4  # Computed at compile time

# Use @parameter for compile-time unrolling
@parameter
for i in range(5):  # Fully unrolled at compile time
    val += shuffle_down(val, UInt32(1 << (4 - i)))
```

### 4. Address Space Annotations

```mojo
# nocompile
# CUDA: __shared__ is implicit
# Mojo: address space must be explicit in types
var smem_ptr = stack_allocation[
    256, DType.float32, address_space=AddressSpace.SHARED  # Explicit!
]()
```

### 5. Kernel Function Signatures

```mojo
# CUDA: __global__ void kernel(float* data, int n)
# Mojo: regular fn, but parameters must be device-passable types
fn kernel(
    data: UnsafePointer[Float32, MutAnyOrigin],  # Pointers need MutAnyOrigin for DevicePassable
    n: Int,                                       # Scalars are device-passable
):
    ...

# Preferred: LayoutTensor IS directly passable to GPU kernels
# Construct from DeviceBuffer on host, pass directly — no raw pointer needed
fn kernel_lt(
    tensor: LayoutTensor[DType.float32, layout, MutAnyOrigin],  # Clean 2D indexing
):
    tensor[row, col] = result  # No manual index math
```

> **Prefer LayoutTensor over UnsafePointer** for kernel parameters. LayoutTensor constructed from a DeviceBuffer is directly passable — it provides type-safe multidimensional indexing and eliminates manual stride calculations.

---

## Import Cheat Sheet

```mojo
# nocompile
# === Core GPU ===
from gpu import thread_idx, block_idx, block_dim, grid_dim, global_idx
from gpu import lane_id, WARP_SIZE
from gpu import warp_id as get_warp_id
from gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from gpu.sync import barrier, syncwarp

# === Memory ===
from memory import UnsafePointer, alloc, stack_allocation, AddressSpace
from gpu.memory import external_memory, async_copy  # AddressSpace also available from gpu.memory

# === Warp Operations ===
from gpu.primitives.warp import shuffle_idx, shuffle_down, shuffle_up, shuffle_xor
from gpu.primitives.warp import vote

# === Layout System ===
from layout import Layout, LayoutTensor, IntTuple
from layout.layout_tensor import LayoutTensorIter
from layout.swizzle import Swizzle

# === Tensor Cores (SM90) ===
from layout.tensor_core_async import TensorCoreAsync, warpgroup_fence
from layout.tma_async import TMATensorTile, SharedMemBarrier

# === Structured Kernel Utilities ===
from linalg.structuring import (
    SMemTile, RegTile, SMemBarrier, PipelineBarrier,
    SharedMemoryManager, NVIDIASharedMemoryManager,
    ScatterGatherAmd,
)

# === Cluster Operations (SM90+) ===
from gpu.primitives.cluster import cluster_sync, elect_one_sync
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
```

---

## Version-Specific Features

### Stable (v26.1)

- Standard CUDA-to-Mojo porting patterns apply
- Use `gpu.block_dim()`, `gpu.thread_idx()`, `gpu.block_idx()` for CUDA equivalents
- LayoutTensor for shared memory access

### Nightly (v26.2+)

- Enhanced GPU intrinsics may be available
- Check latest changelog for new GPU porting utilities

---

## Related Patterns

- [`gpu-porting-cute.md`](gpu-porting-cute.md) — CuTe DSL → Mojo mapping (layout algebra, TiledMMA, TiledCopy)
- [`gpu-porting-rocm.md`](gpu-porting-rocm.md) — ROCm/HIP → Mojo mapping (MFMA, wavefront, scheduling)
- [`gpu-fundamentals.md`](gpu-fundamentals.md) — Mojo GPU programming fundamentals
- [`gpu-structured-kernels.md`](gpu-structured-kernels.md) — ScatterGather/RingBuffer/TileOp architecture
- [`gpu-tensor-cores.md`](gpu-tensor-cores.md) — SM90/SM100 tensor core patterns in detail
- [`gpu-warp-sync.md`](gpu-warp-sync.md) — Barriers, async transactions, pipeline stages
- [`gpu-memory-access.md`](gpu-memory-access.md) — TMA, prefetch, swizzle patterns
- [`gpu-warp-sync.md`](gpu-warp-sync.md) — Warp primitives, shuffle, ballot, reduction
