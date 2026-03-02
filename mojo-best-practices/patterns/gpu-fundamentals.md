---
title: GPU Programming Fundamentals
description: Core GPU programming concepts including thread hierarchy, memory model, kernel patterns, and device context management
impact: CRITICAL
category: gpu
tags: [gpu, cuda, parallel, kernel, memory, coalescing, context]
error_patterns:
  - "GPU OOM"
  - "CUDA error: out of memory"
  - "CUDA error: invalid device"
  - "CUDA error: kernel launch"
  - "device not found"
  - "kernel launch failed"
  - "uncoalesced memory access"
  - "Metal GPU returns zeros"
  - "Metal kernel not executing"
  - "does not support operation: print"
  - "MutAnyOrigin"
scenarios:
  - "Write first GPU kernel"
  - "Fix GPU out of memory error"
  - "Optimize memory coalescing"
  - "Use shared memory for reduction"
  - "Handle GPU device initialization"
  - "Debug GPU kernel without print"
  - "Annotate kernel pointer origins"
consolidates:
  - gpu-fundamentals.md
  - gpu-memory-coalescing.md
  - gpu-memory-optimization.md
  - gpu-native-context.md
  - gpu-native-kernel-patterns.md
  - gpu-state-reset.md
  - gpu-buffer-pooling.md
---

# GPU Programming Fundamentals

**Category:** gpu | **Impact:** CRITICAL

GPU programming in Mojo provides 10-100x speedups for parallel workloads. This pattern covers the thread hierarchy, memory model, coalescing requirements, native kernel patterns, and device context management essential for high-performance GPU code.

## API Availability

| API | Availability | Notes |
|-----|--------------|-------|
| `gpu.thread_idx`, `gpu.block_idx`, `gpu.block_dim`, `gpu.grid_dim` | PUBLIC | Core GPU primitives |
| `gpu.global_idx` | PUBLIC | Pre-computed global thread index (`block_dim * block_idx + thread_idx`) |
| `gpu.barrier` (also `gpu.sync.barrier`) | PUBLIC | Block-level synchronization (both import paths work) |
| `gpu.host.DeviceContext`, `gpu.host.DeviceBuffer` | PUBLIC | Device management |
| `stack_allocation[N, T, address_space=AddressSpace.SHARED]()` | PUBLIC | Shared memory allocation |
| `stack_allocation` with `AddressSpace.SHARED` | PUBLIC | Shared memory allocation (use `from memory import stack_allocation`) |
| `memory.UnsafePointer` | PUBLIC | Standard library |
| `sys.has_accelerator` | PUBLIC | System capability check |
| `sys.is_nvidia_gpu`, `sys.is_amd_gpu`, `sys.is_apple_gpu` | PUBLIC | GPU vendor detection (compile-time) |
| `LayoutTensor`, `Layout` | `from layout import LayoutTensor, Layout` | Type-safe tensor with compile-time layout |

> **Note:** All APIs listed above are available in the Mojo nightly toolchain (v26.2+). Code examples below are documentation snippets — adapt import paths and parameters for your use case.

---

## Core Concepts

### GPU Thread Hierarchy

```
Grid (entire kernel launch)
  |-- Blocks (groups of threads that can synchronize)
        |-- Warps (32 threads NVIDIA / 64 threads AMD executing in lockstep)
              |-- Threads (individual execution units)
```

| Concept | Description | Typical Limits |
|---------|-------------|----------------|
| **Grid** | Collection of all threads launched | Millions of threads |
| **Block** | Threads that share memory and sync | Max 1024 threads |
| **Warp/Wavefront** | Threads executing same instruction | 32 (NVIDIA) / 64 (AMD) |
| **Thread** | Individual execution unit | Private registers |

### Memory Hierarchy

| Memory Type | Bandwidth | Latency | Scope | Use Case |
|-------------|-----------|---------|-------|----------|
| Registers | ~20 TB/s | 0 cycles | Per thread | Local variables |
| Shared Memory | ~10 TB/s | ~5 cycles | Per block | Thread cooperation |
| L1 Cache | ~2-4 TB/s | ~20 cycles | Per SM | Auto-cached |
| L2 Cache | ~1-2 TB/s | ~200 cycles | Device | Auto-cached |
| Global Memory | 500-900 GB/s | ~400 cycles | Device | Main storage |

**Pattern:**

```mojo

from gpu import thread_idx, block_idx, block_dim, grid_dim, global_idx, barrier
from gpu.host import DeviceContext
from memory import UnsafePointer

fn gpu_kernel(
    data: UnsafePointer[Float32, MutAnyOrigin],
    result: UnsafePointer[Float32, MutAnyOrigin],
    size: Int
):
    """Basic GPU kernel pattern."""
    # Calculate global thread index (two equivalent approaches)
    var tid = block_idx.x * block_dim.x + thread_idx.x  # Manual calculation
    # Or use: var tid = global_idx.x  # Equivalent: block_dim * block_idx + thread_idx

    # CRITICAL: Always bounds check (tid is UInt, size is Int — cast to match)
    if tid >= UInt(size):
        return
    result[tid] = data[tid] * 2.0

fn main() raises:
    comptime SIZE: Int = 1_000_000
    comptime BLOCK_SIZE: Int = 256

    var ctx = DeviceContext()

    # Allocate device memory
    var d_data = ctx.enqueue_create_buffer[DType.float32](SIZE)
    var d_result = ctx.enqueue_create_buffer[DType.float32](SIZE)

    # Calculate grid dimensions
    var num_blocks = (SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel — pass DeviceBuffer directly, not .unsafe_ptr()
    ctx.enqueue_function[gpu_kernel, gpu_kernel](
        d_data,
        d_result,
        SIZE,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,)
    )
    ctx.synchronize()
```

> **For element-wise operations**, prefer `algorithm.elementwise` over manual kernel launch — it handles grid/block sizing and striding automatically. See [`gpu-kernels.md`](gpu-kernels.md) for the idiomatic template. The manual pattern above is shown for educational purposes.

---

## Common Patterns

> **Warning: GPU kernel pointer mutability.** For GPU kernel parameters that need mutable (writable) access via raw pointers, `UnsafePointer` may require the `LegacyUnsafePointer` alias with explicit `mut=True`. Import it as: `from memory import LegacyUnsafePointer` and declare as `comptime Ptr = LegacyUnsafePointer[mut=True, type=Float32, origin=MutAnyOrigin]`. Prefer `LayoutTensor` (see [`gpu-layout-tensor.md`](gpu-layout-tensor.md)) over raw pointers when possible -- it handles mutability and origin annotations automatically.

### Memory Coalescing

**When:** All global memory access in GPU kernels

Memory coalescing combines multiple thread memory accesses into single transactions. Adjacent threads must access adjacent memory addresses for optimal bandwidth (10-32x improvement).

**Do:**
```mojo
# nocompile

fn row_major_access_kernel(
    matrix: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int, cols: Int,
):
    """GOOD: Row-major access enables coalesced memory access."""
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x

    if row >= UInt(rows) or col >= UInt(cols):
        return
    # Thread 0 accesses [row*cols], Thread 1 accesses [row*cols+1], etc.
    var val = matrix[row * cols + col]  # COALESCED!
    output[row * cols + col] = val * 2.0
```

**Don't:**
```mojo
# nocompile

fn column_major_access_kernel(
    matrix: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int, cols: Int,
):
    """BAD: Column-major access causes uncoalesced memory access."""
    var col = block_idx.x * block_dim.x + thread_idx.x
    var row = block_idx.y * block_dim.y + thread_idx.y

    if row >= UInt(rows) or col >= UInt(cols):
        return
    # Thread 0 accesses [0], Thread 1 accesses [rows] - stride!
    var val = matrix[col * rows + row]  # UNCOALESCED - 10-32x slower!
    output[col * rows + row] = val * 2.0
```

### Shared Memory for Data Reuse

**When:** Data accessed multiple times or needs inter-thread communication

**Public API pattern using `stack_allocation`:**
```mojo
# nocompile

from gpu.memory import AddressSpace
from memory import stack_allocation

fn matrix_transpose_shared(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    width: Int, height: Int
):
    """Efficient transpose using shared memory."""
    comptime TILE_DIM: Int = 32
    comptime TILE_SIZE: Int = TILE_DIM * (TILE_DIM + 1)  # +1 for bank conflict avoidance

    # Allocate shared memory with stack_allocation and AddressSpace.SHARED
    var tile = stack_allocation[TILE_SIZE, Float32, address_space=AddressSpace.SHARED]()

    var bx = block_idx.x
    var by = block_idx.y
    var tx = thread_idx.x
    var ty = thread_idx.y

    var x_in = bx * TILE_DIM + tx
    var y_in = by * TILE_DIM + ty

    # Load tile (coalesced read)
    if x_in < width and y_in < height:
        tile[ty * (TILE_DIM + 1) + tx] = input[y_in * width + x_in]

    barrier()

    # Store transposed tile (coalesced write)
    var x_out = by * TILE_DIM + tx
    var y_out = bx * TILE_DIM + ty

    if x_out < height and y_out < width:
        output[y_out * height + x_out] = tile[tx * (TILE_DIM + 1) + ty]
```

**LayoutTensor pattern:**
```mojo
# nocompile

from layout import LayoutTensor, Layout
from gpu.memory import AddressSpace

fn matrix_transpose_layout[TILE_DIM: Int](
    input: LayoutTensor[DType.float32, Layout.row_major(...), MutAnyOrigin],
    output: LayoutTensor[DType.float32, Layout.row_major(...), MutAnyOrigin],
):
    """Efficient transpose using LayoutTensor with shared memory."""
    # LayoutTensor provides type-safe multi-dimensional access
    var tile = LayoutTensor[
        DType.float32,
        Layout.row_major(TILE_DIM, TILE_DIM + 1),  # +1 for bank conflict avoidance
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var tx = thread_idx.x
    var ty = thread_idx.y

    # Clean 2D indexing instead of manual offset calculation
    tile[ty, tx] = input[block_idx.y * TILE_DIM + ty, block_idx.x * TILE_DIM + tx]
    barrier()
    output[block_idx.x * TILE_DIM + ty, block_idx.y * TILE_DIM + tx] = tile[tx, ty]
```

### Structure of Arrays (SoA) Layout

**When:** Processing fields across many objects

```mojo
# nocompile

# BAD: Array of Structures (AoS) - uncoalesced
@fieldwise_init
struct ParticleAoS(TrivialRegisterPassable, Copyable):
    var x: Float32
    var y: Float32
    var z: Float32

fn update_aos_slow(particles: UnsafePointer[ParticleAoS, MutAnyOrigin], n: Int):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid >= UInt(n):
        return
    particles[tid].x += 1.0  # Strided access!

# GOOD: Structure of Arrays (SoA) - coalesced
fn update_soa_fast(
    x: UnsafePointer[Float32, MutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    z: UnsafePointer[Float32, MutAnyOrigin],
    n: Int
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid >= UInt(n):
        return
    x[tid] += 1.0  # Coalesced access!
```

### Native DeviceContext Pattern

**When:** Multiple sequential GPU operations (avoids FFI overhead)

```mojo
# nocompile

from gpu.host import DeviceContext, DeviceBuffer
from sys import has_accelerator

fn forward_with_native(x: UnsafePointer[Float32], size: Int) raises:
    if not has_accelerator():
        return forward_cpu(x, size)

    var ctx = DeviceContext()

    # Create GPU buffer once
    var d_x = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_copy(d_x, x)

    # All kernel launches use native dispatch — pass DeviceBuffer directly
    ctx.enqueue_function[my_kernel, my_kernel](d_x, size,
                                     grid_dim=blocks, block_dim=threads)

    # Single sync at end
    ctx.synchronize()
    ctx.enqueue_copy(x, d_x)
```

### Buffer Pooling

**When:** Repeated operations with same-sized buffers

```mojo
# nocompile

from memory import UnsafePointer

struct BufferPool:
    """Pre-allocated buffer pool for repeated operations."""
    var buf1: UnsafePointer[Float32]
    var buf2: UnsafePointer[Float32]
    var buf3: UnsafePointer[Float32]
    var is_allocated: Bool

    fn allocate(mut self, max_size: Int):
        if self.is_allocated:
            self.release()
        from memory import alloc
        self.buf1 = alloc[Float32](max_size)
        self.buf2 = alloc[Float32](max_size)
        self.buf3 = alloc[Float32](max_size)
        self.is_allocated = True

    fn release(mut self):
        if not self.is_allocated:
            return
        self.buf1.free()
        self.buf2.free()
        self.buf3.free()
        self.is_allocated = False

fn process_blocks(blocks: List[Block], hidden: UnsafePointer[Float32]):
    # Allocate pool once before all blocks
    var pool = BufferPool()
    pool.allocate(max_seq * hidden_dim)

    for block in blocks:
        # Reuse pool buffers - zero allocation per block
        process_block_pooled(block, hidden, pool)

    pool.release()
```

---

## Memory Coalescing

**Impact: CRITICAL** — 10-32x bandwidth improvement through coalesced access

Memory coalescing combines multiple thread accesses into a single wide memory transaction. For optimal coalescing, thread N should access address `base + N * sizeof(element)`.

**Incorrect (uncoalesced column-major):**
```mojo
# nocompile

fn column_major_kernel(matrix: UnsafePointer[Float32, MutAnyOrigin], rows: Int, cols: Int):
    var col = block_idx.x * block_dim.x + thread_idx.x
    var row = block_idx.y * block_dim.y + thread_idx.y

    # WRONG: Adjacent threads access addresses rows apart!
    var val = matrix[col * rows + row]  # Strided access - SLOW!
```

**Correct (coalesced row-major):**
```mojo
# nocompile

fn row_major_kernel(matrix: UnsafePointer[Float32, MutAnyOrigin], rows: Int, cols: Int):
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x

    # CORRECT: Adjacent threads access consecutive addresses
    var val = matrix[row * cols + col]  # Coalesced - FAST!
```

**Array of Structures (AoS) vs Structure of Arrays (SoA):**

```mojo
# BAD: AoS - uncoalesced, wastes bandwidth
struct Particle:
    var x: Float32
    var y: Float32
    var z: Float32

fn bad_aos_access(particles: UnsafePointer[Particle, MutAnyOrigin], tid: Int):
    var x = particles[tid].x  # Loads 12 bytes, uses 4

# GOOD: SoA - coalesced, full bandwidth utilization
fn good_soa_access(x: UnsafePointer[Float32, MutAnyOrigin], y: UnsafePointer[Float32, MutAnyOrigin], tid: Int):
    var x_val = x[tid]  # Perfectly coalesced
```

**Vectorized coalesced access:**
```mojo
# nocompile

fn vectorized_kernel(input: UnsafePointer[Float32, MutAnyOrigin], output: UnsafePointer[Float32, MutAnyOrigin], size: Int):
    comptime VECTOR_WIDTH: Int = 4
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var vec_idx = tid * VECTOR_WIDTH

    if vec_idx + UInt(VECTOR_WIDTH) > UInt(size):
        return
    var vec = input.load[width=VECTOR_WIDTH](vec_idx)
    output.store(vec_idx, vec * 2.0)
```

---

## Native Kernel Patterns

Common kernel patterns for GPU workloads:

**1D Element-wise (SiLU, ReLU):**
```mojo
# nocompile

fn gpu_silu_kernel(data: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid >= UInt(n):
        return
    var x = data[tid]
    data[tid] = x / (1.0 + exp(-x))
```

**2D MatMul (naive):**
```mojo
# nocompile

fn gpu_matmul_kernel(C: UnsafePointer[Float32, MutAnyOrigin], A: UnsafePointer[Float32, MutAnyOrigin],
                     B: UnsafePointer[Float32, MutAnyOrigin], M: Int, K: Int, N: Int):
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x

    if row >= UInt(M) or col >= UInt(N):
        return
    var sum: Float32 = 0.0
    for k in range(K):
        # Note: row/col are UInt (from block_idx/thread_idx). Use Int() cast
        # to avoid mixed UInt/Int deprecation warnings: A[Int(row) * K + k]
        sum += A[row * K + k] * B[k * N + col]
    C[row * N + col] = sum
```

**Block size guidelines:**

| Operation | Recommended Block Size |
|-----------|------------------------|
| 1D element-wise | 256 |
| 2D matmul | (16, 16) or (32, 32) |
| Row reduction | 256 (1 block per row) |

### Kernel Restrictions

GPU kernels have constraints not present in CPU code. These cause compilation failures if violated.

**`print()` is unsupported in GPU kernels:**
```
constraint failed: Current compilation target does not support operation: print
```

The `print()` function cannot be compiled for GPU targets. This includes code inside `elementwise[..., target="gpu"]` closures and any function called from a GPU kernel.

```mojo
# nocompile
fn my_kernel(data: LayoutTensor[dtype, layout, MutAnyOrigin]):
    var tid = thread_idx.x
    # print(tid)  # ERROR: Cannot compile for GPU
    _ = tid       # Use _ = x to suppress unused variable warnings during debugging
```

**Pointer origin annotations — `MutAnyOrigin` vs `ImmutAnyOrigin`:**

GPU kernel parameters use origin annotations to declare read/write intent:

```mojo
# nocompile

fn my_kernel[dtype: DType, layout: Layout](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],   # Read-write buffer
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],   # Read-only buffer
):
    var tid = thread_idx.x
    output[tid] = input[tid] * 2
```

| Origin | Meaning | Use For |
|--------|---------|---------|
| `MutAnyOrigin` | Mutable kernel pointer | Output buffers, in-place modification |
| `ImmutAnyOrigin` | Immutable kernel pointer | Read-only input buffers |

Using `ImmutAnyOrigin` for inputs enables compiler optimizations and catches accidental writes at compile time.

### 2D Grid Launch Pattern

**When:** Matrix operations, image processing, 2D convolutions

```mojo

from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext

fn matrix_kernel(
    C: UnsafePointer[Float32, MutAnyOrigin],
    A: UnsafePointer[Float32, MutAnyOrigin],
    B: UnsafePointer[Float32, MutAnyOrigin],
    M: Int, N: Int, K: Int
):
    """2D kernel with row/column indexing."""
    # 2D thread indexing
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x

    if row >= UInt(M) or col >= UInt(N):
        return
    var sum: Float32 = 0.0
    for k in range(K):
        # Note: row/col are UInt (from block_idx/thread_idx). Use Int() cast
        # to avoid mixed UInt/Int deprecation warnings: A[Int(row) * K + k]
        sum += A[row * K + k] * B[k * N + col]
    C[row * N + col] = sum

fn launch_2d_kernel() raises:
    """Launch kernel with 2D grid and block dimensions."""
    comptime M: Int = 1024
    comptime N: Int = 1024
    comptime K: Int = 512

    # 2D block size - (16, 16) = 256 threads per block
    comptime BLOCK_X: Int = 16
    comptime BLOCK_Y: Int = 16

    var ctx = DeviceContext()

    # Allocate matrices
    var d_A = ctx.enqueue_create_buffer[DType.float32](M * K)
    var d_B = ctx.enqueue_create_buffer[DType.float32](K * N)
    var d_C = ctx.enqueue_create_buffer[DType.float32](M * N)

    # Calculate 2D grid dimensions
    var grid_x = (N + BLOCK_X - 1) // BLOCK_X  # Columns
    var grid_y = (M + BLOCK_Y - 1) // BLOCK_Y  # Rows

    # Launch with 2D grid and block dimensions — pass DeviceBuffer directly
    ctx.enqueue_function[matrix_kernel, matrix_kernel](
        d_C,
        d_A,
        d_B,
        M, N, K,
        grid_dim=(grid_x, grid_y),      # 2D grid
        block_dim=(BLOCK_X, BLOCK_Y)    # 2D block
    )
    ctx.synchronize()
```

**Grid/Block Dimension Reference:**

| Dimensions | grid_dim | block_dim | Use Case |
|------------|----------|-----------|----------|
| 1D | `(num_blocks,)` | `(256,)` | Element-wise, reductions |
| 2D | `(grid_x, grid_y)` | `(16, 16)` | Matrix ops, images |
| 3D | `(grid_x, grid_y, grid_z)` | `(8, 8, 8)` | 3D volumes, batched 2D |

**Important:** Block dimensions multiply to total threads per block (max 1024).
- `(16, 16)` = 256 threads
- `(32, 32)` = 1024 threads (maximum)
- `(8, 8, 8)` = 512 threads

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| First GPU kernel | Use basic kernel pattern with bounds check | - |
| Matrix operations | Ensure row-major (coalesced) access | [`gpu-memory-access.md`](gpu-memory-access.md) |
| Data reuse across threads | Use shared memory with bank conflict avoidance | [`gpu-memory-access.md`](gpu-memory-access.md) |
| Many GPU operations | Use native DeviceContext, batch transfers | - |
| Repeated same-size allocations | Use buffer pooling | - |
| Memory pressure between phases | Deallocate buffers between phases, use buffer pooling | - |
| Processing object fields | Convert AoS to SoA layout | - |

---

## Quick Reference

- **Block size**: Use multiples of warp size (32 NVIDIA, 64 AMD) - typically 128, 256, or 512
- **Bounds check**: Always verify `tid >= UInt(size): return` before accessing memory (thread index is `UInt`)
- **Coalescing**: Adjacent threads must access adjacent memory addresses
- **Shared memory padding**: Add +1 to row size to avoid bank conflicts
- **Device sync**: Only synchronize when CPU needs results
- **Buffer reuse**: Allocate once, reuse across iterations

**CUDA Target Architecture (v26.1+):**

```bash
mojo build --target-accelerator=nvidia:sm_80 kernel.mojo   # Ampere (A100)
mojo build --target-accelerator=nvidia:sm_89 kernel.mojo   # Ada Lovelace (L40S, RTX 4090)
mojo build --target-accelerator=nvidia:sm_90 kernel.mojo   # Hopper (H100/H200)
mojo build --target-accelerator=nvidia:sm_100 kernel.mojo  # Blackwell (B100/B200)
mojo build --target-accelerator=amd:gfx942 kernel.mojo     # AMD MI300X
mojo build --target-accelerator=amd:gfx950 kernel.mojo     # AMD MI355X
```

> **`_accelerator_arch()` format:** The runtime function `_accelerator_arch()` returns the **full prefixed string**, matching the `--target-accelerator` value: `"nvidia:sm_80"` (not `"sm_80"`), `"amd:gfx942"` (not `"gfx942"`), `"metal:3"` for Apple GPU, or `""` when no GPU is configured. Always compare against the prefixed form.

---

## Common Pitfalls

### `print()` Generally Unsupported in GPU Kernels

Calling `print()` inside a GPU kernel typically fails with: "Current compilation target does not support operation: print". This affects Apple Metal and most other GPU compilation targets.

> **Note:** Some NVIDIA targets may support limited `printf`-like functionality, but it is not portable. For cross-platform GPU code, avoid `print()` in kernels.

**Workaround:** Write debug values to an output buffer and inspect on the host after kernel completion. Use `_ = expr` to suppress unused variable warnings.

### `is_apple_gpu()` Returns False in Host Code

`sys.is_apple_gpu()` (and `is_nvidia_gpu()`, `is_amd_gpu()`) checks the **compilation target**, not the runtime device. In host-side code compiled for the CPU target, it always returns `False` — even when running on a Mac with an Apple GPU. It only returns `True` inside GPU kernel functions that are compiled for the Apple Metal target (AIR).

```mojo
from sys import is_apple_gpu

# HOST CODE — always False, regardless of hardware
fn host_function():
    if is_apple_gpu():
        pass  # NEVER reached — host is compiled for CPU target

# GPU KERNEL — True when compiled for Metal target
fn my_kernel(data: UnsafePointer[Float32, MutAnyOrigin]):
    @parameter
    if is_apple_gpu():
        pass  # Reached when kernel is compiled for Apple GPU
```

**Alternative for host-side GPU detection:** Use `from sys import _accelerator_arch` and check if the result starts with `"metal:"`:

```mojo
# nocompile
from sys import _accelerator_arch

# Safe at module level — returns "" when no GPU target is configured
comptime _is_metal = _accelerator_arch() in (
    StaticString("metal:1"), StaticString("metal:2"),
    StaticString("metal:3"), StaticString("metal:4"),
    StaticString("metal:5"),
)
```

### Float Literals are Float64

`1.0` in Mojo is `Float64`, not `Float32`. When working with `Scalar[DType.float32]` (common in GPU kernels), always explicitly construct:

```mojo
# nocompile

var val = Scalar[DType.float32](1.0)  # Explicit construction
var flag = Scalar[dtype](1.0) if condition else Scalar[dtype](0.0)  # With generic dtype
```

### LayoutTensor Element Type Mismatch

Indexing ANY LayoutTensor returns `SIMD[dtype, symbolic_size]`, not `Scalar[dtype]`. Always use `rebind[Scalar[dtype]](tensor[i])` for reads. See [`gpu-layout-tensor.md`](gpu-layout-tensor.md) for complete patterns.

### Block-Level Collective Operations

Block primitives require `block_size` parameter (critical on Apple Silicon):

```mojo
# nocompile

from gpu.primitives.block import sum as block_sum, broadcast as block_broadcast, prefix_sum as block_prefix_sum

var total = block_sum[block_size=TPB](my_val)           # Block-wide sum
var shared = block_broadcast[block_size=TPB](my_val, 0) # Broadcast from thread 0
var scan = block_prefix_sum[block_size=TPB](my_val)     # Inclusive prefix sum
```

See [`gpu-block-collectives.md`](gpu-block-collectives.md) for complete patterns.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `GPU OOM` / `out of memory` | Allocating more than GPU VRAM | Reduce batch size, use smaller tensors, or use multi-GPU |
| `CUDA error` | Various CUDA runtime failures | Check device initialization, kernel bounds |
| `kernel launch failed` | Invalid grid/block dimensions | Ensure block size ≤ 1024, grid size > 0 |
| `uncoalesced memory access` | Non-contiguous memory access pattern | Align accesses to 128-byte boundaries |
| `device not found` | No GPU available or driver issue | Check GPU drivers and CUDA installation |
| `illegal memory access` | Out-of-bounds array access | Add bounds checking: `if tid >= UInt(size): return` |
| `cannot implicitly convert 'element_type'` | LayoutTensor read without rebind | Use `rebind[Scalar[dtype]](tensor[i])` — see [`gpu-layout-tensor.md`](gpu-layout-tensor.md) |
| `does not support operation: print` | print() in GPU kernel (not supported on most targets) | Remove print, use output buffer for debug |
| Metal GPU returns zeros/garbage | Missing `@__copy_capture` on nested functions | Add `@__copy_capture(var1, var2, ...)` to ALL functions in the kernel call chain |

---

## Hardware-Specific Guidance

| GPU | Architecture | Key Considerations |
|-----|--------------|-------------------|
| **NVIDIA H100/H200** | SM90 (Hopper) | Full tensor core support, TMA available, use [`gpu-tensor-cores.md`](gpu-tensor-cores.md) |
| **NVIDIA L40S** | SM89 (Ada) | Tensor cores available, no TMA, good for inference |
| **NVIDIA A100** | SM80 (Ampere) | Tensor cores, TMA limited, widely deployed |
| **NVIDIA RTX 4090** | SM89 (Ada) | Consumer GPU, tensor cores, no NVLink |
| **AMD MI300X** | CDNA3 | MFMA instructions, see [`gpu-amd.md`](gpu-amd.md) for details |
| **AMD MI250X** | CDNA2 | Older MFMA, different scheduling |
| **Apple M1/M2/M3** | Apple GPU | Metal shaders, see Apple Metal section below |
| **Intel Arc** | Xe HPG | Not currently supported in Mojo GPU |

### Choosing the Right Pattern

```
Which GPU?
├─ NVIDIA Hopper (H100/H200)
│   └─ Use: gpu-tensor-cores.md (TCGen05, WGMMA)
│   └─ TMA: gpu-memory-access.md
├─ NVIDIA Ampere/Ada (A100, L40S, RTX 4090)
│   └─ Use: gpu-tensor-cores.md (WMMA)
│   └─ Shared memory: gpu-fundamentals.md
├─ AMD CDNA (MI300X, MI250X)
│   └─ Use: gpu-amd.md (MFMA shapes)
│   └─ Different sync model
├─ Apple Silicon (M1/M2/M3)
│   └─ Metal shaders via MAX
│   └─ See Apple Metal section
└─ Intel Arc
    └─ Not currently supported in Mojo GPU
```

### Memory Bandwidth Comparison

| GPU | Memory BW | Recommended Batch Size |
|-----|-----------|----------------------|
| H100 SXM | 3.35 TB/s | 64-256 |
| H100 PCIe | 2.0 TB/s | 32-128 |
| A100 80GB | 2.0 TB/s | 32-128 |
| L40S | 864 GB/s | 16-64 |
| MI300X | 5.3 TB/s | 64-256 |

#### Detailed Architecture Specs

> **Note:** Specifications as of Feb 2026. SM100 (Blackwell/B200) specs omitted — not yet finalized.

| Spec | SM75 (T4) | SM80 (A100) | SM89 (L40S) | SM90 (H100) | CDNA3 (MI300X) | Apple M3 |
|---|---|---|---|---|---|---|
| SMs/CUs | 40 | 108 | 142 | 132 | 304 CUs | 10 cores |
| Shared Mem/SM | 64 KB | 164 KB | 100 KB | 228 KB | 64 KB LDS | 32 KB |
| Max Threads/SM | 1024 | 2048 | 1536 | 2048 | 2048 | varies |
| Warp/Wavefront | 32 | 32 | 32 | 32 | 64 | 32 (SIMD) |
| BF16 Support | No | Yes | Yes | Yes | Yes | Limited |
| Tensor Cores | Gen 1 | Gen 3 | Gen 4 | Gen 4 | MFMA | None |
| TMA Support | No | No | No | Yes | No | No |

#### Precision Decision Guide

Choose the right data type for your workload:

| Data Type | Use When | Accumulator | Notes |
|---|---|---|---|
| FP32 | Baseline correctness, debugging | FP32 | Widest dynamic range |
| FP16 | Inference, memory-bound kernels | FP32 | 2x throughput vs FP32 |
| BF16 | Training, wider exponent range | FP32 | Same range as FP32, less precision |
| FP8 (E4M3) | Inference throughput (SM89+) | FP32 | 4x throughput vs FP16 |
| FP8 (E5M2) | Training gradients (SM89+) | FP32 | Wider range than E4M3 |

**Key rule:** Always accumulate in FP32 to avoid catastrophic cancellation. Cast to storage dtype only for final writes.

#### Mixed-Precision Accumulation Pattern

```mojo

fn mixed_precision_dot[dtype: DType, N: Int](
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
) -> Float32:
    """Dot product with FP32 accumulation regardless of input dtype."""
    var acc = Float32(0)
    for i in range(N):
        acc += a[i].cast[DType.float32]() * b[i].cast[DType.float32]()
    return acc
```

**When to use mixed precision:**
- **Always** for reductions (sum, mean, variance) — FP16 accumulation loses precision rapidly
- **Always** for softmax — exp() amplifies small differences
- **Recommended** for matmul inner loop — accumulate in FP32, store in FP16/BF16

---

## Version-Specific Features

### v25.7 (old) vs v26.1+ (stable)

| Feature | v25.7 (old) | v26.1+ (stable) |
|---------|-------------|-----------------|
| **Constants** | `alias BLOCK = 256` | `comptime BLOCK = 256` (`alias` is deprecated) |
| **GPU imports** | `from gpu.id import ...` | `from gpu import ...` |
| **AddressSpace** | `_GPUAddressSpace` or `AddressSpace` | `AddressSpace` preferred (`_GPUAddressSpace` is a deprecated alias) |
| **DeviceContext** | `DeviceContext()` | `DeviceContext()` |
| **Buffer allocation** | `ctx.enqueue_create_buffer[T](size)` | Same |
| **Shared memory** | `stack_allocation[N, T, address_space=...]` | Same |

**Deprecated GPU import paths (v25.7 only, removed in v26.1+):**
- `gpu.id` - use `from gpu import thread_idx, block_idx, ...`
- `gpu.cluster` - cluster APIs consolidated into `gpu` module
- `gpu.mma` - matrix multiply-accumulate moved to `gpu` module

**v25.7 syntax (deprecated):**
```mojo
# v25.7 (old)
from gpu.id import thread_idx, block_idx  # Deprecated path
from gpu.host import DeviceContext

fn gpu_kernel():
    comptime BLOCK_SIZE = 256
    # ...
```

**v26.1+ syntax (current stable):**
```mojo
# v26.1+ (current stable)
from gpu import thread_idx, block_idx, block_dim, grid_dim, global_idx, barrier
from gpu.host import DeviceContext

fn gpu_kernel():
    comptime BLOCK_SIZE = 256  # alias is deprecated; use comptime
    var tid = global_idx.x   # Equivalent to: block_idx.x * block_dim.x + thread_idx.x
    # ...
```

**Block-level collective operations (v26.1+):**
```mojo
# nocompile

from gpu.primitives.block import sum, max, min, broadcast, prefix_sum

comptime TPB = 256  # Must match your kernel's block_dim

# Block-wide reductions (handle warp+shared memory reduction internally)
# IMPORTANT: Always specify [block_size=N] for portable, correct code
var block_sum = sum[block_size=TPB](my_val)       # Sum across all threads in block
var block_max = max[block_size=TPB](my_val)       # Maximum across all threads
var shared = broadcast[block_size=TPB](my_val, 0) # Broadcast from thread 0 to all threads
var scan = prefix_sum[block_size=TPB](my_val)     # Inclusive prefix sum across block
```

**Notes:**
- Use `comptime` for compile-time constants (`alias` is deprecated in nightly)
- `_GPUAddressSpace` is a deprecated alias for `AddressSpace` in v26.1+; use `AddressSpace` directly (part of prelude)
- Core APIs (`DeviceContext`, `barrier()`) are stable across all versions

---

### GPU Kernel Profiling

**Essential profiling techniques:**

```mojo
# nocompile
# Time a GPU kernel (must synchronize to get accurate timing)
from time import perf_counter_ns

var start = perf_counter_ns()
ctx.enqueue_function[my_kernel, my_kernel](grid_dim, block_dim, args...)
ctx.synchronize()  # Wait for GPU to finish
var elapsed_us = (perf_counter_ns() - start) / 1000
print("Kernel time:", elapsed_us, "μs")
```

**Platform-specific tools:**
- **NVIDIA:** `nsys profile ./my_program` (Nsight Systems), `ncu ./my_program` (Nsight Compute)
- **Apple Metal:** Xcode GPU Frame Capture, Metal System Trace in Instruments
- **AMD:** `rocprof ./my_program`, `omniperf`

---

## Related Patterns

- [`gpu-warp-sync.md`](gpu-warp-sync.md) — Barrier and synchronization patterns
- [`gpu-memory-access.md`](gpu-memory-access.md) — TMA, prefetch, and swizzle patterns
- [`gpu-tensor-cores.md`](gpu-tensor-cores.md) — Tensor core programming for SM90/SM100
- [`gpu-warp-sync.md`](gpu-warp-sync.md) — Warp primitives and reduction patterns


---

## References

- [Mojo GPU Fundamentals](https://docs.modular.com/mojo/manual/gpu/fundamentals)
- [Mojo GPU Block and Warp](https://docs.modular.com/mojo/manual/gpu/block-and-warp)
