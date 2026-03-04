---
title: Structured GPU Kernel Architecture
description: Three-component kernel pattern (ScatterGather, RingBuffer, TileOp) for maintainable high-performance GPU code with 48% code reduction
impact: HIGH
category: gpu
maturity: beta
tags: [gpu, architecture, scattergather, ringbuffer, tileop, pipeline, fp8, sm90, sm100, cdna4]
error_patterns:
  - "Layout heap allocation"
  - "readfirstlane"
  - "per-K accumulation"
  - "RuntimeLayout"
  - "rebind"
  - "bitcast"
  - "SGPR pointer"
  - "s_barrier vs barrier"
---

# Structured GPU Kernel Architecture

**Category:** gpu | **Impact:** HIGH

Structured kernel architecture organizes complex GPU kernels into three composable components: **ScatterGather** (data movement), **RingBuffer** (pipeline coordination), and **TileOp** (computation). This pattern achieves 48% code reduction while maintaining equal performance, and enables AI-assisted kernel generation through consistent, semantic abstractions.

## API Availability

> **Note:** All APIs shown in this pattern are available in the **Mojo nightly toolchain** (v26.2+). Some APIs may not yet be documented on the public docs site, but they are importable and usable.

| API | Import | Notes |
|-----|--------|-------|
| `DeviceContext` | `from gpu.host import DeviceContext` | Device and context management |
| `barrier` | `from gpu.sync import barrier` | Block-level synchronization |
| `thread_idx`, `block_idx` | `from gpu import thread_idx, block_idx` | Thread/block indexing |
| `LayoutTensor` | `from layout import LayoutTensor` | Type-safe tensor with address space |
| `Layout`, `RuntimeLayout` | `from layout import Layout` | Compile-time layout computation |
| `SharedMemBarrier` | `from layout.tma_async import SharedMemBarrier` | Pipeline barrier primitives |
| `RingBuffer` | `from .ring_buffer import RingBuffer` | Ring buffer abstraction |
| `TensorCoreAsync` | `from layout.tensor_core_async import TensorCoreAsync` | SM90 warp group MMA (WGMMA) |
| `TensorCore` | `from layout.tensor_core import TensorCore` | AMD MFMA wrapper |
| `TMATensorTile` | `from layout.tma_async import TMATensorTile` | TMA tensor tile descriptor |
| `ScatterGatherAmd` | `from linalg.structuring import ScatterGatherAmd` | AMD DRAM↔register data movement |
| `SMemTile`, `RegTile` | `from linalg.structuring import SMemTile, RegTile` | Shared memory and register tile type aliases |
| `SharedMemoryManager` | `from linalg.structuring import SharedMemoryManager` | Shared memory bump allocator |

> **Porting guides:** If you're coming from CUDA, CuTe, or ROCm, see [`gpu-porting-cuda.md`](gpu-porting-cuda.md), [`gpu-porting-cute.md`](gpu-porting-cute.md), or [`gpu-porting-rocm.md`](gpu-porting-rocm.md) for side-by-side examples.

## Core Concepts

### The Three-Component Architecture

Structured kernels decompose GPU workloads into three orthogonal concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Global Memory (DRAM)                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
              ┌───────────────────────────────┐
              │      ScatterGather            │
              │   (TMA/DMA Load Operations)   │
              └───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Shared Memory (SMEM/LDS)                            │
│         ← RingBuffer manages tile staging →                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
              ┌───────────────────────────────┐
              │          TileOp               │
              │   (Matrix Unit Operations)    │
              └───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           Registers / Tensor Memory (TMEM)                       │
│                    Accumulators                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
              ┌───────────────────────────────┐
              │          TileOp               │
              │    (Store to SMEM)            │
              └───────────────────────────────┘
                              ↓
              ┌───────────────────────────────┐
              │      ScatterGather            │
              │   (TMA/DMA Store Operations)  │
              └───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Global Memory (DRAM)                         │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Hardware Mapping |
|-----------|---------------|------------------|
| **ScatterGather** | Data movement between memory hierarchies | TMA (NVIDIA), buffer resources (AMD) |
| **RingBuffer** | Producer-consumer synchronization | mbarriers (NVIDIA), shared barriers (AMD) |
| **TileOp** | Tile-based computation | tcgen05_mma (SM100), wgmma (SM90), mfma (AMD) |

### Benefits of Structured Architecture

| Metric | Monolithic | Structured | Improvement |
|--------|------------|------------|-------------|
| Code duplication | ~1,500 lines | ~200 lines | 87% reduction |
| Shared components | 2 | 8+ | 4x reuse |
| Cross-variant changes | Touch all kernels | Touch one component | Isolation |
| AI-assisted generation | Difficult | Consistent patterns | Enabled |

## Common Patterns

### 1. ScatterGather: Data Movement Abstraction

**When:** Loading tiles from global memory to shared memory with automatic masking and swizzling.

**Architecture-Specific Implementations:**

| Architecture | Load Mechanism | Store Mechanism |
|--------------|----------------|-----------------|
| NVIDIA SM100 | TMA descriptors, `tcgen05_ld/st` | TMA store |
| NVIDIA SM90 | `cp_async`, TMA operations | TMA store |
| AMD MI3xx | `load_to_lds`, buffer resources | Global store |

```mojo
# nocompile

# ScatterGather abstraction for tile loading
struct ScatterGather[
    arch: Architecture,
    tile_shape: TileShape,
    dtype: DType,
]:
    """Handles data movement between global and shared memory."""

    fn load_tile(
        self,
        src: UnsafePointer[Scalar[dtype]],
        dst: LayoutTensor[dtype, ..., AddressSpace.SHARED],
        barrier: UnsafePointer[SharedMemBarrier, address_space=AddressSpace.SHARED],
        expected_bytes: Int,
    ):
        """Load tile with automatic masking and layout transformation."""
        barrier.expect_bytes(expected_bytes)

        @parameter
        if arch == Architecture.SM100:
            # TMA descriptor-based load
            self.tma_load(src, dst, barrier)
        elif arch == Architecture.SM90:
            # cp_async with TMA
            self.cp_async_load(src, dst, barrier)
        else:  # AMD
            # Buffer resource load to LDS
            self.buffer_load_to_lds(src, dst)
            barrier.arrive()
```

### 2. RingBuffer: Producer-Consumer Pipeline

**When:** Coordinating asynchronous data loading with computation to hide memory latency.

**Do:** Use explicit producer-consumer semantics with barrier-based handoff.

```mojo
# nocompile

# Producer-consumer pipeline pattern
struct ProducerConsumerPipeline[
    Payload: TilePayload,
    num_stages: Int,
]:
    """Ring buffer for producer-consumer synchronization."""

    var barriers: UnsafePointer[MbarType.StorageType, AddressSpace.SHARED]
    var payload: Payload
    var stage: UInt32

    fn producer(self) -> ProducerContext:
        """Get producer context for loading tiles."""
        return ProducerContext(self)

    fn consumer(self) -> ConsumerContext:
        """Get consumer context for processing tiles."""
        return ConsumerContext(self)


# Usage pattern - Producer warp
with input_pipeline.producer() as producer:
    for k in range(0, num_k_iters, k_group_size):
        with producer.acquire() as stage:
            # Signal expected bytes to barrier
            stage.expect_bytes(expected_bytes)
            # Get tile storage for this stage
            var tiles = stage.payload().get_tiles(stage.stage())
            # Load tiles via ScatterGather
            scatter_gather.load_tile(src_a, tiles.a_tile, stage.barrier())
            scatter_gather.load_tile(src_b, tiles.b_tile, stage.barrier())


# Usage pattern - Consumer warp (MMA)
with input_pipeline.consumer() as consumer:
    for k in range(0, num_k_iters, k_group_size):
        with consumer.acquire() as stage:
            # Get tiles loaded by producer
            var tiles = stage.payload().get_tiles(stage.stage())
            # Execute matrix multiply
            tile_op.mma(tiles.a_tile, tiles.b_tile, accumulator)
```

**Critical:** Producer steps AFTER work, consumer steps BEFORE work (software pipelining):

```mojo
# nocompile

# Producer: step after completing work
while has_work():
    with work_iter.next() as current:
        do_load(current)
    # fetch_next + step() happen in __exit__

# Consumer (MMA): step before work for prefetch overlap
while has_work():
    with work_iter.next_prefetch():
        do_mma(work_iter.work_info)
    # prefetched value assigned in __exit__
```

### 3. TileOp: Computation Abstraction

**When:** Performing tile-based matrix operations with architecture-specific matrix units.

**Architecture-Specific Operations:**

| Architecture | Instruction | Accumulator Storage | Tile Size |
|--------------|-------------|---------------------|-----------|
| NVIDIA SM100 | `tcgen05_mma` | TMEM (256KB/SM) | 64×256×K |
| NVIDIA SM90 | `wgmma` | Registers | 64×N×K |
| AMD MI3xx | `mfma` | Registers | 16×16×32 (BF16) |

```mojo
# nocompile

# TileOp abstraction for matrix computation
struct TileOp[
    arch: Architecture,
    mma_shape: MMAShape,
    dtype: DType,
    accum_dtype: DType,
]:
    """Handles tile-based matrix multiplication."""

    fn mma(
        self,
        a_tile: ATileType,
        b_tile: BTileType,
        inout accumulator: AccumulatorType,
    ):
        """Execute matrix multiply-accumulate on architecture-specific units."""

        @parameter
        if arch == Architecture.SM100:
            # SM100: tcgen05_mma with TMEM accumulators
            tcgen05_mma[mma_shape](
                accumulator.tmem_offset,
                a_tile,
                b_tile,
            )
        elif arch == Architecture.SM90:
            # SM90: wgmma with register accumulators
            wgmma_async[mma_shape](
                accumulator.registers,
                a_tile.descriptor(),
                b_tile.descriptor(),
            )
        else:  # AMD
            # AMD: mfma with register accumulators
            mfma[mma_shape](
                accumulator.registers,
                a_tile.to_registers(),
                b_tile.to_registers(),
            )
```

### 4. Parameterized Tile Payload Pattern

**When:** Supporting multiple kernel variants (standard matmul, block-scaled, FP8) with shared infrastructure.

**Do:** Use composition to separate synchronization from payload structure.

```mojo
# nocompile

# Trait for all tile payloads
trait TilePayload(TrivialRegisterPassable):
    """Marker trait for tile payload types."""
    pass


# Standard matmul payload
struct StandardTilePayload[...](TilePayload, TrivialRegisterPassable):
    var a_tiles: UnsafePointer[ATileStorage, AddressSpace.SHARED]
    var b_tiles: UnsafePointer[BTileStorage, AddressSpace.SHARED]

    fn get_tiles(self, stage: UInt32) -> Tuple[ATile, BTile]:
        return (self.a_tiles[stage], self.b_tiles[stage])


# Block-scaled matmul payload (FP8 with per-block scales)
struct BlockScaledTilePayload[...](TilePayload, TrivialRegisterPassable):
    var a_tiles: UnsafePointer[ATileStorage, AddressSpace.SHARED]
    var b_tiles: UnsafePointer[BTileStorage, AddressSpace.SHARED]
    var sfa_tiles: UnsafePointer[SFATileStorage, AddressSpace.SHARED]  # A scales
    var sfb_tiles: UnsafePointer[SFBTileStorage, AddressSpace.SHARED]  # B scales

    fn get_tiles(self, stage: UInt32) -> Tuple[ATile, BTile, SFATile, SFBTile]:
        return (
            self.a_tiles[stage],
            self.b_tiles[stage],
            self.sfa_tiles[stage],
            self.sfb_tiles[stage],
        )


# Generic pipeline works with any payload type
struct InputTilePipeline[
    Payload: TilePayload,
    num_stages: Int,
]:
    """Pipeline parameterized by payload type."""
    var barriers: UnsafePointer[MbarStorage, AddressSpace.SHARED]
    var payload: Payload
```

### 5. RAII Warp Lifecycle Pattern

**When:** Managing warp-specialized resources (TMEM allocation, pipeline stages) with automatic cleanup.

```mojo

# MMA warp context with RAII resource management
struct MmaWarpContext:
    var tmem: TmemAllocation
    var output_pipeline: OutputPipeline
    var dealloc_barrier: TmemDeallocBarrier

    fn __enter__(self) -> Self:
        # Allocate TMEM, signal epilogue warp
        self.tmem.allocate()
        self.output_pipeline.signal_ready()
        return self

    fn __exit__(self):
        # Wait for epilogue to finish reading, deallocate TMEM
        self.dealloc_barrier.wait()
        self.tmem.deallocate()


# Usage
with MmaWarpContext(tmem, output_pipeline, dealloc_barrier) as ctx:
    while work_iter.has_work():
        process_tile(ctx.tmem, ...)
    # __exit__ automatically handles cleanup
```

### 6. Address-Space-Qualified Methods

**When:** Accessing shared memory (SMEM/LDS) with correct compiler instruction generation.

**Critical Pattern:** Use `ref[AddressSpace.SHARED] self` to ensure compiler generates optimal load/store instructions.

```mojo
# nocompile

struct B200MatmulSmem[...]:
    var a_tiles_storage: Self.ATileArray.StorageType
    var b_tiles_storage: Self.BTileArray.StorageType

    @always_inline
    fn a_tiles(ref[AddressSpace.SHARED] self) -> Self.ATileArray:
        """Access A tiles with SMEM-qualified pointer."""
        return Self.ATileArray(self.a_tiles_storage)

    @always_inline
    fn b_tiles(ref[AddressSpace.SHARED] self) -> Self.BTileArray:
        """Access B tiles with SMEM-qualified pointer."""
        return Self.BTileArray(self.b_tiles_storage)

# Usage - compiler generates ld.shared instruction (fast)
ref smem = external_memory[AddressSpace.SHARED, SmemType](smem_addr)[]
var tiles = smem.a_tiles()  # ld.shared instruction (~30 cycles)

# Without address space qualification, compiler might generate:
# ld.global instruction (~500 cycles) - 16x slower!
```

**Why Necessary:**

| Aspect | With Qualification | Without Qualification |
|--------|-------------------|---------------------|
| Instruction | `ld.shared.b32` | `ld.global.b32` (incorrect) |
| Latency | ~30 cycles | ~500 cycles |
| Type safety | Compile-time enforcement | Runtime errors possible |
| Performance | Optimal | 16x slower |

**Guidelines:**
- Always use `ref[AddressSpace.SHARED] self` for SMEM access methods
- Use `ref [AddressSpace.GLOBAL]self` for global memory access
- Address space mismatches caught at compile time
- Store pointers with proper address space: `UnsafePointer[T, AddressSpace.SHARED]`

### 7. Zero-Cost Abstractions

**When:** Building reusable kernel components without runtime overhead.

**Core Techniques for Zero-Cost Abstractions:**

```mojo
# nocompile

# 1. TrivialRegisterPassable - Lives in registers, no heap allocation
struct ProducerConsumerPipeline[num_stages: Int](TrivialRegisterPassable):
    var full: UnsafePointer[SharedMemBarrier, address_space=AddressSpace.SHARED]
    var empty: UnsafePointer[SharedMemBarrier, address_space=AddressSpace.SHARED]
    var _producer_stage: UInt32 # Single register
    var _consumer_phase: UInt32 # Single register

    # Total: 4 register values, zero heap allocation

# 2. @always_inline - Function calls evaporate
@always_inline
fn wait_producer(self):
    self.full[self._consumer_stage].wait(self._consumer_phase)
    # Inlined directly at call site - no function call overhead

# 3. comptime parameters - Compile-time evaluation
@always_inline
fn process[comptime num_stages: Int](self):
    @parameter
    for stage in range(num_stages):  # Loop unrolled at compile time
        process_stage(stage)
```

**Verification - Check Generated Assembly:**

```bash
# Compile with PTX output to verify zero overhead
mojo build --emit asm kernel.mojo

# Inspect PTX - abstractions should not appear in assembly
# - No function calls for @always_inline functions
# - No heap allocations for TrivialRegisterPassable
# - Same instruction count as hand-written code
```

**Performance Results from Production Kernels:**

| Metric | Hand-Written | Structured | Overhead |
|--------|--------------|------------|----------|
| PTX instructions | ~2,400 | ~2,400 | **0%** |
| Register usage | 255 | 255 | **0%** |
| SMEM usage | 228KB | 228KB | **0%** |
| Runtime (TFLOPS) | ~1770 | ~1770 | **0%** |

**Key Principle:** Abstractions exist only in source code, disappear completely at compile time.

### 8. FP8 Blockwise Quantization Pattern

**When:** Implementing FP8 matmul with per-block scale factors applied during the K-loop.

**Key Difference from Standard Matmul:**

| Aspect | Standard | Blockwise FP8 |
|--------|----------|---------------|
| Accumulation | Hardware TMEM/registers | Register tiles across K |
| Scale application | N/A | Per-K in CUDA cores |
| A-scales loading | N/A | TMA (special 1D per row) |
| B-scales access | N/A | Global memory (not SMEM) |
| Epilogue source | TMEM directly | Register accumulators |

```mojo
# nocompile

# Blockwise FP8 accumulation pattern
struct BlockwiseFP8Accumulator[
    accum_type: DType,
    accum_layout: Layout,
    num_stages: Int,
]:
    var upper: UpperAccumTile
    var lower: LowerAccumTile

    fn promote(
        inout self,
        b_scales: LayoutTensor[DType.float32, ..., AddressSpace.GLOBAL],
        a_scales_tile: SMemTileArray,
        epi_stage: EpilogueKStage,
        work_coord: TileCoord,
        k_iter: Int,
    ):
        """Apply scales and accumulate partial results per-K iteration."""
        # 1. Get TMEM offset from bundled stage
        var tmem_offset = epi_stage.output_stage.tmem.offset()

        # 2. Get input stage index for A-scales
        var input_stage_idx = epi_stage.input_stage_index

        # 3. Load partial from TMEM, apply scales, accumulate
        for stage in range(num_stages):
            var tmem = TmemAddress(tmem_offset + stage * stage_stride)
            var partial = tmem.load_fragment[fragment_shape]()
            TmemAddress.wait_load()

            # Read scales: A from SMEM, B from global
            var a_scale = a_scales_tile[input_stage_idx][0, row_offset]
            var b_scale = b_scales[bn, k_iter]

            # Apply scales and accumulate in registers
            self.upper[stage] += partial * a_scale * b_scale

        # 4. Signal input pipeline (lane-guarded)
        if lane_id() < cluster_size:
            epi_stage.arrive_input()
```

### 9. Block-Scaled Quantization with Hardware Acceleration

**When:** Implementing MXFP8 or NVFP4 quantization with per-block scaling factors stored in TMEM (SM100).

**Pattern - Hardware-Accelerated Scaling:**

```mojo
# nocompile

# Block-scaled payload extends standard payload
struct BlockScaledTilePayload[...](TilePayload, TrivialRegisterPassable):
    var a_tiles: UnsafePointer[ATileStorage, AddressSpace.SHARED]
    var b_tiles: UnsafePointer[BTileStorage, AddressSpace.SHARED]
    var sfa_tiles: UnsafePointer[SFATileStorage, AddressSpace.SHARED]  # A scale factors
    var sfb_tiles: UnsafePointer[SFBTileStorage, AddressSpace.SHARED]  # B scale factors

    fn get_tiles(self, stage: UInt32) -> Tuple[ATile, BTile, SFATile, SFBTile]:
        return (
            self.a_tiles[stage],
            self.b_tiles[stage],
            self.sfa_tiles[stage],
            self.sfb_tiles[stage],
        )

# MMA operation with hardware-accelerated scaling
fn mma_block_scaled[...](
    a_tile: ATile,
    b_tile: BTile,
    sfa_tile: SFATile,  # Scale factor A
    sfb_tile: SFBTile,  # Scale factor B
    accumulator: TmemAddress,
):
    """Hardware applies scales during MMA operation."""
    # SM100 tcgen05 MMA with scale factors in TMEM
    # Hardware automatically applies: output = (A * sfa) @ (B * sfb)
    tcgen05_mma_block_scaled[mma_shape](
        accumulator,
        a_tile,
        b_tile,
        sfa_tile,  # TMEM offset for A scales
        sfb_tile,  # TMEM offset for B scales
    )
```

**Code Reduction with Payload Composition:**

| Metric | Copy-Paste Approach | Payload Composition | Reduction |
|--------|---------------------|---------------------|-----------|
| Lines added | ~1,500 | ~200 | **87%** |
| Shared components | 0 (duplicate all) | All pipeline logic | **100%** |
| Maintainability | Change all variants | Change one payload | **Isolated** |

**Supported Quantization Schemes:**

| Scheme | Format | Hardware Support | Scale Granularity |
|--------|--------|------------------|-------------------|
| MXFP8 | E4M3 | SM100 (Blackwell) | Per-block (N×K blocks) |
| NVFP4 | E2M1 | SM100 (Blackwell) | Per-block |
| FP8 | E4M3/E5M2 | SM90 (Hopper), SM100 | Row/column/block |

**Performance Impact:**

| Configuration | SM100 TFLOPS | Notes |
|---------------|--------------|-------|
| BF16 baseline | ~1770 | No quantization |
| FP8 FBGEMM | ~1770 | Same performance, 2x less memory |
| Block-scaled MXFP8 | ~1700 | 4% overhead, highest accuracy |

## Hardware-Specific Patterns

### NVIDIA SM100 (Blackwell B200)

**7-Warp Specialization Model (224 threads):**

| Warps | Threads | Role | Responsibility |
|-------|---------|------|----------------|
| 0-3 | 128 | Epilogue | Load TMEM→Registers, apply epilogue, store to GMEM |
| 4 | 32 | Scheduler | Issue CLC queries, manage throttle pipeline |
| 5 | 32 | Load | TMA load A/B tiles, signal barriers |
| 6 | 32 | MMA | Allocate TMEM, execute tcgen05_mma, deallocate |

**Memory Subsystem:**

| Memory | Capacity | Latency | Use Case |
|--------|----------|---------|----------|
| TMEM | 256KB/SM | Low | MMA accumulators |
| SMEM | ~228KB | Medium | Tile staging |
| L2 | Large | Higher | Prefetch buffer |

```mojo
# nocompile

# SM100 warp-specialized kernel structure
fn sm100_matmul_kernel[...](
    ctx: DeviceContext,
    a: TensorMap,
    b: TensorMap,
    c: LayoutTensor,
):
    var warp_id = thread_idx() // 32

    if warp_id < 4:
        # Epilogue warps (0-3)
        with EpilogueWarpContext(...) as epi_ctx:
            epilogue_loop(epi_ctx, c)
    elif warp_id == 4:
        # Scheduler warp
        scheduler_loop(...)
    elif warp_id == 5:
        # Load warp
        with LoadWarpContext(...) as load_ctx:
            load_loop(load_ctx, a, b)
    else:  # warp_id == 6
        # MMA warp
        with MmaWarpContext(...) as mma_ctx:
            mma_loop(mma_ctx)
```

### NVIDIA SM90 (Hopper H100)

**Ring Buffer Pipeline Pattern:**

```mojo
# nocompile

# SM90 ring buffer usage
with ring_buffer.producer() as producer:
    while has_work():
        with producer.get_tiles() as tiles:
            # Async copy with cp_async
            cp_async(tiles.a_tile, src_a, barrier)
            cp_async(tiles.b_tile, src_b, barrier)


with ring_buffer.consumer() as consumer:
    while has_work():
        with consumer.get_tiles() as tiles:
            # Warp group MMA
            wgmma_async(accumulator, tiles.a_tile, tiles.b_tile)
```

### AMD MI355X (CDNA4)

**Critical Hardware Differences:**

| Aspect | NVIDIA | AMD |
|--------|--------|-----|
| Wave/warp size | 32 | 64 |
| Register allocation | Dynamic | Static, evenly partitioned |
| Matrix operands | From SMEM | From registers only |
| Async memory | TMA descriptors | `load_to_lds` with buffer resources |
| Shared memory | ~228KB | ~160KB |

**8-Wave Ping-Pong Pattern (Optimal for CDNA4):**

```
SIMD Layout:
  SIMD 0: Wave 0 (Group 0), Wave 4 (Group 1)
  SIMD 1: Wave 1 (Group 0), Wave 5 (Group 1)
  SIMD 2: Wave 2 (Group 0), Wave 6 (Group 1)
  SIMD 3: Wave 3 (Group 0), Wave 7 (Group 1)

Phase Alternation:
  Phase A: Group 0 computes, Group 1 loads
  Phase B: Group 0 loads, Group 1 computes
```

**Why 8-wave ping-pong:** AMD's static register allocation means 8 waves with only 4 active wastes 50% register capacity. Ping-pong uses all 8 waves productively.

```mojo
# nocompile

# AMD 8-wave ping-pong pattern
fn amd_matmul_kernel[...](
    ctx: DeviceContext,
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    c: UnsafePointer[Scalar[dtype]],
):
    var wave_id = thread_idx() // 64
    var group = wave_id // 4  # 0 or 1
    var phase = 0

    while has_work():
        if (group == 0) == (phase == 0):
            # Compute phase
            mfma[16, 16, 32](accumulator, a_regs, b_regs)
        else:
            # Load phase
            load_to_lds(a_lds, a_ptr)
            load_to_lds(b_lds, b_ptr)

        s_barrier()  # Synchronize all waves
        phase = 1 - phase  # Alternate
```

**AMD Synchronization Primitives:**

| Primitive | Scope | Purpose |
|-----------|-------|---------|
| `s_waitcnt[vmcnt=N]()` | Per-wave | Wait for global memory ops |
| `s_waitcnt[lgkmcnt=N]()` | Per-wave | Wait for LDS operations |
| `s_barrier()` | Workgroup | All waves synchronized |
| `schedule_barrier()` | Compiler | Prevent instruction reordering |

### Apple Metal (GPU)

**Important:** Apple structured kernels target **Metal GPU** compute, not CPU NEON SIMD. These are distinct architectures.

| Aspect | Apple Metal GPU | NEON CPU |
|--------|----------------|----------|
| Target hardware | GPU shader cores | CPU SIMD units |
| Thread model | GPU threads/threadgroups | CPU vector lanes |
| Memory | GPU VRAM, threadgroup memory | CPU cache hierarchy |
| Programming model | Metal Shading Language | ARM NEON intrinsics |

**Structured Kernel Components for Metal:**

```mojo
# nocompile

# Apple Metal structured matmul kernel
struct AppleMatmulKernel[
    tile_m: Int,
    tile_n: Int,
    tile_k: Int,
    dtype: DType,
]:
    """Metal GPU matmul using structured pattern."""

    fn kernel(
        self,
        a: UnsafePointer[Scalar[dtype]],
        b: UnsafePointer[Scalar[dtype]],
        c: UnsafePointer[Scalar[dtype]],
        m: Int, n: Int, k: Int,
    ):
        # TileLoader: Load tiles to threadgroup memory (SMEM equivalent)
        var tile_loader = MetalTileLoader[tile_m, tile_k, dtype]()
        var a_tile = tile_loader.load(a, thread_idx())

        var tile_loader_b = MetalTileLoader[tile_k, tile_n, dtype]()
        var b_tile = tile_loader_b.load(b, thread_idx())

        # TileOp: Outer product accumulation pattern
        var accumulator = register_tile[tile_m, tile_n, dtype]()
        tile_op_outer_product(a_tile, b_tile, accumulator)

        # Store results
        store_tile(c, accumulator, thread_idx())
```

**Metal-Specific Patterns:**

| Component | Metal Implementation | Notes |
|-----------|---------------------|-------|
| ScatterGather | Standard memory copy to threadgroup | No specialized DMA like TMA |
| RingBuffer | Barrier-based synchronization | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| TileOp | Outer product loops | No hardware matrix units (use vectorization) |

**Performance Characteristics:**

| Operation | Apple M3 Max | Notes |
|-----------|--------------|-------|
| FP32 matmul | ~3-5 TFLOPS | Depends on tile size optimization |
| Memory bandwidth | ~400 GB/s | Unified memory architecture |
| Threadgroup memory | 32KB | Limited vs NVIDIA SMEM |

**File Locations (Reference):**
```
max/kernels/src/linalg/matmul/gpu/apple_structured/
└── __init__.mojo            # Module stub (implementation in development)
```

> **Note:** Apple Metal structured kernel support is in active development. The directory currently contains only the module init file.

## Performance Results

### Production Kernel Metrics

**NVIDIA SM100 (Blackwell B200) - BF16 Matmul:**

| Metric | Monolithic Kernel | Structured Kernel | Improvement |
|--------|------------------|-------------------|-------------|
| Total lines of code | 14,683 | 7,634 | **-48%** |
| Main kernel file | 3,721 | 1,843 | **-50%** |
| Throughput (TFLOPS) | ~1770 | ~1770 | **Parity** |
| Code to add FP8 | ~1,500 lines | ~200 lines | **-87%** |
| Shared components | 2 | 8+ | **4x reuse** |

**AMD MI355 (CDNA4) - 8-Wave Ping-Pong:**

| Configuration | Throughput | Memory Bandwidth | Register Utilization |
|---------------|------------|------------------|---------------------|
| BF16 matmul | ~1.55 TFLOPS | ~1.5 TB/s | ~100% (8-wave) |
| FP8 matmul | ~2.72 TFLOPS | ~2.7 TB/s | ~100% (8-wave) |
| 4-wave (legacy) | ~0.8 TFLOPS | ~0.75 TB/s | ~50% (wasted) |

**Apple Metal (M3 Max) - GPU Matmul:**

| Configuration | Throughput | Notes |
|---------------|------------|-------|
| FP32 matmul (optimized) | ~3-5 TFLOPS | Tile size dependent |
| Memory bandwidth | ~400 GB/s | Unified memory |

### Maintainability Gains

| Task | Before (Monolithic) | After (Structured) | Time Savings |
|------|---------------------|-------------------|--------------|
| Understand codebase | 2-3 weeks | 2-3 days | **10x faster** |
| Add quantization variant | 3-5 days | 4-6 hours | **6x faster** |
| Fix cross-variant bug | Touch all kernels | Touch one component | **Isolated** |
| Onboard new engineer | Weeks | Days | **Faster** |

### Zero-Cost Verification

**PTX Analysis (NVIDIA SM100):**

| Metric | Hand-Written PTX | Structured Kernel PTX | Overhead |
|--------|------------------|----------------------|----------|
| Instruction count | ~2,400 | ~2,400 | **0%** |
| Register usage | 255 | 255 | **0%** |
| SMEM usage | 228KB | 228KB | **0%** |
| Function calls | 0 | 0 | **0%** (inlined) |

**Conclusion:** Structured architecture achieves the "holy trinity" of GPU programming: **Performance + Productivity + Portability** with zero runtime cost.

## Decision Guide

| Scenario | Recommended Approach |
|----------|---------------------|
| New complex kernel from scratch | Use 3-component architecture |
| Refactor monolithic kernel | Extract ScatterGather first, then RingBuffer, then TileOp |
| Add FP8 support to existing kernel | Use parameterized TilePayload pattern |
| Port NVIDIA kernel to AMD | Keep ScatterGather/TileOp interfaces, reimplement with AMD primitives |
| Debug pipeline stalls | Check RingBuffer barrier signaling order |
| Optimize memory bandwidth | Focus on ScatterGather coalescing and swizzle patterns |
| Apple Metal kernel | Use structured pattern with outer product TileOp |

## Quick Reference

### Component Checklist

**ScatterGather:**
- [ ] Uses architecture-appropriate DMA (TMA/cp_async/load_to_lds)
- [ ] Handles boundary masking automatically
- [ ] Applies swizzle for bank conflict avoidance
- [ ] Signals barrier with expected bytes
- [ ] Methods use correct address space qualifiers (`ref[AddressSpace.SHARED] self`)

**RingBuffer:**
- [ ] Producer signals AFTER loading
- [ ] Consumer waits BEFORE consuming
- [ ] Stage count matches pipeline depth
- [ ] Barrier types match architecture (mbarrier vs shared)
- [ ] Components use TrivialRegisterPassable (no heap allocation)

**TileOp:**
- [ ] Uses architecture-specific MMA instruction
- [ ] Accumulator storage matches hardware (TMEM vs registers)
- [ ] Tile shapes align with MMA requirements
- [ ] Handles accumulator lifecycle (alloc/dealloc for TMEM)
- [ ] Functions marked `@always_inline` for zero overhead

**Zero-Cost Abstractions:**
- [ ] Components inherit from TrivialRegisterPassable
- [ ] Hot path functions marked `@always_inline`
- [ ] Compile-time parameters use `comptime` or `@parameter`
- [ ] Verify PTX shows no function call or heap overhead

### Barrier Naming Convention

| Name | Purpose |
|------|---------|
| `input_barriers()` | Producer-consumer for TMA→MMA |
| `accum_barriers()` | Producer-consumer for MMA→Epilogue |
| `clc_mbars_full/empty()` | Scheduler CLC queries |
| `tmem_dealloc()` | TMEM lifecycle barrier |

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Layout heap allocation on GPU | Using `Layout(idx)` dynamically | Use `RuntimeLayout[layout]()(idx)` |
| Non-uniform pointer error | VGPR pointer where SGPR expected | Use `readfirstlane()` for uniform pointers |
| Barrier deadlock | Wrong barrier type on AMD | Use `s_barrier()` for control-flow, `barrier()` for memory fence |
| Slow SMEM access (16x) | Missing address space qualification | Use `ref[AddressSpace.SHARED] self` for SMEM methods |
| Env var not working | Using `get_defined_bool` for runtime flag | `get_defined_bool` reads values at compile time and bakes them into the binary; cached `.mojopkg` files serve stale values. Use `getenv()` for runtime flags or `mojo -D FLAG=value` for compile-time defines |
| Type mismatch in generic | Compiler can't prove type equality | Use `rebind[TargetType]()` for same-layout types |
| Pointer element type error | Need to reinterpret pointer type | Use `bitcast` for pointer element conversion |
| TMEM not deallocated | Missing dealloc barrier wait | Ensure MMA warp waits on dealloc barrier in __exit__ |
| 50% register waste on AMD | 8 waves with 4 idle | Use 8-wave ping-pong pattern |
| Payload not extensible | Copy-pasting entire kernel | Use TilePayload trait with composition |
| Runtime overhead | Not using TrivialRegisterPassable | Ensure components use TrivialRegisterPassable |
| Function call overhead | Missing @always_inline | Add @always_inline to hot path functions |

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **Constants** | `alias` or `comptime` | Both work in v26.1+ |
| **Compile-time params** | `@parameter` decorator | Stable |
| **Pipeline context** | Manual enter/exit | Stable |

**Example (v26.1+):**

```mojo
# nocompile

comptime NUM_STAGES = 3
comptime TILE_SIZE = 128

fn kernel[dtype: DType]():
    @parameter
    if dtype == DType.float16:
        process_fp16()
```

**Notes:**
- Use `comptime` for compile-time constants (`alias` is deprecated in nightly)
- `@parameter` for compile-time branching is stable
- Pipeline context managers work identically across versions

## Related Patterns

- [`gpu-fundamentals.md`](gpu-fundamentals.md) — Thread hierarchy and memory model basics
- [`gpu-warp-sync.md`](gpu-warp-sync.md) — Barrier primitives used by RingBuffer
- [`gpu-memory-access.md`](gpu-memory-access.md) — TMA and memory coalescing for ScatterGather
- [`gpu-tensor-cores.md`](gpu-tensor-cores.md) — MMA instructions used by TileOp
- [`gpu-amd.md`](gpu-amd.md) — AMD-specific 8-wave ping-pong details

## References

### Official Documentation

- [NVIDIA Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA Hopper Architecture Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [AMD CDNA3 Architecture](https://www.amd.com/en/technologies/cdna)

### Reference Implementations

**Note:** The patterns documented here are based on production structured kernel implementations in the MAX repository. These APIs are available in the Mojo nightly toolchain (v26.2+).

**NVIDIA SM100 Structured Kernels:**
```
max/kernels/src/linalg/matmul/gpu/sm100_structured/
├── structured_kernels/
│   ├── pipeline.mojo              # Producer-consumer pipeline
│   ├── tile_pipeline.mojo         # Input/output tile staging
│   ├── tile_loader.mojo           # TMA-based loading
│   ├── output_writer.mojo         # Output operations
│   ├── tmem.mojo                  # Tensor memory management
│   ├── barriers.mojo              # Type-safe barrier wrappers
│   └── tile_scheduler.mojo        # CLC work distribution
├── default/                       # Standard FP8/BF16 matmul
├── block_scaled/                  # MXFP8/NVFP4 per-block scaling
├── blockwise_fp8/                 # Register-based per-K scaling
└── grouped_block_scaled_1d1d/     # Grouped matmul variants
```

**Apple Structured Kernels:**
```
max/kernels/src/linalg/matmul/gpu/apple_structured/
└── __init__.mojo                  # Module stub
```

> **Note:** Apple Metal structured kernel support is in active development. The directory currently contains only the module init file. The structured kernel patterns shown in the Apple Metal section above are design targets, not yet implemented as separate files.

**Documentation (Internal):**
```
Kernels/cursor_docs/
├── anatomy/
│   └── anatomy_of_structured_kernels.md          # 1808 lines
├── blog_series/
│   ├── part1_why_structured_kernels.md          # Motivation
│   ├── part2_three_pillars.md                   # Component details
│   ├── part3_platform_implementations.md         # Platform specifics
│   └── part4_composition_unification.md          # Extensibility
└── sm100/
    ├── sm100_structured_architecture.md          # Implementation guide
    └── structured_kernel_porting_recipe.md
```

**Key Insight:** The structured kernel architecture demonstrates that maintainable GPU code with zero runtime overhead is achievable through careful abstraction design. The patterns shown here are language-agnostic and applicable to any GPU kernel development.
