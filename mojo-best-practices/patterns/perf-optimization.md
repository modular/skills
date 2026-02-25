---
title: General Optimization & Memory Patterns
description: Comprehensive guide to caching strategies, lazy loading, mmap patterns, compile-time computation, buffer management, memory alignment, data layout, prefetching, tiling, and avoiding overhead
impact: HIGH
category: perf
tags: [caching, mmap, compile-time, buffers, allocation, python-interop, memory, alignment, layout, prefetch, cache, tiling, accumulators]
error_patterns:
  - "slow startup"
  - "allocation overhead"
  - "compilation time"
  - "memory mapped"
  - "mmap"
  - "lazy loading"
  - "buffer"
  - "cache"
  - "cache miss"
  - "alignment"
  - "unaligned"
  - "memory bandwidth"
  - "false sharing"
  - "cache line"
  - "slow memory access"
  - "prefetch"
scenarios:
  - "Implement caching strategy"
  - "Use memory-mapped files"
  - "Precompute at compile time"
  - "Manage buffers efficiently"
  - "Reduce startup overhead"
  - "Optimize model loading"
  - "Use zero-copy loading"
  - "Optimize memory access pattern"
  - "Align data for SIMD"
  - "Use prefetching"
  - "Choose stack vs heap"
  - "Implement tiled processing"
  - "Reduce cache misses"
  - "Use multiple accumulators"
consolidates:
  - perf-tokenizer-caching.md
  - perf-dimension-dependent-caching.md
  - perf-mmap-loading.md
  - perf-mmap-zero-copy.md
  - perf-avoid-python-objects.md
  - perf-benchmark-methodology.md
  - perf-compile-time-compute.md
  - perf-dynamic-allocation.md
  - perf-lazy-gpu-warmup.md
  - perf-raw-pointers.md
  - perf-trivial-type-optimization.md
  - perf-weight-preloading.md
  - perf-persistent-buffers.md
  - perf-algorithm-shortcuts.md
  - perf-ffi-overhead.md
  - perf-gpu-cache-preservation.md
  - perf-gpu-vs-cpu-tradeoffs.md
  - perf-mpsgraph-compilation-overhead.md
  - perf-memory-alignment.md
  - perf-memory-layout.md
  - perf-memory-prefetch.md
  - memory-borrow-vs-copy.md
  - perf-multiple-accumulators.md
  - perf-tiled-processing.md
  - perf-memory.md
---
<!-- PATTERN QUICK REF
WHEN: Optimizing Mojo code performance, memory layout, cache efficiency, GPU/CPU tradeoffs
KEY_TYPES: UnsafePointer, SIMD, StaticTuple, PrefetchOptions, LayoutTensor, Float32Ptr
SYNTAX:
  - comptime VALUE = expr (compile-time constants)
  - @align(N) struct (cache-line / SIMD alignment)
  - prefetch[PrefetchOptions().for_read().high_locality()](ptr)
  - data.load[width=WIDTH](i) / data.store(i, vec)
  - T.__copyinit__is_trivial (trivial type check)
  - external_call["mmap", UInt8Ptr](...) (FFI mmap)
PITFALLS: mmap leak (missing munmap), false sharing (unpadded parallel counters), prefetch too late, single accumulator dependency chain, GPU cache invalidation between phases, FFI overhead misattribution
RELATED: perf-vectorization, perf-parallelization, memory-ownership
-->

# General Optimization & Memory Patterns

**Category:** perf | **Impact:** HIGH

This pattern covers essential optimization strategies: caching, lazy loading, memory-mapped files, compile-time computation, buffer management, memory alignment, data layout, prefetching, tiling, and avoiding common overhead sources.

---

## Core Concepts

### The Optimization Hierarchy

1. **Algorithm-level** (10-100x): Choose the right algorithm, skip unnecessary work
2. **Memory access** (2-10x): Cache-friendly layouts, prefetching
3. **Parallelization** (4-16x): Multi-core, SIMD
4. **Micro-optimizations** (1.1-2x): Inline, unroll, reduce overhead

Always start from the top. A better algorithm beats micro-optimizations every time.

### Memory Hierarchy Impact

| Level | Latency | Size | Optimization |
|-------|---------|------|--------------|
| L1 Cache | ~1 ns | 32-64 KB | Sequential access, prefetch |
| L2 Cache | ~4 ns | 256-512 KB | Tiling, blocking |
| L3 Cache | ~10 ns | 4-32 MB | Data layout |
| Main Memory | ~100 ns | GBs | Minimize access |

### Cache Line Alignment

Modern CPUs fetch memory in 64-byte cache lines. Proper alignment ensures:
- Single cache line fetch for SIMD vectors
- No false sharing in parallel code
- Optimal prefetcher behavior

```mojo
# nocompile
@align(64)  # Align to cache line (64 bytes)
struct AlignedBuffer:
    var data: SIMD[DType.float32, 16]

    fn __init__(out self):
        self.data = SIMD[DType.float32, 16]()
```

---

## Common Patterns

### Pattern 1: Algorithm-Level Shortcuts

**When:** Before any micro-optimization, look for macro-optimization

**Impact:** CRITICAL (20-50% speedup by mathematically eliminating work)

**Benchmark (Mandelbrot 1920x1080):**
- Without shortcuts: 11.2 ms
- With cardioid/bulb skip: 3.9 ms (**65% faster**)

```mojo
@always_inline
fn in_cardioid_or_bulb(x0: Float64, y0: Float64) -> Bool:
    """Check if point is in main cardioid or period-2 bulb."""
    # Main cardioid: |c - 1/4| < 1/2(1 - cos(theta))
    var q = (x0 - 0.25) * (x0 - 0.25) + y0 * y0
    if q * (q + (x0 - 0.25)) <= 0.25 * y0 * y0:
        return True
    # Period-2 bulb: |c + 1| < 1/4
    if (x0 + 1.0) * (x0 + 1.0) + y0 * y0 <= 0.0625:
        return True
    return False


fn mandelbrot_optimized(x0: Float64, y0: Float64, max_iter: Int) -> Int:
    # Skip points mathematically known to be in the set
    if in_cardioid_or_bulb(x0, y0):
        return max_iter  # In the set, no need to iterate

    var x: Float64 = 0
    var y: Float64 = 0
    var iter = 0
    while x*x + y*y <= 4.0 and iter < max_iter:
        var xtemp = x*x - y*y + x0
        y = 2*x*y + y0
        x = xtemp
        iter += 1
    return iter
```

**Common algorithm shortcuts:**

| Domain | Shortcut |
|--------|----------|
| Mandelbrot | Cardioid/bulb skip, periodicity checking |
| Distance | Compare squared distances (avoid sqrt) |
| Trig | Use identities to reduce calls |
| Matrix | Exploit symmetry, skip zeros |
| Search | Binary search, skip sorted ranges |
| Graphics | Bounding box early-out |

**Trig identities for optimization:**

| Expression | Optimized Form |
|------------|----------------|
| `sin(x) * cos(x)` | `sin(2x) / 2` |
| `sin^2(x)` | `(1 - cos(2x)) / 2` |
| `sin^2(x) + cos^2(x)` | `1` (eliminate entirely!) |

---

### Pattern 2: Tokenizer and Resource Caching

**When:** Resources loaded multiple times during inference

**Impact:** HIGH (100x faster tokenization by caching)

```mojo
struct PipelineContext(Movable):
    # Cached tokenizer (loaded once, reused)
    var cached_tokenizer: Tokenizer
    var tokenizer_cached: Bool

    fn __init__(out self):
        self.cached_tokenizer = Tokenizer()
        self.tokenizer_cached = False

    fn __moveinit__(out self, deinit take: Self):
        self.cached_tokenizer = take.cached_tokenizer^
        self.tokenizer_cached = take.tokenizer_cached


fn encode_text(mut ctx: Context, prompt: String) -> Embeddings:
    var tokenizer_path = ctx.model_dir + "/tokenizer/tokenizer.json"

    # Load tokenizer once and cache
    if not ctx.tokenizer_cached:
        print("  Loading tokenizer...")
        ctx.cached_tokenizer = load_tokenizer(tokenizer_path)
        ctx.tokenizer_cached = True
    else:
        print("  Using cached tokenizer...")

    # Tokenize with cached instance
    var tokens = ctx.cached_tokenizer.tokenize(prompt)
    # ... continue with encoding
```

**Performance impact:**

| Scenario | Tokenization Time | Improvement |
|----------|------------------|-------------|
| Reload each time | 1-2s | - |
| Cached tokenizer | <10ms | 100x+ faster |

---

### Pattern 3: Dimension-Dependent Caching

**When:** Values depend only on input dimensions and are constant throughout processing

```mojo
# nocompile
struct ProcessingContext:
    # Positional encoding cache
    var pos_enc_cos: Float32Ptr
    var pos_enc_sin: Float32Ptr
    var cached_seq_len: Int

    fn ensure_positional_encoding(mut self, seq_len: Int):
        """Compute positional encoding only if dimensions changed."""
        if self.cached_seq_len == seq_len:
            return  # Cache hit

        # Cache miss - recompute
        if self.pos_enc_cos:
            self.pos_enc_cos.free()
            self.pos_enc_sin.free()

        self.pos_enc_cos = alloc[Float32](seq_len * dim)
        self.pos_enc_sin = alloc[Float32](seq_len * dim)
        compute_positional_encoding(self.pos_enc_cos, self.pos_enc_sin, seq_len)

        self.cached_seq_len = seq_len
```

**Common dimension-dependent values:**

| Value | Depends On | Cache When |
|-------|------------|------------|
| Positional encodings | seq_len | Constant per batch |
| Frequency tables | seq_len, dim | Constant per dimension |
| Causal masks | seq_len | Constant per seq_len |
| Sinusoidal embeddings | seq_len, dim | Constant per dimension |

---

### Pattern 4: Memory-Mapped Files (mmap)

**When:** Large files (100MB+) accessed multiple times, random access patterns

**Zero-copy pattern for model weights:**

```mojo
# nocompile
struct SafetensorsFile:
    var data: UInt8Ptr
    var is_mmapped: Bool

    fn get_bf16(self, tensor: TensorInfo) -> UInt16Ptr:
        """Copy version - caller owns returned memory."""
        var result = alloc[UInt16](tensor.numel())
        memcpy(result, self.get_data(tensor), tensor.numel() * 2)
        return result

    fn get_bf16_direct(self, tensor: TensorInfo) -> UInt16Ptr:
        """Zero-copy version - pointer into mmap'd region.

        Caller must NOT free. Valid while file is open.
        """
        return self.get_data(tensor).bitcast[UInt16]()
```

**Memory comparison:**

| Approach | 4GB Model Memory |
|----------|------------------|
| Copy all weights | 8GB |
| Zero-copy (mmap) | 4GB |

**mmap FFI bindings:**
```mojo
# nocompile
comptime PROT_READ: Int32 = 1
comptime MAP_PRIVATE: Int32 = 2
comptime MAP_FAILED: Int = -1

fn libc_mmap(addr: UInt8Ptr, length: Int, prot: Int32, flags: Int32,
             fd: Int32, offset: Int64) -> UInt8Ptr:
    return external_call["mmap", UInt8Ptr](addr, Int64(length), prot, flags, fd, offset)

fn libc_munmap(addr: UInt8Ptr, length: Int) -> Int32:
    return external_call["munmap", Int32](addr, Int64(length))
```

**Best practices:**
- Track mmap state for correct cleanup
- Handle MAP_FAILED
- Close fd after mmap (mmap keeps internal reference)
- Use munmap in destructor

---

### Pattern 5: Avoid Python Objects in Hot Paths

**When:** Performance-critical loops

**Impact:** CRITICAL (100-1000x speedup vs Python object operations)

**Don't:**
```mojo
# nocompile
from python import Python

fn slow_sum() -> Float64:
    var np = Python.import_module("numpy")
    var arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    var total: Float64 = 0.0
    for i in range(5):
        # Python call each iteration - extremely slow
        total += arr[i].to_float64()
    return total
```

**Do (convert at boundaries, compute natively):**
```mojo
# nocompile
fn process_numpy_data(py_array: PythonObject) -> Float64:
    # Convert to Mojo at the boundary
    var size = int(py_array.shape[0])
    var data = List[Float64](capacity=size)

    for i in range(size):
        data.append(py_array[i].to_float64())

    # Now use native Mojo operations
    return compute_statistics(data)

fn compute_statistics(data: List[Float64]) -> Float64:
    # Pure Mojo - no Python overhead
    var sum: Float64 = 0.0
    for item in data:
        sum += item
    return sum / len(data)
```

---

### Pattern 6: Compile-Time Computation

**When:** Values can be determined at compile time

**Impact:** HIGH (zero runtime cost for compile-time computable values)

```mojo
from utils import StaticTuple

@parameter
fn factorial[n: Int]() -> Int:
    @parameter
    if n <= 1:
        return 1
    else:
        return n * factorial[n - 1]()

fn main():
    # Computed at compile time - zero runtime cost
    comptime fact_10 = factorial[10]()  # 3628800
    comptime fact_5 = factorial[5]()    # 120
    print(fact_10, fact_5)

# Compile-time lookup table generation
@parameter
fn generate_squares[size: Int]() -> StaticTuple[Int, size]:
    var result = StaticTuple[Int, size]()
    @parameter
    for i in range(size):
        result[i] = i * i
    return result

comptime SQUARES = generate_squares[256]()
```

**Use comptime for:**
- Mathematical constants
- Lookup tables with fixed values
- Type definitions
- Array sizes and bounds
- Configuration that doesn't change at runtime

---

### Pattern 7: Persistent Buffers

**When:** Functions called repeatedly with same buffer sizes

**Impact:** HIGH (5-15% speedup by eliminating allocation overhead in hot loops)

**Don't (allocate per call):**
```mojo
fn layer_forward(mut model: Model, layer: Layer, seq_len: Int):
    # Allocates buffers every call - 36 layers = 36 allocations per forward
    var scores = alloc[Float32](seq_len * seq_len)
    var q_buf = alloc[Float32](seq_len * dim)

    for h in range(num_heads):
        # ... use buffers ...
        pass

    scores.free()
    q_buf.free()
```

**Do (persistent buffers in struct):**
```mojo
struct Model:
    # Persistent work buffers (allocated once)
    var work_scores: Float32Ptr
    var work_q: Float32Ptr
    var max_seq_len: Int

    fn allocate_buffers(mut self, max_seq: Int):
        self.max_seq_len = max_seq
        self.work_scores = alloc[Float32](max_seq * max_seq)
        self.work_q = alloc[Float32](max_seq * dim)


fn layer_forward(mut model: Model, layer: Layer, seq_len: Int):
    # Reuses persistent buffers - zero allocation overhead
    var scores = model.work_scores
    var q_buf = model.work_q

    for h in range(num_heads):
        # ... use buffers ...
        pass
    # No free needed - buffers persist across calls
```

**Buffer pool pattern:**
```mojo
# nocompile
@align(64)  # Cache-line alignment for SIMD
struct ModelBuffers:
    var norm_out: Float32Ptr
    var q: Float32Ptr
    var k: Float32Ptr
    var v: Float32Ptr
    var attn_out: Float32Ptr
    var mlp_gate: Float32Ptr
    var mlp_up: Float32Ptr
    var is_allocated: Bool

    fn allocate(mut self, max_seq: Int, hidden: Int, kv_dim: Int, mlp_dim: Int):
        self.norm_out = alloc[Float32](max_seq * hidden)
        self.q = alloc[Float32](max_seq * hidden)
        self.k = alloc[Float32](max_seq * kv_dim)
        self.v = alloc[Float32](max_seq * kv_dim)
        self.attn_out = alloc[Float32](max_seq * hidden)
        self.mlp_gate = alloc[Float32](max_seq * mlp_dim)
        self.mlp_up = alloc[Float32](max_seq * mlp_dim)
        self.is_allocated = True
```

This eliminates ~7 allocations per layer x 36 layers = 252 malloc/free calls per forward pass.

---

### Pattern 8: Dynamic vs Pre-Allocation

**When:** Variable-size inputs on memory-constrained systems

**Don't (pre-allocate for max):**
```mojo
# nocompile
struct Transformer:
    var work_buf: Float32Ptr

fn load() -> Transformer:
    var tf = Transformer()
    # BAD: 8.4GB pre-allocation for max 4096+512 sequence
    tf.work_buf = alloc[Float32](4608 * 3072 * 4)  # OOM on 16GB!
    return tf^
```

**Do (allocate based on actual size):**
```mojo
fn forward(tf: Transformer, img_h: Int, img_w: Int, txt_seq: Int):
    var img_seq = img_h * img_w
    var total_seq = img_seq + txt_seq

    # GOOD: Allocate for actual sequence length
    var work_buf = alloc[Float32](total_seq * hidden)
    # ... use buffer ...
    work_buf.free()
```

---

### Pattern 9: Lazy GPU Warmup

**When:** GPU operations cache weight conversions automatically

**Impact:** HIGH (3-4 second savings by avoiding redundant warmup)

**Don't (redundant explicit warming):**
```mojo
fn load_model_gpu(model: Model, mps: MPSContext):
    # Explicit pre-warming: loops through ALL weights
    print("Warming GPU weights...")
    mps.gpu_chain_begin()
    for layer in model.layers:
        mps.warm_weight_cache(layer.q_weight_bf16, q_dim * hidden)
        mps.warm_weight_cache(layer.k_weight_bf16, kv_dim * hidden)
        # ... more weights ...
    mps.gpu_chain_end()
    # Forward pass ALSO does caching, so warming was redundant!
```

**Do (rely on lazy caching):**
```mojo
# nocompile
fn load_model_gpu(model: Model, mps: MPSContext):
    # Skip explicit warming - forward pass will warm on first use
    # First forward pass warms caches inline with ~100ms overhead
    output = forward_gpu(model, input)
```

**Benchmark:**

| Approach | Load Time | Forward Time | Total |
|----------|-----------|--------------|-------|
| Explicit warming | 3.1s + 3.9s | 3.9s | **10.9s** |
| Lazy warming | 3.1s | 4.0s | **7.1s** |
| **Savings** | - | - | **3.8s (35%)** |

---

### Pattern 10: Weight Preloading

**When:** Multi-model pipelines with hidden disk I/O

**Impact:** HIGH (30-60% pipeline speedup)

```mojo
# nocompile
struct PipelineContext:
    var model_a: ModelA
    var model_a_cached: Bool
    var model_b: ModelB
    var model_b_cached: Bool

    fn __init__(out self, model_dir: String, preload: Bool) raises:
        # Load primary model (always needed)
        self.model_a = load_model_a(model_dir)
        self.model_a_cached = True

        # Preload secondary model during init (KEY OPTIMIZATION)
        if preload:
            self.model_b = load_model_b(model_dir)
            self.model_b_cached = True
        else:
            self.model_b = ModelB()
            self.model_b_cached = False
```

**Real-world impact:**

| Component | Without Preload | With Preload | Savings |
|-----------|-----------------|--------------|---------|
| Phase 1 | 5.1s | 2.0s | **60%** |
| **Pipeline** | **9.6s** | **6.5s** | **32%** |

---

### Pattern 11: Trivial Type Optimization

**When:** Generic container implementations

**Impact:** HIGH (2-10x faster using memcpy for trivial types)

```mojo
# nocompile
fn copy_array[T: Copyable](
    dest: UnsafePointer[T],
    src: UnsafePointer[T],
    count: Int,
):
    @parameter
    if T.__copyinit__is_trivial:
        # Trivial: single memcpy (uses SIMD, very fast)
        memcpy(dest=dest, src=src, count=count)
    else:
        # Non-trivial: must call copy constructor
        for i in range(count):
            (dest + i).init_pointee_copy(src[i])
```

**Available trivial flags:**

| Flag | Meaning | True For |
|------|---------|----------|
| `T.__moveinit__is_trivial` | Move is bitwise copy | Int, Float32, SIMD, Pointer |
| `T.__copyinit__is_trivial` | Copy is bitwise copy | Int, Float32, SIMD, Pointer |
| `T.__del__is_trivial` | Destructor is no-op | Int, Float32, SIMD, Pointer |

---

### Pattern 12: Raw Pointers for Maximum Performance

**When:** Hot loops where bounds checking overhead matters

**Impact:** HIGH (2-3x faster than List-based operations)

```mojo
from memory import UnsafePointer
from memory import UnsafePointer
from builtin.type_aliases import MutAnyOrigin

comptime Int64Ptr = UnsafePointer[mut=True, type=Int64, origin=MutAnyOrigin]

fn scale_simd(
    result: Int64Ptr,
    data: Int64Ptr,
    size: Int
):
    comptime WIDTH: Int = 8

    var i = 0
    while i + WIDTH <= size:
        var vec = data.load[width=WIDTH](i)
        result.store(i, vec * 2)
        i += WIDTH

    while i < size:
        result[i] = data[i] * 2
        i += 1
```

**Safety checklist:**

| Concern | Mitigation |
|---------|------------|
| Memory leak | Always call `.free()` |
| Use-after-free | Don't access after free |
| Buffer overflow | Track size manually |
| Data races | Use with parallelization carefully |

---

### Pattern 13: Proper Benchmarking

**When:** Measuring performance for optimization decisions

```mojo
# nocompile
from time import perf_counter_ns

fn benchmark_proper() -> Float64:
    """Benchmark with warmup and best-of-10 methodology."""
    # Setup - not included in timing
    var a = List[Int](capacity=1000000)
    for i in range(1000000):
        a.append(i)

    # Warmup run - warms CPU caches and JIT
    process(a)

    # Best of 10 iterations
    var best: UInt = 9223372036854775807
    for _ in range(10):
        var start = perf_counter_ns()
        process(a)
        var elapsed = UInt(perf_counter_ns() - start)
        if elapsed < best:
            best = elapsed

    return Float64(best) / 1_000_000.0  # Return ms
```

**Benchmarking checklist:**

| Step | Purpose |
|------|---------|
| Separate setup from timing | Don't measure allocation time |
| Warmup run | Fill CPU caches, trigger any JIT |
| Multiple iterations | Reduce measurement noise |
| Take best time | Eliminates OS scheduling interference |
| Verify correctness | Check result values match |

---

### Pattern 14: FFI Call Overhead Optimization

**When:** GPU operations via FFI in hot paths

**Impact:** MEDIUM — FFI overhead is real but GPU speedup (10-100x) usually outweighs it

FFI calls to C/Objective-C have per-call overhead (~0.5-1μs each). With thousands of GPU calls per inference, this adds up. However, **GPU acceleration provides 10-100x speedup over CPU, which far exceeds this overhead**.

**Critical insight: DO NOT replace GPU FFI with CPU-native code.**

```mojo
# BAD: "Native" but uses CPU BLAS - actually SLOWER
fn forward_native_cpu(...):
    cpu_linear(...)  # CPU BLAS - no FFI but slow

# GOOD: FFI but uses GPU - actually FASTER
fn forward_gpu_ffi(...):
    mps.gpu_linear_bf16(...)  # GPU via FFI - has overhead but 10x faster
```

**Optimization strategies (in order of effectiveness):**

1. **Enable weight caching** (most important - 1.8x speedup):
```mojo
# nocompile
# Cache weights across inference steps
if not ctx.model_cached:
    ctx.model = load_model(path)
    ctx.model_cached = True
```

2. **Batch/chain GPU operations** (reduce sync points):
```mojo
mps.gpu_batch_begin()
# ... multiple operations ...
mps.gpu_batch_end()  # Single sync
```

3. **Fuse operations** (reduce FFI calls):
```mojo
# nocompile
# BAD: 3 separate FFI calls
q = mps.gpu_linear(input, q_weight)
k = mps.gpu_linear(input, k_weight)
v = mps.gpu_linear(input, v_weight)

# GOOD: 1 fused FFI call
qkv = mps.gpu_fused_qkv(input, qkv_weight)
```

**Performance reference:**

| Path | Relative Time | Notes |
|------|---------------|-------|
| Native GPU (Metal) | 1x | No FFI, native GPU |
| GPU via FFI + cache | 1.2x | FFI + GPU + caching |
| GPU via FFI, no cache | 2.2x | Cache miss every step |
| Native CPU | 3.5x+ | No FFI but CPU only |

---

### Pattern 15: GPU Cache Preservation

**When:** Multi-phase GPU pipelines with weight conversion caching (e.g., BF16 to F16)

**Impact:** CRITICAL — Prevents 4+ second warmup overhead when GPU caches are invalidated

**Don't (clears cache between phases):**
```mojo
# nocompile
fn run_pipeline(mut ctx: Context):
    result1 = compute_phase1_gpu(ctx)

    # BAD: This clears ALL GPU caches including weights needed for phase 2!
    ctx.mps.reset()

    # Phase 2: Must re-convert weights (4+ seconds!)
    for step in range(num_steps):
        compute_step_gpu(ctx, step)  # First step is 5x slower
```

**Do (preserve cache when weights are shared):**
```mojo
# nocompile
fn run_pipeline(mut ctx: Context):
    result1 = compute_phase1_gpu(ctx)

    # GOOD: Only reset if subsequent phases don't need the cached weights
    if not ctx.weights_preloaded:
        ctx.mps.reset()  # Safe to reset - weights not cached yet
    else:
        print("[GPU] Keeping weight cache for next phase")

    # Phase 2: Uses cached weights (fast!)
    for step in range(num_steps):
        compute_step_gpu(ctx, step)  # All steps are fast
```

**Benchmark results:**

| Scenario | First Step Time | Steady State |
|----------|-----------------|--------------|
| With reset between phases | 5448ms | 1040ms |
| Without reset (cache preserved) | 1215ms | 1040ms |
| Savings | **4.2 seconds** | - |

---

### Pattern 16: GPU vs CPU Path Selection

**When:** Choosing between GPU FFI and "native" CPU paths

**Impact:** HIGH — GPU 10-100x faster than CPU even with FFI overhead

**Anti-Pattern: Native CPU Path**
```mojo
# "Native" path - eliminates FFI but uses CPU BLAS
fn transformer_forward_native(...) raises:
    flux_linear_nobias(q, x, weight, ...)  # CPU BLAS - no FFI but ~10-100x slower
```

**Correct: GPU FFI Path**
```mojo
# FFI GPU path - has FFI overhead but uses GPU acceleration
fn transformer_forward_gpu(...) raises:
    var q = mps.gpu_linear_bf16(x, weight_bf16, ...)  # 10-100x faster
```

**Apple Silicon Exception:** On unified memory architecture, simple memory operations (concat, copy) may be faster on CPU:

| Operation | GPU Kernel | CPU memcpy | Winner |
|-----------|------------|------------|--------|
| Matrix multiply | Fast | Slow | GPU |
| Concat/copy | 830ms overhead | 662ms | CPU (25% faster) |

**Recommendation for Apple Silicon:**
- Use GPU for compute-intensive ops (matmul, attention, convolution)
- Use CPU for memory operations (concat, transpose, copy)
- Always measure with timing to verify assumptions

---

### Pattern 17: MPSGraph Compilation Overhead

**When:** Using MPSGraph with varying configurations (batch, channels, H, W)

**Impact:** HIGH — Runtime graph compilation can be slower than CPU fallback

MPSGraph caches compiled graphs per configuration. Each unique configuration compiles a new graph at first use (50-200ms).

**Case Study (Image Decoder with ~30 unique configs):**
```
GPU Path (first run):
- Graph compilations: 30 × ~100ms = ~3000ms
- Actual GPU compute: ~200ms
- Total: ~3200ms

CPU BLAS Path:
- No compilation
- Total: ~1900ms (actually faster!)
```

**Solution: Pre-compile all configs at startup:**
```mojo
# nocompile
fn precompile_conv_graphs(mps: MPSContext, configs: List[ConvConfig]):
    """Pre-compile convolution graphs at startup."""
    for cfg in configs:
        var dummy_in = mps.alloc_tensor(cfg.in_ch * cfg.h * cfg.w)
        var dummy_w = mps.alloc_tensor(cfg.out_ch * cfg.in_ch * cfg.k * cfg.k)
        _ = mps.conv2d_native(dummy_in, dummy_w, cfg)
        dummy_in.free()
        dummy_w.free()

fn get_decoder_configs(max_resolution: Int) -> List[ConvConfig]:
    var configs = List[ConvConfig]()
    var h = max_resolution // 16 * 2
    var w = max_resolution // 16 * 2

    configs.append(ConvConfig(32, 512, h, w, 3, 1))
    for level in range(4):
        var ch = 512 if level < 2 else 256 if level == 2 else 128
        configs.append(ConvConfig(ch, ch, h, w, 3, 1))
        h *= 2
        w *= 2
    return configs^
```

**Alternative: Adaptive GPU/CPU selection:**
```mojo
# nocompile
fn conv2d_adaptive(cfg: ConvConfig, mps: MPSContext) -> Bool:
    # GPU wins when spatial > 64x64
    if cfg.h >= 64 and cfg.w >= 64:
        return mps.try_conv2d_native(output, inp, weight, bias, cfg)

    # CPU BLAS is faster for small sizes (avoids compile overhead)
    conv2d_blas(output, inp, weight, bias, cfg)
    return False
```

---

## Memory Optimization

This section covers memory-level optimizations: alignment, data layout, prefetching, multiple accumulators, tiled processing, and borrowing.

### Pattern 18: Struct Alignment for SIMD

**When:** Structs containing SIMD vectors or performance-critical data

**Common alignment values:**

| Alignment | Use Case |
|-----------|----------|
| 16 bytes | SSE vectors (128-bit) |
| 32 bytes | AVX vectors (256-bit) |
| 64 bytes | AVX-512 vectors, cache lines |
| 128 bytes | Some GPU requirements |

**Do:**
```mojo
# nocompile
@align(32)
struct GoodBuffer:
    var data: SIMD[DType.float32, 8]  # 32 bytes, properly aligned
    var header: Int8

fn process(b: GoodBuffer):
    # Aligned loads are fast and safe
    pass
```

**Don't:**
```mojo
struct BadBuffer:
    var header: Int8
    var data: SIMD[DType.float32, 8]  # Likely unaligned

fn process(b: BadBuffer):
    # Unaligned loads are slower or may crash on some architectures
    pass
```

**Prevent false sharing in parallel code:**
```mojo
# nocompile
@align(64)  # Cache line alignment
struct CacheAlignedCounter:
    var count: Int
    # Padding to fill cache line, prevent false sharing
    var _padding: SIMD[DType.int64, 7]

    fn __init__(out self):
        self.count = 0
        self._padding = SIMD[DType.int64, 7]()

    fn increment(mut self):
        self.count += 1
```

---

### Pattern 19: Data Layout (AoS vs SoA)

**When:** Batch operations on large datasets

**Benchmark (particle simulation):**
- AoS (Array of Structs): Poor cache utilization for single-field access
- SoA (Struct of Arrays): 2-10x faster for batch field operations

**Don't (AoS with strided access):**
```mojo
# nocompile
struct ParticleAoS:
    var x: Float64
    var y: Float64
    var z: Float64
    var vx: Float64
    var vy: Float64
    var vz: Float64
    var mass: Float64

fn sum_x_aos(particles: List[ParticleAoS]) -> Float64:
    var total: Float64 = 0.0
    for p in particles:
        # Accessing x values jumps 56 bytes between particles
        # Cache lines (64 bytes) mostly wasted on unused fields
        total += p[].x
    return total
```

**Do (SoA for batch operations):**
```mojo
struct ParticlesSoA:
    var x: List[Float64]
    var y: List[Float64]
    var z: List[Float64]
    var vx: List[Float64]
    var vy: List[Float64]
    var vz: List[Float64]
    var mass: List[Float64]

fn sum_x_soa(particles: ParticlesSoA) -> Float64:
    var total: Float64 = 0.0
    # Sequential memory access - perfect cache utilization
    for i in range(len(particles.x)):
        total += particles.x[i]
    return total
```

**Layout decision guide:**

| Layout | Best For | Cache Behavior |
|--------|----------|----------------|
| AoS | Single particle operations, infrequent access | All fields of one item cached together |
| SoA | Batch operations on single fields | Sequential access, perfect prefetching |
| Hybrid | Mixed access patterns | Balance between both |

---

### Pattern 20: Matrix Transpose for Coalesced Access

**When:** Matrix multiplication where B is accessed column-wise

**Benchmark (1024x1024, Apple M-series):**
- Without transpose: 2.4 GFLOP/s
- With transpose + SIMD: 105 GFLOP/s (**44x faster**)
- With transpose + SIMD + parallel + unroll: **117 GFLOP/s**

```mojo
# nocompile
# SLOW: Column-wise access to B causes cache misses
fn matmul_slow(C: Matrix, A: Matrix, B: Matrix):
    for m in range(M):
        for n in range(N):
            var acc: Float64 = 0.0
            for k in range(K):
                acc += A.get(m, k) * B.get(k, n)  # B[k,n] is strided!
            C.set(m, n, acc)

# FAST: Transpose B, then both accesses are row-wise
fn transpose(B: Matrix) -> Matrix:
    var BT = Matrix(B.cols, B.rows)
    for i in range(B.rows):
        for j in range(B.cols):
            BT.set(j, i, B.get(i, j))
    return BT^

fn matmul_fast(C: Matrix, A: Matrix, B: Matrix):
    var BT = transpose(B)  # One-time O(n^2) cost
    for m in range(M):
        var a_row = A.data + m * K
        for n in range(N):
            var bt_row = BT.data + n * K  # Now coalesced!
            # SIMD dot product - both accesses sequential
            var acc = SIMD[DType.float64, 8]()
            var k = 0
            while k + 8 <= K:
                acc += (a_row + k).load[width=8]() * (bt_row + k).load[width=8]()
                k += 8
            C.set(m, n, acc.reduce_add())
```

---

### Pattern 21: Memory Prefetching

**When:** Large sequential or strided array traversals

**Benchmark (100M element traversal):**
- Without prefetch: 8.5 ms
- With prefetch: 6.2 ms (**27% faster**)

```mojo
# nocompile
from memory import prefetch, PrefetchOptions

fn sum_with_prefetch(data: UnsafePointer[Float64], size: Int) -> Float64:
    comptime WIDTH: Int = 8
    comptime PREFETCH_DISTANCE: Int = 256  # Elements ahead to prefetch

    var acc = SIMD[DType.float64, WIDTH]()
    var i = 0

    while i + WIDTH <= size:
        # Prefetch data that will be needed in future iterations
        if i + PREFETCH_DISTANCE < size:
            prefetch[PrefetchOptions().for_read().high_locality()](
                data.offset(i + PREFETCH_DISTANCE)
            )

        acc += data.load[width=WIDTH](i)
        i += WIDTH

    return acc.reduce_add()
```

**Prefetch options:**
```mojo
# nocompile
# Read prefetch (most common)
prefetch[PrefetchOptions().for_read().high_locality()](ptr)

# Write prefetch (when you'll write to memory soon)
prefetch[PrefetchOptions().for_write().high_locality()](ptr)

# Low locality (data used once, don't pollute cache)
prefetch[PrefetchOptions().for_read().low_locality()](ptr)
```

**Prefetch distance guidelines:**

| Access Pattern | Recommended Distance | Reasoning |
|----------------|---------------------|-----------|
| Sequential read | 256-512 elements | ~2-4 cache lines ahead |
| Strided access | 128-256 elements | Account for stride |
| Random access | Don't prefetch | Unpredictable, may hurt |

---

### Pattern 22: Multiple Accumulators for ILP

**When:** Reduction operations with independent iterations

Modern CPUs can execute multiple independent instructions simultaneously. Using a single accumulator creates a dependency chain that limits throughput.

**Benchmark (100M element sum):**
- Single accumulator: 12.0 ms
- 4 accumulators: 8.2 ms (**1.5x faster**)
- 8 accumulators: 7.0 ms (**1.7x faster**)

**Don't (single accumulator creates dependency chain):**
```mojo
fn sum_single_acc(data: UnsafePointer[Float64], size: Int) -> Float64:
    comptime WIDTH: Int = 8
    var acc = SIMD[DType.float64, WIDTH]()

    var i = 0
    while i + WIDTH <= size:
        acc += data.load[width=WIDTH](i)  # Must wait for previous acc update
        i += WIDTH

    return acc.reduce_add()
```

**Do (multiple accumulators enable parallel execution):**
```mojo
fn sum_multi_acc(data: UnsafePointer[Float64], size: Int) -> Float64:
    comptime WIDTH: Int = 8

    # 8 independent accumulators for maximum ILP
    var acc0 = SIMD[DType.float64, WIDTH]()
    var acc1 = SIMD[DType.float64, WIDTH]()
    var acc2 = SIMD[DType.float64, WIDTH]()
    var acc3 = SIMD[DType.float64, WIDTH]()
    var acc4 = SIMD[DType.float64, WIDTH]()
    var acc5 = SIMD[DType.float64, WIDTH]()
    var acc6 = SIMD[DType.float64, WIDTH]()
    var acc7 = SIMD[DType.float64, WIDTH]()

    var i = 0
    # Process 8 SIMD vectors per iteration (64 elements)
    while i + WIDTH * 8 <= size:
        acc0 += data.load[width=WIDTH](i)
        acc1 += data.load[width=WIDTH](i + WIDTH)
        acc2 += data.load[width=WIDTH](i + WIDTH * 2)
        acc3 += data.load[width=WIDTH](i + WIDTH * 3)
        acc4 += data.load[width=WIDTH](i + WIDTH * 4)
        acc5 += data.load[width=WIDTH](i + WIDTH * 5)
        acc6 += data.load[width=WIDTH](i + WIDTH * 6)
        acc7 += data.load[width=WIDTH](i + WIDTH * 7)
        i += WIDTH * 8

    # Combine accumulators
    acc0 = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7

    # Handle remaining elements
    while i + WIDTH <= size:
        acc0 += data.load[width=WIDTH](i)
        i += WIDTH

    var sum = acc0.reduce_add()
    while i < size:
        sum += data[i]
        i += 1

    return sum
```

**Accumulator count guidelines:**

| CPU Type | Recommended Accumulators | Reasoning |
|----------|-------------------------|-----------|
| Apple M-series | 4-8 | Wide execution units |
| Intel/AMD x86 | 4-8 | Multiple ALUs per core |
| Memory-bound ops | 2-4 | Diminishing returns |

---

### Pattern 23: Tiled Processing for Large Data

**When:** Large image/tensor processing that exceeds cache or GPU memory

**Impact:** 75% memory reduction for large image processing (1024x1024+)

```mojo
# nocompile
@fieldwise_init
struct TiledConfig(Copyable, Movable):
    var tile_size: Int      # Tile size (e.g., 512 pixels)
    var overlap: Int        # Overlap between tiles (e.g., 64 pixels)
    var enabled: Bool

fn compute_blend_weight(y: Int, x: Int, h: Int, w: Int, overlap: Int) -> Float32:
    """Linear falloff weight at tile edges."""
    var wt: Float32 = 1.0

    # Top edge
    if y < overlap:
        wt *= Float32(y) / Float32(overlap)
    # Bottom edge
    if y >= h - overlap:
        wt *= Float32(h - 1 - y) / Float32(overlap)
    # Left edge
    if x < overlap:
        wt *= Float32(x) / Float32(overlap)
    # Right edge
    if x >= w - overlap:
        wt *= Float32(w - 1 - x) / Float32(overlap)

    return wt

fn process_tiled(input: Float32Ptr, h: Int, w: Int, config: TiledConfig) -> Float32Ptr:
    """Process input in overlapping tiles."""
    var stride = config.tile_size - config.overlap
    var output = alloc[Float32](h * w)
    var weight_sum = alloc[Float32](h * w)

    # Initialize accumulators
    for i in range(h * w):
        output[i] = 0.0
        weight_sum[i] = 0.0

    # Process tiles
    var y = 0
    while y < h:
        var tile_h = min(config.tile_size, h - y)
        var x = 0
        while x < w:
            var tile_w = min(config.tile_size, w - x)

            # Extract and process tile
            var tile = extract_tile(input, y, x, tile_h, tile_w, w)
            var processed = process_single_tile(tile, tile_h, tile_w)

            # Accumulate with blending
            for ty in range(tile_h):
                for tx in range(tile_w):
                    var wt = compute_blend_weight(ty, tx, tile_h, tile_w, config.overlap)
                    var oy = y + ty
                    var ox = x + tx
                    var idx = oy * w + ox
                    output[idx] += wt * processed[ty * tile_w + tx]
                    weight_sum[idx] += wt

            x += stride
        y += stride

    # Normalize
    for i in range(h * w):
        if weight_sum[i] > 0:
            output[i] /= weight_sum[i]

    weight_sum.free()
    return output
```

**Recommended tiling settings:**

| Resolution | tile_size | overlap | Memory |
|------------|-----------|---------|--------|
| 512x512 | No tiling needed | - | - |
| 1024x1024 | 512 | 64 | ~128MB |
| 2048x2048 | 512 | 64 | ~128MB |

---

### Pattern 24: Prefer Borrowing Over Copying

**When:** Passing large data structures to functions

**Impact:** 10-100x faster for large data structures

```mojo
fn analyze(data: List[Float64]) -> Float64:
    # data is borrowed immutably - no copy occurs
    # This is the default behavior with 'read' convention
    var sum: Float64 = 0.0
    for item in data:
        sum += item
    return sum / len(data)

fn main():
    var measurements = List[Float64]()
    for i in range(1000000):
        measurements.append(Float64(i))

    var avg = analyze(measurements)  # Borrowed, not copied
    print(avg)
    print(measurements[0])  # Still valid - we only borrowed
```

**When copying is appropriate:**
- Small types that fit in registers (Int, Float64, Bool)
- When you need an independent copy to modify
- Types conforming to `TrivialRegisterType` trait

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Mathematically skip-able work | Algorithm shortcuts | Pattern 1 |
| Expensive resource loading | Cache in context struct | Pattern 2 |
| Values depending on dimensions | Dimension-dependent caching | Pattern 3 |
| Large file access (100MB+) | mmap with zero-copy | Pattern 4 |
| Python in hot paths | Convert at boundary, use native | Pattern 5 |
| Fixed constants | comptime | Pattern 6 |
| Repeated function calls | Persistent buffers | Pattern 7 |
| Memory-constrained | Dynamic allocation based on actual size | Pattern 8 |
| GPU vs CPU choice | GPU for compute, CPU for memory ops | Pattern 16 |
| FFI overhead in hot paths | Batch/fuse operations, cache weights | Pattern 14 |
| Multi-phase GPU pipelines | Preserve cache between phases | Pattern 15 |
| MPSGraph varying configs | Pre-compile or use adaptive selection | Pattern 17 |
| SIMD data structures | @align(32) or @align(64) | Pattern 18 |
| Batch field operations | SoA layout | Pattern 19 |
| Matrix multiply | Transpose B for coalesced access | Pattern 20 |
| Large sequential scans | Prefetch 256-512 elements ahead | Pattern 21 |
| Reduction operations | Multiple accumulators (4-8) | Pattern 22 |
| Large images (1024+) | Tiled processing with overlap | Pattern 23 |
| Parallel counters | Cache-line aligned to prevent false sharing | Pattern 18 |
| Large data to functions | Borrow instead of copy | Pattern 24 |

---

## Quick Reference

- **Caching pattern**: Check flag, load if miss, return cached
- **mmap cleanup**: Track is_mmapped, use munmap vs free appropriately
- **Compile-time**: Use `comptime` (alias is deprecated)
- **Trivial check**: `T.__copyinit__is_trivial`
- **Benchmark**: Warmup + best of 10, separate setup
- **Cache line size**: 64 bytes (typical)
- **AVX alignment**: 32 bytes
- **AVX-512 alignment**: 64 bytes
- **Prefetch distance**: 256-512 elements for sequential
- **Accumulator count**: 4-8 for modern CPUs
- **Tile overlap**: At least 1/8 of tile size

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `cache miss overhead` | Missing or stale cache | Implement cache invalidation; check TTL |
| `lazy init race condition` | Multiple threads initializing | Use atomic compare-and-swap for first init |
| `buffer pool exhaustion` | All buffers in use | Increase pool size; add blocking wait for buffer |
| `mmap leak` | Missing munmap | Track is_mmapped flag; call munmap in destructor |
| `benchmark variance too high` | Cold cache or background work | Add warmup iterations; isolate benchmark process |
| `trivial copy check wrong` | Using wrong trait check | Use `T.__copyinit__is_trivial` for trivial copy detection |
| `unaligned access crash` | Pointer not aligned for operation | Use `@align(N)` on struct; use aligned allocation |
| `cache thrashing` | Stride matches cache line size | Change array layout; add padding to break stride |
| `prefetch no effect` | Prefetch too late or wrong distance | Prefetch 256-512 elements ahead for sequential access |
| `false sharing` | Different threads modify same cache line | Pad data to 64-byte boundaries between threads |
| `stack overflow` | Large stack allocation | Use heap for large buffers; reduce local array sizes |
| `memory bandwidth saturated` | Too many concurrent memory streams | Reduce active streams; improve data locality |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **Constants** | `alias` or `comptime` | Both work in v26.1+ |
| **Heap allocation** | `from memory import alloc; alloc[T](n)` | v26.1+ |
| **mmap** | `mmap.mmap()` | Stable |
| **FFI** | `external_call[]` | Stable |
| **Prefetch** | `prefetch[PrefetchOptions()]` | Stable |
| **Struct alignment** | `@align(64)` | v26.2+ nightly |

**Example (v26.1+):**
```mojo
# nocompile
from memory import alloc, prefetch, PrefetchOptions

comptime CACHE_LINE = 64
comptime POOL_SIZE = 16
comptime SIMD_WIDTH = 8

fn allocate_buffer(size: Int) -> UnsafePointer[Float32]:
    return alloc[Float32](size)
```

**Notes:**
- Both `alias` and `comptime` work for compile-time constants in v26.1+
- `alloc()` is available in v26.1+ (not nightly-only)
- `@align(N)` decorator is v26.2+ nightly
- Prefetching APIs (`prefetch`, `PrefetchOptions`) are stable
- Caching, buffer pooling, memory-mapped file, tiled processing, and compile-time computation patterns are all stable across versions

---

## Related Patterns

- [`perf-vectorization.md`](perf-vectorization.md) — SIMD patterns that benefit from alignment
- [`perf-parallelization.md`](perf-parallelization.md) — Multi-core patterns with memory considerations

---

## References

- [Mojo Parameters and Metaprogramming](https://docs.modular.com/mojo/manual/parameters/)
- [Mojo Python Interoperability](https://docs.modular.com/mojo/manual/python/)
- [Mojo Manual](https://docs.modular.com/mojo/manual/)
- [Mojo UnsafePointer](https://docs.modular.com/mojo/std/memory/unsafe_pointer/)
- [Mojo Memory](https://docs.modular.com/mojo/std/memory/)
- [Mojo Decorators](https://docs.modular.com/mojo/manual/decorators/)
- [Mojo Value Semantics](https://docs.modular.com/mojo/manual/values/)
