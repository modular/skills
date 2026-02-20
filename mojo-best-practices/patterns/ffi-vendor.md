---
title: FFI Vendor Library Integration
description: Patterns for integrating vendor libraries, GPU frameworks, Apple BLAS/AMX, dynamic loading, Python GIL management, and platform-specific FFI considerations
impact: CRITICAL
category: ffi
tags: [ffi, vendor-libraries, gpu, blas, apple, metal, mps, gil, dynamic-loading]
error_patterns:
  - "Accelerate"
  - "cuBLAS"
  - "rocBLAS"
  - "GIL deadlock"
  - "MPS error"
  - "BF16"
scenarios:
  - "Use Apple BLAS for matrix multiply"
  - "Integrate GPU libraries"
  - "Manage Python GIL"
  - "Handle BF16/F16 conversion"
  - "Optimize MPS performance"
consolidates:
  - ffi-prefer-vendor-libraries.md
  - ffi-apple-amx-blas.md
  - ffi-gpu-libraries.md
  - ffi-gpu-program-caching.md
  - ffi-python-gil.md
  - ffi-mps-bf16-limitations.md
  - ffi-apple-bf16-to-f16.md
---

# FFI Vendor Library Integration

**Category:** ffi | **Impact:** CRITICAL

Patterns for integrating vendor-optimized libraries including Apple BLAS/AMX, GPU libraries (cuBLAS, rocBLAS), Python GIL management, and platform-specific considerations. Vendor libraries provide 10-100x performance gains over custom implementations through undocumented hardware features.

---

## Vendor Library Integration

### Why Vendor Libraries

Vendor libraries use undocumented hardware features and years of optimization. Custom implementations rarely beat them for standard operations.

| Operation | Custom | Vendor Library | Winner |
|-----------|--------|----------------|--------|
| Dense Matmul | 1.0x | 2-10x | Library |
| FFT | 1.0x | 5-20x | Library |
| Convolution | 1.0x | 3-10x | Library |
| Fused Ops | 1.2-2x | 1.0x | Custom |

### When Custom Kernels Help

```mojo
# nocompile
# GOOD: Fuse operations that libraries can't combine
fn fused_layernorm_linear(x: Ptr, weight: Ptr, scale: Ptr, bias: Ptr, output: Ptr):
    # LayerNorm + Linear in one kernel = fewer memory round-trips
    pass

# GOOD: Custom operations not in vendor libraries
fn rotary_position_embedding(...):
    # Libraries don't provide this
    pass
```

---

## Apple Accelerate BLAS (AMX)

### Benchmark Results

| Matrix Size | Custom Mojo | BLAS (AMX) | Speedup |
|-------------|-------------|------------|---------|
| 512x512 | 5.5 ms | 0.45 ms | **12x** |
| 1024x1024 | 48 ms | 0.85 ms | **56x** |
| 2048x2048 | 390 ms | 6.5 ms | **60x** |
| 4096x4096 | 3200 ms | 51 ms | **63x** |

### Basic BLAS Integration

```mojo
# nocompile
from ffi import external_call
from memory import UnsafePointer
from builtin.type_aliases import MutAnyOrigin

comptime Float64Ptr = UnsafePointer[mut=True, type=Float64, origin=MutAnyOrigin]
comptime Float32Ptr = UnsafePointer[mut=True, type=Float32, origin=MutAnyOrigin]

# BLAS constants
comptime CblasRowMajor: Int32 = 101
comptime CblasColMajor: Int32 = 102
comptime CblasNoTrans: Int32 = 111
comptime CblasTrans: Int32 = 112


fn blas_dgemm(M: Int, N: Int, K: Int, alpha: Float64,
    A: Float64Ptr, B: Float64Ptr, beta: Float64, C: Float64Ptr):
    """
    Double-precision matrix multiply: C = alpha*A*B + beta*C
    A: MxK, B: KxN, C: MxN (row-major)
    """
    external_call["cblas_dgemm", NoneType](
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        Int32(M), Int32(N), Int32(K),
        alpha, A, Int32(K),  # A is MxK, lda=K
        B, Int32(N),         # B is KxN, ldb=N
        beta, C, Int32(N)    # C is MxN, ldc=N
    )


fn blas_sgemm(M: Int, N: Int, K: Int, alpha: Float32,
    A: Float32Ptr, B: Float32Ptr, beta: Float32, C: Float32Ptr):
    """Single-precision matrix multiply."""
    external_call["cblas_sgemm", NoneType](
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        Int32(M), Int32(N), Int32(K),
        alpha, A, Int32(K),
        B, Int32(N),
        beta, C, Int32(N)
    )
```

### Using BLAS with Mojo

**With `mojo run` (uses dynamic linking - recommended):**
```bash
# mojo run handles dynamic linking automatically
mojo run main.mojo
```

BLAS functions from Apple Accelerate are available automatically via `external_call` when using `mojo run`.

**With `mojo build` (static linking - has limitations):**
```bash
# mojo build does not support -Xlinker flags for framework linking
# FFI with external libraries works best with mojo run
mojo build main.mojo -o app
```

> **Note:** `mojo build` has limitations with FFI linking. For production use of BLAS, consider using `mojo run` or wrapping BLAS calls in a dynamic library loaded at runtime with `OwnedDLHandle`.

### Common BLAS Operations

```mojo
# nocompile
fn blas_dgemv(M: Int, N: Int, alpha: Float64,
    A: Float64Ptr, x: Float64Ptr, beta: Float64, y: Float64Ptr):
    """y = alpha*A*x + beta*y, where A is MxN."""
    external_call["cblas_dgemv", NoneType](
        CblasRowMajor, CblasNoTrans,
        Int32(M), Int32(N),
        alpha, A, Int32(N),
        x, Int32(1),  # incX
        beta, y, Int32(1)  # incY
    )

fn blas_ddot(n: Int, x: Float64Ptr, y: Float64Ptr) -> Float64:
    """Dot product of two vectors."""
    return external_call["cblas_ddot", Float64](
        Int32(n), x, Int32(1), y, Int32(1)
    )

fn blas_dscal(n: Int, alpha: Float64, x: Float64Ptr):
    """Scale vector: x = alpha * x."""
    external_call["cblas_dscal", NoneType](
        Int32(n), alpha, x, Int32(1)
    )

fn blas_daxpy(n: Int, alpha: Float64, x: Float64Ptr, y: Float64Ptr):
    """y = alpha*x + y."""
    external_call["cblas_daxpy", NoneType](
        Int32(n), alpha, x, Int32(1), y, Int32(1)
    )
```

### Neural Network Layer Example

```mojo
# nocompile
fn linear_layer_blas(
    weights: Float32Ptr,  # out_features x in_features
    bias: Float32Ptr,     # out_features
    input: Float32Ptr,    # batch x in_features
    output: Float32Ptr,   # batch x out_features
    batch: Int, in_features: Int, out_features: Int
):
    """Linear layer: output = input @ weights.T + bias using BLAS."""

    # Matrix multiply: output = input @ weights.T
    external_call["cblas_sgemm", NoneType](
        CblasRowMajor, CblasNoTrans, CblasTrans,  # Transpose B
        Int32(batch), Int32(out_features), Int32(in_features),
        Float32(1.0),
        input, Int32(in_features),
        weights, Int32(in_features),  # Stored as outxin, accessed as inxout
        Float32(0.0),
        output, Int32(out_features)
    )

    # Add bias to each row
    for b in range(batch):
        for o in range(out_features):
            output[b * out_features + o] += bias[o]
```

### Cross-Platform BLAS

**Dynamic linking approach (works with `mojo run`):**

| Platform | BLAS Library | Notes |
|----------|--------------|-------|
| macOS | Apple Accelerate | Available by default |
| Linux | OpenBLAS | Install: `apt install libopenblas-dev` |
| Linux | Intel MKL | Install via Intel oneAPI |

**Runtime library loading (for `mojo build` builds):**
```mojo
# nocompile
from ffi import OwnedDLHandle, RTLD

fn load_blas() raises -> OwnedDLHandle:
    @parameter
    if os_is_macos():
        return OwnedDLHandle("/System/Library/Frameworks/Accelerate.framework/Accelerate", RTLD.NOW)
    elif os_is_linux():
        return OwnedDLHandle("libopenblas.so", RTLD.NOW)
    else:
        raise Error("Unsupported platform for BLAS")
```

---

## Dynamic Library Loading

### Load Library and Get Functions

```mojo
# nocompile
from ffi import OwnedDLHandle, RTLD

# Function type aliases
comptime CreateCtxFn = fn() -> NonePtr
comptime DestroyCtxFn = fn(NonePtr) -> None
comptime MatmulFn = fn(NonePtr, Int32, Int32, Int32, Float32Ptr, Float32Ptr, Float32Ptr) -> Int32

fn main() raises:
    # Load the dynamic library
    var lib = OwnedDLHandle("./libcustom.so", RTLD.NOW)  # .dylib on macOS

    # Get function pointers
    var create_ctx = lib.get_function[CreateCtxFn]("create_context")
    var destroy_ctx = lib.get_function[DestroyCtxFn]("destroy_context")
    var matmul = lib.get_function[MatmulFn]("matmul_f32")

    # Create context and use
    var ctx = create_ctx()
    _ = matmul(ctx, 1024, 1024, 1024, A, B, C)
    destroy_ctx(ctx)
```

### Platform-Specific Loading

```mojo
# nocompile
fn load_platform_library() raises -> OwnedDLHandle:
    """Load appropriate library for current platform."""
    @parameter
    if os_is_macos():
        return OwnedDLHandle("./libcustom.dylib", RTLD.NOW)
    elif os_is_linux():
        return OwnedDLHandle("./libcustom.so", RTLD.NOW)
    else:
        return OwnedDLHandle("./custom.dll", RTLD.NOW)
```

### Building Shared Libraries

```bash
# Linux
gcc -shared -fPIC -o libcustom.so custom.c

# macOS
clang -dynamiclib -o libcustom.dylib custom.c

# macOS with frameworks (e.g., Metal)
clang -dynamiclib -o libgpu_wrapper.dylib gpu_wrapper.m \
    -framework Metal -fobjc-arc
```

---

## GPU Library Integration

### cuBLAS Integration Pattern

```mojo
# nocompile
from ffi import external_call, DLHandle
from memory import UnsafePointer

# Load cuBLAS library
var cublas = DLHandle("libcublas.so")

# cuBLAS handle type
struct cublasHandle_t:
    var handle: UnsafePointer[NoneType]

# Initialize cuBLAS
fn cublas_create() -> cublasHandle_t:
    var handle = cublasHandle_t(UnsafePointer[NoneType]())
    var status = external_call["cublasCreate", Int32](
        UnsafePointer(to=handle.handle)
    )
    debug_assert(status == 0, "cuBLAS create failed")
    return handle

# SGEMM: C = alpha * A * B + beta * C
fn cublas_sgemm(
    handle: cublasHandle_t,
    m: Int, n: Int, k: Int,
    alpha: Float32,
    A: UnsafePointer[Float32], lda: Int,
    B: UnsafePointer[Float32], ldb: Int,
    beta: Float32,
    C: UnsafePointer[Float32], ldc: Int,
):
    # cuBLAS uses column-major, so we compute C^T = B^T * A^T
    # which gives row-major C = A * B
    var status = external_call["cublasSgemm", Int32](
        handle.handle,
        0,  # CUBLAS_OP_N (no transpose)
        0,  # CUBLAS_OP_N
        Int32(n), Int32(m), Int32(k),  # Note: n, m swapped for row-major
        UnsafePointer(to=alpha),
        B, Int32(ldb),
        A, Int32(lda),
        UnsafePointer(to=beta),
        C, Int32(ldc),
    )
    debug_assert(status == 0, "cuBLAS SGEMM failed")
```

### Platform Abstraction Pattern

```mojo
# nocompile
struct GPUBlas:
    var _handle: UnsafePointer[NoneType]
    var _is_nvidia: Bool

    fn __init__(out self, ctx: DeviceContext):
        @parameter
        if is_nvidia_gpu():
            self._handle = _cublas_create()
            self._is_nvidia = True
        elif is_amd_gpu():
            self._handle = _rocblas_create()
            self._is_nvidia = False
        else:
            abort("No GPU BLAS available")

    fn gemm(
        self,
        A: DeviceBuffer[DType.float32],
        B: DeviceBuffer[DType.float32],
        C: DeviceBuffer[DType.float32],
        m: Int, n: Int, k: Int,
    ):
        if self._is_nvidia:
            _cublas_sgemm(self._handle, m, n, k, ...)
        else:
            _rocblas_sgemm(self._handle, m, n, k, ...)

    fn __del__(deinit self):
        if self._is_nvidia:
            _cublas_destroy(self._handle)
        else:
            _rocblas_destroy(self._handle)
```

### Library vs Custom Kernel Decision

| Operation | Library | Custom Kernel |
|-----------|---------|---------------|
| Dense GEMM | cuBLAS/rocBLAS | Only for small matrices |
| Batched GEMM | Library | Fused with other ops |
| Convolution | cuDNN/MIOpen | Non-standard strides |
| FFT | cuFFT/rocFFT | Always use library |
| Sparse ops | cuSPARSE/rocSPARSE | Custom sparsity patterns |
| Element-wise | Custom kernel | Simpler, can fuse |
| Reductions | Custom kernel | Better for fused ops |

---

## GPU Program Caching

Cache compiled GPU programs for 10-50x speedup.

### Caching Pattern

```c
#define MAX_CACHE_SIZE 64

typedef struct {
    int param1, param2, param3;  // Parameters that define the program
    GPUProgram *program;         // Cached compiled program
} cache_entry_t;

static cache_entry_t g_cache[MAX_CACHE_SIZE];
static int g_cache_count = 0;

GPUProgram* get_cached_program(int param1, int param2, int param3) {
    // Search for existing program with same parameters
    for (int i = 0; i < g_cache_count; i++) {
        if (g_cache[i].param1 == param1 &&
            g_cache[i].param2 == param2 &&
            g_cache[i].param3 == param3) {
            return g_cache[i].program;  // Reuse!
        }
    }
    // Create new program only if not found
    GPUProgram *program = create_gpu_program(param1, param2, param3);
    // Add to cache and return...
    return program;
}
```

### What to Cache by Platform

| Platform | Cacheable Object | Typical Speedup |
|----------|------------------|-----------------|
| Apple/MPS | MPSGraph, MPSCommandBuffer | 10-50x |
| CUDA | CUDA Graphs, Compiled PTX | 5-20x |
| Vulkan | VkPipeline objects | 10-30x |
| OpenCL | cl_program, cl_kernel | 5-15x |

---

## Python GIL Management

### GIL States

```mojo
# nocompile
struct GILAcquired(TrivialRegisterType):
    """Marker type indicating the GIL is held by this thread."""
    pass

struct GILReleased(TrivialRegisterType):
    """Marker type indicating the GIL is NOT held by this thread."""
    pass
```

### Release GIL During Mojo Computation

```mojo
# nocompile
fn long_computation() raises -> PythonObject:
    var result: MyResult

    # Release GIL while doing pure Mojo work
    Python.release_gil()
    result = expensive_mojo_function()  # GIL released - Python threads can run
    Python.acquire_gil()

    # Re-acquire GIL for Python interop
    return Python.import_module("json").dumps(result)
```

### Context Manager Pattern (Preferred)

```mojo
# nocompile
fn process_batch(data: PythonObject) raises -> PythonObject:
    # Convert Python input to Mojo (GIL needed)
    var mojo_data = convert_to_mojo(data)

    var result: MojoResult
    with Python.GILReleased():
        # Pure Mojo computation - GIL automatically released/re-acquired
        result = heavy_processing(mojo_data)

    # Convert back to Python (GIL re-acquired)
    return convert_to_python(result)
```

### GIL Rules

| Situation | GIL Status | Action |
|-----------|------------|--------|
| Calling from Python | Held | Keep it for Python calls |
| Long Mojo computation | Held | Release it |
| Calling Python API | Required | Acquire if not held |
| Parallel Mojo threads | Don't need | Release for parallelism |
| Python callbacks | Not held | Acquire before calling |

### Common GIL Mistakes

```mojo
# nocompile
# BAD: Deadlock - releasing when already released
Python.release_gil()
Python.release_gil()  # Double release - undefined behavior!

# BAD: Deadlock - Python call without GIL
Python.release_gil()
var x = Python.import_module("os")  # CRASH - no GIL!
Python.acquire_gil()

# BAD: Blocking Python threads unnecessarily
fn handle_request(request: PythonObject) raises -> PythonObject:
    # Should release GIL during file I/O
    var data = read_large_file("data.bin")  # Blocks Python threads
    return process(data)
```

---

## MPS/Metal GPU Performance Optimization

Critical learnings from optimizing transformer inference on Apple Silicon with MPSGraph and Metal.

### Counter-Intuitive Finding: GPU Beats CPU Even for Tiny Matrices

**Hypothesis:** CPU BLAS (Apple Accelerate) would be faster for small resolutions due to avoiding GPU dispatch overhead.

**Reality:** CPU BLAS is **6.6x SLOWER** than GPU path even at 64x64 resolution.

| Path | Time (64x64, 4 steps) | Relative |
|------|----------------------|----------|
| GPU (MPS) | 1839ms | 1x |
| CPU (Accelerate) | 12133ms | **6.6x slower** |

**Why:**
- Apple's MPS is highly optimized even for small matrices
- CPU BLAS has function call overhead and cache misses
- BF16->F32 weight conversion adds CPU latency
- Modern Apple GPUs have very low dispatch latency (~1.5ms)

**Lesson:** Don't assume CPU is faster for small workloads--benchmark first.

### MPSGraph Overhead Analysis

At small resolutions, dispatch overhead dominates compute time.

**Key Numbers (FLUX.2 transformer):**
- 310 GPU operations per denoising step
- ~1.5ms overhead per operation = 465ms fixed overhead
- At 64x64: actual compute ~75ms, overhead ~465ms
- At 512x512: compute dominates, overhead negligible

**Per-operation overhead sources:**
1. MPSGraphTensorData wrapper creation (3 per linear op)
2. NSDictionary creation for feeds/results (2 per linear op)
3. `encodeToCommandBuffer` call
4. Objective-C message dispatch

**Total:** ~550 Obj-C allocations per step, inherent to MPSGraph API.

### Kernel Fusion Effectiveness

**What Works:**

| Fusion | Ops Saved | Impact |
|--------|-----------|--------|
| Triple linear (QKV) | 3->1 | Good |
| Fused QKNorm + RoPE | 3->1 | Good |
| Flash attention threadgroups 128->256 | N/A | 10-15% |

**What Doesn't Work:**

| Fusion | Result | Why |
|--------|--------|-----|
| MLP gate+up with split kernel | Neutral/negative | Split kernel overhead offsets fusion |
| BF16 weight caching | No improvement | Small weights (512 bytes), mutex overhead |
| Custom fused AdaLN+QKV | 2x slower | MPSGraph's matmul is more optimized |

**Lesson:** Fusion only helps if the fused kernel is more efficient than the dispatch overhead saved. MPSGraph is often more optimized than custom Metal kernels for standard ops.

### Command Encoder Batching

**Problem:** Each custom Metal kernel created its own command encoder (67 create/end cycles per step).

**Solution:** Persistent encoder mode--share encoder across consecutive custom kernels.

**Result:** Reduced to ~15 encoder cycles per step.

**Caveat:** MPSGraph operations use different API and interrupt encoder sharing.

### Flash Attention Threadgroup Optimization

**Discovery:** M3 Max supports 1024 threads per threadgroup, but flash attention was using only 128 (12.5% utilization).

**Fix:** Increase to 256 threads.

**Impact:** 10-15% attention speedup.

### MPSGraph Pre-compilation

**Problem:** JIT compilation adds ~100-150ms on first use per resolution.

**Solution:** Pre-compile graphs for target resolution only (not all possible resolutions).

**Trade-off:**
- Pre-compile everything: +3s startup, faster first inference
- JIT on demand: Fast startup, slower first inference

**Best approach:** Pre-compile when weights are preloaded with known target resolution.

### Graph Cache Sizing

MPSGraph caches compiled graphs by input shapes. At small sizes, cache eviction causes re-compilation.

**Recommended cache sizes:**
- SDPA graphs: 16 (up from 8)
- Linear graphs: 64 (up from 32)
- Threshold for MPSGraph vs custom: seq_len >= 8

### MLP Fusion is Resolution-Dependent

**Critical Discovery:** MLP gate+up fusion behaves differently at different resolutions.

| Resolution | Fusion Impact | Why |
|------------|---------------|-----|
| Small (64-128) | Neutral/positive | Fewer dispatches helps |
| Large (512+) | **Negative** | Split kernel overhead exceeds dispatch savings |

**Pattern:**
```mojo
# nocompile
# Fused (faster at small resolutions, slower at large)
_ = gpu_linear(input, fused_out, fused_weight, seq, hidden, mlp*2)
split_gate_up(fused_out, gate, up, seq, mlp)  # Extra kernel!

# Separate (slower at small, faster at large resolutions)
_ = gpu_linear(input, gate, gate_weight, seq, hidden, mlp)
_ = gpu_linear(input, up, up_weight, seq, hidden, mlp)
```

**Recommendation:** Profile at your target resolution before committing to fusion. MPSGraph may already optimize consecutive matmuls internally.

### Direct MPS vs MPSGraph Threshold

MPSGraph has ~1.0-1.6ms overhead per operation. Direct MPS has ~0.3-0.5ms.

**Optimal threshold (FLOPS-based):**
```c
static int use_mpsgraph(int seq, int in_dim, int out_dim) {
    long long flops = (long long)seq * in_dim * out_dim;
    if (flops < 50000000) return 0;  // Direct MPS for <50M FLOPS
    if (seq < 64) return 0;          // Small sequences
    if (in_dim < 512 && out_dim < 512) return 0;  // Small matrices
    return 1;  // MPSGraph only for large operations
}
```

### Performance Results Summary

| Resolution | Optimization Impact | vs PyTorch |
|------------|---------------------|------------|
| 64x64 | 45% faster (2905->1595ms) | 2.0x slower |
| 128x128 | 38% faster (2716->1695ms) | 1.4x slower |
| 256x256 | 27% faster (3062->2230ms) | **11% faster** |
| 512x512 | 17% faster (4838->4040ms) | **16% faster** |

**Bottom line:** Now beating PyTorch at 256x256 and 512x512 through mega-kernel fusion and optimized dispatch thresholds. Small resolutions still limited by per-operation overhead.

---

## BF16/F16 Data Type Considerations

### Float Format Comparison

- **BF16 (Brain Float 16):** sign(1) + exp(8) + mant(7) - Same dynamic range as F32
- **F16 (IEEE Half):** sign(1) + exp(5) + mant(10) - Lower dynamic range, higher precision

### BF16 to F16 Conversion (Lossy)

```c
static inline uint16_t bf16_to_f16(uint16_t bf16) {
    uint32_t sign = (bf16 >> 15) & 0x1;
    int32_t exp = (bf16 >> 7) & 0xFF;  // bf16 exponent (bias 127)
    uint32_t mant = bf16 & 0x7F;       // bf16 mantissa (7 bits)

    if (exp == 0) return (uint16_t)(sign << 15);  // Zero/denormal
    if (exp == 0xFF) return (uint16_t)((sign << 15) | 0x7C00 | (mant << 3));  // Inf/NaN

    int32_t new_exp = exp - 127 + 15;  // Rebias from 127 to 15
    if (new_exp <= 0) return (uint16_t)(sign << 15);  // Underflow
    if (new_exp >= 31) return (uint16_t)((sign << 15) | 0x7C00);  // Overflow

    return (uint16_t)((sign << 15) | (new_exp << 10) | (mant << 3));
}
```

### MPS Data Type Constraints

| Input A | Weights B | Output C | Supported |
|---------|-----------|----------|-----------|
| F32 | F32 | F32 | Yes |
| F32 | F16 | F32 | Yes |
| BF16 | BF16 | BF16 | Yes |
| F32 | BF16 | F32 | **NO** |

### Workarounds

```mojo
# nocompile
# Option 1: Convert BF16 to F16 (lossy but compatible)
var f16_weights = convert_bf16_to_f16(bf16_weights)
vendor_matmul(f32_input, f16_weights, f32_output)

# Option 2: Use consistent precision pipeline
var bf16_input = convert_f32_to_bf16(f32_input)
vendor_matmul_bf16(bf16_input, bf16_weights, bf16_output)
var f32_result = convert_bf16_to_f32(bf16_output)

# Option 3: Use F32 throughout (simple but uses more memory)
var f32_weights = convert_bf16_to_f32(bf16_weights)  # Lossless
vendor_matmul(f32_input, f32_weights, f32_output)
```

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Matrix multiply | Use vendor BLAS (10-100x faster) | Apple Accelerate, cuBLAS |
| Loading optional library | Use `OwnedDLHandle` with RTLD.NOW | Dynamic Loading |
| BF16 data on MPS | Convert to F16 or F32 | Data Type Limitations |
| Long Mojo computation from Python | Release GIL | GIL Management |
| GPU program compilation slow | Cache compiled programs | GPU Program Caching |
| Small matrices on Apple Silicon | Use GPU (MPS), not CPU BLAS | MPS Performance |

---

## Quick Reference

- **BLAS**: Use `mojo run` for dynamic linking, or `OwnedDLHandle` for runtime loading
- **GPU caching**: Cache compiled programs by parameters for 10-50x speedup
- **GIL**: Release during long Mojo computation, acquire for Python calls
- **BF16 on MPS**: Convert to F16 (lossy) or F32 (lossless)
- **MPS vs CPU**: GPU wins even for small matrices on Apple Silicon
- **Fusion trade-off**: Profile at target resolution; MPSGraph often beats custom kernels

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `library not found` | DLHandle can't locate shared library | Use full path or set `DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH` |
| `segfault in BLAS` | Wrong matrix dimensions or leading dimension | Verify M, N, K match matrix sizes; lda >= max(1, M) for column-major |
| `GIL deadlock` | Mojo code calling Python without GIL | Always acquire GIL before Python calls: `Python.acquire_gil()` |
| `MPS BF16 error` | Mixing BF16 weights with F32 input | Convert BF16 to F16 or use consistent BF16 pipeline |
| `cuBLAS/rocBLAS error` | Handle not initialized or wrong dimensions | Check cublasCreate() return code; verify matrix dimensions |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **Compile-time constants** | `alias` or `comptime` | Both work in v26.1+ |
| **external_call** | `external_call[name, ret_type](args)` | Stable |
| **OwnedDLHandle** | Same API | Stable |
| **Python GIL** | Same API | Stable |

**Notes:**
- Both `alias` and `comptime` work for compile-time constants in v26.1+
- FFI core APIs (`external_call`, `OwnedDLHandle`) are stable across versions
- Python GIL management patterns unchanged between versions

---

## Related Patterns

- [`ffi-interop.md`](ffi-interop.md) — Core FFI patterns: CString, libc, binary data
- [`python-interop.md`](python-interop.md) — Python-specific interop patterns
- [`gpu-fundamentals.md`](gpu-fundamentals.md) — Native Mojo GPU programming

---

## References

- [Apple Accelerate Documentation](https://developer.apple.com/documentation/accelerate)
- [BLAS Reference](https://developer.apple.com/documentation/accelerate/blas)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [rocBLAS Documentation](https://rocm.docs.amd.com/projects/rocBLAS/)
