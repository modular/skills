# Mojo Best Practices Scenario Index

This index maps common tasks and scenarios to relevant patterns.

## Task Lookup

| Task/Scenario | Pattern | Category | Impact |
|--------------|---------|----------|--------|
| Achieve 1000x speedup vs Python | [perf-parallelization](../patterns/perf-parallelization.md) | perf | CRITICAL |
| Add conditional trait conformance | [meta-programming](../patterns/meta-programming.md) | meta | MEDIUM |
| Add debug assertions | [error-handling](../patterns/error-handling.md) | error | HIGH |
| Add error handling to function | [error-handling](../patterns/error-handling.md) | error | HIGH |
| Add implicit conversion | [type-system](../patterns/type-system.md) | type | CRITICAL |
| Add iterator to collection | [struct-design](../patterns/struct-design.md) | struct | HIGH |
| Add trait bounds to generic function | [type-traits](../patterns/type-traits.md) | type | HIGH |
| Align data for SIMD | [perf-optimization](../patterns/perf-optimization.md) | perf | HIGH |
| Annotate kernel pointer origins | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu | CRITICAL |
| Avoid memory leaks with shared data | [memory-refcounting](../patterns/memory-refcounting.md) | memory | HIGH |
| Benchmark function performance | [testing](../patterns/testing.md) | test | HIGH |
| Block normalize pattern | [gpu-block-collectives](../patterns/gpu-block-collectives.md) | gpu | HIGH |
| Block-wide reduction without shared memory | [gpu-block-collectives](../patterns/gpu-block-collectives.md) | gpu | HIGH |
| Bulk copy or fill memory | [memory-collections](../patterns/memory-collections.md) | memory | CRITICAL |
| Call C library from Mojo | [ffi](../patterns/ffi.md) | ffi | CRITICAL |
| Call Python from Mojo | [python-interop](../patterns/python-interop.md) | python | MEDIUM |
| Call custom op from Python graph code | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu | HIGH |
| Chain multiple GPU kernels in a single operation | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu | HIGH |
| Chain multiple GPU kernels in pipeline | [gpu-kernels](../patterns/gpu-kernels.md) | gpu | HIGH |
| Choose argument convention | [fn-design](../patterns/fn-design.md) | fn | HIGH |
| Choose between pointer types | [memory-safety](../patterns/memory-safety.md) | memory | CRITICAL |
| Choose stack vs heap | [perf-optimization](../patterns/perf-optimization.md) | perf | HIGH |
| Compare CPU vs GPU results | [debug-debugging](../patterns/debug-debugging.md) | debug | HIGH |
| Compare Rust and Mojo ownership models | [porting-rust](../patterns/porting-rust.md) | porting | HIGH |
| Compare algorithm implementations | [testing](../patterns/testing.md) | test | HIGH |
| Construct LayoutTensor from DeviceBuffer | [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu | CRITICAL |
| Convert AMD scheduling to Mojo | [gpu-porting-rocm](../patterns/gpu-porting-rocm.md) | gpu | HIGH |
| Convert CUDA tensor core code to Mojo | [gpu-porting-cuda](../patterns/gpu-porting-cuda.md) | gpu | HIGH |
| Convert CUTLASS GEMM to Mojo structured kernel | [gpu-porting-cute](../patterns/gpu-porting-cute.md) | gpu | HIGH |
| Convert OutputTensor/InputTensor to LayoutTensor | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu | HIGH |
| Convert Python types to Mojo | [python-interop](../patterns/python-interop.md) | python | MEDIUM |
| Create SIMD-friendly data structure | [type-simd](../patterns/type-simd.md) | type | HIGH |
| Create TMA descriptors for WGMMA | [gpu-tensor-cores](../patterns/gpu-tensor-cores.md) | gpu | CRITICAL |
| Create custom GPU operation | [gpu-kernels](../patterns/gpu-kernels.md) | gpu | HIGH |
| Create custom error type | [error-handling](../patterns/error-handling.md) | error | HIGH |
| Create custom trait | [type-traits](../patterns/type-traits.md) | type | HIGH |
| Create generic container | [memory-ownership](../patterns/memory-ownership.md) | memory | CRITICAL |
| Create generic type | [meta-programming](../patterns/meta-programming.md) | meta | MEDIUM |
| Create hashable dictionary key | [type-system](../patterns/type-system.md) | type | CRITICAL |
| Create model layer struct with weights | [struct-design](../patterns/struct-design.md) | struct | HIGH |
| Create performance regression tests | [testing](../patterns/testing.md) | test | HIGH |
| Create shared memory LayoutTensor | [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu | CRITICAL |
| Create simple data struct | [struct-design](../patterns/struct-design.md) | struct | HIGH |
| Create test suite | [testing](../patterns/testing.md) | test | HIGH |
| Create thread-safe reference counted type | [memory-refcounting](../patterns/memory-refcounting.md) | memory | HIGH |
| Debug GPU kernel correctness | [debug-debugging](../patterns/debug-debugging.md) | debug | HIGH |
| Debug GPU kernel without print | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu | CRITICAL |
| Debug numerical accuracy issues | [debug-debugging](../patterns/debug-debugging.md) | debug | HIGH |
| Define type alias | [type-system](../patterns/type-system.md) | type | CRITICAL |
| Design function API | [fn-design](../patterns/fn-design.md) | fn | HIGH |
| Dispatch between CPU and GPU implementations | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu | HIGH |
| Distribute work evenly | [perf-parallelization](../patterns/perf-parallelization.md) | perf | CRITICAL |
| Eliminate bank conflicts with swizzling | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu | HIGH |
| Find Mojo equivalent of CuTe Layout | [gpu-porting-cute](../patterns/gpu-porting-cute.md) | gpu | HIGH |
| Find Mojo equivalent of HIP API | [gpu-porting-rocm](../patterns/gpu-porting-rocm.md) | gpu | HIGH |
| Find the Mojo equivalent of C++ syntax | [porting-cpp](../patterns/porting-cpp.md) | porting | HIGH |
| Find the Mojo equivalent of Python syntax | [porting-python](../patterns/porting-python.md) | porting | HIGH |
| Find the Mojo equivalent of Rust syntax | [porting-rust](../patterns/porting-rust.md) | porting | HIGH |
| Find the Mojo equivalent of a CUDA API | [gpu-porting-cuda](../patterns/gpu-porting-cuda.md) | gpu | HIGH |
| Fix GPU out of memory error | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu | CRITICAL |
| Fix LayoutTensor element type error | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu | HIGH |
| Fix LayoutTensor type mismatch | [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu | CRITICAL |
| Fix SIMD alignment issue | [type-simd](../patterns/type-simd.md) | type | HIGH |
| Fix alignment issues | [perf-vectorization](../patterns/perf-vectorization.md) | perf | HIGH |
| Fix dangling reference error | [memory-safety](../patterns/memory-safety.md) | memory | CRITICAL |
| Fix data race in parallel code | [perf-parallelization](../patterns/perf-parallelization.md) | perf | CRITICAL |
| Fix double-free bug | [memory-refcounting](../patterns/memory-refcounting.md) | memory | HIGH |
| Fix failing test | [testing](../patterns/testing.md) | test | HIGH |
| Fix floating-point precision | [debug-debugging](../patterns/debug-debugging.md) | debug | HIGH |
| Fix integer size mismatch | [ffi](../patterns/ffi.md) | ffi | CRITICAL |
| Fix linker error | [ffi](../patterns/ffi.md) | ffi | CRITICAL |
| Fix missing trait conformance error | [type-traits](../patterns/type-traits.md) | type | HIGH |
| Fix race condition in shared memory | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu | HIGH |
| Fix type mismatch error | [type-system](../patterns/type-system.md) | type | CRITICAL |
| Fix use-after-move error | [memory-ownership](../patterns/memory-ownership.md) | memory | CRITICAL |
| Fix warp divergence issue | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu | HIGH |
| Fuse multiple GPU operations | [gpu-kernels](../patterns/gpu-kernels.md) | gpu | HIGH |
| Handle BF16/F16 conversion | [ffi](../patterns/ffi.md) | ffi | CRITICAL |
| Handle C strings safely | [ffi](../patterns/ffi.md) | ffi | CRITICAL |
| Handle GPU device initialization | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu | CRITICAL |
| Handle Optional values safely | [type-system](../patterns/type-system.md) | type | CRITICAL |
| Handle Python errors in Mojo | [python-interop](../patterns/python-interop.md) | python | MEDIUM |
| Handle errors gracefully | [error-handling](../patterns/error-handling.md) | error | HIGH |
| Handle uninitialized memory | [memory-collections](../patterns/memory-collections.md) | memory | CRITICAL |
| Implement Copyable for struct | [type-traits](../patterns/type-traits.md) | type | HIGH |
| Implement SM100 UMMA pattern | [gpu-tensor-cores](../patterns/gpu-tensor-cores.md) | gpu | CRITICAL |
| Implement async copy pipeline | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu | HIGH |
| Implement butterfly all-reduce | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu | HIGH |
| Implement caching strategy | [perf-optimization](../patterns/perf-optimization.md) | perf | MEDIUM |
| Implement collection destructor | [memory-collections](../patterns/memory-collections.md) | memory | CRITICAL |
| Implement double-buffered prefetch | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu | HIGH |
| Implement early exit for SIMD | [perf-vectorization](../patterns/perf-vectorization.md) | perf | HIGH |
| Implement model configuration struct | [struct-design](../patterns/struct-design.md) | struct | HIGH |
| Implement model state transfer | [memory-ownership](../patterns/memory-ownership.md) | memory | CRITICAL |
| Implement move constructor | [memory-ownership](../patterns/memory-ownership.md) | memory | CRITICAL |
| Implement operator overloading | [struct-design](../patterns/struct-design.md) | struct | HIGH |
| Implement parallel attention | [perf-parallelization](../patterns/perf-parallelization.md) | perf | CRITICAL |
| Implement producer-consumer pipeline | [gpu-kernels](../patterns/gpu-kernels.md) | gpu | HIGH |
| Implement shared ownership | [memory-refcounting](../patterns/memory-refcounting.md) | memory | HIGH |
| Implement tiled processing | [perf-optimization](../patterns/perf-optimization.md) | perf | HIGH |
| Implement variadic function | [meta-programming](../patterns/meta-programming.md) | meta | MEDIUM |
| Implement warp-level reduction | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu | HIGH |
| Import Python modules | [python-interop](../patterns/python-interop.md) | python | MEDIUM |
| Integrate GPU libraries | [ffi](../patterns/ffi.md) | ffi | CRITICAL |
| Integrate custom op with Python graph | [gpu-kernels](../patterns/gpu-kernels.md) | gpu | HIGH |
| Investigate output differences | [debug-debugging](../patterns/debug-debugging.md) | debug | HIGH |
| Load 2D tiles with TMA | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu | HIGH |
| Load SIMD vectors from LayoutTensor | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu | HIGH |
| Make type printable with Writable | [type-traits](../patterns/type-traits.md) | type | HIGH |
| Manage Python GIL | [ffi](../patterns/ffi.md) | ffi | CRITICAL |
| Manage buffers efficiently | [perf-optimization](../patterns/perf-optimization.md) | perf | MEDIUM |
| Manage weight matrices with UnsafePointer | [memory-ownership](../patterns/memory-ownership.md) | memory | CRITICAL |
| Map CuTe TiledCopy to Mojo TMA | [gpu-porting-cute](../patterns/gpu-porting-cute.md) | gpu | HIGH |
| Map CuTe TiledMMA to Mojo WGMMA | [gpu-porting-cute](../patterns/gpu-porting-cute.md) | gpu | HIGH |
| Measure SIMD speedup | [testing](../patterns/testing.md) | test | HIGH |
| Measure throughput metrics | [testing](../patterns/testing.md) | test | HIGH |
| Migrate C++ class to Mojo struct | [porting-cpp](../patterns/porting-cpp.md) | porting | HIGH |
| Migrate Python class to Mojo struct | [porting-python](../patterns/porting-python.md) | porting | HIGH |
| Migrate Rust struct to Mojo | [porting-rust](../patterns/porting-rust.md) | porting | HIGH |
| Minimize boundary crossings | [python-interop](../patterns/python-interop.md) | python | MEDIUM |
| Normalize values across a block | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu | HIGH |
| Optimize MPS performance | [ffi](../patterns/ffi.md) | ffi | CRITICAL |
| Optimize kernel launch configuration | [gpu-kernels](../patterns/gpu-kernels.md) | gpu | HIGH |
| Optimize memory access pattern | [perf-optimization](../patterns/perf-optimization.md) | perf | HIGH |
| Optimize memory coalescing | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu | CRITICAL |
| Optimize memory coalescing | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu | HIGH |
| Optimize model loading | [perf-optimization](../patterns/perf-optimization.md) | perf | MEDIUM |
| Optimize multi-core performance | [perf-parallelization](../patterns/perf-parallelization.md) | perf | CRITICAL |
| Optimize numeric computation | [perf-vectorization](../patterns/perf-vectorization.md) | perf | HIGH |
| Organize struct code properly | [struct-design](../patterns/struct-design.md) | struct | HIGH |
| Overload function for different types | [fn-design](../patterns/fn-design.md) | fn | HIGH |
| Package Mojo kernel as mojopkg | [gpu-kernels](../patterns/gpu-kernels.md) | gpu | HIGH |
| Package custom op as mojopkg | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu | HIGH |
| Parallel stream compaction | [gpu-block-collectives](../patterns/gpu-block-collectives.md) | gpu | HIGH |
| Parallelize loop across CPU cores | [perf-parallelization](../patterns/perf-parallelization.md) | perf | CRITICAL |
| Port C code to Mojo | [debug-debugging](../patterns/debug-debugging.md) | debug | HIGH |
| Port CUDA C++ kernel to Mojo | [porting-cpp](../patterns/porting-cpp.md) | porting | HIGH |
| Port CUDA kernel to AMD | [gpu-amd](../patterns/gpu-amd.md) | gpu | MEDIUM |
| Port CuTe DSL kernel to Mojo | [gpu-porting-cute](../patterns/gpu-porting-cute.md) | gpu | HIGH |
| Port ROCm warp operations to Mojo | [gpu-porting-rocm](../patterns/gpu-porting-rocm.md) | gpu | HIGH |
| Port Rust SIMD code to Mojo | [porting-rust](../patterns/porting-rust.md) | porting | HIGH |
| Port a C++ program to Mojo | [porting-cpp](../patterns/porting-cpp.md) | porting | HIGH |
| Port a CUDA kernel to Mojo | [gpu-porting-cuda](../patterns/gpu-porting-cuda.md) | gpu | HIGH |
| Port a HIP kernel to Mojo | [gpu-porting-rocm](../patterns/gpu-porting-rocm.md) | gpu | HIGH |
| Port a Python script to Mojo | [porting-python](../patterns/porting-python.md) | porting | HIGH |
| Port a Rust program to Mojo | [porting-rust](../patterns/porting-rust.md) | porting | HIGH |
| Precompute at compile time | [perf-optimization](../patterns/perf-optimization.md) | perf | MEDIUM |
| Property-based testing | [testing](../patterns/testing.md) | test | HIGH |
| Read from LayoutTensor without rebind error | [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu | CRITICAL |
| Reduce cache misses | [perf-optimization](../patterns/perf-optimization.md) | perf | HIGH |
| Reduce startup overhead | [perf-optimization](../patterns/perf-optimization.md) | perf | MEDIUM |
| Register a custom GPU operation for MAX Graph | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu | HIGH |
| Replace C++ templates with Mojo parameters | [porting-cpp](../patterns/porting-cpp.md) | porting | HIGH |
| Replace NumPy with native Mojo SIMD | [porting-python](../patterns/porting-python.md) | porting | HIGH |
| Reshape LayoutTensor across ranks | [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu | CRITICAL |
| Reshape LayoutTensor between ranks | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu | HIGH |
| Return value with ownership transfer | [memory-ownership](../patterns/memory-ownership.md) | memory | CRITICAL |
| Return reference from function safely | [memory-safety](../patterns/memory-safety.md) | memory | CRITICAL |
| Rewrite CUDA shared memory pattern in Mojo | [gpu-porting-cuda](../patterns/gpu-porting-cuda.md) | gpu | HIGH |
| Select optimal MFMA shape | [gpu-amd](../patterns/gpu-amd.md) | gpu | MEDIUM |
| Specialize warps for producer/consumer | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu | HIGH |
| Speed up Python numeric code with Mojo | [porting-python](../patterns/porting-python.md) | porting | HIGH |
| Store reference in struct | [memory-safety](../patterns/memory-safety.md) | memory | CRITICAL |
| Synchronize threads in block | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu | HIGH |
| Test GPU kernel correctness | [testing](../patterns/testing.md) | test | HIGH |
| Test lifecycle methods | [testing](../patterns/testing.md) | test | HIGH |
| Tile a LayoutTensor | [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu | CRITICAL |
| Transfer ownership to function | [memory-ownership](../patterns/memory-ownership.md) | memory | CRITICAL |
| Unpack parameter lists | [meta-programming](../patterns/meta-programming.md) | meta | MEDIUM |
| Unroll loop for performance | [perf-vectorization](../patterns/perf-vectorization.md) | perf | HIGH |
| Use @always_inline for hot path | [fn-design](../patterns/fn-design.md) | fn | HIGH |
| Use @fieldwise_init decorator | [struct-design](../patterns/struct-design.md) | struct | HIGH |
| Use Apple BLAS for matrix multiply | [ffi](../patterns/ffi.md) | ffi | CRITICAL |
| Use Python libraries from Mojo | [python-interop](../patterns/python-interop.md) | python | MEDIUM |
| Use Span for non-owning view | [memory-collections](../patterns/memory-collections.md) | memory | CRITICAL |
| Use async copy for latency hiding | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu | HIGH |
| Use block collectives for reduction | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu | HIGH |
| Use compile-time parameters | [meta-programming](../patterns/meta-programming.md) | meta | MEDIUM |
| Use context manager for cleanup | [error-handling](../patterns/error-handling.md) | error | HIGH |
| Use double-buffering for latency hiding | [gpu-kernels](../patterns/gpu-kernels.md) | gpu | HIGH |
| Use grid-stride pattern | [perf-vectorization](../patterns/perf-vectorization.md) | perf | HIGH |
| Use mbarrier for TMA operations | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu | HIGH |
| Use memory-mapped files | [perf-optimization](../patterns/perf-optimization.md) | perf | MEDIUM |
| Use multiple accumulators | [perf-optimization](../patterns/perf-optimization.md) | perf | HIGH |
| Use prefetching | [perf-optimization](../patterns/perf-optimization.md) | perf | HIGH |
| Use register-passable types | [type-simd](../patterns/type-simd.md) | type | HIGH |
| Use s_waitcnt correctly | [gpu-amd](../patterns/gpu-amd.md) | gpu | MEDIUM |
| Use shared memory for reduction | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu | CRITICAL |
| Use shuffle for fast communication | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu | HIGH |
| Use shuffle_xor for pair communication | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu | HIGH |
| Use tensor cores for matrix multiply | [gpu-tensor-cores](../patterns/gpu-tensor-cores.md) | gpu | CRITICAL |
| Use vectorize in GPU kernel | [perf-vectorization](../patterns/perf-vectorization.md) | perf | HIGH |
| Use zero-copy loading | [perf-optimization](../patterns/perf-optimization.md) | perf | MEDIUM |
| Vectorize loop with SIMD | [perf-vectorization](../patterns/perf-vectorization.md) | perf | HIGH |
| Vectorize numeric loop | [type-simd](../patterns/type-simd.md) | type | HIGH |
| Work with binary data | [ffi](../patterns/ffi.md) | ffi | CRITICAL |
| Write MFMA kernel in Mojo | [gpu-porting-rocm](../patterns/gpu-porting-rocm.md) | gpu | HIGH |
| Write WGMMA kernel for H100 | [gpu-tensor-cores](../patterns/gpu-tensor-cores.md) | gpu | CRITICAL |
| Write a GPU kernel in Mojo | [gpu-porting-cuda](../patterns/gpu-porting-cuda.md) | gpu | HIGH |
| Write first GPU kernel | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu | CRITICAL |
| Write kernel for AMD MI300X | [gpu-amd](../patterns/gpu-amd.md) | gpu | MEDIUM |
| Write unit tests for Mojo code | [testing](../patterns/testing.md) | test | HIGH |
| Write zero-cost abstractions | [meta-programming](../patterns/meta-programming.md) | meta | MEDIUM |
| Zero output buffer before kernel launch | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu | HIGH |

## Patterns by Scenario Count

| Pattern | Category | Scenarios Covered |
|---------|----------|-------------------|
| [gpu-kernels](../patterns/gpu-kernels.md) | gpu | 8 |
| [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu | 8 |
| [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu | 8 |
| [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu | 7 |
| [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu | 7 |
| [memory-ownership](../patterns/memory-ownership.md) | memory | 7 |
| [perf-optimization](../patterns/perf-optimization.md) | perf | 14 |
| [perf-vectorization](../patterns/perf-vectorization.md) | perf | 7 |
| [struct-design](../patterns/struct-design.md) | struct | 7 |
| [debug-debugging](../patterns/debug-debugging.md) | debug | 6 |
| [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu | 6 |
| [meta-programming](../patterns/meta-programming.md) | meta | 6 |
| [perf-parallelization](../patterns/perf-parallelization.md) | perf | 6 |
| [python-interop](../patterns/python-interop.md) | python | 6 |
| [testing](../patterns/testing.md) | test | 11 |
| [error-handling](../patterns/error-handling.md) | error | 5 |
| [ffi](../patterns/ffi.md) | ffi | 10 |
| [gpu-porting-cuda](../patterns/gpu-porting-cuda.md) | gpu | 5 |
| [gpu-porting-cute](../patterns/gpu-porting-cute.md) | gpu | 5 |
| [gpu-porting-rocm](../patterns/gpu-porting-rocm.md) | gpu | 5 |
| [porting-cpp](../patterns/porting-cpp.md) | porting | 5 |
| [porting-python](../patterns/porting-python.md) | porting | 5 |
| [porting-rust](../patterns/porting-rust.md) | porting | 5 |
| [type-system](../patterns/type-system.md) | type | 5 |
| [type-traits](../patterns/type-traits.md) | type | 5 |
| [fn-design](../patterns/fn-design.md) | fn | 4 |
| [gpu-amd](../patterns/gpu-amd.md) | gpu | 4 |
| [gpu-tensor-cores](../patterns/gpu-tensor-cores.md) | gpu | 4 |
| [memory-collections](../patterns/memory-collections.md) | memory | 4 |
| [memory-refcounting](../patterns/memory-refcounting.md) | memory | 4 |
| [memory-safety](../patterns/memory-safety.md) | memory | 4 |
| [type-simd](../patterns/type-simd.md) | type | 4 |
| [gpu-block-collectives](../patterns/gpu-block-collectives.md) | gpu | 3 |

---
