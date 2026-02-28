# Mojo Best Practices Error Index

This index maps common error messages to relevant patterns.

## Error Message Lookup

| Error Message | Pattern | Category |
|--------------|---------|----------|
| `@compiler.register` | [gpu-kernels](../patterns/gpu-kernels.md) | gpu |
| `AMD GPU kernel` | [gpu-porting-rocm](../patterns/gpu-porting-rocm.md) | gpu |
| `Accelerate` | [ffi](../patterns/ffi.md) | ffi |
| `Apple Silicon` | [gpu-block-collectives](../patterns/gpu-block-collectives.md) | gpu |
| `Arc` | [memory-refcounting](../patterns/memory-refcounting.md) | memory |
| `BF16` | [ffi](../patterns/ffi.md) | ffi |
| `C++ equivalent` | [porting-cpp](../patterns/porting-cpp.md) | porting |
| `CString` | [ffi](../patterns/ffi.md) | ffi |
| `CUDA equivalent` | [gpu-porting-cuda](../patterns/gpu-porting-cuda.md) | gpu |
| `CUDA error: invalid device` | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu |
| `CUDA error: kernel launch` | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu |
| `CUDA error: out of memory` | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu |
| `CUTLASS to Mojo` | [gpu-porting-cute](../patterns/gpu-porting-cute.md) | gpu |
| `CuTe equivalent` | [gpu-porting-cute](../patterns/gpu-porting-cute.md) | gpu |
| `OwnedDLHandle` | [ffi](../patterns/ffi.md) | ffi |
| `DeviceBuffer` | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu |
| `GIL` | [python-interop](../patterns/python-interop.md) | python |
| `GIL deadlock` | [ffi](../patterns/ffi.md) | ffi |
| `GPU OOM` | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu |
| `HIP equivalent` | [gpu-porting-rocm](../patterns/gpu-porting-rocm.md) | gpu |
| `HIP error` | [gpu-amd](../patterns/gpu-amd.md) | gpu |
| `IndexList` | [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu |
| `InputTensor` | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu |
| `Layout heap allocation` | [gpu-structured-kernels](../patterns/gpu-structured-kernels.md) | gpu |
| `MFMA error` | [gpu-amd](../patterns/gpu-amd.md) | gpu |
| `MFMA in Mojo` | [gpu-porting-rocm](../patterns/gpu-porting-rocm.md) | gpu |
| `MI300` | [gpu-amd](../patterns/gpu-amd.md) | gpu |
| `MMA shape mismatch` | [gpu-tensor-cores](../patterns/gpu-tensor-cores.md) | gpu |
| `MPS error` | [ffi](../patterns/ffi.md) | ffi |
| `Metal GPU returns zeros` | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu |
| `Metal kernel not executing` | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu |
| `MutAnyOrigin` | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu |
| `NaN` | [debug-debugging](../patterns/debug-debugging.md) | debug |
| `OutputTensor` | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu |
| `Python` | [python-interop](../patterns/python-interop.md) | python |
| `Python equivalent` | [porting-python](../patterns/porting-python.md) | porting |
| `PythonObject` | [python-interop](../patterns/python-interop.md) | python |
| `QuickBench` | [testing](../patterns/testing.md) | test |
| `ROCm` | [gpu-amd](../patterns/gpu-amd.md) | gpu |
| `RuntimeLayout` | [gpu-structured-kernels](../patterns/gpu-structured-kernels.md) | gpu |
| `Rust equivalent` | [porting-rust](../patterns/porting-rust.md) | porting |
| `Rust vs Mojo` | [porting-rust](../patterns/porting-rust.md) | porting |
| `SGPR pointer` | [gpu-structured-kernels](../patterns/gpu-structured-kernels.md) | gpu |
| `SIMD width` | [perf-vectorization](../patterns/perf-vectorization.md) | perf |
| `SIMD width mismatch` | [type-simd](../patterns/type-simd.md) | type |
| `Self.T vs T` | [struct-design](../patterns/struct-design.md) | struct |
| `TCGEN05` | [gpu-tensor-cores](../patterns/gpu-tensor-cores.md) | gpu |
| `TMA error` | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu |
| `TestSuite` | [testing](../patterns/testing.md) | test |
| `TiledCopy` | [gpu-porting-cute](../patterns/gpu-porting-cute.md) | gpu |
| `TiledMMA` | [gpu-porting-cute](../patterns/gpu-porting-cute.md) | gpu |
| `UMMA` | [gpu-tensor-cores](../patterns/gpu-tensor-cores.md) | gpu |
| `WGMMA error` | [gpu-tensor-cores](../patterns/gpu-tensor-cores.md) | gpu |
| `abandoned without being explicitly destroyed` | [memory-ownership](../patterns/memory-ownership.md) | memory |
| `alias` | [meta-programming](../patterns/meta-programming.md) | meta |
| `alignment` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `alignment` | [perf-vectorization](../patterns/perf-vectorization.md) | perf |
| `alignment error` | [type-simd](../patterns/type-simd.md) | type |
| `allocation overhead` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `argument convention` | [fn-design](../patterns/fn-design.md) | fn |
| `assert_equal` | [testing](../patterns/testing.md) | test |
| `assert_false` | [testing](../patterns/testing.md) | test |
| `assert_true` | [testing](../patterns/testing.md) | test |
| `assertion failed` | [testing](../patterns/testing.md) | test |
| `bank conflict` | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu |
| `barrier` | [gpu-block-collectives](../patterns/gpu-block-collectives.md) | gpu |
| `barrier deadlock` | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu |
| `benchmark` | [testing](../patterns/testing.md) | test |
| `bitcast` | [gpu-structured-kernels](../patterns/gpu-structured-kernels.md) | gpu |
| `block reduction` | [gpu-block-collectives](../patterns/gpu-block-collectives.md) | gpu |
| `block size` | [gpu-kernels](../patterns/gpu-kernels.md) | gpu |
| `block_size parameter` | [gpu-block-collectives](../patterns/gpu-block-collectives.md) | gpu |
| `block_size required` | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu |
| `boundary crossing` | [python-interop](../patterns/python-interop.md) | python |
| `buffer` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `buffer overflow` | [memory-collections](../patterns/memory-collections.md) | memory |
| `cache` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `cache line` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `cache miss` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `cannot be converted from.*capturing.*to.*func` | [perf-vectorization](../patterns/perf-vectorization.md) | perf |
| `cannot borrow .* as mutable` | [memory-ownership](../patterns/memory-ownership.md) | memory |
| `cannot convert .* to` | [type-system](../patterns/type-system.md) | type |
| `cannot implicitly convert 'LayoutTensor` | [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu |
| `cannot implicitly convert 'LayoutTensor` | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu |
| `cannot implicitly convert 'SIMD` | [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu |
| `cannot infer` | [meta-programming](../patterns/meta-programming.md) | meta |
| `cannot pass .* to` | [fn-design](../patterns/fn-design.md) | fn |
| `cannot use .* as mutable` | [struct-design](../patterns/struct-design.md) | struct |
| `cannot vectorize` | [perf-vectorization](../patterns/perf-vectorization.md) | perf |
| `cannot vectorize` | [type-simd](../patterns/type-simd.md) | type |
| `compilation time` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `comptime` | [meta-programming](../patterns/meta-programming.md) | meta |
| `concurrent` | [perf-parallelization](../patterns/perf-parallelization.md) | perf |
| `convert C++ to Mojo` | [porting-cpp](../patterns/porting-cpp.md) | porting |
| `convert CUDA to Mojo` | [gpu-porting-cuda](../patterns/gpu-porting-cuda.md) | gpu |
| `convert Python to Mojo` | [porting-python](../patterns/porting-python.md) | porting |
| `convert ROCm to Mojo` | [gpu-porting-rocm](../patterns/gpu-porting-rocm.md) | gpu |
| `convert Rust to Mojo` | [porting-rust](../patterns/porting-rust.md) | porting |
| `cuBLAS` | [ffi](../patterns/ffi.md) | ffi |
| `custom op registration` | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu |
| `dangling reference` | [memory-safety](../patterns/memory-safety.md) | memory |
| `data race` | [memory-refcounting](../patterns/memory-refcounting.md) | memory |
| `data race` | [perf-parallelization](../patterns/perf-parallelization.md) | perf |
| `descriptor invalid` | [gpu-tensor-cores](../patterns/gpu-tensor-cores.md) | gpu |
| `device not found` | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu |
| `divergence` | [debug-debugging](../patterns/debug-debugging.md) | debug |
| `does not conform to trait` | [type-traits](../patterns/type-traits.md) | type |
| `does not implement Copyable` | [type-traits](../patterns/type-traits.md) | type |
| `does not implement Equatable` | [type-system](../patterns/type-system.md) | type |
| `does not implement Hashable` | [type-system](../patterns/type-system.md) | type |
| `does not implement Movable` | [type-traits](../patterns/type-traits.md) | type |
| `does not implement Writable` | [type-traits](../patterns/type-traits.md) | type |
| `does not match result type` | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu |
| `does not support operation: print` | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu |
| `double free` | [memory-collections](../patterns/memory-collections.md) | memory |
| `double free` | [memory-refcounting](../patterns/memory-refcounting.md) | memory |
| `element_type` | [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu |
| `enqueue_function` | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu |
| `expected 'Scalar' but got 'SIMD'` | [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu |
| `expected .* argument` | [fn-design](../patterns/fn-design.md) | fn |
| `expected .* but got` | [testing](../patterns/testing.md) | test |
| `expected .* but got` | [type-system](../patterns/type-system.md) | type |
| `failed to infer parameter 'mut'` | [memory-ownership](../patterns/memory-ownership.md) | memory |
| `false sharing` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `field .* not initialized` | [struct-design](../patterns/struct-design.md) | struct |
| `floating-point` | [debug-debugging](../patterns/debug-debugging.md) | debug |
| `generic` | [meta-programming](../patterns/meta-programming.md) | meta |
| `grid size` | [gpu-kernels](../patterns/gpu-kernels.md) | gpu |
| `illegal memory access` | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu |
| `import` | [python-interop](../patterns/python-interop.md) | python |
| `incorrect result` | [debug-debugging](../patterns/debug-debugging.md) | debug |
| `inf` | [debug-debugging](../patterns/debug-debugging.md) | debug |
| `interpreter` | [python-interop](../patterns/python-interop.md) | python |
| `invalid SIMD operation` | [type-simd](../patterns/type-simd.md) | type |
| `invalid lane` | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu |
| `kernel changes not reflected` | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu |
| `kernel launch failed` | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu |
| `kernel launch failed` | [gpu-kernels](../patterns/gpu-kernels.md) | gpu |
| `layout algebra` | [gpu-porting-cute](../patterns/gpu-porting-cute.md) | gpu |
| `lazy loading` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `library not found` | [ffi](../patterns/ffi.md) | ffi |
| `lifetime of .* does not outlive` | [memory-safety](../patterns/memory-safety.md) | memory |
| `linker error` | [ffi](../patterns/ffi.md) | ffi |
| `mbarrier` | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu |
| `memory bandwidth` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `memory leak` | [memory-collections](../patterns/memory-collections.md) | memory |
| `memory leak` | [memory-refcounting](../patterns/memory-refcounting.md) | memory |
| `memory mapped` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `migrate from C++` | [porting-cpp](../patterns/porting-cpp.md) | porting |
| `migrate from CUDA` | [gpu-porting-cuda](../patterns/gpu-porting-cuda.md) | gpu |
| `migrate from Python` | [porting-python](../patterns/porting-python.md) | porting |
| `migrate from Rust` | [porting-rust](../patterns/porting-rust.md) | porting |
| `misaligned address` | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu |
| `misaligned address` | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu |
| `missing __init__` | [struct-design](../patterns/struct-design.md) | struct |
| `missing trait bound` | [type-traits](../patterns/type-traits.md) | type |
| `missing try block` | [error-handling](../patterns/error-handling.md) | error |
| `missing type annotation` | [type-system](../patterns/type-system.md) | type |
| `mmap` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `mojopkg` | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu |
| `mut` | [fn-design](../patterns/fn-design.md) | fn |
| `no matching method in call to 'enqueue_function'` | [gpu-kernels](../patterns/gpu-kernels.md) | gpu |
| `no matching method in call to 'load'` | [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu |
| `no matching method in call to 'load'` | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu |
| `no matching method in call to 'store'` | [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu |
| `num_physical_cores` | [perf-parallelization](../patterns/perf-parallelization.md) | perf |
| `numerical accuracy` | [debug-debugging](../patterns/debug-debugging.md) | debug |
| `occupancy` | [gpu-kernels](../patterns/gpu-kernels.md) | gpu |
| `op not found` | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu |
| `origin mismatch` | [memory-safety](../patterns/memory-safety.md) | memory |
| `out of bounds` | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu |
| `ownership transfer required` | [memory-ownership](../patterns/memory-ownership.md) | memory |
| `parallelize` | [perf-parallelization](../patterns/perf-parallelization.md) | perf |
| `parameter` | [meta-programming](../patterns/meta-programming.md) | meta |
| `per-K accumulation` | [gpu-structured-kernels](../patterns/gpu-structured-kernels.md) | gpu |
| `perf_counter` | [testing](../patterns/testing.md) | test |
| `port C++ to Mojo` | [porting-cpp](../patterns/porting-cpp.md) | porting |
| `port CUDA kernel` | [gpu-porting-cuda](../patterns/gpu-porting-cuda.md) | gpu |
| `port HIP kernel` | [gpu-porting-rocm](../patterns/gpu-porting-rocm.md) | gpu |
| `port Python to Mojo` | [porting-python](../patterns/porting-python.md) | porting |
| `port Rust to Mojo` | [porting-rust](../patterns/porting-rust.md) | porting |
| `precision` | [debug-debugging](../patterns/debug-debugging.md) | debug |
| `prefetch` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `prefix sum` | [gpu-block-collectives](../patterns/gpu-block-collectives.md) | gpu |
| `race condition` | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu |
| `raises not declared` | [error-handling](../patterns/error-handling.md) | error |
| `readfirstlane` | [gpu-structured-kernels](../patterns/gpu-structured-kernels.md) | gpu |
| `rebind` | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu |
| `rebind` | [gpu-structured-kernels](../patterns/gpu-structured-kernels.md) | gpu |
| `rebind input type` | [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu |
| `rebind input type` | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu |
| `reduction` | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu |
| `reference count underflow` | [memory-refcounting](../patterns/memory-refcounting.md) | memory |
| `reference to local variable` | [memory-safety](../patterns/memory-safety.md) | memory |
| `register spill` | [gpu-kernels](../patterns/gpu-kernels.md) | gpu |
| `resource leak` | [error-handling](../patterns/error-handling.md) | error |
| `results differ` | [debug-debugging](../patterns/debug-debugging.md) | debug |
| `rewrite C++ in Mojo` | [porting-cpp](../patterns/porting-cpp.md) | porting |
| `rewrite Rust in Mojo` | [porting-rust](../patterns/porting-rust.md) | porting |
| `rewrite in Mojo` | [porting-python](../patterns/porting-python.md) | porting |
| `rewrite kernel in Mojo` | [gpu-porting-cuda](../patterns/gpu-porting-cuda.md) | gpu |
| `rocBLAS` | [ffi](../patterns/ffi.md) | ffi |
| `s_barrier vs barrier` | [gpu-structured-kernels](../patterns/gpu-structured-kernels.md) | gpu |
| `s_waitcnt` | [gpu-amd](../patterns/gpu-amd.md) | gpu |
| `shared memory` | [gpu-block-collectives](../patterns/gpu-block-collectives.md) | gpu |
| `shuffle error` | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu |
| `shuffle_xor` | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu |
| `simd_width_of` | [perf-vectorization](../patterns/perf-vectorization.md) | perf |
| `slow memory access` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `slow performance` | [perf-parallelization](../patterns/perf-parallelization.md) | perf |
| `slow performance` | [perf-vectorization](../patterns/perf-vectorization.md) | perf |
| `slow startup` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `speed up Python` | [porting-python](../patterns/porting-python.md) | porting |
| `struct .* has no member` | [struct-design](../patterns/struct-design.md) | struct |
| `swizzle` | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu |
| `symbol not found` | [ffi](../patterns/ffi.md) | ffi |
| `sync error` | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu |
| `template to parameter` | [porting-cpp](../patterns/porting-cpp.md) | porting |
| `tensor core` | [gpu-tensor-cores](../patterns/gpu-tensor-cores.md) | gpu |
| `test failed` | [testing](../patterns/testing.md) | test |
| `thread` | [perf-parallelization](../patterns/perf-parallelization.md) | perf |
| `throughput` | [testing](../patterns/testing.md) | test |
| `timing` | [testing](../patterns/testing.md) | test |
| `to_layout_tensor` | [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu |
| `trait bound` | [meta-programming](../patterns/meta-programming.md) | meta |
| `type conversion` | [python-interop](../patterns/python-interop.md) | python |
| `type mismatch` | [type-system](../patterns/type-system.md) | type |
| `type parameter` | [meta-programming](../patterns/meta-programming.md) | meta |
| `unaligned` | [perf-optimization](../patterns/perf-optimization.md) | perf |
| `unaligned access` | [perf-vectorization](../patterns/perf-vectorization.md) | perf |
| `uncaught error` | [error-handling](../patterns/error-handling.md) | error |
| `uncoalesced access` | [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu |
| `uncoalesced memory access` | [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu |
| `undefined symbol` | [ffi](../patterns/ffi.md) | ffi |
| `unhandled exception` | [error-handling](../patterns/error-handling.md) | error |
| `uninitialized memory` | [memory-collections](../patterns/memory-collections.md) | memory |
| `use after free` | [memory-collections](../patterns/memory-collections.md) | memory |
| `use after free` | [memory-refcounting](../patterns/memory-refcounting.md) | memory |
| `use after free` | [memory-safety](../patterns/memory-safety.md) | memory |
| `use of moved value` | [memory-ownership](../patterns/memory-ownership.md) | memory |
| `value borrowed after move` | [memory-ownership](../patterns/memory-ownership.md) | memory |
| `var` | [fn-design](../patterns/fn-design.md) | fn |
| `variadic` | [meta-programming](../patterns/meta-programming.md) | meta |
| `vector type` | [perf-vectorization](../patterns/perf-vectorization.md) | perf |
| `vote mask` | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu |
| `warp divergence` | [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu |
| `wavefront` | [gpu-amd](../patterns/gpu-amd.md) | gpu |
| `width must be` | [type-simd](../patterns/type-simd.md) | type |
| `work distribution` | [perf-parallelization](../patterns/perf-parallelization.md) | perf |
| `wrong output` | [debug-debugging](../patterns/debug-debugging.md) | debug |

## Patterns by Error Count

| Pattern | Category | Error Patterns Covered |
|---------|----------|----------------------|
| [gpu-fundamentals](../patterns/gpu-fundamentals.md) | gpu | 11 |
| [gpu-custom-ops](../patterns/gpu-custom-ops.md) | gpu | 10 |
| [gpu-memory-access](../patterns/gpu-memory-access.md) | gpu | 10 |
| [debug-debugging](../patterns/debug-debugging.md) | debug | 9 |
| [gpu-layout-tensor](../patterns/gpu-layout-tensor.md) | gpu | 8 |
| [gpu-structured-kernels](../patterns/gpu-structured-kernels.md) | gpu | 8 |
| [meta-programming](../patterns/meta-programming.md) | meta | 8 |
| [perf-optimization](../patterns/perf-optimization.md) | perf | 8 |
| [perf-optimization](../patterns/perf-optimization.md) | perf | 8 |
| [perf-vectorization](../patterns/perf-vectorization.md) | perf | 8 |
| [gpu-kernels](../patterns/gpu-kernels.md) | gpu | 7 |
| [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu | 7 |
| [perf-parallelization](../patterns/perf-parallelization.md) | perf | 7 |
| [python-interop](../patterns/python-interop.md) | python | 7 |
| [testing](../patterns/testing.md) | test | 7 |
| [ffi](../patterns/ffi.md) | ffi | 6 |
| [ffi](../patterns/ffi.md) | ffi | 6 |
| [gpu-amd](../patterns/gpu-amd.md) | gpu | 6 |
| [gpu-block-collectives](../patterns/gpu-block-collectives.md) | gpu | 6 |
| [gpu-warp-sync](../patterns/gpu-warp-sync.md) | gpu | 6 |
| [gpu-tensor-cores](../patterns/gpu-tensor-cores.md) | gpu | 6 |
| [memory-ownership](../patterns/memory-ownership.md) | memory | 6 |
| [memory-refcounting](../patterns/memory-refcounting.md) | memory | 6 |
| [porting-cpp](../patterns/porting-cpp.md) | porting | 6 |
| [porting-python](../patterns/porting-python.md) | porting | 6 |
| [porting-rust](../patterns/porting-rust.md) | porting | 6 |
| [type-system](../patterns/type-system.md) | type | 6 |
| [error-handling](../patterns/error-handling.md) | error | 5 |
| [fn-design](../patterns/fn-design.md) | fn | 5 |
| [gpu-porting-cuda](../patterns/gpu-porting-cuda.md) | gpu | 5 |
| [gpu-porting-cute](../patterns/gpu-porting-cute.md) | gpu | 5 |
| [gpu-porting-rocm](../patterns/gpu-porting-rocm.md) | gpu | 5 |
| [memory-collections](../patterns/memory-collections.md) | memory | 5 |
| [memory-safety](../patterns/memory-safety.md) | memory | 5 |
| [struct-design](../patterns/struct-design.md) | struct | 5 |
| [testing](../patterns/testing.md) | test | 5 |
| [type-simd](../patterns/type-simd.md) | type | 5 |
| [type-traits](../patterns/type-traits.md) | type | 5 |

---
