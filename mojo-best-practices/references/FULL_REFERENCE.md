# Mojo Best Practices - Full Reference

> **Auto-generated.** Do not edit manually.
>
> **Note:** Start with [SKILL.md](../SKILL.md) for the recommended entry point. This file is a complete index for deep dives.

## Table of Contents

**40 patterns** across **13 categories**

- [C Interoperability](#c-interoperability) (2 patterns)
- [GPU Programming](#gpu-programming) (15 patterns)
- [Memory Safety & Ownership](#memory-safety--ownership) (4 patterns)
- [Type System](#type-system) (3 patterns)
- [Debugging](#debugging) (1 patterns)
- [Function Design](#function-design) (1 patterns)
- [Struct Design](#struct-design) (1 patterns)
- [Testing](#testing) (2 patterns)
- [Error Handling](#error-handling) (1 patterns)
- [Performance Optimization](#performance-optimization) (4 patterns)
- [Python Interoperability](#python-interoperability) (1 patterns)
- [Advanced Metaprogramming](#advanced-metaprogramming) (2 patterns)
- [Other](#other) (3 patterns)

---

## C Interoperability

**Priority:** CRITICAL | **Patterns:** 2

### FFI and C Interoperability

**Pattern:** `ffi-interop` | **Impact:** CRITICAL

Core FFI patterns including C strings, libc functions, binary data, integer type safety, string handling, and dynamic library loading

See: [../patterns/ffi-interop.md](../patterns/ffi-interop.md)

### FFI Vendor Library Integration

**Pattern:** `ffi-vendor` | **Impact:** CRITICAL

Patterns for integrating vendor libraries, GPU frameworks, Apple BLAS/AMX, dynamic loading, Python GIL management, and platform-specific FFI considerations

See: [../patterns/ffi-vendor.md](../patterns/ffi-vendor.md)

---

## GPU Programming

**Priority:** CRITICAL | **Patterns:** 15

### AMD GPU Programming

**Pattern:** `gpu-amd` | **Impact:** MEDIUM

MFMA shapes, scheduling barriers, and waitcnt for AMD CDNA GPUs

See: [../patterns/gpu-amd.md](../patterns/gpu-amd.md)

### Block-Level Collective Operations

**Pattern:** `gpu-block-collectives` | **Impact:** HIGH

Block-wide sum, prefix_sum, broadcast operations that replace manual shared memory + barrier patterns

See: [../patterns/gpu-block-collectives.md](../patterns/gpu-block-collectives.md)

### Custom GPU Operations for MAX Graph

**Pattern:** `gpu-custom-ops` | **Impact:** HIGH

Custom op registration, OutputTensor/InputTensor types, Python graph integration, multi-kernel orchestration

See: [../patterns/gpu-custom-ops.md](../patterns/gpu-custom-ops.md)

### GPU Programming Fundamentals

**Pattern:** `gpu-fundamentals` | **Impact:** CRITICAL

Core GPU programming concepts including thread hierarchy, memory model, kernel patterns, and device context management

See: [../patterns/gpu-fundamentals.md](../patterns/gpu-fundamentals.md)

### GPU Kernel Optimization

**Pattern:** `gpu-kernels` | **Impact:** HIGH

Kernel fusion, producer-consumer pipelines, and double-buffering patterns

See: [../patterns/gpu-kernels.md](../patterns/gpu-kernels.md)

### LayoutTensor API and Patterns

**Pattern:** `gpu-layout-tensor` | **Impact:** CRITICAL

LayoutTensor indexing, rebind pattern, tile API, shared memory allocation, cross-rank reshape

See: [../patterns/gpu-layout-tensor.md](../patterns/gpu-layout-tensor.md)

### GPU Memory Access Patterns

**Pattern:** `gpu-memory-access` | **Impact:** HIGH

TMA hardware loading, prefetch patterns, shared memory swizzling, and dynamic data caching

See: [../patterns/gpu-memory-access.md](../patterns/gpu-memory-access.md)

### CUDA to Mojo Porting Guide

**Pattern:** `gpu-porting-cuda` | **Impact:** HIGH

Side-by-side CUDA→Mojo porting guide with complete examples from vector add to tensor core GEMM

See: [../patterns/gpu-porting-cuda.md](../patterns/gpu-porting-cuda.md)

### CuTe DSL to Mojo Porting Guide

**Pattern:** `gpu-porting-cute` | **Impact:** HIGH

Map NVIDIA CuTe DSL layout algebra, TiledCopy, and TiledMMA to Mojo LayoutTensor, TMA, and WGMMA equivalents

See: [../patterns/gpu-porting-cute.md](../patterns/gpu-porting-cute.md)

### ROCm/HIP to Mojo Porting Guide

**Pattern:** `gpu-porting-rocm` | **Impact:** HIGH

Side-by-side ROCm/HIP→Mojo porting guide covering MFMA tensor cores, wavefront operations, LDS, and AMD scheduling

See: [../patterns/gpu-porting-rocm.md](../patterns/gpu-porting-rocm.md)

### Structured GPU Kernel Architecture

**Pattern:** `gpu-structured-kernels` | **Impact:** HIGH

Three-component kernel pattern (ScatterGather, RingBuffer, TileOp) for maintainable high-performance GPU code with 48% code reduction

See: [../patterns/gpu-structured-kernels.md](../patterns/gpu-structured-kernels.md)

### GPU Synchronization and Async Operations

**Pattern:** `gpu-synchronization` | **Impact:** HIGH

Synchronization primitives including barriers, mbarriers, async transactions, and async copy patterns

See: [../patterns/gpu-synchronization.md](../patterns/gpu-synchronization.md)

### Tensor Core Programming for SM90 and SM100

**Pattern:** `gpu-tensor-cores` | **Impact:** CRITICAL

WGMMA (SM90), UMMA/TCGEN05 (SM100), tensor memory, and descriptor patterns for maximum tensor core throughput

See: [../patterns/gpu-tensor-cores.md](../patterns/gpu-tensor-cores.md)

### GPU Troubleshooting

**Pattern:** `gpu-troubleshooting` | **Impact:** HIGH

Systematic diagnosis of GPU build failures, performance issues, and integration problems

**Consolidates:** 

See: [../patterns/gpu-troubleshooting.md](../patterns/gpu-troubleshooting.md)

### Warp Primitives and Reduction Patterns

**Pattern:** `gpu-warp` | **Impact:** HIGH

Warp shuffle operations, warp specialization, row reduction, and block reduction patterns

See: [../patterns/gpu-warp.md](../patterns/gpu-warp.md)

---

## Memory Safety & Ownership

**Priority:** CRITICAL | **Patterns:** 4

### Memory Collections Patterns

**Pattern:** `memory-collections` | **Impact:** CRITICAL

Span usage, MaybeUninitialized, collection destructors, and bulk memory operations in Mojo

See: [../patterns/memory-collections.md](../patterns/memory-collections.md)

### Memory Ownership and Lifecycle Management

**Pattern:** `memory-ownership` | **Impact:** CRITICAL

Comprehensive guide to ownership transfer, borrowing vs copying, implicit traits, and lifecycle methods in Mojo

See: [../patterns/memory-ownership.md](../patterns/memory-ownership.md)

### Reference Counting Implementation Patterns

**Pattern:** `memory-refcounting` | **Impact:** HIGH

Thread-safe reference counting with atomic operations and correct memory ordering for shared ownership in Mojo

See: [../patterns/memory-refcounting.md](../patterns/memory-refcounting.md)

### Memory Safety Patterns

**Pattern:** `memory-safety` | **Impact:** CRITICAL

Preventing dangling references, origin tracking, safe pointer types, and allocation strategies in Mojo

See: [../patterns/memory-safety.md](../patterns/memory-safety.md)

---

## Type System

**Priority:** CRITICAL | **Patterns:** 3

### SIMD Types and Vectorization

**Pattern:** `type-simd` | **Impact:** HIGH

SIMD type patterns for high-performance numerical operations including register-passable types, vectorization, and alignment

See: [../patterns/type-simd.md](../patterns/type-simd.md)

### Type System Fundamentals

**Pattern:** `type-system` | **Impact:** CRITICAL

Core type system patterns including annotations, conversions, Optional handling, numeric precision, and hashable keys

See: [../patterns/type-system.md](../patterns/type-system.md)

### Traits and Generic Programming

**Pattern:** `type-traits` | **Impact:** HIGH

Trait definition, conformance, parametric traits, trait bounds, and conditional conformance patterns

See: [../patterns/type-traits.md](../patterns/type-traits.md)

---

## Debugging

**Priority:** HIGH | **Patterns:** 1

### Debugging Patterns

**Pattern:** `debug-debugging` | **Impact:** HIGH

Systematic debugging of numerical accuracy issues and GPU numerical correctness

See: [../patterns/debug-debugging.md](../patterns/debug-debugging.md)

---

## Function Design

**Priority:** HIGH | **Patterns:** 1

### Function Design Patterns

**Pattern:** `fn-design` | **Impact:** HIGH

Comprehensive patterns for designing Mojo functions including argument conventions, keyword arguments, overloading, inlining, and target-specific code

See: [../patterns/fn-design.md](../patterns/fn-design.md)

---

## Struct Design

**Priority:** HIGH | **Patterns:** 1

### Struct Design Patterns

**Pattern:** `struct-design` | **Impact:** HIGH

Comprehensive patterns for designing Mojo structs including initialization, encapsulation, composition, operators, and iterators

See: [../patterns/struct-design.md](../patterns/struct-design.md)

---

## Testing

**Priority:** HIGH | **Patterns:** 2

### Mojo Benchmarking Patterns

**Pattern:** `test-benchmarking` | **Impact:** HIGH

Performance benchmarking patterns including QuickBench, proper timing methodology, and performance testing best practices

See: [../patterns/test-benchmarking.md](../patterns/test-benchmarking.md)

### Mojo Testing Patterns

**Pattern:** `test-testing` | **Impact:** HIGH

Unit testing patterns including test suites, assertions, lifecycle counters, property-based testing, and GPU test patterns

See: [../patterns/test-testing.md](../patterns/test-testing.md)

---

## Error Handling

**Priority:** MEDIUM-HIGH | **Patterns:** 1

### Error Handling Patterns

**Pattern:** `error-handling` | **Impact:** HIGH

Comprehensive error handling in Mojo including raises annotation, try/except/finally, context managers, error messages, and debug assertions

See: [../patterns/error-handling.md](../patterns/error-handling.md)

---

## Performance Optimization

**Priority:** MEDIUM | **Patterns:** 4

### Memory Optimization Patterns

**Pattern:** `perf-memory` | **Impact:** HIGH

Comprehensive guide to memory alignment, data layout, prefetching, stack vs heap allocation, multiple accumulators, and tiled processing

See: [../patterns/perf-memory.md](../patterns/perf-memory.md)

### General Optimization Patterns

**Pattern:** `perf-optimization` | **Impact:** MEDIUM

Comprehensive guide to caching strategies, lazy loading, mmap patterns, compile-time computation, buffer management, and avoiding overhead

See: [../patterns/perf-optimization.md](../patterns/perf-optimization.md)

### Multi-Core Parallelization Patterns

**Pattern:** `perf-parallelization` | **Impact:** CRITICAL

Comprehensive guide to parallelize[], work distribution, and parallel attention patterns for multi-core CPU execution

See: [../patterns/perf-parallelization.md](../patterns/perf-parallelization.md)

### SIMD Vectorization Patterns

**Pattern:** `perf-vectorization` | **Impact:** HIGH

Comprehensive guide to SIMD vectorization, alignment, early exit, loop unrolling, and grid-stride patterns for maximum CPU/GPU throughput

See: [../patterns/perf-vectorization.md](../patterns/perf-vectorization.md)

---

## Python Interoperability

**Priority:** MEDIUM | **Patterns:** 1

### Python Interoperability

**Pattern:** `python-interop` | **Impact:** MEDIUM

Patterns for efficient Python/Mojo integration including boundary minimization, import optimization, type conversion, and error handling

See: [../patterns/python-interop.md](../patterns/python-interop.md)

---

## Advanced Metaprogramming

**Priority:** LOW | **Patterns:** 2

### API Canary Tests

**Pattern:** `meta-canary-apis` | **Impact:** LOW

Quick compilation tests for critical APIs - first to break when Mojo changes

See: [../patterns/meta-canary-apis.md](../patterns/meta-canary-apis.md)

### Mojo Metaprogramming Patterns

**Pattern:** `meta-programming` | **Impact:** MEDIUM

Compile-time parameters, variadic parameters, conditional conformance, and parameter unpacking for zero-cost generics

See: [../patterns/meta-programming.md](../patterns/meta-programming.md)

---

## Other

**Priority:** LOW | **Patterns:** 3

### C++ to Mojo Porting Guide

**Pattern:** `porting-cpp` | **Impact:** HIGH

Side-by-side C++→Mojo porting guide with memory safety, template→parameter mapping, and unified CPU+GPU patterns

See: [../patterns/porting-cpp.md](../patterns/porting-cpp.md)

### Python to Mojo Porting Guide

**Pattern:** `porting-python` | **Impact:** HIGH

Side-by-side Python→Mojo porting guide with performance showcases, unique Mojo capabilities, and CI-checkable examples

See: [../patterns/porting-python.md](../patterns/porting-python.md)

### Rust to Mojo Porting Guide

**Pattern:** `porting-rust` | **Impact:** HIGH

Side-by-side Rust→Mojo porting guide comparing ownership models, trait systems, SIMD, and GPU programming

See: [../patterns/porting-rust.md](../patterns/porting-rust.md)

---
