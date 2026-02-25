# Error Disambiguation Guide

When an error message matches multiple patterns, use this guide to determine which pattern to consult first.

---

## "alignment" errors

**Priority 1: Check [`perf-optimization.md`](../patterns/perf-optimization.md) if:**
- Error mentions "memory alignment" or "pointer alignment"
- You're working with raw memory operations or `UnsafePointer`
- The error occurs during allocation or pointer arithmetic

**Priority 2: Check [`perf-vectorization.md`](../patterns/perf-vectorization.md) if:**
- Error mentions "SIMD alignment" or "vector alignment"
- You're working with SIMD types or vectorized loops
- The error occurs during `load()` or `store()` operations

---

## "cannot vectorize" errors

**Priority 1: Check [`perf-vectorization.md`](../patterns/perf-vectorization.md) if:**
- The compiler reports loop vectorization failure
- You're trying to use SIMD on an existing loop
- Error mentions "loop carried dependency" or "stride"

**Priority 2: Check [`type-simd.md`](../patterns/type-simd.md) if:**
- You're directly using SIMD types (`SIMD[DType.float32, 4]`)
- Error mentions type constraints or SIMD width issues
- The error is about SIMD type compatibility

---

## "data race" errors

**Priority 1: Check [`perf-parallelization.md`](../patterns/perf-parallelization.md) if:**
- You're using `parallelize()` or manual threading
- Multiple threads access the same data
- The error occurs in parallel loops

**Priority 2: Check [`memory-refcounting.md`](../patterns/memory-refcounting.md) if:**
- You're using `ArcPointer` or shared references
- The race involves reference count operations
- Multiple threads share an `Arc`-wrapped value

---

## "double free" / "use after free" / "memory leak" errors

**Priority 1: Check [`memory-safety.md`](../patterns/memory-safety.md) if:**
- You're using `UnsafePointer` directly
- Error occurs in custom container implementation
- You're managing `init_pointee_*` and `destroy_pointee` manually

**Priority 2: Check [`memory-refcounting.md`](../patterns/memory-refcounting.md) if:**
- You're using `ArcPointer` or reference counting
- The error involves cycles or weak references
- Ownership is shared between multiple owners

---

## "expected .* but got" errors

**Priority 1: Check [`type-system.md`](../patterns/type-system.md) if:**
- Error is a type mismatch between different types
- You're passing wrong argument types
- Error occurs in generic code or trait implementations

**Priority 2: Check [`testing.md`](../patterns/testing.md) if:**
- Error comes from `assert_equal` or `assert_almost_equal`
- You're writing or running tests
- The message format matches test assertion output

---

## "kernel launch failed" errors

**Priority 1: Check [`gpu-fundamentals.md`](../patterns/gpu-fundamentals.md) if:**
- You're new to GPU programming in Mojo
- Error occurs during kernel dispatch setup
- The error mentions grid/block dimensions

**Priority 2: Check [`gpu-kernels.md`](../patterns/gpu-kernels.md) if:**
- You have complex kernel logic
- Error occurs inside the kernel function
- The error mentions shared memory or synchronization

---

## "misaligned address" errors

**Priority 1: Check [`gpu-memory-access.md`](../patterns/gpu-memory-access.md) if:**
- Error occurs during memory load/store operations
- You're accessing global or shared memory
- The error mentions coalescing or memory transactions

**Priority 2: Check [`gpu-warp-sync.md`](../patterns/gpu-warp-sync.md) if:**
- Error occurs after a barrier or sync point
- Multiple threads access the same memory location
- The error involves `barrier()` or `fence()` operations

---

## "slow performance" errors

**Priority 1: Check [`perf-vectorization.md`](../patterns/perf-vectorization.md) if:**
- Performance issue is in compute-bound code
- You're processing arrays or matrices
- You want to leverage SIMD instructions

**Priority 2: Check [`perf-parallelization.md`](../patterns/perf-parallelization.md) if:**
- Performance issue involves multiple CPU cores
- You're processing independent data in parallel
- You want to use threading or `parallelize()`

---

## General Disambiguation Strategy

When unsure which pattern to check:

1. **Check the error location** - Is it in memory management, GPU, testing, or general code?
2. **Check the operation type** - What were you trying to do when the error occurred?
3. **Check both patterns** - Start with Priority 1, then move to Priority 2 if unresolved
4. **Use ERROR_INDEX.md** - For errors not listed here, check the full error index

---

*This file is maintained manually. If you discover new ambiguous error patterns, please update this file.*
