---
title: "C++ to Mojo Porting Guide"
description: Side-by-side C++→Mojo porting guide with memory safety, template→parameter mapping, and unified CPU+GPU patterns
impact: HIGH
category: porting
tags: [cpp, c++, porting, migration, templates, memory-safety, gpu, raii]
error_patterns:
  - "port C\\+\\+ to Mojo"
  - "convert C\\+\\+ to Mojo"
  - "C\\+\\+ equivalent"
  - "migrate from C\\+\\+"
  - "rewrite C\\+\\+ in Mojo"
  - "template to parameter"
scenarios:
  - "Port a C++ program to Mojo"
  - "Find the Mojo equivalent of C++ syntax"
  - "Replace C++ templates with Mojo parameters"
  - "Migrate C++ class to Mojo struct"
  - "Port CUDA C++ kernel to Mojo"
---

# C++ to Mojo Porting Guide

**Category:** Porting | **Impact:** HIGH

Complete guide for porting C++ code to Mojo with side-by-side examples. Covers syntax mapping, memory safety improvements, template→parameter migration, RAII patterns, and unified CPU+GPU programming.

> **Source language version:** C++23 (ISO/IEC 14882:2024, ratified Feb 2024)
> **Target:** Mojo stable (v26.1+) and nightly (v26.2+)
> **CI check:** Verify with `g++ --version` (GCC 14+) or `clang++ --version` (Clang 18+) and `mojo --version`

---

## Quick Reference: C++ → Mojo Mapping

### Basic Syntax

| C++23 | Mojo | Notes |
|-------|------|-------|
| `int x = 42;` | `var x: Int = 42` | No semicolons |
| `const int X = 42;` | `alias X = 42` | Compile-time constant |
| `constexpr int X = 42;` | `alias X = 42` | Same — compile-time |
| `auto x = 42;` | `var x = 42` | Type inference |
| `float x = 3.14f;` | `var x: Float32 = 3.14` | Explicit precision |
| `double x = 3.14;` | `var x: Float64 = 3.14` | Explicit precision |
| `std::string s = "hi";` | `var s: String = "hi"` | UTF-8 |
| `// comment` | `# comment` | Python-style |
| `/* block */` | `# line comments only` | No block comments |
| `#include <header>` | `from module import name` | No headers |
| `namespace foo { }` | Module files | File = module |
| `using T = int;` | `alias T = Int` | Type alias |
| `nullptr` | `UnsafePointer[T]()` | Null pointer |
| `static_assert(cond)` | `comptime assert cond` | Compile-time assert |
| `sizeof(T)` | `size_of[T]()` | Compile-time size (`from sys import size_of`) |

### Functions

| C++23 | Mojo | Notes |
|-------|------|-------|
| `int foo(int x)` | `fn foo(x: Int) -> Int` | Args are immutable by default |
| `void foo(int& x)` | `fn foo(mut x: Int)` | Mutable reference |
| `void foo(const int& x)` | `fn foo(x: Int)` | Immutable borrow (default) |
| `void foo(int&& x)` | `fn foo(var x: Int)` | Takes ownership |
| `int foo(int x = 10)` | `fn foo(x: Int = 10) -> Int` | Default values |
| `template<int N> void foo()` | `fn foo[N: Int]()` | Compile-time parameter |
| `auto foo(auto x)` | `fn foo[T: Writable](x: T)` | Constrained generic |
| `constexpr int foo()` | `fn foo() -> Int:` with `alias` result | Compile-time evaluation |
| `[[nodiscard]] int foo()` | Return values must be used by default | Built-in |
| `noexcept` | Default — `fn` doesn't throw | Add `raises` to throw |
| `throw std::runtime_error("msg")` | `raise "msg"` | With `fn foo() raises:` |
| `try { } catch (const std::exception& e) { }` | `try: ... except e:` | Similar |
| `[](int x) { return x*2; }` | `fn(x: Int) -> Int: return x*2` | Closure |

### Types and Templates

| C++23 | Mojo | Notes |
|-------|------|-------|
| `class Foo { };` | `struct Foo:` | Value semantics |
| `struct Foo { };` | `struct Foo:` | Same in Mojo |
| `template<typename T>` | `struct Foo[T: AnyType]:` | Parametric |
| `template<typename T> requires Concept<T>` | `struct Foo[T: Trait]:` | Constrained |
| `concept Printable = requires(T t) { t.print(); }` | `trait Writable: fn write_to[W: Writer](self, mut writer: W):` | Explicit trait |
| `std::variant<int, string>` | `Variant[Int, String]` | Sum type |
| `std::optional<int>` | `Optional[Int]` | Nullable |
| `std::expected<int, Error>` | `fn foo() raises -> Int` | Error handling |
| `std::array<int, 4>` | `InlineArray[Int, 4]` | Stack-allocated |
| `std::vector<int>` | `List[Int]` | Dynamic array. `T` must be `Copyable` (not just `Movable`); move-only types cannot go in `List` |
| `std::unordered_map<K, V>` | `Dict[K, V]` | Hash map |
| `std::unique_ptr<T>` | Ownership semantics built-in | No wrapper needed |
| `std::shared_ptr<T>` | Manual ref counting | `ArcPointer` pattern |
| `std::span<T>` | `Span[T]` | Non-owning view |
| `std::tuple<A, B, C>` | `Tuple[A, B, C]` | Heterogeneous |

> **Warning: List and InlineArray construction syntax.** `List[Int](1, 2, 3)` and `InlineArray[Int, 3](1, 2, 3)` do NOT compile. Use list literal syntax with a type annotation: `var v: List[Int] = [1, 2, 3]` or `var a: InlineArray[Int, 3] = [1, 2, 3]`. For empty lists use `List[Int]()`. For pre-sized lists use `List[Int](length=N, fill=0)`.

### Memory and Pointers

| C++23 | Mojo | Notes |
|-------|------|-------|
| `T* ptr` | `UnsafePointer[T]` | Raw pointer |
| `new T(args)` | `alloc[T](1)` + `__init__` | `from memory import alloc` |
| `delete ptr` | `ptr.free()` | Manual deallocation |
| `ptr->member` | `ptr[].member` | Dereference |
| `*ptr` | `ptr[]` | Dereference |
| `ptr + offset` | `ptr + offset` | Pointer arithmetic |
| `reinterpret_cast<U*>(p)` | `ptr.bitcast[U]()` | Type cast |
| `memcpy(dst, src, n)` | `memcpy(dst, src, count=n)` | `from memory import memcpy` |
| `malloc(n)` | `alloc[T](n)` | `from memory import alloc` |
| `free(ptr)` | `ptr.free()` | Deallocation |

---

## Key Differences: C++ vs Mojo

### 1. Memory Safety Without Runtime Overhead

C++ has no borrow checker. Memory errors are caught by sanitizers (runtime) or static analysis (limited).

**C++23 (unsafe by default):**

```cpp
// C++23 — compiles fine, undefined behavior at runtime
std::vector<int> vec = {1, 2, 3};
int& ref = vec[0];
vec.push_back(4);      // May reallocate!
std::cout << ref;       // DANGLING REFERENCE — undefined behavior
// No compiler error. ASAN catches it at runtime (if you're lucky).
```

**Mojo (safe by default):**

```mojo
fn main():
    var vec: List[Int] = [1, 2, 3]
    # Mojo's ownership system prevents dangling references.
    # Borrowing vec[0] while mutating vec is a compile-time error.
    # The borrow checker catches this before your code ever runs.
```

### 2. Templates → Parameters (Simpler, Faster Compilation)

C++ templates are Turing-complete but produce cryptic errors and slow compilation. Mojo parameters are explicit and fast.

**C++23 (templates):**

```cpp
// C++23 — template metaprogramming
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T, int Width>
struct SIMDVector {
    std::array<T, Width> data;

    SIMDVector operator+(const SIMDVector& other) const {
        SIMDVector result;
        for (int i = 0; i < Width; ++i)
            result.data[i] = data[i] + other.data[i];
        return result;
    }
};

// Error messages from template failures can be 100+ lines
```

**Mojo (parameters):**

```mojo
# nocompile
# Mojo — parametric types with clear constraints
struct SIMDVector[T: DType, Width: Int]:
    var data: SIMD[T, Width]

    fn __add__(self, other: Self) -> Self:
        return Self {data: self.data + other.data}

# Error messages reference the parameter constraint directly
# SIMD is a built-in type — no manual loop needed
```

### 3. RAII → Same Pattern, Better Ergonomics

Both C++ and Mojo use deterministic destruction. Mojo simplifies it.

**C++23:**

```cpp
class FileHandle {
    int fd_;
public:
    explicit FileHandle(const std::string& path)
        : fd_(open(path.c_str(), O_RDONLY)) {
        if (fd_ < 0) throw std::runtime_error("Failed to open");
    }
    ~FileHandle() { if (fd_ >= 0) close(fd_); }

    // Rule of Five boilerplate:
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
    FileHandle(FileHandle&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    FileHandle& operator=(FileHandle&& other) noexcept {
        if (this != &other) { if (fd_ >= 0) close(fd_); fd_ = other.fd_; other.fd_ = -1; }
        return *this;
    }
};
```

**Mojo:**

```mojo
struct FileHandle:
    var fd: Int

    fn __init__(out self, path: String) raises:
        self.fd = 1  # Open file
        if self.fd < 0:
            raise "Failed to open"

    fn __init__(out self, *, deinit take: Self):
        self.fd = take.fd
        take.fd = -1

    fn __del__(deinit self):
        if self.fd >= 0:
            pass  # Close file
    # No Rule of Five — Mojo generates safe defaults
    # No copy constructor needed (non-copyable by default)
    # Copyable → explicit .copy() method; ImplicitlyCopyable → var b = a works
    # For implicit copy (var b = a), structs need the ImplicitlyCopyable trait
```

---

## Performance Showcase: C++ vs Mojo

### Level 1: SIMD — Built-In vs Intrinsics

**C++23 (manual intrinsics):**

```cpp
// C++23 — requires platform-specific intrinsics
#include <immintrin.h>  // x86 only!

void add_arrays_avx(float* a, float* b, float* c, int n) {
    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    // Handle remainder...
    // This code is x86-ONLY. ARM needs NEON intrinsics.
    // std::experimental::simd exists but is rarely used.
}
```

**Mojo (portable, first-class SIMD):**

```mojo
# nocompile
from sys import simd_width_of
from memory import UnsafePointer

fn add_arrays(a: UnsafePointer[Float32], b: UnsafePointer[Float32],
              mut c: UnsafePointer[Float32], n: Int):
    comptime WIDTH = simd_width_of[DType.float32]()  # Adapts to AVX/NEON/SVE

    var i = 0
    while i + WIDTH <= n:
        c.store(i, a.load[width=WIDTH](i) + b.load[width=WIDTH](i))
        i += WIDTH
    while i < n:
        c[i] = a[i] + b[i]
        i += 1
    # Works on x86, ARM, GPU — same code
```

### Level 2: Compile-Time Computation

**C++23 (constexpr/consteval):**

```cpp
// C++23
consteval int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

// consteval forces compile-time evaluation
constexpr int fib_10 = fibonacci(10);  // 55, computed at compile time

// But constexpr functions have many restrictions:
// - No dynamic allocation in consteval
// - No I/O
// - No reinterpret_cast
// - Limited container support
```

**Mojo (alias — simpler):**

```mojo
fn fibonacci(n: Int) -> Int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# alias forces compile-time evaluation — same function works at both
comptime FIB_10 = fibonacci(10)  # 55, computed at compile time

# The SAME function works at runtime too:
fn main():
    var fib_runtime = fibonacci(20)  # Computed at runtime
    print(fib_runtime)
```

### Level 3: GPU Kernels — Unified vs Separate Toolchain

**C++23 + CUDA (separate toolchain):**

```cpp
// kernel.cu — MUST be a .cu file, compiled with nvcc
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = a[tid] + b[tid];
}

// main.cpp — different file, different compiler
#include <cuda_runtime.h>
int main() {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    vector_add<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
// Build: nvcc kernel.cu -c && g++ main.cpp kernel.o -lcudart
// Two compilers, two languages, two build steps
```

**Mojo (unified — ONE file, ONE compiler):**

```mojo
# nocompile
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim

fn vector_add_kernel(
    a: UnsafePointer[Float32], b: UnsafePointer[Float32],
    c: UnsafePointer[Float32], n: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < n:
        c[tid] = a[tid] + b[tid]

fn main() raises:
    var ctx = DeviceContext()
    var n = 1024
    var a = ctx.enqueue_create_buffer[DType.float32](n)
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    var c = ctx.enqueue_create_buffer[DType.float32](n)

    ctx.enqueue_function[vector_add_kernel](
        a.unsafe_ptr(), b.unsafe_ptr(), c.unsafe_ptr(), n,
        grid_dim=(n // 256 + 1,), block_dim=(256,),
    )
    ctx.synchronize()
# Build: mojo run file.mojo — one tool, one language
```

> **Production note:** For simple element-wise kernels like `vector_add`, prefer `algorithm.elementwise` — it handles grid/block sizing automatically. See [`gpu-kernels.md`](gpu-kernels.md).

---

## Things Only Mojo Can Do (That C++ Cannot)

### 1. Borrow Checker Without Runtime Cost

C++ static analyzers catch some issues. Mojo catches all ownership violations at compile time.

```mojo
fn takes_ownership(var data: List[Int]):
    pass  # data is destroyed here

fn main():
    var list: List[Int] = [1, 2, 3]
    takes_ownership(list^)
    # print(list[0])  # COMPILE ERROR: list was moved
    # C++ would compile this and crash at runtime
```

### 2. Python Interop Built-In

C++ requires SWIG, pybind11, or Boost.Python for Python interop.

```mojo
# nocompile
from python import Python

fn use_python_library() raises:
    var np = Python.import_module("numpy")
    var arr = np.array([1.0, 2.0, 3.0])
    print(arr.mean())  # Seamless Python interop — no bindings needed
```

### 3. SIMD as First-Class Type (Not Intrinsics)

```mojo
# SIMD is a type, not a platform-specific intrinsic
var a = SIMD[DType.float32, 8](1.0)
var b = SIMD[DType.float32, 8](2.0)
var c = a + b                     # Operator overloading on SIMD
var d = c.reduce_add()            # Built-in reductions
var mask = a.gt(SIMD[DType.float32, 8](1.5))
var selected = mask.select(a, b)  # Branchless select
# Same code works on x86 (AVX), ARM (NEON), GPU
```

### 4. No Header Files, No Forward Declarations

```mojo
# Mojo — just import and use
from math import sqrt
from memory import UnsafePointer

fn foo() -> Int:
    return bar()  # Can call bar() even though it's defined below

fn bar() -> Int:
    return 42
# No .h files, no include guards, no circular dependency hell
```

### 5. Unified CPU+GPU in Same Language

C++ requires CUDA (NVIDIA proprietary) or HIP (AMD) — different toolchains.

```mojo
# nocompile
# ONE file, ONE language, works on CPU and GPU
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim
from algorithm import parallelize

fn cpu_work(data: UnsafePointer[Float32], n: Int):
    """CPU parallel work."""
    @parameter
    fn worker(i: Int):
        data[i] = data[i] * 2.0
    parallelize[worker](n)

fn gpu_kernel(data: UnsafePointer[Float32], n: Int):
    """GPU kernel — same file."""
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < n:
        data[tid] = data[tid] + 1.0
```

---

## Common Migration Patterns

### Pattern 1: std::vector → List

**C++23:**

```cpp
std::vector<int> vec;
vec.push_back(1);
vec.push_back(2);
vec.reserve(100);
for (int x : vec) { std::cout << x; }
auto size = vec.size();
```

**Mojo:**

```mojo
var vec = List[Int]()
vec.append(1)
vec.append(2)
vec.reserve(100)
for x in vec:
    print(x)
var size = len(vec)
```

> **Warning:** `Copyable` and `ImplicitlyCopyable` are different traits. `Copyable` only provides `.copy()` (explicit clone). For `var b = a` (implicit copy assignment), the type must also conform to `ImplicitlyCopyable`. Use `.copy()` as the safe default when porting C++ copy semantics.

### Pattern 2: std::unique_ptr → Ownership

**C++23:**

```cpp
auto ptr = std::make_unique<Widget>(42);
process(std::move(ptr));  // Transfer ownership
// ptr is now nullptr
```

**Mojo:**

```mojo
# nocompile
var widget = Widget(42)
process(widget^)  # Transfer ownership with ^
# widget is invalid — compiler enforces this
```

### Pattern 3: Concepts → Traits

**C++23:**

```cpp
template<typename T>
concept Printable = requires(T t, std::ostream& os) {
    { os << t } -> std::same_as<std::ostream&>;
};

template<Printable T>
void print_item(const T& item) {
    std::cout << item << std::endl;
}
```

**Mojo:**

```mojo
# nocompile
trait Writable:
    fn write_to[W: Writer](self, mut writer: W):
        ...

fn print_item[T: Writable](item: T):
    print(item)
```

### Pattern 4: constexpr/consteval → alias/comptime

**C++23:**

```cpp
constexpr int BUFFER_SIZE = 1024;
consteval int compute_table_size(int entries) { return entries * 2 + 1; }
constexpr int TABLE_SIZE = compute_table_size(100);  // 201

template<int N>
struct FixedBuffer {
    std::array<float, N> data{};
};

FixedBuffer<BUFFER_SIZE> buf;
```

**Mojo:**

```mojo
# nocompile
comptime BUFFER_SIZE = 1024

fn compute_table_size(entries: Int) -> Int:
    return entries * 2 + 1

comptime TABLE_SIZE = compute_table_size(100)  # 201

struct FixedBuffer[N: Int]:
    var data: InlineArray[Float32, Self.N]

    fn __init__(out self):
        self.data = InlineArray[Float32, Self.N](fill=0.0)

var buf = FixedBuffer[BUFFER_SIZE]()
```

---

## Build System Comparison

| Aspect | C++23 | Mojo |
|--------|-------|------|
| Build tool | CMake, Bazel, Meson, Make... | `mojo build` / `mojo run` |
| Package manager | vcpkg, Conan, apt, brew... | Built-in module system |
| Header management | `#include`, include guards, modules (C++20) | `import` |
| Compilation model | Translation units → linking | Module-based |
| Build time (100K LOC) | Minutes | Seconds |
| Cross-compilation | Complex toolchain setup | MLIR-based targeting |

---

## CI Validation

### Version Checking

```bash
# Check C++ compiler (source language reference)
g++ --version     # Expect: GCC 14+ for C++23 support
clang++ --version # Expect: Clang 18+ for C++23 support

# Check Mojo version
mojo --version    # Expect: mojo 26.1.x or 26.2.x
```

### What to Watch For

| Language | Version | Breaking Changes to Monitor |
|----------|---------|---------------------------|
| C++ | C++23 → C++26 | `std::simd`, contracts, static reflection, sender/receiver |
| C++ | GCC/Clang | New standard library features |
| Mojo | v26.1 → v26.2 | `comptime` syntax, trait changes |

---

## Version-Specific Features

### Stable (v26.1) vs Nightly (v26.2+)

| Feature | Stable (v26.1) | Nightly (v26.2+) |
|---------|----------------|-------------------|
| **Parameters** | `[N: Int]` | Same |
| **Traits** | `trait Foo:` | Same |
| **Constants** | `alias` or `comptime` | Both work |
| **SIMD** | `SIMD[DType, Width]` | Same |
| **GPU** | `DeviceContext`, `enqueue_function` | Same |
| **Struct alignment** | N/A | `@align(N)` decorator |

---

## Related Patterns

- [`gpu-porting-cuda.md`](gpu-porting-cuda.md) — Detailed CUDA C++→Mojo GPU kernel porting
- [`gpu-porting-cute.md`](gpu-porting-cute.md) — CUTLASS/CuTe→Mojo mapping
- [`memory-ownership.md`](memory-ownership.md) — Deep dive on ownership
- [`type-traits.md`](type-traits.md) — Trait system details
- [`perf-vectorization.md`](perf-vectorization.md) — SIMD vectorization patterns

---

## References

- [Mojo Manual](https://docs.modular.com/mojo/manual/)
- [C++23 Standard (cppreference)](https://en.cppreference.com/w/cpp/23)
- [C++ Compiler Support](https://en.cppreference.com/w/cpp/compiler_support)
