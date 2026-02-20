---
title: "Rust to Mojo Porting Guide"
description: Side-by-side Rust→Mojo porting guide comparing ownership models, trait systems, SIMD, and GPU programming
impact: HIGH
category: porting
tags: [rust, porting, migration, ownership, traits, simd, gpu]
error_patterns:
  - "port Rust to Mojo"
  - "convert Rust to Mojo"
  - "Rust equivalent"
  - "migrate from Rust"
  - "rewrite Rust in Mojo"
  - "Rust vs Mojo"
scenarios:
  - "Port a Rust program to Mojo"
  - "Find the Mojo equivalent of Rust syntax"
  - "Compare Rust and Mojo ownership models"
  - "Migrate Rust struct to Mojo"
  - "Port Rust SIMD code to Mojo"
---

# Rust to Mojo Porting Guide

<!-- staleness-ok-file -->

**Category:** Porting | **Impact:** HIGH

Complete guide for porting Rust code to Mojo with side-by-side examples. Both languages share ownership semantics and compile-time safety, but Mojo adds built-in GPU kernels, first-class SIMD, Python interop, and simpler lifetime handling.

> **Source language version:** Rust 1.93 (released Jan 2026, Edition 2024)
> **Target:** Mojo stable (v26.1+) and nightly (v26.2+)
> **CI check:** Verify with `rustc --version` (expect 1.93+) and `mojo --version`

---

## Quick Reference: Rust → Mojo Mapping

### Basic Syntax

| Rust 1.93 | Mojo | Notes |
|-----------|------|-------|
| `let x = 42;` | `var x = 42` | Mojo `var` is mutable by default |
| `let mut x = 42;` | `var x = 42` | All `var` is mutable |
| `let x: i32 = 42;` | `var x: Int32 = 42` | Explicit type |
| `const X: i32 = 42;` | `alias X: Int = 42` | Compile-time constant |
| `fn foo() -> i32 { 42 }` | `fn foo() -> Int: return 42` | Explicit `return` |
| `fn foo() { }` | `fn foo():` | No return type needed |
| `println!("{}", x);` | `print(x)` | No format macro needed |
| `// comment` | `# comment` | Python-style |
| `use std::io;` | `from io import ...` | Import |
| `mod foo;` | Module files | File = module |
| `pub fn foo()` | `fn foo()` | Public by default |
| `type Alias = Vec<i32>;` | `alias Alias = List[Int]` | Type alias |
| `todo!()` | `pass` or `...` | Placeholder |
| `unreachable!()` | `abort()` | Unreachable |
| `assert!(cond);` | `debug_assert(cond)` | Runtime assert |
| `assert_eq!(a, b);` | `debug_assert(a == b)` | Equality assert |

### Ownership and Borrowing

| Rust 1.93 | Mojo | Notes |
|-----------|------|-------|
| `fn foo(x: String)` | `fn foo(var x: String)` | Takes ownership |
| `fn foo(x: &String)` | `fn foo(x: String)` | Immutable borrow (default) |
| `fn foo(x: &mut String)` | `fn foo(mut x: String)` | Mutable borrow |
| `let y = x;` (move) | `var y = x^` | Explicit transfer with `^` |
| `x.clone()` | Copy (if `Copyable`) | Traits control copying |
| `drop(x)` | `x^.__del__()` or scope end | Deterministic |
| `'a` (lifetime) | Origins (inferred) | Often no annotation needed |
| `&'a str` | `StringSlice[origin]` | Origin-tracked reference |

### Types

| Rust 1.93 | Mojo | Notes |
|-----------|------|-------|
| `i8/i16/i32/i64` | `Int8/Int16/Int32/Int64` | Same sizes |
| `u8/u16/u32/u64` | `UInt8/UInt16/UInt32/UInt64` | Same sizes |
| `f32/f64` | `Float32/Float64` | Same sizes |
| `bool` | `Bool` | Same |
| `char` | Not yet | UTF-32 character |
| `String` | `String` | Heap UTF-8 |
| `&str` | `StringSlice` or `StringLiteral` | Non-owning view |
| `Vec<T>` | `List[T]` | Dynamic array |
| `[T; N]` | `InlineArray[T, N]` | Stack array |
| `Box<T>` | `UnsafePointer[T]` or stack alloc | Heap or stack |
| `Rc<T>` / `Arc<T>` | Manual ref counting | No built-in RC yet |
| `Option<T>` | `Optional[T]` | Nullable |
| `Result<T, E>` | `fn foo() raises -> T` | Error handling |
| `HashMap<K, V>` | `Dict[K, V]` | Hash map |
| `HashSet<T>` | `Set[T]` | Hash set |
| `(A, B, C)` | `Tuple[A, B, C]` | Tuple |
| `&[T]` | `Span[T]` | Slice view |

### Structs and Traits

| Rust 1.93 | Mojo | Notes |
|-----------|------|-------|
| `struct Foo { x: i32 }` | `struct Foo: var x: Int` | Fields declared with `var` |
| `#[derive(Debug, Clone)]` | `@fieldwise_init` | Different derive model |
| `impl Foo { }` | Methods inside `struct Foo:` | No separate impl blocks |
| `impl Trait for Foo { }` | `struct Foo(Trait):` | Parenthetical conformance |
| `trait Foo { fn bar(&self); }` | `trait Foo: fn bar(self):` | Similar |
| `trait Foo: Bar { }` | `trait Foo(Bar):` | Trait inheritance |
| `where T: Display + Clone` | `[T: Display & Clone]` | Combined constraints |
| `enum Color { Red, Blue }` | `alias Red = 0; alias Blue = 1` | No sum types yet |
| `enum Result<T, E> { Ok(T), Err(E) }` | `Variant[T, E]` or `raises` | Different approach |
| `match x { ... }` | `if`/`elif` chains | No pattern matching yet |

### Error Handling

| Rust 1.93 | Mojo | Notes |
|-----------|------|-------|
| `Result<T, E>` | `fn foo() raises -> T` | `raises` keyword |
| `Ok(value)` | `return value` | Just return |
| `Err(e)` | `raise "message"` | Raise string error |
| `?` operator | No equivalent yet | Must use try/except |
| `unwrap()` | `value()` on Optional | Panics if None |
| `expect("msg")` | `value()` | |
| `panic!("msg")` | `abort("msg")` | Aborts program |

### Closures and Iterators

| Rust 1.93 | Mojo | Notes |
|-----------|------|-------|
| `\|x\| x * 2` | `fn(x: Int) -> Int: return x * 2` | Typed closures |
| `\|x: i32\| -> i32 { x * 2 }` | `fn(x: Int) -> Int: return x * 2` | Same |
| `iter.map(\|x\| x*2)` | `for x in iter: result.append(x[]*2)` | Manual iteration |
| `iter.filter(\|x\| x > 0)` | `for x in iter: if x[] > 0: ...` | Manual filtering |
| `iter.collect::<Vec<_>>()` | Build List manually | No collect |
| `for x in &vec { }` | `for x in vec:` | Iteration (x is reference) |

---

## Ownership: Rust vs Mojo — A Close Comparison

Both languages use ownership + borrowing for memory safety. The key differences are in syntax and lifetime annotations.

### Move Semantics

**Rust:**
```rust
fn process(data: Vec<i32>) {
    // data is owned, dropped at end of scope
    for x in &data {
        println!("{}", x);
    }
}

fn main() {
    let data = vec![1, 2, 3];
    process(data);          // Implicit move
    // println!("{:?}", data); // ERROR: value used after move
}
```

**Mojo:**
```mojo
# nocompile
fn process(var data: List[Int]):
    # data is owned, destroyed at end of scope
    for x in data:
        print(x[])

fn main():
    var data = List[Int](1, 2, 3)
    process(data^)          # Explicit move with ^
    # print(data[0])        # ERROR: value used after move
```

**Key differences:**
- Mojo uses `^` to make moves EXPLICIT. Rust moves implicitly.
- Mojo uses **ASAP destruction** (last-use semantics) — values are freed at their last use point, not at scope end. This is especially valuable for GPU memory where early deallocation allows larger models to fit.

### Borrowing

**Rust:**
```rust
fn analyze(data: &Vec<f64>) -> f64 {
    // Immutable borrow — &
    data.iter().sum()
}

fn modify(data: &mut Vec<f64>) {
    // Mutable borrow — &mut
    data.push(42.0);
}

fn main() {
    let mut data = vec![1.0, 2.0, 3.0];
    let sum = analyze(&data);       // Borrow immutably
    modify(&mut data);              // Borrow mutably
}
```

**Mojo:**
```mojo
# nocompile
fn analyze(data: List[Float64]) -> Float64:
    # Immutable borrow — default (no sigil needed!)
    var total: Float64 = 0.0
    for x in data:
        total += x[]
    return total

fn modify(mut data: List[Float64]):
    # Mutable borrow — mut keyword
    data.append(42.0)

fn main():
    var data = List[Float64](1.0, 2.0, 3.0)
    var sum = analyze(data)     # Borrow immutably (no & needed)
    modify(data)                # Borrow mutably (no &mut needed)
```

**Key difference:** Mojo infers borrow mode from the parameter convention (`read` is default, `mut` for mutable). No `&` or `&mut` sigils.

### Lifetimes → Origins

**Rust (explicit lifetimes):**
```rust
// Rust requires lifetime annotations when the compiler can't infer
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// Complex lifetime annotations for structs
struct Parser<'input> {
    text: &'input str,
    pos: usize,
}

impl<'input> Parser<'input> {
    fn new(text: &'input str) -> Self {
        Parser { text, pos: 0 }
    }
}
```

**Mojo (origins — often inferred):**
```mojo
# nocompile
# Mojo infers origins in most cases — no annotations needed
fn longest(x: StringSlice, y: StringSlice) -> StringSlice:
    if len(x) > len(y):
        return x
    return y

# Struct with reference — origin inferred from usage
struct Parser:
    var text: String
    var pos: Int

    fn __init__(out self, text: String):
        self.text = text
        self.pos = 0
```

**Key difference:** Mojo's origin system handles most lifetime cases automatically. You rarely write lifetime annotations.

---

## Things Only Mojo Can Do (That Rust Cannot)

### 1. Native GPU Kernels

Rust's GPU story is fragmented: `rust-gpu` (experimental), `wgpu` (high-level), CUDA FFI.

**Rust (no native GPU):**
```rust
// Rust has NO stable way to write GPU kernels.
// Options: rust-gpu (experimental), wgpu (shader-level), CUDA FFI
// None are first-class. All require separate toolchains.
```

**Mojo (native GPU):**
```mojo
# nocompile
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim

fn gpu_kernel(data: UnsafePointer[Float32], n: Int):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < n:
        data[tid] = data[tid] * 2.0

fn main() raises:
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.float32](1024)
    ctx.enqueue_function[gpu_kernel](
        buf.unsafe_ptr(), 1024,
        grid_dim=(4,), block_dim=(256,),
    )
    ctx.synchronize()
```

> **Production note:** For simple element-wise kernels, prefer `algorithm.elementwise` — it handles grid/block sizing automatically. See [`gpu-kernels.md`](gpu-kernels.md).

### 2. First-Class SIMD Type

Rust's `std::simd` has been unstable/nightly-only for YEARS (since 2021). Not usable in stable Rust.

**Rust 1.93 (stable — no portable SIMD):**
```rust
// std::simd is NIGHTLY-ONLY — not available in stable Rust
// #![feature(portable_simd)] // Requires nightly compiler
// use std::simd::*;

// Stable Rust: You must use platform-specific intrinsics
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

unsafe fn add_avx(a: *const f32, b: *const f32, c: *mut f32) {
    let va = _mm256_loadu_ps(a);  // x86 only!
    let vb = _mm256_loadu_ps(b);
    let vc = _mm256_add_ps(va, vb);
    _mm256_storeu_ps(c, vc);
}
// Not portable. ARM needs different code.
```

**Mojo (stable — first-class SIMD):**
```mojo
# nocompile
# SIMD is a built-in type in stable Mojo — always available
var a = SIMD[DType.float32, 8](1.0)
var b = SIMD[DType.float32, 8](2.0)
var c = a + b                     # 8 additions in one instruction
var sum = c.reduce_add()          # Built-in reduction
var mask = a > 1.5
var selected = mask.select(a, b)  # Branchless select
# Works on x86 (AVX), ARM (NEON), and GPU — same code
```

### 3. Python Interop Built-In

Rust requires PyO3 (complex setup, `maturin` build tool, Python FFI overhead).

**Rust (PyO3 — complex):**
```rust
// Cargo.toml: pyo3 = { version = "0.22", features = ["auto-initialize"] }
use pyo3::prelude::*;

fn use_numpy() -> PyResult<()> {
    Python::with_gil(|py| {
        let np = py.import("numpy")?;
        let arr = np.call_method1("array", ([1, 2, 3],))?;
        let mean = arr.call_method0("mean")?;
        println!("Mean: {}", mean);
        Ok(())
    })
}
// Requires: pip install maturin, Cargo.toml config, GIL management
```

**Mojo (built-in):**
```mojo
# nocompile
from python import Python

fn use_numpy() raises:
    var np = Python.import_module("numpy")
    var arr = np.array([1, 2, 3])
    print("Mean:", arr.mean())
# No build tool, no FFI config, no GIL management
```

### 4. Unified CPU + GPU in Same File

```mojo
from algorithm import parallelize
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim

# CPU parallel work
fn cpu_process(data: UnsafePointer[Float32], n: Int):
    @parameter
    fn worker(i: Int):
        data[i] *= 2.0
    parallelize[worker](n)

# GPU kernel — same file, same language
fn gpu_process(data: UnsafePointer[Float32], n: Int):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < n:
        data[tid] += 1.0

# Both used from the same main function
fn main() raises:
    var ctx = DeviceContext()
    # ... allocate, copy, launch, synchronize
```

### 5. Simpler Compile-Time Parameters

Rust's const generics are limited (only primitive types, no complex expressions).

**Rust (const generics — limited):**
```rust
// Rust const generics: only primitive types, no complex expressions
struct Matrix<const ROWS: usize, const COLS: usize> {
    data: [[f64; COLS]; ROWS],
}

// Can't do: const TOTAL: usize = ROWS * COLS  (in some contexts)
// Can't do: const generic over custom types
// Can't do: const generic functions that call other functions
```

**Mojo (parameters — fully general):**
```mojo
# nocompile
struct Matrix[ROWS: Int, COLS: Int, dtype: DType = DType.float64]:
    var data: InlineArray[SIMD[dtype, COLS], ROWS]

    fn __init__(out self):
        self.data = InlineArray[SIMD[dtype, COLS], ROWS](
            fill=SIMD[dtype, COLS](0)
        )

    # Parameters can be computed from other parameters
    comptime TOTAL = ROWS * COLS
    comptime IS_SQUARE = ROWS == COLS

    @parameter
    if Self.IS_SQUARE:
        fn trace(self) -> Scalar[dtype]:
            var t = Scalar[dtype](0)
            for i in range(ROWS):
                t += self.data[i][i]
            return t
```

---

## Performance Comparison

Both Rust and Mojo compile to native code and achieve similar raw performance for equivalent algorithms. The key Mojo advantages are:

| Area | Rust 1.93 | Mojo | Mojo Advantage |
|------|-----------|------|---------------|
| Scalar loops | Fast | Fast | Similar |
| SIMD | Nightly-only `std::simd` or unsafe intrinsics | First-class `SIMD[T, W]` | Portable, safe, stable |
| Parallelism | `rayon` crate | Built-in `parallelize` | No dependency |
| GPU kernels | FFI to CUDA/Vulkan | Native `enqueue_function` | Same language |
| Compile times | Minutes for large crates | Seconds (MLIR-based) | 5-10x faster |
| Python interop | PyO3 (complex) | Built-in | Zero setup |
| Build system | Cargo (excellent) | `mojo build` (simple) | Similar quality |

---

## Common Migration Patterns

### Pattern 1: Result<T, E> → raises

**Rust:**
```rust
fn parse_number(s: &str) -> Result<i64, String> {
    s.parse::<i64>().map_err(|e| e.to_string())
}

fn main() {
    match parse_number("42") {
        Ok(n) => println!("Got: {}", n),
        Err(e) => println!("Error: {}", e),
    }
}
```

**Mojo:**
```mojo
fn parse_number(s: String) raises -> Int:
    return Int(atol(s))

fn main():
    try:
        var n = parse_number("42")
        print("Got:", n)
    except e:
        print("Error:", e)
```

### Pattern 2: impl Block → Methods Inside Struct

**Rust:**
```rust
struct Circle {
    radius: f64,
}

impl Circle {
    fn new(radius: f64) -> Self {
        Circle { radius }
    }

    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
}

impl std::fmt::Display for Circle {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Circle(r={})", self.radius)
    }
}
```

**Mojo:**
```mojo
# nocompile
from math import pi

struct Circle(Writable):
    var radius: Float64

    fn __init__(out self, radius: Float64):
        self.radius = radius

    fn area(self) -> Float64:
        return pi[DType.float64]() * self.radius * self.radius

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Circle(r=", self.radius, ")")
```

### Pattern 3: Vec<T> Operations → List[T]

**Rust:**
```rust
let mut numbers: Vec<i32> = vec![3, 1, 4, 1, 5, 9];
numbers.sort();
numbers.dedup();
let sum: i32 = numbers.iter().sum();
let evens: Vec<&i32> = numbers.iter().filter(|x| *x % 2 == 0).collect();
```

**Mojo:**
```mojo
# nocompile
fn main():
    var numbers = List[Int](3, 1, 4, 1, 5, 9)
    sort(numbers)
    # dedup manually
    var sum = 0
    for x in numbers:
        sum += x[]
    var evens = List[Int]()
    for x in numbers:
        if x[] % 2 == 0:
            evens.append(x[])
```

### Pattern 4: Generic Functions

**Rust:**
```rust
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in &list[1..] {
        if item > largest {
            largest = item;
        }
    }
    largest
}
```

**Mojo:**
```mojo
# nocompile
fn largest[T: ComparableCollectionElement](list: List[T]) -> T:
    var result = list[0]
    for i in range(1, len(list)):
        if list[i] > result:
            result = list[i]
    return result
```

### Pattern 5: Trait Implementations

**Rust:**
```rust
trait Summary {
    fn summarize(&self) -> String;

    fn preview(&self) -> String {
        format!("{}...", &self.summarize()[..50])
    }
}

struct Article {
    title: String,
    content: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{}: {}", self.title, self.content)
    }
}
```

**Mojo:**
```mojo
trait Summary:
    fn summarize(self) -> String:
        ...

    fn preview(self) -> String:
        return self.summarize()  # Default implementation

struct Article(Summary):
    var title: String
    var content: String

    fn __init__(out self, title: String, content: String):
        self.title = title
        self.content = content

    fn summarize(self) -> String:
        return self.title + ": " + self.content
```

---

## CI Validation

### Version Checking

```bash
# Check Rust version (source language reference)
rustc --version    # Expect: rustc 1.93.x or later
cargo --version    # Expect: cargo 1.93.x or later

# Check Mojo version
mojo --version     # Expect: mojo 26.1.x or 26.2.x
```

### Validating Examples

```bash
# Verify Rust examples compile
rustc --edition 2024 --check example.rs

# Verify Mojo examples compile
# mojo run porting-examples/rust_to_mojo_test.mojo
```

### What to Watch For

| Language | Version | Breaking Changes to Monitor |
|----------|---------|---------------------------|
| Rust | 1.93 → 1.85+ | Edition 2024 changes, std::simd stabilization |
| Rust | std::simd | If stabilized, update SIMD comparison section |
| Mojo | v26.1 → v26.2 | Syntax changes, new traits |

---

## Version-Specific Features

### Stable (v26.1) vs Nightly (v26.2+)

| Feature | Stable (v26.1) | Nightly (v26.2+) |
|---------|----------------|-------------------|
| **Ownership** | `var` params, `^` transfer | Same |
| **Traits** | `trait Foo:`, `struct Bar(Foo):` | Same |
| **Constants** | `alias` or `comptime` | Both work |
| **SIMD** | `SIMD[DType, Width]` | Same |
| **GPU** | `DeviceContext`, `enqueue_function` | Same |
| **Struct alignment** | N/A | `@align(N)` |
| **Typed errors** | N/A | `fn foo() raises CustomError` |

---

## Related Patterns

- [`memory-ownership.md`](memory-ownership.md) — Deep dive on Mojo ownership
- [`memory-safety.md`](memory-safety.md) — Safe pointer patterns, origins
- [`type-traits.md`](type-traits.md) — Trait system details
- [`perf-vectorization.md`](perf-vectorization.md) — SIMD patterns
- [`gpu-fundamentals.md`](gpu-fundamentals.md) — GPU kernel development

---

## References

- [Mojo Manual](https://docs.modular.com/mojo/manual/)
- [Rust Reference](https://doc.rust-lang.org/reference/)
- [Rust Edition 2024](https://doc.rust-lang.org/edition-guide/rust-2024/)
