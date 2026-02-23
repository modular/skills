# Mojo - Common Mistakes & Gotchas

Quick reference for common Mojo pitfalls. Each entry shows the wrong approach, the correct approach, and explains why.

**See also:** [ERROR_INDEX.md](ERROR_INDEX.md) for error message lookup, pattern files for full context.

---

## Table of Contents

- [Memory & Ownership](#memory--ownership)
- [Type System](#type-system)
- [Structs & Initialization](#structs--initialization)
- [FFI & Strings](#ffi--strings)
- [Testing & Benchmarks](#testing--benchmarks)
- [GPU Programming](#gpu-programming)
- [Performance](#performance)
- [Version-Specific](#version-specific)

---

## Memory & Ownership

### Use After Move

**❌ WRONG:** Accessing a value after ownership transfer
```mojo
# nocompile - Demonstrates anti-pattern
fn bad_example():
    var data = String("hello")
    consume(data^)  # Ownership transferred with ^
    print(data)     # Use after move - data is invalid!
```

**✅ CORRECT:** Copy before transferring ownership
```mojo
fn good_example():
    var data = String("hello")
    var copy = data  # Create a copy first
    consume(data^)   # Transfer original
    print(copy)      # Use the copy
```

**Why?** The `^` transfer operator moves ownership. After move, the original variable is consumed and cannot be used.

---

### Dangling Reference from Temporary

**❌ WRONG:** Returning a reference to a temporary
```mojo
# nocompile - Demonstrates anti-pattern
fn get_data() -> Pointer[Int]:
    var temp = 42
    return Pointer.address_of(temp)  # temp dies at function end!
```

**✅ CORRECT:** Return by value or use caller-provided storage
```mojo
fn get_data() -> Int:
    return 42  # Return by value

# Or use out parameter:
fn get_data(out result: Int):
    result = 42
```

**Why?** Local variables are destroyed when the function returns. References to them become dangling.

---

### Missing `^` in Ownership Transfer

**❌ WRONG:** Forgetting transfer operator
```mojo
# nocompile - Demonstrates anti-pattern
fn bad_transfer(var value: String):
    pass

fn caller():
    var s = String("hello")
    bad_transfer(s)  # Error: would need to copy value, use ^ to transfer
```

**✅ CORRECT:** Use `^` to transfer ownership
```mojo
fn good_transfer(var value: String):
    pass

fn caller():
    var s = String("hello")
    good_transfer(s^)  # Explicitly transfer ownership
```

**Why?** Mojo requires explicit `^` for ownership transfer to prevent accidental moves. Use `var` for function parameters that take ownership.

---

## Type System

### SIMD Width Mismatch

**❌ WRONG:** Hardcoded SIMD width
```mojo
# nocompile - Demonstrates anti-pattern
fn process[T: DType](data: UnsafePointer[Scalar[T]]):
    var vec = data.load[width=8]()  # Assumes 8 works everywhere
```

**✅ CORRECT:** Use `simdwidthof` for portable code
```mojo
fn process[T: DType](data: UnsafePointer[Scalar[T]]):
    alias width = simdwidthof[T]()
    var vec = data.load[width=width]()
```

**Why?** SIMD width varies by hardware and dtype. `simdwidthof` returns the optimal width for the target.

---

### Missing Trait Implementation

**❌ WRONG:** Using a type without required trait
```mojo
# nocompile - Demonstrates anti-pattern
struct MyType:
    var value: Int

fn use_it():
    var x = MyType(42)
    var y = x  # Error: MyType doesn't implement Copyable
```

**✅ CORRECT:** Implement required traits
```mojo
@fieldwise_init
struct MyType(Copyable, Movable):
    var value: Int

fn use_it():
    var x = MyType(42)
    var y = x  # Works - Copyable implemented
```

**Why?** Mojo requires explicit trait implementation. Common traits: `Copyable`, `Movable`, `Stringable`, `Writable`.

---

## Structs & Initialization

### `@value` vs `@fieldwise_init`

**❌ WRONG:** Using deprecated `@value` decorator (pre-26.1)
```mojo
# nocompile - Deprecated syntax
@value
struct OldStyle:
    var x: Int
```

**✅ CORRECT:** Use `@fieldwise_init` (26.1+)
```mojo
@fieldwise_init
struct NewStyle:
    var x: Int
```

**Why?** `@value` is deprecated. `@fieldwise_init` only generates the initializer, giving you control over copy/move.

---

### Missing `out` in `__init__`

**❌ WRONG:** Forgetting `out self` parameter
```mojo
# nocompile - Demonstrates anti-pattern
struct Bad:
    var value: Int

    fn __init__(self, v: Int):  # Missing 'out'
        self.value = v
```

**✅ CORRECT:** Use `out self`
```mojo
struct Good:
    var value: Int

    fn __init__(out self, v: Int):
        self.value = v
```

**Why?** `out self` indicates the initializer is creating a new instance. Without it, `self` is borrowed (read-only).

---

## FFI & Strings

### `String.unsafe_ptr()` for FFI (not `as_c_string_slice().unsafe_ptr()`)

**❌ WRONG:** Chaining through `as_c_string_slice()` on a rvalue
```mojo
# nocompile - Demonstrates anti-pattern
fn call_c(s: String):
    external_call["c_func", NoneType](s.as_c_string_slice().unsafe_ptr())
    # Error: invalid use of mutating method on rvalue of type 'String'
```

**✅ CORRECT:** Call `unsafe_ptr()` directly on the `String`
```mojo
fn call_c(s: String):
    external_call["c_func", NoneType](s.unsafe_ptr())
```

**Why?** `as_c_string_slice()` is a mutating method (`mut self`) and cannot be called on a temporary/rvalue. `s.unsafe_ptr()` returns the same pointer to the string's null-terminated buffer without the mutation requirement.

---

### `StringSlice` vs `String` — Conversions Required

**❌ WRONG:** Passing slice results directly to functions expecting `String`
```mojo
# nocompile - Demonstrates anti-pattern
fn needs_string(s: String): ...

fn example(line: String):
    needs_string(line[:10].strip())  # strip() returns StringSlice, not String
    var hk = _lower(line[:colon])    # slicing returns StringSlice, not String
```

**✅ CORRECT:** Wrap with `String(...)` to convert
```mojo
fn example(line: String):
    needs_string(String(line[:10].strip()))    # Explicit conversion
    needs_string(String(String(line[:10]).strip()))  # Chain: slice → String → strip → String
    var hk = _lower(String(line[:colon]))      # Convert slice first
```

**Why?** String slicing (`s[a:b]`), `.strip()`, and similar methods return `StringSlice` (a borrowed view), not an owned `String`. Functions declared with `s: String` parameters do not accept `StringSlice` implicitly. Double-wrapping is needed when chaining: first convert the slice to `String`, then call the method.

---

### `List[UInt8]` vs `Span[Byte]` from `as_bytes()`

**❌ WRONG:** Expecting `as_bytes()` to return a `List`
```mojo
# nocompile - Demonstrates anti-pattern
fn takes_list(body: List[UInt8]): ...

fn example(s: String):
    takes_list(s.as_bytes())  # Error: Span[Byte] ≠ List[UInt8]
```

**✅ CORRECT:** Copy into a `List` explicitly
```mojo
fn example(s: String):
    var body = List[UInt8](s.as_bytes())  # Copy Span into owned List
    takes_list(body^)
```

**Why?** `String.as_bytes()` returns a `Span[Byte]` — a non-owning view into the string's buffer. `List[UInt8]` is an owned, heap-allocated buffer. When a function needs `List[UInt8]`, use `List[UInt8](span)` to copy.

---

### Default Arguments Must Not Call Runtime Functions

**❌ WRONG:** Default argument that calls `getenv` (or any syscall)
```mojo
# nocompile - Demonstrates anti-pattern
fn connect(url: String, config: TlsConfig = TlsConfig()) raises -> Connection:
    # TlsConfig.__init__ calls getenv() → evaluated at COMPILE TIME → fails
    ...
```

**✅ CORRECT:** Use function overloads to defer runtime default construction
```mojo
fn connect(url: String) raises -> Connection:
    # TlsConfig() created in function BODY → runtime → fine
    return _connect_impl(url, TlsConfig())

fn connect(url: String, config: TlsConfig) raises -> Connection:
    return _connect_impl(url, config)

fn _connect_impl(url: String, config: TlsConfig) raises -> Connection:
    ...
```

**Why?** Mojo evaluates default argument expressions at compile time during function type-checking and `TestSuite.discover_tests`. Any default that calls `getenv`, reads files, or performs I/O will fail with `failed to compile-time evaluate function call`. Use overloads instead, constructing the default value in the function body at runtime.

**Symptoms:** `note: failed to compile-time evaluate function call`, `unable to interpret call to unknown external function: getenv`

---

### Partial Move from Struct Fields

**❌ WRONG:** Moving one field out of a struct via `^`
```mojo
# nocompile - Demonstrates anti-pattern
struct Result(Movable):
    var frame: WsFrame   # non-trivial, has List[UInt8] payload
    var consumed: Int

fn example(result: Result) -> WsFrame:
    return result.frame^  # Error: field 'result.frame.payload' destroyed
                          # out of the middle of a value
```

**✅ CORRECT:** Add a `deinit self` method that consumes the whole struct
```mojo
struct Result(Movable):
    var frame: WsFrame
    var consumed: Int

    fn take_frame(deinit self) -> WsFrame:
        """Consume this result and return the frame."""
        return self.frame^  # OK: self is consumed, remaining fields are dropped

fn example(var result: Result) -> WsFrame:
    return result^.take_frame()
```

**Why?** When you move a field out via `result.field^`, the struct `result` is partially destroyed — its other fields still need destruction but `field` is gone. Mojo refuses partial moves unless the whole struct is consumed. `deinit self` parameters declare "I consume this value entirely; you don't need to destroy it after the call." The remaining `Int` fields are then destroyed normally at the end of `take_frame`.

**Symptoms:** `field 'result.X.Y' destroyed out of the middle of a value, preventing the overall value from being destroyed`

---

### Re-Raising a Caught Error Requires `^` (Transfer)

**❌ WRONG:** Re-raising without ownership transfer
```mojo
# nocompile - Demonstrates anti-pattern
try:
    risky_call()
except e:
    if some_condition:
        handle(e)
    else:
        raise e  # Error: value of type 'Error' cannot be implicitly copied
```

**✅ CORRECT:** Use `^` to transfer the error
```mojo
try:
    risky_call()
except e:
    if some_condition:
        handle(e)
    else:
        raise e^  # Transfer ownership
```

**Why?** `Error` does not conform to `ImplicitlyCopyable`. Re-raising it with `raise e` would need a copy. The `^` transfer operator moves ownership into the `raise`, which is what you want anyway — you don't need `e` after raising it.

**Symptoms:** `value of type 'Error' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'`

---

## Testing & Benchmarks

### Test Imports Must Be Top-Level with `TestSuite.discover_tests`

**❌ WRONG:** Importing inside a test function body
```mojo
# nocompile - Demonstrates anti-pattern
def test_connect():
    from flare.ws import WsClient  # Import inside function body
    var ws = WsClient.connect("ws://echo.example.com")
    ...

def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
```

**✅ CORRECT:** All imports at the top of the file
```mojo
from flare.ws import WsClient  # Top-level import

def test_connect():
    var ws = WsClient.connect("ws://echo.example.com")
    ...

def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
```

**Why?** `TestSuite.discover_tests[__functions_in_module()]` resolves all test functions at compile time. If a function-body import resolves to types with default arguments that call runtime syscalls (like `getenv`), Mojo tries to evaluate them at compile time and fails with `failed to compile-time evaluate function call`.

---

### Benchmark Functions Need `raises capturing` Signature

**❌ WRONG:** Plain non-raising benchmark function
```mojo
# nocompile - Demonstrates anti-pattern
fn my_bench() -> None:
    keep(expensive_op())

# Error: 'run' candidate not viable: value passed to 'func3' cannot be
# converted from 'fn() -> None' to 'fn() raises capturing -> None'
var r = run[my_bench]()
```

**✅ CORRECT:** Use the `Bench`/`Bencher` API with correct signature
```mojo
from benchmark import Bench, BenchConfig, Bencher, BenchId, keep

fn my_bench(mut b: Bencher) raises capturing:
    @parameter
    @always_inline
    fn call_fn() raises:
        keep(expensive_op())
    b.iter[call_fn]()

fn main() raises:
    var m = Bench(BenchConfig()^)
    m.bench_function[my_bench](BenchId("my operation"))
    m.dump_report()
```

**Why?** The `Bench.bench_function` API requires `fn(mut b: Bencher) raises capturing` as the function signature. The `run[fn]()` shorthand works only for `fn() raises capturing -> None` (closures). Use `@parameter @always_inline fn call_fn()` for the hot inner loop inside `b.iter[call_fn]()`.

Also: `BenchConfig` is not `ImplicitlyCopyable` — pass to `Bench` with `^`: `Bench(BenchConfig()^)`.

---

## GPU Programming

### LayoutTensor Reads Need `rebind` (Most Common Error)

**❌ WRONG:** Reading a LayoutTensor value directly via indexing
```mojo
# nocompile - Demonstrates anti-pattern
fn my_kernel[layout: Layout](tensor: LayoutTensor[dtype, layout, MutAnyOrigin]):
    var val: Scalar[dtype] = tensor[i]  # Error: element_type != Scalar[dtype]
    shared[i] = shared[i] + shared[i + stride]  # Error: can't add element_types
```

**✅ CORRECT:** Use `rebind[Scalar[dtype]]()` for LayoutTensor indexed reads
```mojo
fn my_kernel[layout: Layout](tensor: LayoutTensor[dtype, layout, MutAnyOrigin]):
    var val = rebind[Scalar[dtype]](tensor[i])  # Always rebind reads
    shared[i] = rebind[Scalar[dtype]](shared[i]) + rebind[Scalar[dtype]](shared[i + stride])  # Rebind both sides
    tensor[i] = val  # Writes work WITHOUT rebind
```

**Why?** LayoutTensor indexing returns `SIMD[dtype, symbolic_size]` where `symbolic_size` is a compile-time expression, not `Scalar[dtype]`. This happens with most layouts — concrete (`Layout.row_major(8)`) and generic (`layout: Layout`). Writes accept the symbolic type, but indexed reads need explicit conversion. Alternative APIs like `load_scalar` may avoid this, but `rebind` on indexed reads is the most common pattern.

**See:** [`gpu-layout-tensor.md`](../patterns/gpu-layout-tensor.md) for complete LayoutTensor patterns.

---

### Float Literals are Float64

**❌ WRONG:** Assigning float literals to Float32 variables
```mojo
# nocompile - Demonstrates anti-pattern
var val: Scalar[DType.float32] = 1.0  # Error: Float64 can't convert to Float32
var flag: Scalar[dtype] = 1.0 if condition else 0.0  # Same error
```

**✅ CORRECT:** Explicitly construct the target type
```mojo
var val = Scalar[DType.float32](1.0)
var flag = Scalar[dtype](1.0) if condition else Scalar[dtype](0.0)
```

**Why?** In Mojo, `1.0` is `Float64` by default. GPU kernels typically use `DType.float32`, so explicit construction is required.

---

### `print()` Unsupported in GPU Kernels

**❌ WRONG:** Using print for GPU kernel debugging
```mojo
# nocompile - Demonstrates anti-pattern
fn my_kernel(data: UnsafePointer[Float32]):
    var tid = thread_idx.x
    print("tid:", tid)  # Error: print not supported on GPU target
```

**✅ CORRECT:** Write to output buffer or use `_ = expr` to suppress warnings
```mojo
fn my_kernel(data: UnsafePointer[Float32], debug: UnsafePointer[Float32]):
    var tid = thread_idx.x
    debug[tid] = data[tid]  # Write debug values to a buffer, inspect on host
    _ = data[tid]  # Suppress unused variable warning without print
```

**Why?** GPU kernels compile to Metal/CUDA/ROCm shader code which typically has no console I/O. The error message: "Current compilation target does not support operation: print". Some NVIDIA targets may support limited `printf`, but it is not portable across GPU backends.

---

### Block Primitives Need `block_size` Parameter

**❌ WRONG:** Calling block primitives without block_size (fails on Apple Silicon)
```mojo
# nocompile - Demonstrates anti-pattern
from gpu.primitives.block import sum as block_sum
var total = block_sum(my_val)  # Missing block_size!
```

**✅ CORRECT:** Always provide `block_size` keyword parameter
```mojo
from gpu.primitives.block import sum as block_sum, broadcast as block_broadcast
var total = block_sum[block_size=TPB](my_val)
var shared = block_broadcast[block_size=TPB](my_val, 0)
```

**Why?** Block primitives need to know the block size at compile time to generate correct synchronization code. This is required on Apple Silicon and recommended on all targets.

**See:** [`gpu-block-collectives.md`](../patterns/gpu-block-collectives.md)

---

### Uncoalesced Memory Access

**❌ WRONG:** Non-contiguous thread access pattern
```mojo
# nocompile - Demonstrates anti-pattern
@gpu_kernel
fn bad_kernel(data: UnsafePointer[Float32]):
    var tid = gpu.thread_idx.x
    var val = data[tid * STRIDE]  # Strided access - uncoalesced!
```

**✅ CORRECT:** Adjacent threads access adjacent memory
```mojo
@gpu_kernel
fn good_kernel(data: UnsafePointer[Float32]):
    var tid = gpu.thread_idx.x
    var val = data[tid]  # Coalesced - adjacent threads, adjacent memory
```

**Why?** GPU memory is accessed in large chunks. Strided access wastes bandwidth and causes multiple transactions.

---

### Missing Barrier After Shared Memory Write

**❌ WRONG:** Reading shared memory without synchronization
```mojo
# nocompile - Demonstrates anti-pattern
@gpu_kernel
fn bad_kernel():
    var shared = gpu.shared_memory[128, Float32]()
    shared[gpu.thread_idx.x] = compute()
    # Missing barrier!
    var result = shared[other_idx]  # Race condition!
```

**✅ CORRECT:** Barrier before reading
```mojo
@gpu_kernel
fn good_kernel():
    var shared = gpu.shared_memory[128, Float32]()
    shared[gpu.thread_idx.x] = compute()
    gpu.barrier()  # Ensure all writes complete
    var result = shared[other_idx]  # Safe to read
```

**Why?** Without barrier, some threads may read before others have written. This causes race conditions.

---

### Wrong Thread Hierarchy

**❌ WRONG:** Confusing block and grid dimensions
```mojo
# nocompile - Demonstrates anti-pattern
@gpu_kernel
fn bad_kernel():
    # Wrong: using block_dim for grid-level indexing
    var global_id = gpu.block_idx.x * gpu.block_idx.y  # Nonsense!
```

**✅ CORRECT:** Proper global thread ID calculation
```mojo
@gpu_kernel
fn good_kernel():
    var global_id = gpu.block_idx.x * gpu.block_dim.x + gpu.thread_idx.x
```

**Why?**
- `block_idx`: Which block (0 to grid_dim-1)
- `block_dim`: Threads per block
- `thread_idx`: Thread within block (0 to block_dim-1)

---

## Performance

### Repeated Allocation in Loop

**❌ WRONG:** Allocating inside hot loop
```mojo
# nocompile - Demonstrates anti-pattern
fn bad_loop(n: Int):
    for i in range(n):
        var buffer = List[Int](capacity=1024)  # Allocates every iteration!
        process(buffer)
```

**✅ CORRECT:** Allocate once, reuse
```mojo
fn good_loop(n: Int):
    var buffer = List[Int](capacity=1024)  # Allocate once
    for i in range(n):
        buffer.clear()  # Reuse without reallocation
        process(buffer)
```

**Why?** Memory allocation is expensive. Reusing buffers avoids allocation overhead in hot paths.

---

### Missing `@always_inline` for Small Functions

**❌ WRONG:** Small helper without inline hint
```mojo
fn add(a: Int, b: Int) -> Int:
    return a + b

fn hot_path():
    for i in range(1000000):
        _ = add(i, 1)  # Function call overhead
```

**✅ CORRECT:** Inline small functions
```mojo
@always_inline
fn add(a: Int, b: Int) -> Int:
    return a + b
```

**Why?** Function call overhead matters in tight loops. `@always_inline` eliminates it.

---

## Version-Specific

### Stable vs Nightly Syntax Differences

**❌ WRONG:** Using nightly syntax on stable
```mojo
# nocompile - Nightly v26.2+ only
struct MyType(TrivialRegisterType):  # Error on stable!
    var value: Int
```

**✅ CORRECT:** Use version-appropriate syntax
```mojo
# Stable v26.1.0.0.0
@register_passable("trivial")
struct MyType:
    var value: Int
```

**Why?** `TrivialRegisterType` trait is nightly-only. Check your version with `mojo --version`.

---

### `comptime` vs `alias`

**❌ WRONG:** Using `comptime` on stable
```mojo
# nocompile - Nightly v26.2+ only
comptime SIZE: Int = 64
```

**✅ CORRECT:** Use `alias` (works on both)
```mojo
alias SIZE = 64  # Works on stable and nightly
```

**Why?** `comptime` keyword is nightly-only. `alias` is the stable equivalent for compile-time constants.

---

## Quick Lookup Table

| Error/Symptom | Likely Cause | Section |
|---------------|--------------|---------|
| "cannot implicitly convert 'LayoutTensor...element_type'" | LayoutTensor read without rebind | [LayoutTensor Reads](#layouttensor-reads-need-rebind-most-common-error) |
| "Float64 can't convert to Float32" | Float literal `1.0` is Float64 | [Float Literals](#float-literals-are-float64) |
| "does not support operation: print" | print() in GPU kernel | [Print Unsupported](#print-unsupported-in-gpu-kernels) |
| Block primitive fails on Apple | Missing `block_size` param | [Block Primitives](#block-primitives-need-block_size-parameter) |
| "use of moved value" | Missing copy before `^` | [Use After Move](#use-after-move) |
| "does not implement Copyable" | Missing trait | [Missing Trait](#missing-trait-implementation) |
| SIMD runtime error | Hardcoded width | [SIMD Width](#simd-width-mismatch) |
| GPU wrong results | Missing barrier | [Missing Barrier](#missing-barrier-after-shared-memory-write) |
| "would need to copy" / missing `^` | Missing transfer operator | [Missing Transfer](#missing--in-ownership-transfer) |
| Slow GPU kernel | Uncoalesced access | [Uncoalesced Memory](#uncoalesced-memory-access) |
| Nightly-only API error | Version mismatch | [Version-Specific](#version-specific) |
| "invalid use of mutating method on rvalue" | `as_c_string_slice()` on String | [FFI String unsafe_ptr](#stringunsafe_ptr-for-ffi-not-as_c_string_sliceunsafe_ptr) |
| `StringSlice` passed to `String` param | `.strip()` / slicing returns `StringSlice` | [StringSlice vs String](#stringslice-vs-string--conversions-required) |
| `Span[Byte]` passed to `List[UInt8]` param | `as_bytes()` returns Span, not List | [List vs Span](#listuint8-vs-spanbyte-from-as_bytes) |
| "failed to compile-time evaluate function call" | Default arg calls `getenv`/syscall | [Default Args Runtime](#default-arguments-must-not-call-runtime-functions) |
| "field 'X.Y' destroyed out of the middle of a value" | Partial struct field move with `^` | [Partial Move](#partial-move-from-struct-fields) |
| "value of type 'Error' cannot be implicitly copied" | `raise e` without `^` | [Re-Raise Error](#re-raising-a-caught-error-requires--transfer) |
| TestSuite test fails with compile-time eval error | Import inside test function body | [Test Imports](#test-imports-must-be-top-level-with-testsuitediscover_tests) |
| `run[fn]()` benchmark fails with "candidate not viable" | Wrong benchmark function signature | [Benchmark Signature](#benchmark-functions-need-raises-capturing-signature) |

---

## Related

- [ERROR_INDEX.md](ERROR_INDEX.md) - Full error message lookup
- [memory-ownership.md](../patterns/memory-ownership.md) - Complete ownership guide
- [gpu-fundamentals.md](../patterns/gpu-fundamentals.md) - GPU programming patterns
