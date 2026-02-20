---
title: API Canary Tests
description: Quick compilation tests for critical APIs - first to break when Mojo changes
impact: LOW
category: meta
maturity: experimental
tags: [testing, canary, api-validation]
error_patterns: []
scenarios: []
---

# API Canary Tests

**Category:** meta | **Impact:** LOW

This file contains minimal compilable examples of critical APIs. It serves as an early warning system - if these examples fail to compile, patterns likely need updates.

**Note:** This is an internal testing file used by CI for early detection of API changes. These examples are kept minimal and focused on compilation validation.

---

## Version-Specific Features

| Feature | Status | Notes |
|---------|--------|-------|
| Heap allocation | `from memory import alloc` | v26.1+ |
| Constants | `alias` or `comptime` | Both work in v26.1+ |
| memcpy args | Keyword supported | Named params recommended |
| Struct alignment | `@align(N)` | v26.2+ nightly |
| TrivialRegisterType | Trait-based register passable | v26.2+ nightly |

**Note:** This canary file uses syntax compatible with both versions. If compilation fails on nightly, check the changelog for API changes.

---

## Memory APIs

### Heap Allocation with alloc()

```mojo
from memory import alloc

fn test_heap_alloc():
    var ptr = alloc[Int](10)
    ptr[0] = 42
    var val = ptr[0]
    ptr.free()

fn main():
    test_heap_alloc()
```

### memcpy with keyword arguments

```mojo
from memory import memcpy, alloc

fn test_memcpy():
    var src = alloc[Int](4)
    var dst = alloc[Int](4)

    for i in range(4):
        src[i] = i

    # Both v26.1 and v26.2+ support keyword arguments
    memcpy(dest=dst, src=src, count=4)

    src.free()
    dst.free()

fn main():
    test_memcpy()
```

---

## SIMD Types

### Basic SIMD

```mojo
fn test_simd():
    # Use fixed width for canary test - 4 is portable across all platforms
    comptime width = 4
    var vec = SIMD[DType.float32, width](1.0, 2.0, 3.0, 4.0)
    var sum = vec.reduce_add()  # Should be 10.0

fn main():
    test_simd()
```

---

## Struct Decorators

### @fieldwise_init (preferred over @value)

```mojo
@fieldwise_init
struct Point:
    var x: Int
    var y: Int

fn test_fieldwise_init():
    var p = Point(x=10, y=20)
    var sum = p.x + p.y

fn main():
    test_fieldwise_init()
```

### TrivialRegisterType (replaces @register_passable)

```mojo
# nocompile - Nightly v26.2+ only
# For stable v26.1, use: @register_passable("trivial")
struct SmallValue(TrivialRegisterType):
    var value: Int

    fn __init__(out self, v: Int):
        self.value = v

fn takes_by_value(v: SmallValue) -> Int:
    return v.value

fn test_trivial_register_type():
    var sv = SmallValue(42)
    var result = takes_by_value(sv)

fn main():
    test_trivial_register_type()
```

---

## Compile-Time Constants

### alias and comptime (both v26.1+)

```mojo
from memory import alloc

# Both alias and comptime work in v26.1+
comptime BUFFER_SIZE = 1024

fn test_comptime():
    var buffer = alloc[UInt8](BUFFER_SIZE)
    buffer.free()

fn main():
    test_comptime()
```

---

## FFI

### external_call

```mojo
# nocompile
from ffi import external_call

fn test_external_call():
    # Simple libc call
    var result = external_call["abs", Int32](Int32(-42))

fn main():
    test_external_call()
```

---

## Traits

### Writable trait

```mojo
struct MyType(Writable):
    var value: Int

    fn __init__(out self, value: Int):
        self.value = value

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("MyType(", self.value, ")")

fn test_writable():
    var obj = MyType(42)
    print(obj)

fn main():
    test_writable()
```

---

## Collections

### List

```mojo
fn test_list():
    var items = List[Int]()
    items.append(1)
    items.append(2)
    items.append(3)

    var sum = 0
    for item in items:
        sum += item

fn main():
    test_list()
```

### Dict

```mojo
fn test_dict():
    var d = Dict[String, Int]()
    d["one"] = 1
    d["two"] = 2

    var val = d.get("one", 0)

fn main():
    test_dict()
```

---

## Error Handling

### raises and Error

```mojo
fn might_fail(x: Int) raises -> Int:
    if x < 0:
        raise Error("negative value")
    return x * 2

fn test_error_handling():
    try:
        var result = might_fail(10)
    except e:
        print("Error:", e)

fn main():
    test_error_handling()
```

---

## Memory Safety

### Span (safe view into contiguous memory)

```mojo
# nocompile - variadic List constructor nightly-only
# For stable v26.1, use: var data = List[Int](); data.append(1); ...
from memory import Span

fn test_span():
    var data = List[Int](1, 2, 3, 4, 5)
    var span = Span(data)

    var sum = 0
    for i in range(len(span)):
        sum += span[i]

fn main():
    test_span()
```

### UnsafePointer (low-level memory access)

```mojo
from memory import alloc

fn test_unsafe_pointer():
    var ptr = alloc[Int](4)

    for i in range(4):
        ptr[i] = i * 10

    var sum = 0
    for i in range(4):
        sum += ptr[i]

    ptr.free()
    _ = sum

fn main():
    test_unsafe_pointer()
```

---

## Stack Allocation

### InlineArray (fixed-size stack array)

```mojo
fn test_inline_array():
    # v26.2+ requires bracket literal syntax - old InlineArray[T, N](1, 2, 3) no longer works
    var arr: InlineArray[Int, 4] = [1, 2, 3, 4]
    var sum = 0
    for i in range(4):
        sum += arr[i]

fn main():
    test_inline_array()
```

---

## String Operations

### String methods

```mojo
# nocompile - ascii_ljust/ascii_rjust are nightly v26.2+ only
fn test_string_ops():
    var s = String("Hello, World!")
    var upper = s.upper()
    var lower = s.lower()
    var contains = "World" in s
    var split = s.split(",")
    var length = len(s)
    
    # v26.2+ renamed ljust/rjust to ascii_ljust/ascii_rjust
    var padded = s.ascii_ljust(20, "*")
    var right_padded = s.ascii_rjust(20, "*")
    
    # v26.2+ Safety: use .as_bytes() for byte-level access
    # Direct subscripting returns Unicode codepoints now
    var bytes = s.as_bytes()
    var first_byte = bytes[0]
    
    _ = upper
    _ = lower
    _ = contains
    _ = split
    _ = length
    _ = padded
    _ = right_padded
    _ = first_byte

fn main():
    test_string_ops()
```

### StringLiteral vs String

```mojo
fn test_string_literal():
    comptime greeting = "Hello"
    var runtime_str = String(greeting) + ", World!"
    var length = len(runtime_str)
    _ = length

fn main():
    test_string_literal()
```

---

## Trait Implementation

### Movable trait

```mojo
struct Resource(Movable):
    var data: Int

    fn __init__(out self, value: Int):
        self.data = value

    fn __moveinit__(out self, deinit take: Self):
        self.data = take.data

fn takes_ownership(var r: Resource) -> Int:
    return r.data

fn test_movable():
    var r = Resource(42)
    var result = takes_ownership(r^)
    _ = result

fn main():
    test_movable()
```

### Copyable trait

```mojo
struct Value(Copyable):
    var data: Int

    fn __init__(out self, value: Int):
        self.data = value

    fn __copyinit__(out self, copy: Self):
        self.data = copy.data

fn takes_copy(v: Value) -> Int:
    return v.data

fn test_copyable():
    var v = Value(42)
    var result = takes_copy(v)
    _ = result
    _ = v.data

fn main():
    test_copyable()
```

---

## Parameters and Aliases

### Compile-time parameters

```mojo
comptime DEFAULT_SIZE = 16

fn parameterized_func[size: Int = DEFAULT_SIZE]() -> Int:
    return size * 2

struct Container[size: Int]:
    var data: InlineArray[Int, Self.size]

    fn __init__(out self, value: Int):
        self.data = InlineArray[Int, Self.size](fill=value)

    fn sum(self) -> Int:
        var result = 0
        for i in range(Self.size):
            result = result + self.data[i]
        return result

fn test_parameters():
    var result1 = parameterized_func()
    var result2 = parameterized_func[32]()
    var container = Container[4](10)
    var sum = container.sum()
    _ = result1
    _ = result2
    _ = sum

fn main():
    test_parameters()
```

---

## GPU Constructs

### Basic GPU indexing (compiles without GPU)

```mojo
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext

fn simple_kernel():
    var tid = thread_idx.x
    var bid = block_idx.x
    var bdim = block_dim.x
    var global_id = bid * bdim + tid
    _ = global_id

fn main():
    comptime THREADS_PER_BLOCK = 256
    comptime NUM_BLOCKS = 128
    _ = THREADS_PER_BLOCK
    _ = NUM_BLOCKS
```

---

## Python Interop

### Basic Python import

```mojo
# nocompile - Python interop type inference issue in stable v26.1
from python import Python

fn test_python_interop() raises:
    var np = Python.import_module("numpy")
    var arr = np.array([1, 2, 3, 4, 5])
    var sum_val = np.sum(arr)
    print("Sum:", sum_val)

fn main() raises:
    test_python_interop()
```

---

## Atomic Operations

### Atomic counter (from os.atomic)

```mojo
from os.atomic import Atomic

fn test_atomic():
    var counter = Atomic[DType.int64](0)
    _ = counter.fetch_add(1)
    _ = counter.fetch_add(5)
    var value = counter.load()
    _ = value

fn main():
    test_atomic()
```

---

## Math Functions with Constraints

### Floating-point functions (v26.2+ constraint changes)

```mojo
# nocompile - where clauses are nightly v26.2+ only
from math import exp, sin, cos, log

fn test_math_constraints[T: DType](x: SIMD[T, 1]) -> SIMD[T, 1] where T.is_floating_point():
    # v26.2+ uses where clauses instead of __comptime_assert in function body
    var result = exp(x) + sin(x) + cos(x) + log(x + 1.0)
    return result

fn main():
    var val = test_math_constraints[DType.float32](1.0)
    _ = val
```

### Custom Range with Slice Literals

```mojo
# nocompile - __slice_literal__ argument is nightly v26.2+ only
struct CustomRange:
    var start: Int
    var stop: Int
    var step: Int

    # v26.2+ requires __slice_literal__ parameter to disambiguate from collections
    fn __init__(out self, start: Int, stop: Int, step: Int = 1, __slice_literal__: () = ()):
        self.start = start
        self.stop = stop
        self.step = step

fn test_slice_literals():
    # This would be used in subscript context like: collection[CustomRange(0, 10, 2)]
    var r = CustomRange(0, 10, 2)
    _ = r.start

fn main():
    test_slice_literals()
```

---

## Additional Collections

### Set

```mojo
from collections import Set

fn test_set():
    var s = Set[Int]()
    s.add(1)
    s.add(2)
    s.add(3)
    s.add(2)  # Duplicate, won't be added

    var contains = 2 in s
    var size = len(s)

    _ = contains
    _ = size

fn main():
    test_set()
```

### Optional

```mojo
# nocompile - variadic List constructor nightly-only
# For stable v26.1, use: var items = List[Int](); items.append(10); ...
fn find_value(items: List[Int], target: Int) -> Optional[Int]:
    for i in range(len(items)):
        if items[i] == target:
            return i
    return None

fn test_optional():
    var items = List[Int](10, 20, 30, 40)
    var result = find_value(items, 30)
    if result:
        var idx = result.value()
        _ = idx

fn main():
    test_optional()
```

### Variant

```mojo
from utils import Variant

fn test_variant():
    var v = Variant[Int, String](42)

    if v.isa[Int]():
        var val = v[Int]
        _ = val

fn main():
    test_variant()
```

---

## Argument Conventions

### mut parameter (mutable reference)

```mojo
fn modify_value(mut x: Int):
    x = x + 10

fn test_mut():
    var value = 42
    modify_value(value)
    _ = value

fn main():
    test_mut()
```

---

## Version Info

This canary file is tested against:
- **Stable**: v26.1 (via https://github.com/modular/modular/releases)
- **Nightly**: Latest from conda.modular.com/max-nightly

Last updated: 2026-01-27
