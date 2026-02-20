---
title: Mojo Metaprogramming Patterns
description: Compile-time parameters, variadic parameters, conditional conformance, and parameter unpacking for zero-cost generics
impact: MEDIUM
category: meta
tags: [metaprogramming, parameters, generics, variadic, traits, compile-time]
error_patterns:
  - "parameter"
  - "comptime"
  - "alias"
  - "generic"
  - "variadic"
  - "trait bound"
  - "cannot infer"
  - "type parameter"
scenarios:
  - "Create generic type"
  - "Use compile-time parameters"
  - "Implement variadic function"
  - "Add conditional trait conformance"
  - "Unpack parameter lists"
  - "Write zero-cost abstractions"
consolidates:
  - meta-conditional-conformance.md
  - meta-parameter-unpacking.md
  - meta-variadic-params.md
  - meta-parameters.md
---

# Mojo Metaprogramming Patterns

**Category:** meta | **Impact:** MEDIUM

Metaprogramming in Mojo enables zero-cost generics through compile-time parameters. Parameters (in square brackets) are resolved at compile time, eliminating runtime overhead while providing type-safe, reusable code. This pattern covers compile-time parameters, variadic parameters, conditional conformance, and modern unpacking syntax.

---

## Core Concepts

### Compile-Time Parameters

Parameters enable type-safe generic programming with no runtime overhead. Types and values in square brackets are monomorphized at compile time.

**Pattern:**

```mojo
# nocompile
# Generic container using type parameter
# Note: List requires Copyable, so we add it to bounds
struct Stack[T: Movable & Copyable]:
    var data: List[Self.T]
    var _size: Int

    fn __init__(out self):
        self.data = List[Self.T]()
        self._size = 0

    fn push(mut self, var value: Self.T):
        self.data.append(value^)
        self._size += 1

    fn pop(mut self) raises -> Self.T:
        if self._size == 0:
            raise "stack underflow"
        self._size -= 1
        return self.data.pop()

    fn size(self) -> Int:
        return self._size

    fn is_empty(self) -> Bool:
        return self._size == 0

# Monomorphized at compile time - no runtime generics overhead
var int_stack = Stack[Int]()
int_stack.push(1)
int_stack.push(2)

var string_stack = Stack[String]()
string_stack.push("hello")
```

### Parameter Inference

The compiler can infer parameters from arguments.

**Pattern:**

```mojo
# nocompile
# Compiler infers parameters from arguments
fn print_type[T: Stringable](value: T):
    print(value)

print_type(42)      # T inferred as Int
print_type("hello") # T inferred as String
```

### Combining Type and Value Parameters

**Pattern:**

```mojo
# nocompile
from utils import StaticTuple

# Combining type and value parameters
struct FixedArray[T: Copyable, Size: Int]:
    var data: StaticTuple[T, Size]

    fn __init__(out self, default: T):
        self.data = StaticTuple[T, Size]()
        @parameter
        for i in range(Size):
            self.data[i] = default

    fn __getitem__(self, index: Int) -> T:
        return self.data[index]

    fn __setitem__(mut self, index: Int, value: T):
        self.data[index] = value

var arr = FixedArray[Int, 10](0)
arr[5] = 42
```

---

## Conditional Conformance

### Trait Bounds for Generic Types

Add methods to generic types only when type parameters satisfy certain traits. Use `Self.T` to reference type parameters inside struct bodies.

**Pattern:**

```mojo
# nocompile
# Modern Mojo requires Self.T syntax and explicit traits
struct Container[T: Movable & ImplicitlyDestructible & ImplicitlyCopyable](
    ImplicitlyDestructible
):
    var value: Self.T

    fn __init__(out self, var value: Self.T):
        self.value = value^

    fn get(self) -> Self.T:
        return self.value

fn main():
    var c = Container(42)
    print(c.get())  # Works
```

### Multiple Trait Constraints

Use `&` to combine trait requirements.

**Pattern:**

```mojo
# nocompile
# Use & to combine trait requirements
struct Pair[T: Movable & ImplicitlyDestructible & Equatable](
    ImplicitlyDestructible
):
    var first: Self.T
    var second: Self.T

    fn __init__(out self, var first: Self.T, var second: Self.T):
        self.first = first^
        self.second = second^

    fn __eq__(self, other: Self) -> Bool:
        return self.first == other.first and self.second == other.second
```

### Forwarding Trait Requirements

**Pattern:**

```mojo
# nocompile
# Wrapper that forwards trait requirements
struct Wrapper[T: Movable & ImplicitlyDestructible & Writable](
    ImplicitlyDestructible, Writable
):
    var value: Self.T

    fn __init__(out self, var value: Self.T):
        self.value = value^

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Wrapper(")
        self.value.write_to(writer)
        writer.write(")")
```

### Key Traits for Generic Types

| Trait | Purpose | When Needed |
|-------|---------|-------------|
| `Movable` | Value can be moved | Almost always |
| `ImplicitlyDestructible` | Auto-cleanup | Struct contains generic field |
| `ImplicitlyCopyable` | Return by value | Methods return `Self.T` |
| `Equatable` | `==` / `!=` operators | Comparison methods |
| `Writable` | Print support | Output methods |

---

## Variadic Parameters

### Variadic Value Parameters

Variadic parameters allow functions to accept a variable number of compile-time arguments with zero runtime overhead.

**Pattern:**

```mojo
# Function with variadic integer parameters
fn sum_compile_time[*values: Int]() -> Int:
    var result: Int = 0
    @parameter
    for i in range(len(VariadicList(values))):
        result += values[i]
    return result

comptime SUM = sum_compile_time[1, 2, 3, 4, 5]()  # 15, computed at compile time
```

### Direct Tuple Creation

**Pattern:**

```mojo
# Tuple literals work directly
var t = (1, "hello", 3.14)
print(t[0])  # 1
print(t[1])  # hello
print(t[2])  # 3.14

# Type is Tuple[Int, String, Float64]
```

### Multiple Type Parameters (Explicit Approach)

For a fixed number of different types, use explicit parameters for clearer error messages.

**Pattern:**

```mojo
# nocompile
# For a fixed number of different types, use explicit parameters
fn process_two[T1: Stringable, T2: Stringable](a: T1, b: T2):
    print(a)
    print(b)

fn process_three[T1: Movable, T2: Movable, T3: Movable](
    var a: T1, var b: T2, var c: T3
) -> Tuple[T1, T2, T3]:
    return (a^, b^, c^)

process_two(42, "hello")
var result = process_three(1, "test", 3.14)
```

### Variadic Type Parameters (Advanced)

**Pattern:**

```mojo
# nocompile
# Processing variadic type arguments
# Note: Accessing elements for printing may require additional trait bounds
fn count_args[*Ts: Movable](*args: *Ts) -> Int:
    return len(VariadicList(Ts))

var n = count_args(1, "hello", 3.14, True)  # n = 4
```

---

## Parameter Unpacking Syntax (v0.26.1+)

### Modern Unpacking with `...`

Mojo v0.26.1+ uses `...` syntax for parameter unpacking, replacing the older `*_` and `**_` syntax.

**Deprecated syntax:**

```mojo
# nocompile
# OLD - deprecated in v0.26.1
fn process[T: AnyType, *_Ts: AnyType, **_Kwargs: AnyType](...):
    pass

# OLD - explicitly unpacked parameters
comptime MyType = SomeGeneric[Int, String, *_, **_]
```

**New syntax:**

```mojo
# NEW - use ... for unpacking
fn process[T: AnyType, ...Ts: AnyType](...):
    pass

# NEW - explicitly unpacked parameters
comptime MyType = SomeGeneric[Int, String, ...]
```

### Usage Patterns

**Pattern:**

```mojo
# Variadic type parameters
struct Tuple[...Ts: AnyType]:
    # Ts is a variadic pack of types
    pass

# Using ... to forward all remaining parameters
fn wrapper[T: AnyType, ...Args](*args: *Args) -> T:
    return inner_fn[T, ...Args](*args)

# Partial type application with ...
comptime PartialDict = Dict[String, ...]  # Remaining params unbound
```

### Migration Guide

| Old Syntax | New Syntax | Meaning |
|------------|------------|---------|
| `*_` | `...` | Unpack remaining positional type params |
| `**_` | `...` | Unpack remaining keyword type params |
| `*_Name` | `...Name` | Named variadic pack |

**Migration steps:**

1. Replace `*_` with `...`
2. Replace `**_` with `...`
3. Replace `*_Name` with `...Name`
4. The compiler will warn on deprecated syntax

---

## Common Patterns

### Parameterized Functions with Trait Bounds

**When:** You need generic functions that work with any type meeting certain requirements.

**Do:**
```mojo
# nocompile
from utils import StaticTuple

# Generic function with trait bounds (needs Copyable for List, ImplicitlyCopyable for assignments)
fn find_max[T: Comparable & Copyable & ImplicitlyCopyable](items: List[T]) -> T:
    var max_val = items[0]
    for i in range(1, len(items)):
        if items[i] > max_val:
            max_val = items[i]
    return max_val

# Value parameter for compile-time sizes
fn create_identity_matrix[N: Int]() -> StaticTuple[StaticTuple[Float64, N], N]:
    var result = StaticTuple[StaticTuple[Float64, N], N]()
    @parameter
    for i in range(N):
        @parameter
        for j in range(N):
            result[i][j] = 1.0 if i == j else 0.0
    return result

comptime Identity4x4 = create_identity_matrix[4]()
```

**Don't:**
```mojo
# nocompile
# BAD: No trait bounds, won't compile for types without comparison
fn find_max_bad[T](items: List[T]) -> T:
    var max_val = items[0]
    for i in range(1, len(items)):
        if items[i] > max_val:  # Error: T doesn't guarantee comparison
            max_val = items[i]
    return max_val
```

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Generic container with any type | Use type parameter with trait bounds | `struct Stack[T: Movable & Copyable]` |
| Fixed-size array at compile time | Use value parameter | `struct FixedArray[T, Size: Int]` |
| Known small number of types | Use explicit parameters | `fn process[T1, T2](a: T1, b: T2)` |
| Unknown number of same-type args | Use variadic value params | `fn sum[*values: Int]()` |
| Forward all parameters | Use `...` syntax | `fn wrapper[...Args](*args: *Args)` |
| Methods only for certain types | Conditional conformance with trait bounds | `struct Pair[T: Equatable]` |

---

## Quick Reference

- **`Self.T` syntax**: Use inside struct bodies to reference type parameters
- **`&` operator**: Combine multiple trait requirements
- **`@parameter` loops**: Unroll loops at compile time for variadic iteration
- **`comptime`**: Declare compile-time computed constants
- **`...` unpacking**: Modern syntax for variadic parameters (v0.26.1+)
- **Monomorphization**: Each unique parameter combination generates specialized code

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `comptime expression not constant` | Runtime value in comptime context | Ensure all inputs are compile-time known |
| `@parameter loop not unrolling` | Non-constant bounds | Use `comptime` value for loop bounds |
| `variadic unpacking failed` | Using old `*_` syntax | Use `...` for unpacking in v0.26.1+ |
| `monomorphization explosion` | Too many parameter combinations | Reduce parameter space; use runtime dispatch for some cases |
| `compile-time recursion limit` | Deep recursive templates | Increase limit or restructure to iteration |
| `alias deprecated warning` | Using `alias` for constants | Both `alias` and `comptime` work in v26.1+ |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **Compile-time constants** | `alias` or `comptime` | Both work in v26.1+ |
| **Parameter unpacking** | `...` syntax | `*_` is deprecated |
| **Compile-time loops** | `@parameter for` | Stable |
| **Trait bounds** | `T: Movable & Copyable` | Stable |
| **Self.T syntax** | Required in structs | Stable |

**Example (v26.1+):**
```mojo
# nocompile
# Compile-time constant (both alias and comptime work)
comptime BUFFER_SIZE = 1024

# Parameter unpacking (new syntax)
fn forward[*Ts: AnyType](*args: *Ts):
    other_fn[...Ts](*args)  # Use ... for unpacking
```

**Notes:**
- Both `alias` and `comptime` work for compile-time constants in v26.1+
- Parameter unpacking uses `...` instead of deprecated `*_`
- `@parameter for` loops work the same across versions
- Trait bounds and Self.T syntax unchanged

---

## Related Patterns

- [`memory-ownership.md`](memory-ownership.md) — Implement lifecycle methods for generic types
- [`struct-design.md`](struct-design.md) — Simplify struct initialization with @fieldwise_init
- [`perf-vectorization.md`](perf-vectorization.md) — Use SIMD width as compile-time parameter

---

## References

- [Mojo Parameters and Metaprogramming](https://docs.modular.com/mojo/manual/parameters/)
- [Mojo Traits Documentation](https://docs.modular.com/mojo/manual/traits)
- [Mojo Changelog v0.26.1](https://docs.modular.com/mojo/changelog)
