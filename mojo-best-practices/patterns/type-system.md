---
title: Type System Fundamentals
description: Core type system patterns including annotations, conversions, Optional handling, numeric precision, and hashable keys
impact: CRITICAL
category: type
tags: [types, annotations, implicit, optional, numeric, hashable, precision]
error_patterns:
  - "does not implement Hashable"
  - "does not implement Equatable"
  - "type mismatch"
  - "cannot convert .* to"
  - "missing type annotation"
  - "expected .* but got"
scenarios:
  - "Create hashable dictionary key"
  - "Handle Optional values safely"
  - "Define type alias"
  - "Add implicit conversion"
  - "Fix type mismatch error"
consolidates:
  - type-explicit-annotations.md
  - type-implicit-conversions.md
  - type-optional-patterns.md
  - type-numeric-precision.md
  - type-keyelement-composition.md
  - type-use-aliases.md
---

# Type System Fundamentals

**Category:** type | **Impact:** CRITICAL

Mojo's type system enables both compile-time safety and maximum performance. Explicit type annotations unlock 10-100x performance improvements, while proper use of Optional, numeric types, and hashable patterns ensures safe, efficient code.

---

## Core Concepts

### Explicit Type Annotations

Always annotate function parameters and return types. This enables compiler optimizations and catches errors at compile time.

**Pattern:**

```mojo
# nocompile
# Incorrect: Missing type annotations
def process(data):  # What type is data?
    result = data * 2  # What operations are valid?
    return result  # What type is returned?

# Correct: Explicit type annotations
fn process(data: Int) -> Int:
    # Compiler knows exact types - can optimize fully
    return data * 2

fn process_list(data: List[Float64]) -> Float64:
    var sum: Float64 = 0.0
    for item in data:
        sum += item
    return sum
```

**Benefits:**
- Compiler generates optimized machine code for specific types
- Type errors caught at compile time, not runtime
- Better IDE support and documentation
- Enables function overloading

### Type Aliases for Complex Types

Use `comptime` to define type aliases for improved readability and maintainability.

**Pattern:**

```mojo
# nocompile
# Incorrect: Repeated complex type signatures
fn process(
    data: Dict[String, List[Tuple[Int, Float64, String]]]
) -> Dict[String, List[Tuple[Int, Float64, String]]]:
    pass

# Correct: Type aliases for clarity
comptime Record = Tuple[Int, Float64, String]
comptime RecordList = List[Record]
comptime Database = Dict[String, RecordList]

fn process(data: Database) -> Database:
    # Clear and maintainable
    pass
```

**Common type alias patterns:**

```mojo
# nocompile
from utils import StaticTuple

# Numeric types
comptime Vec3 = SIMD[DType.float32, 4]  # 3D vector (4 for alignment)
comptime Mat4 = StaticTuple[Vec3, 4]    # 4x4 matrix

# Callback types
comptime Callback = fn(Int) -> Bool
comptime ErrorHandler = fn(String) raises -> None

# Collection types
comptime StringList = List[String]
comptime IntSet = Set[Int]
```

> **Note**: `alias` is deprecated in nightly; use `comptime` for compile-time constants and type aliases.

### Writable Trait

Use the `Writable` trait for output handling.

| Trait | Method | Use Case |
|-------|--------|----------|
| `Writable` | `write_to(writer)` | Efficient streaming output (preferred) |

**Pattern:**

```mojo
from format import Writable, Writer

@fieldwise_init
struct Point(Writable):
    var x: Int
    var y: Int

    # Writable: Efficient for print() and streaming
    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Point(", self.x, ", ", self.y, ")")
```

**When to use each:**

```mojo
# nocompile
var p = Point(10, 20)

# Writable: Used by print() - no intermediate String allocation
print(p)  # Calls write_to() internally

# When you need the String itself
var s: String = String(p)
var log_msg = "Created: " + String(p)
```

> **Best Practice**: Implement `Writable` for types you'll print frequently. `print()` uses `Writable`, so implementing `write_to` makes your type printable.

---

## Common Patterns

### Implicit Conversions with @implicit

Use `@implicit` for safe, lossless type conversions that improve API ergonomics.

**When:** Type conversion is always safe and expected

**Do:**
```mojo
struct Meters:
    var value: Float64

    fn __init__(out self, value: Float64):
        self.value = value

    # Allow implicit conversion from Int (lossless)
    @implicit
    fn __init__(out self, value: Int):
        self.value = Float64(value)

fn measure_distance(distance: Meters):
    print("Distance:", distance.value, "meters")

fn main():
    measure_distance(Meters(5.5))  # Explicit
    measure_distance(100)          # Implicit from Int - works!
```

**Don't:**
```mojo
struct UserId:
    var id: Int

    # WRONG: Silently converts any Int to UserId
    @implicit
    fn __init__(out self, id: Int):
        self.id = id

# This compiles but is probably a bug
fn process_user(user: UserId):
    pass

fn main():
    process_user(42)  # Did we mean to pass a UserId here?
```

**When to use @implicit:**
- Lossless numeric widening (Int -> Float64)
- String literal to String type conversions
- Unit types (meters, seconds) from raw numbers
- Wrapper types where conversion is always safe

**When NOT to use @implicit:**
- Lossy conversions (Float64 -> Int)
- Conversions that might fail or truncate
- When explicit conversion documents intent
- Between unrelated semantic types

### Optional Type Handling

Use `Optional[T]` instead of sentinel values for type-safe nullable handling.

**When:** A value may or may not exist

**Do:**
```mojo
# nocompile
fn find_index(items: List[Int], target: Int) -> Optional[Int]:
    for i in range(len(items)):
        if items[i] == target:
            return Optional(i)
    return Optional[Int](None)

# Type system forces handling the missing case
var idx = find_index(items, 42)
if idx:  # or: if idx is not None
    print(items[idx.value()])
```

**Don't:**
```mojo
# nocompile
fn find_index(items: List[Int], target: Int) -> Int:
    for i in range(len(items)):
        if items[i] == target:
            return i
    return -1  # Magic sentinel value - easy to forget check

# Caller must remember to check for -1
var idx = find_index(items, 42)
if idx != -1:  # Easy to forget this check
    print(items[idx])
```

**Key Optional patterns:**

```mojo
# nocompile
# Construction
var some_value = Optional(42)           # Contains value
var no_value = Optional[Int](None)      # Empty

# Checking for value
if optional:              # Bool conversion
    ...
if optional is not None:  # Pythonic identity check
    ...

# Safe access (aborts on empty)
var val = optional.value()

# Default value with or_else
var val = optional.or_else(default_value)
var val = optional.or_else(0)  # Returns 0 if empty

# Take value (moves out of Optional)
var val = optional.take()  # Optional is now empty

# Iterate at most once
for value in optional:
    print(value)  # Only executes if value present
```

**OptionalReg for trivial types:**

```mojo
# More efficient for small, trivial types
struct OptionalReg[T: __TypeOfAllTypes](TrivialRegisterPassable, Boolable, Defaultable):
    # Use for pointers, integers, etc.
    pass

# Example usage
fn find_pointer(target: Int) -> OptionalReg[UnsafePointer[Int]]:
    ...
```

### Numeric Type Selection

Choose numeric types based on range, precision, and performance requirements.

**When:** Processing numeric data

**Do:**
```mojo
# Use specific sizes when range is known
fn compute_index(x: Int32, y: Int32, width: Int32) -> Int64:
    # Use Int64 for result to prevent overflow
    return Int64(y) * Int64(width) + Int64(x)

# Use Float32 for graphics (sufficient precision, 2x throughput)
fn calculate_color(r: Float32, g: Float32, b: Float32) -> Float32:
    return (r + g + b) / 3.0

# Use BFloat16 for ML inference (memory bandwidth limited)
fn ml_layer(
    weights: SIMD[DType.bfloat16, 8],
    inputs: SIMD[DType.bfloat16, 8]
) -> SIMD[DType.bfloat16, 8]:
    return weights * inputs

# Use UInt8 for byte data
fn process_image(pixels: UnsafePointer[UInt8], size: Int):
    pass
```

**Don't:**
```mojo
fn compute_index(x: Int, y: Int, width: Int) -> Int:
    return y * width + x  # May overflow for large grids

fn calculate_color(r: Float64, g: Float64, b: Float64) -> Float64:
    # Float64 is overkill for 0-255 color values
    return (r + g + b) / 3.0
```

### Dictionary Key Requirements

Implement `ImplicitlyCopyable & Hashable & Equatable` for custom dictionary keys. Dict requires `ImplicitlyCopyable` (not just `Copyable`) because key lookups and insertions perform implicit copies internally.

**When:** Creating types to use as Dict keys

**Do:**
```mojo
# nocompile
from hashlib import Hasher  # Required for __hash__ signatures

@fieldwise_init
struct PersonId(ImplicitlyCopyable, Hashable, Equatable):
    var id: Int
    var department: String

    # Hashable: caller provides hasher, you feed data into it
    fn __hash__[H: Hasher](self, mut hasher: H):
        hasher.update(self.id)
        hasher.update(self.department)

    # Equatable: compare all fields
    fn __eq__(self, other: Self) -> Bool:
        return self.id == other.id and self.department == other.department

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

# Now works as Dict key
var employees = Dict[PersonId, String]()
employees[PersonId(123, "Engineering")] = "Alice"
```

> **Note**: The `Hashable` trait has a default `__hash__` implementation using reflection, so simple structs with all-`Hashable` fields can just declare `(Hashable)` without implementing `__hash__` manually. Use `hash(value)` to get the `UInt64` hash of a `Hashable` value.

**Don't:**
```mojo
# nocompile
struct PersonId:
    var id: Int
    var department: String

    fn __init__(out self, id: Int, department: String):
        self.id = id
        self.department = department

# Error: PersonId doesn't conform to ImplicitlyCopyable & Hashable & Equatable
var employees = Dict[PersonId, String]()  # Compile error!
```

**Hash quality guidelines:**

```mojo
# nocompile
from hashlib import Hasher  # Required import for Hasher type

# Good: Uses all fields, good distribution
fn __hash__[H: Hasher](self, mut hasher: H):
    hasher.update(self.field1)
    hasher.update(self.field2)
    hasher.update(self.field3)

# Bad: Only uses one field - many collisions
fn __hash__[H: Hasher](self, mut hasher: H):
    hasher.update(self.field1)  # Ignores field2, field3!

# Simplest: Let reflection handle it (all fields must be Hashable)
@fieldwise_init
struct MyKey(Hashable, Equatable, ImplicitlyCopyable):
    var field1: Int
    var field2: String
    # Default __hash__ uses all fields automatically
```

**Hash contract (must follow):**
- If `a == b`, then `hash(a) == hash(b)` (REQUIRED)
- Hash should be consistent for the lifetime of the object
- All fields used in `__eq__` should be used in `__hash__`

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Performance-critical code | Use `fn` with explicit types | [`fn-design.md`](fn-design.md) |
| Complex generic types | Define `comptime` type aliases | [`meta-programming.md`](meta-programming.md) |
| Safe nullable values | Use `Optional[T]` | [`error-handling.md`](error-handling.md) |
| Small trivial optionals | Use `OptionalReg[T]` | [`type-simd.md`](type-simd.md) |
| Custom Dict keys | Implement ImplicitlyCopyable & Hashable & Equatable traits | [`type-traits.md`](type-traits.md) |
| Lossless conversions | Use `@implicit` decorator | [`fn-design.md`](fn-design.md) |
| Graphics/ML workloads | Use Float32/BFloat16 | [`type-simd.md`](type-simd.md) |
| Scientific computing | Use Float64 | [`perf-optimization.md`](perf-optimization.md) |

---

## Quick Reference

- **Type annotations**: Always use `fn` with explicit types for 10-100x performance
- **Type aliases**: Use `comptime` (`alias` is deprecated)
- **@implicit**: Only for safe, lossless conversions (Int -> Float64, StringLiteral -> String)
- **Optional**: Use `.value()` for safe access, `.or_else(default)` for defaults
- **Numeric precision**: Float32 for graphics, Float64 for science, BFloat16 for ML
- **Dict keys**: Types must conform to ImplicitlyCopyable & Hashable & Equatable - all three required
- **Hash contract**: Equal objects must have equal hashes

---

## Version-Specific Features

### Constants: alias and comptime (v26.1+)

Use `comptime` for compile-time constants (`alias` is deprecated in nightly):

```mojo
# Preferred: comptime (v26.1+)
comptime PI: Float64 = 3.14159265358979323846
comptime TAU: Float64 = PI * 2
comptime MAX_BUFFER_SIZE: Int = 1024 * 1024

# Type aliases
comptime Vec4f = SIMD[DType.float32, 4]
comptime Predicate = fn(Int) -> Bool

# comptime: preferred keyword (v26.1+) — no warnings
comptime LEGACY_CONSTANT = 42
```

**Note:** `alias` is deprecated in nightly. Use `comptime` for all compile-time constants.

### v26.1+: comptime(expr)

Mojo supports forced compile-time evaluation with `comptime(expr)` syntax:

```mojo
# nocompile
# Force compile-time evaluation of expressions (v26.1+)
fn aligned_size[T: AnyType]() -> Int:
    return comptime((size_of[T]() + 63) & ~63)

# Compile-time assertions (v26.1+)
fn require_power_of_two[N: Int]():
    comptime assert (N & (N - 1)) == 0, "N must be power of 2"
```

### v26.1+: Linear Types with ImplicitlyDestructible

Types no longer need to implement `__del__()` by default. Use `ImplicitlyDestructible` for automatic cleanup.

```mojo
# Simple value types - no __del__ needed
struct Point(ImplicitlyDestructible):
    var x: Float64
    var y: Float64

# Generic constraints
fn process[T: AnyType](value: T):
    pass  # T doesn't need __del__()

fn take_ownership[T: ImplicitlyDestructible](var value: T):
    pass  # T will be properly destroyed
```

### v26.1+: Never Type

The `Never` type represents functions that never return normally.

```mojo
# nocompile
fn abort_program() -> Never:
    """Function that never returns."""
    ...

fn divide(a: Int, b: Int) -> Int:
    if b == 0:
        abort_program()  # Compiler knows this never returns
    return a // b

# With typed raises - non-raising via Never
fn non_raising_via_never() raises Never -> Int:
    return 42  # Compiles as non-raising
```

---

## v26.1+: String UTF-8 Safety

Mojo v26.1 provides safe constructors for creating Strings from byte data with explicit UTF-8 validation strategies.

### UTF-8 Constructors

```mojo
from memory import Span

# Option 1: Raising constructor - validates UTF-8, raises on invalid
fn parse_utf8_strict(data: Span[Byte]) raises -> String:
    return String(from_utf8=data)  # Raises if invalid UTF-8

# Option 2: Lossy constructor - replaces invalid bytes with U+FFFD
fn parse_utf8_lossy(data: Span[Byte]) -> String:
    return String(from_utf8_lossy=data)  # Never fails

# Option 3: Unsafe constructor - for pre-validated/trusted input
fn parse_utf8_trusted(data: Span[Byte]) -> String:
    return String(unsafe_from_utf8=data)  # No validation, fastest
```

### When to Use Each

| Constructor | Use Case | Safety |
|-------------|----------|--------|
| `from_utf8=` | User input, network data, files | Safe, raises on invalid |
| `from_utf8_lossy=` | Display purposes, logging | Safe, replaces invalid |
| `unsafe_from_utf8=` | Internal buffers, already-validated data | Unsafe but fast |

### Unicode Iteration

```mojo
# nocompile
var s = String("Hello 世界")

# Forward iteration over codepoints
for cp in s.codepoints():
    print(cp)

# Reverse iteration (replaces deprecated __reversed__)
for cp in s.codepoint_slices_reversed():
    print(cp)

# ASCII padding (renamed from ljust/rjust)
var padded = s.ascii_ljust(20, " ")
var right_padded = s.ascii_rjust(20, "0")
```

---

## v26.1+: Consuming Methods (Linear Types)

For types that must be explicitly consumed (not implicitly destroyed), use consuming methods with `var self`:

```mojo
# nocompile
struct UniqueHandle:
    """A handle that must be explicitly closed."""
    var fd: Int
    var closed: Bool

    fn __init__(out self, fd: Int):
        self.fd = fd
        self.closed = False

    # Consuming method - takes ownership of self
    fn close(var self):
        """Consume and close the handle."""
        if not self.closed:
            # Close the file descriptor
            self.closed = True
        # self is consumed here - cannot be used after

    # Regular method
    fn read(mut self, buffer: UnsafePointer[UInt8], size: Int) -> Int:
        return 0

fn main():
    var handle = UniqueHandle(42)
    _ = handle.read(ptr, 100)
    handle.close()  # Explicitly consume - required!
    # handle cannot be used after close()
```

### Linear vs ImplicitlyDestructible

| Type | Trait | Destruction |
|------|-------|-------------|
| Simple values | `ImplicitlyDestructible` | Automatic, no `__del__()` needed |
| Resource owners | `Destructible` | Implement `__del__()` for cleanup |
| Linear types | None | Must be explicitly consumed via `var self` methods |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `cannot infer type` | Missing type annotation | Add explicit type: `var x: Int = 5` |
| `type mismatch` | Incompatible types in expression | Cast explicitly or use correct type |
| `constrained` error | Compile-time constraint failed | Check parameter values meet requirements |
| `alias cannot be reassigned` | Trying to mutate alias | Use `var` for mutable values |
| `comptime expected` | Runtime value in comptime context | Ensure value is known at compile time |
| `invalid UTF-8` | Invalid bytes in String constructor | Use `from_utf8_lossy=` or validate input |
| `Optional has no value` | Calling `.value()` on None Optional | Check with `if opt:` before accessing value |

---

## Related Patterns

- [`type-simd.md`](type-simd.md) — SIMD types and vectorization
- [`type-traits.md`](type-traits.md) — Trait bounds and conformance
- [`memory-ownership.md`](memory-ownership.md) — Ownership and borrowing
- [`fn-design.md`](fn-design.md) — Argument passing and function design

---

## References

- [Mojo Type System](https://docs.modular.com/mojo/manual/types)
- [Mojo Functions Documentation](https://docs.modular.com/mojo/manual/functions)
- [Mojo Decorators](https://docs.modular.com/mojo/manual/decorators/)
