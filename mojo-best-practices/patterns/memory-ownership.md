---
title: Memory Ownership and Lifecycle Management
description: Comprehensive guide to ownership transfer, borrowing vs copying, implicit traits, and lifecycle methods in Mojo
impact: CRITICAL
category: memory
tags: [ownership, transfer, borrowing, copying, implicit-traits, lifecycle, destructor]
error_patterns:
  - "use of moved value"
  - "value borrowed after move"
  - "cannot borrow .* as mutable"
  - "ownership transfer required"
  - "abandoned without being explicitly destroyed"
  - "failed to infer parameter 'mut'"
scenarios:
  - "Transfer ownership to function"
  - "Return value with ownership transfer"
  - "Implement move constructor"
  - "Create generic container"
  - "Fix use-after-move error"
  - "Manage weight matrices with UnsafePointer"
  - "Implement model state transfer"
consolidates:
  - memory-ownership-transfer.md
  - memory-borrow-vs-copy.md
  - memory-implicit-traits.md
  - memory-lifecycle-methods.md
---

# Memory Ownership and Lifecycle Management

**Category:** memory | **Impact:** CRITICAL

This pattern covers Mojo's ownership system: how to transfer ownership with the `^` operator, when to borrow vs copy, using implicit traits for generic type safety, and implementing proper lifecycle methods. Mastering these concepts prevents use-after-free bugs, memory leaks, double-frees, and achieves 10-100x performance improvements for large data structures.

---

## Core Concepts

### Ownership Transfer with the ^ Operator

When passing a value to a function that takes ownership (`var` argument), use the `^` transfer operator to explicitly end the variable's lifetime. This makes ownership flow explicit and auditable.

**Pattern:**

```mojo
fn process_data(var data: List[Int]):
    # Function takes ownership of data
    for item in data:
        print(item)  # Iterate over list elements

fn main():
    var my_list = List[Int]()
    my_list.append(1)
    my_list.append(2)
    process_data(my_list^)  # Explicitly transfer ownership with ^
    # my_list is no longer valid here - compiler enforces this
```

**Anti-pattern (ambiguous ownership):**

```mojo
# nocompile
fn main():
    var my_list = List[Int]()
    my_list.append(1)
    process_data(my_list)  # Error: my_list must be transferred
    print(my_list[0])  # Potential use-after-move - compiler catches this
```

> **Note**: The `owned` keyword is deprecated. Use `var` for taking ownership of arguments, and `deinit` for destructors/move constructors.

### Borrowing vs Copying

The default `read` convention borrows immutably without copying. Use this for large structs to avoid unnecessary allocations and memory copies.

**Pattern (borrowing - 10-100x faster for large data):**

```mojo
fn analyze(data: List[Float64]) -> Float64:
    # data is borrowed immutably - no copy occurs
    # This is the default behavior with 'read' convention
    var sum: Float64 = 0.0
    for item in data:
        sum += item
    return sum / Float64(len(data))

fn main():
    var measurements = List[Float64]()
    for i in range(1000000):
        measurements.append(Float64(i))

    var avg = analyze(measurements)  # Borrowed, not copied
    print(avg)
    print(measurements[0])  # Still valid - we only borrowed
```

**When copying is appropriate:**

- Small types that fit in registers (Int, Float64, Bool)
- When you need an independent copy to modify
- Types conforming to `TrivialRegisterPassable` trait

### Parameter Convention Quick Reference

| Convention | Syntax | Ownership | Use When |
|-----------|--------|-----------|----------|
| Immutable borrow | `read` (default for `fn`) | Borrowed, read-only | Most parameters — safe, zero-cost |
| Mutable borrow | `mut` | Borrowed, writable | Need to modify caller's value |
| Ownership transfer | `owned` / `var` | Caller transfers ownership | Taking ownership of large values |
| Explicit origin | `ref [origin]` | Borrowed with named origin | Returning references, origin tracking |

> **Naming note:** `inout` is now `mut`, `owned` is now `var` (for parameter convention). Both old names still compile but produce warnings.

### Implicit Traits for Generic Type Safety

Modern Mojo requires explicit trait bounds for generic types. The `Implicit*` traits enable automatic behavior that would otherwise require manual handling.

**Key implicit traits:**

| Trait | Purpose | Required For |
|-------|---------|--------------|
| `ImplicitlyDestructible` | Auto-destructor calls | Structs with generic fields |
| `ImplicitlyCopyable` | Auto-copy on return (refines `Copyable` + `ImplicitlyDestructible`) | Methods returning `Self.T` by value |

> **Note**: There is no `ImplicitlyMovable` trait in the stdlib. For move semantics, use the `Movable` trait directly.

**Pattern (generic container):**

```mojo
# Generic container requires ImplicitlyDestructible
struct Box[T: Movable & ImplicitlyDestructible](ImplicitlyDestructible):
    var value: Self.T

    fn __init__(out self, var value: Self.T):
        self.value = value^

fn main():
    var b = Box(42)
    # b automatically destroyed when scope ends
```

**Pattern (returning generic values):**

```mojo
# Need ImplicitlyCopyable to return Self.T by value
struct Container[T: Movable & ImplicitlyDestructible & ImplicitlyCopyable](
    ImplicitlyDestructible
):
    var value: Self.T

    fn __init__(out self, var value: Self.T):
        self.value = value^

    fn get(self) -> Self.T:
        return self.value  # Requires ImplicitlyCopyable

fn main():
    var c = Container(42)
    var v = c.get()  # Works because Int is ImplicitlyCopyable
    print(v)
```

**Anti-pattern (missing implicit traits):**

```mojo
# nocompile - intentional error example
# WRONG: Missing ImplicitlyDestructible
struct BadBox[T: Movable]:
    var value: Self.T  # Error: 'self' abandoned without being explicitly destroyed

    fn __init__(out self, var value: Self.T):
        self.value = value^
```

### Lifecycle Methods

Types that manage resources (memory, file handles, sockets) must implement appropriate lifecycle methods: `__init__`, `__del__`.

**Pattern (complete lifecycle implementation):**

```mojo
from memory import UnsafePointer

# alias is deprecated; use comptime
comptime UInt8Ptr = UnsafePointer[mut=True, type=UInt8, origin=MutAnyOrigin]

struct FileBuffer(Movable):
    var data: UInt8Ptr
    var size: Int

    fn __init__(out self, size: Int):
        self.size = size
        from memory import alloc
        self.data = alloc[UInt8](size)

    fn __del__(deinit self):
        # Clean up allocated memory
        # NOTE: free() is a METHOD on UnsafePointer, not standalone function
        if self.data:
            self.data.free()

    fn __moveinit__(out self, deinit take: Self, /):
        # Transfer ownership - take source's resources
        self.data = take.data
        self.size = take.size

    # Explicitly NOT implementing Copyable means copying is disallowed
    # This is correct for unique resource ownership

    # Note on lifecycle method syntax:
    # - v26.1 (stable): `__moveinit__` and `__copyinit__` are the standard forms.
    # - v26.2+ (nightly): The preferred form uses unified `__init__` overloads:
    #     fn __init__(out self, *, deinit take: Self)   # replaces __moveinit__
    #     fn __init__(out self, *, copy: Self)           # replaces __copyinit__
    #   Both old and new forms are accepted, but nightly prefers the __init__ overloads.
```

> **Note**: The `owned` keyword is deprecated. Use `deinit` in destructors and move constructors, and `var` for regular function parameters that take ownership.

> **Tip**: When using `UnsafePointer` as a struct field, consider using the full type specification with named parameters: `UnsafePointer[mut=True, type=T, origin=MutAnyOrigin]`. The simpler `UnsafePointer[T]` syntax may fail with "failed to infer parameter 'mut'" in some contexts, though many stdlib uses employ shorter forms successfully.

---

## Common Patterns

### Copyable Value Types

**When:** Creating types that should be freely copyable (like Point, Color, etc.)

**Do:**

```mojo
@fieldwise_init
struct Point(Copyable):
    var x: Float64
    var y: Float64
```

To copy a `Copyable` value, use the explicit `.copy()` method:

```mojo
# nocompile
var p1 = Point(1.0, 2.0)
var p2 = p1.copy()  # Explicit copy; implicit copy requires ImplicitlyCopyable
```

> **Note:** `var p2 = p1` (implicit copy) only works if the type also conforms
> to `ImplicitlyCopyable`. Using `.copy()` is the safer default pattern.

Mojo synthesizes implementations of the copy and move constructor that is
correct for most cases. It is better to use this than reimplement it.

> **Warning:** `List[T]` requires `T: Copyable` — move-only types (only `Movable`) cannot be stored in `List`. If you need a collection of move-only types, use `UnsafePointer`-based storage with manual lifecycle management.

### Full-Featured Generic Container

**When:** Building containers that need printing, comparison, and value return

**Do:**

```mojo
struct SmartContainer[
    T: Movable & ImplicitlyDestructible & ImplicitlyCopyable & Writable
](ImplicitlyDestructible, Writable):
    var value: Self.T

    fn __init__(out self, var value: Self.T):
        self.value = value^

    fn get(self) -> Self.T:
        return self.value

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("SmartContainer(")
        self.value.write_to(writer)
        writer.write(")")

fn main():
    var sc = SmartContainer(42)
    print(sc)  # SmartContainer(42)
    print(sc.get())  # 42
```

### Complete Allocation Lifecycle

**When:** Managing raw memory with UnsafePointer

**Do:**

```mojo
# nocompile
# 1. Allocate
from memory import alloc
var ptr = alloc[MyType](count)

# 2. Initialize (memory is uninitialized after alloc!)
ptr.init_pointee_move(value^)           # Move into uninitialized
ptr.init_pointee_copy(value)            # Copy into uninitialized
ptr.init_pointee_move_from(other_ptr)   # Move from another pointer

# 3. Use
var val = ptr[]                          # Read
ptr[] = new_value                        # Write (to initialized memory)

# 4. Destroy pointees (for non-trivial types)
ptr.destroy_pointee()                    # Call destructor

# 5. Free memory
ptr.free()                               # Return memory to allocator
```

### @no_inline on Destructors

**When:** Reducing code size for types used frequently

**Do:**

```mojo
# nocompile
struct OwnedPointer[T: Movable]:
    # CRITICAL: Use full type spec with named parameters
    var _inner: UnsafePointer[mut=True, type=Self.T, origin=MutAnyOrigin]

    @no_inline  # Reduces code bloat from inlining destructor everywhere
    fn __del__(deinit self):
        self._inner.destroy_pointee()
        self._inner.free()
```

---

## Decision Guide

| Scenario | Approach | Trait Requirements |
|----------|----------|-------------------|
| Pass large data to function | Use default `read` (borrowing) | None |
| Transfer ownership permanently | Use `^` operator with `var` param | `Movable` |
| Allow copying of custom type | Implement copy ctor if custom; use `.copy()` for explicit copies | `Copyable` |
| Generic container with cleanup | Add `ImplicitlyDestructible` bound | `Movable & ImplicitlyDestructible` |
| Return generic value by value | Add `ImplicitlyCopyable` bound | `+ ImplicitlyCopyable` |
| Resource management (files, memory) | Implement `__del__` | None (manual) |
| Prevent copying | Don't implement `Copyable` | `Movable` only |

### Common Trait Combinations

| Use Case | Trait Bounds |
|----------|--------------|
| Basic container | `Movable & ImplicitlyDestructible` |
| With value return | `+ ImplicitlyCopyable` |
| With printing | `+ Writable` |
| With comparison | `+ Equatable` |

---

## Quick Reference

- **`^` operator**: Transfers ownership, ends variable lifetime
- **`var` parameter**: Function takes ownership of argument
- **`read` (default)**: Immutable borrow, no copy
- **`mut`**: Mutable borrow
- **`deinit`**: Consuming convention for `__del__` and `__init__(*, take=)` (value is destroyed when method returns)
- **`ImplicitlyDestructible`**: Required for generic types with automatic cleanup
- **`ImplicitlyCopyable`**: Required for returning generic types by value
- **`@no_inline` on destructors**: Reduces code bloat

---

## Key Traits for Lifecycle

| Trait | Purpose |
|-------|---------|
| `Movable` | Type can be moved (transfer ownership) |
| `Copyable` | Type can be copied via explicit `.copy()` call. Does NOT allow `var b = a` |
| `ImplicitlyDestructible` | Destructor called automatically (needed for generic containers) |
| `ImplicitlyCopyable` | Refines `Copyable` — allows implicit `var b = a` assignment and return-by-value in generics |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `use of moved value` | Value ownership transferred, then used | Use `^` only when done with value, or borrow instead |
| `abandoned without being explicitly destroyed` | Generic field missing `ImplicitlyDestructible` | Add `ImplicitlyDestructible` to trait bounds |
| `failed to infer parameter 'mut'` | UnsafePointer missing full type spec | Use `UnsafePointer[mut=True, type=T, origin=MutAnyOrigin]` |
| `cannot borrow as mutable` | Multiple mutable references | Restructure to single mutable ref at a time |
| `value borrowed after move` | Borrowed then transferred | Transfer ownership last, not mid-use |
| `field 'X.Y' destroyed out of the middle of a value` | Partial struct field move via `result.field^` | Add `fn take_field(deinit self) -> FieldType` method |
| `cannot transfer out of immutable reference` | `^` on `var` param field without `deinit self` | Change `fn f(var self)` parameter to `fn f(deinit self)` |

### Partial Move from Struct Fields (Use `deinit self`)

Moving a single field out of a struct with `result.field^` leaves the struct
partially destroyed. Mojo refuses this unless the whole struct is consumed.
The solution is a `deinit self` method that moves the desired field and lets the
remainder be dropped implicitly.

```mojo
# nocompile
struct ParseResult(Movable):
    var value: MyValue   # non-trivial (has List payload)
    var consumed: Int    # trivial

    # CORRECT: consume self entirely, return the interesting field
    fn take_value(deinit self) -> MyValue:
        return self.value^   # OK: self is consumed, consumed (Int) is dropped

fn example(var result: ParseResult) -> MyValue:
    return result^.take_value()  # move result then extract

    # WRONG — partial move:
    # return result.value^
    # Error: field 'result.value.X' destroyed out of the middle of a value
```

---

## Version-Specific Features

### Ownership Keywords

| Context | Keyword | Purpose |
|---------|---------|---------|
| Function parameters taking ownership | `var value: T` | Caller transfers or copies value; callee owns it |
| Move constructor (v26.1) | `fn __moveinit__(out self, deinit take: Self, /)` | Source is consumed and destroyed after move |
| Move constructor (v26.2+ preferred) | `fn __init__(out self, *, deinit take: Self)` | Unified `__init__` overload replacing `__moveinit__` |
| Copy constructor (v26.1) | `fn __copyinit__(out self, copy: Self, /)` | Makes a copy of the source value |
| Copy constructor (v26.2+ preferred) | `fn __init__(out self, *, copy: Self)` | Unified `__init__` overload replacing `__copyinit__` |
| `__del__` argument | `deinit self` | Value is destroyed when destructor returns |
| Consuming methods | `var self` | Method consumes `self`, taking ownership |

**Taking ownership in functions:**

```mojo
fn take_ownership(var value: String):  # 'var' takes ownership
    print(value)

fn main():
    var s = String("hello")
    take_ownership(s^)  # Use ^ to transfer ownership
    # s is no longer valid here
```

**Lifecycle methods:**

```mojo
# nocompile
struct Container[T: Movable](Movable):
    var data: T

    fn __init__(out self, var value: T):
        self.data = value^

    # only include this if custom; Mojo will synthesize a default impl.
    fn __moveinit__(out self, deinit take: Self, /):  # 'deinit' for move source
        self.data = take.data^

    fn __del__(deinit self):  # 'deinit' in destructor
        pass

    fn consume(var self) -> T:  # 'var self' for consuming methods
        return self.data^
```

### Constants: alias and comptime

Use `comptime` for compile-time constants (`alias` is deprecated in nightly).

```mojo
# Both syntaxes work in v26.1+
comptime BUFFER_SIZE = 1024
comptime MAX_ITEMS: Int = 100

fn main():
    print(BUFFER_SIZE, MAX_ITEMS)
```

**Version guidance:**

- Use `comptime` for compile-time constants (`alias` is deprecated in nightly)
- `alias` is deprecated in nightly

---

## Related Patterns

- [`memory-safety.md`](memory-safety.md) — Dangling references, origin tracking, Span usage
- [`memory-refcounting.md`](memory-refcounting.md) — Reference counting implementation

---

## References

- [Mojo Ownership Documentation](https://docs.modular.com/mojo/manual/values/ownership)
- [Mojo Value Semantics](https://docs.modular.com/mojo/manual/values/)
- [Mojo Traits Documentation](https://docs.modular.com/mojo/manual/traits)
- [Mojo Lifecycle Documentation](https://docs.modular.com/mojo/manual/lifecycle/)
