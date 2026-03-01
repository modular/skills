---
title: Traits and Generic Programming
description: Trait definition, conformance, parametric traits, trait bounds, and conditional conformance patterns
impact: HIGH
category: type
tags: [types, traits, generics, bounds, conformance, parametric]
error_patterns:
  - "does not implement Copyable"
  - "does not implement Movable"
  - "does not implement Writable"
  - "does not conform to trait"
  - "missing trait bound"
scenarios:
  - "Make type printable with Writable"
  - "Implement Copyable for struct"
  - "Add trait bounds to generic function"
  - "Create custom trait"
  - "Fix missing trait conformance error"
consolidates:
  - type-trait-bounds.md
  - struct-trait-conformance.md
  - meta-conditional-conformance.md
---

# Traits and Generic Programming

**Category:** type | **Impact:** HIGH

Traits define contracts that types must fulfill, enabling type-safe generic programming. Proper trait conformance enables use with generic algorithms, collections, and debugging utilities while maintaining compile-time safety.

---

## Core Concepts

### Trait Conformance

Implementing standard library traits provides consistent behavior and enables use with generic algorithms and collections.

**Pattern:**

```mojo
# Incorrect: Missing useful traits
struct Vector2D:
    var x: Float64
    var y: Float64

    fn __init__(out self, x: Float64, y: Float64):
        self.x = x
        self.y = y

# Can't print it
# print(vec)  # Error: Vector2D doesn't implement Writable

# Can't compare
# if vec1 == vec2:  # Error: no __eq__ method
```

```mojo
# nocompile
# Correct: Implement appropriate traits
struct Vector2D(Copyable, Movable, Writable, Equatable):
    var x: Float64
    var y: Float64

    fn __init__(out self, x: Float64, y: Float64):
        self.x = x
        self.y = y

    fn __copyinit__(out self, copy: Self, /):
        self.x = copy.x
        self.y = copy.y

    fn __moveinit__(out self, deinit take: Self, /):
        self.x = take.x
        self.y = take.y

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Vector2D(", self.x, ", ", self.y, ")")

    fn __eq__(self, other: Self) -> Bool:
        return self.x == other.x and self.y == other.y

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

# Now it works everywhere
var v1 = Vector2D(1.0, 2.0)
var v2 = v1.copy()  # Explicit copy (use ImplicitlyCopyable trait for var v2 = v1)
print(v1)    # Printing works (via Writable)
if v1 == v2: # Comparison works
    print("Equal")
```

> **Warning: `Copyable` vs `ImplicitlyCopyable`** — This is the #1 source of porting errors.
> - `Copyable`: provides `.copy()` method only. `var b = a` does NOT compile.
> - `ImplicitlyCopyable` (refines `Copyable` + `ImplicitlyDestructible`): allows `var b = a` and implicit return-by-value in generic functions.
> - `List[T]` requires `T: Copyable` — move-only types cannot go in `List`.
> - Use `.copy()` as the safe default; only rely on implicit copy when the type is known to be `ImplicitlyCopyable`.

### Trait Bounds for Generics

Trait bounds (`[T: TraitName]`) specify what capabilities a generic type must have.

**Pattern:**

```mojo
# T must implement Writable to be printed
fn print_value[T: Writable](value: T):
    print(value)

# T must be comparable (and ImplicitlyCopyable to return by value)
fn find_max[T: Comparable & ImplicitlyCopyable](a: T, b: T) -> T:
    if a > b:
        return a
    return b

# Combining multiple trait bounds with &
fn process[T: Movable & ImplicitlyDestructible & Writable](var value: T):
    print(value)
    # value automatically destroyed
```

---

## Common Patterns

### Implementing Standard Traits

Implement the appropriate traits based on how your type will be used.

**When:** Creating custom types

**Do:**

```mojo
struct TrivialPoint(Copyable, Movable):
    var x: Float32
    var y: Float32

    # Declare that copy/move are just bitwise copies
    comptime __copyinit__is_trivial: Bool = True
    comptime __moveinit__is_trivial: Bool = True
    comptime __del__is_trivial: Bool = True
```

> **Note**: `__copyinit__is_trivial` enables optimized bitwise copies but does NOT enable implicit copy assignment (`var b = a`). For that, add the `ImplicitlyCopyable` trait to the struct declaration, or use explicit `var b = a.copy()`.

Generic code can check these flags:

```mojo
# nocompile
fn copy_array[T: Copyable](dest: UnsafePointer[T], src: UnsafePointer[T], n: Int):
    @parameter
    if T.__copyinit__is_trivial:
        memcpy(dest=dest, src=src, count=n)  # Fast path
    else:
        for i in range(n):
            (dest + i).init_pointee_copy(src[i])
```

**Common traits to implement:**

| Trait | Purpose | Required Methods |
|-------|---------|-----------------|
| Copyable | Allow explicit `.copy()` (does NOT enable `var b = a`) | `__copyinit__(out self, copy: Self, /)` |
| Movable | Allow moving | `__moveinit__(out self, deinit take: Self, /)` |
| Writable | Print/output support | `write_to[W: Writer]` |
| Representable | Debug representation | `__repr__` |
| Equatable | Equality comparison | `__eq__`, `__ne__` |
| Hashable | Use in Dict/Set | `__hash__` |
| Sized | Report size | `__len__` |

> **Note**: For `print()` to work, implement `Writable` trait with `write_to`. `Stringable` provides `__str__` for explicit string conversion.

> **Note**: In v26.1 (stable), use `__copyinit__`/`__moveinit__`. In v26.2+ (nightly), the preferred form is unified `__init__` overloads: `fn __init__(out self, *, copy: Self)` (replaces `__copyinit__`) and `fn __init__(out self, *, deinit take: Self)` (replaces `__moveinit__`). Both forms are accepted in nightly.

### Minimal Bounds Principle

Only require the traits you actually use.

**When:** Writing generic functions

**Do:**

```mojo
# Minimal requirements
fn just_print[T: Writable](value: T):
    print(value)
```

**Don't:**

```mojo
# Over-constrained
fn just_print[T: Movable & Copyable & Hashable & Writable](value: T):
    print(value)  # Only needs Writable
```

### Struct with Trait Bounds

Define generic structs with explicit trait requirements.

**When:** Creating generic containers

**Do:**

```mojo
# nocompile
struct SortedList[T: Movable & ImplicitlyDestructible & Comparable](
    ImplicitlyDestructible
):
    var data: List[Self.T]

    fn __init__(out self):
        self.data = List[Self.T]()

    fn add(mut self, var value: Self.T):
        # Can use < because T: Comparable
        var i = 0
        for item in self.data:
            if value < item:
                break
            i += 1
        self.data.insert(i, value^)
```

> **Note**: Use `Self.T` to reference type parameters inside struct bodies. Generic types often require `ImplicitlyDestructible` trait conformance.

### Trait Composition

Combine traits to define reusable requirement sets.

**When:** Multiple functions need the same trait combination

**Do:**

```mojo
# nocompile
# Custom trait compositions for reuse
comptime PrintableValue = Copyable & ImplicitlyDestructible & Writable
comptime SortableElement = Copyable & Comparable

# Usage in generic code — Dict keys require Copyable & Hashable & Equatable
fn print_dict[K: Copyable & Hashable & Equatable, V: PrintableValue](d: Dict[K, V]):
    for entry in d.items():
        print(entry.key, ":", entry.value)

fn lookup_or_insert[K: Copyable & Hashable & Equatable, V: Copyable](
    mut dict: Dict[K, V],
    key: K,
    default: V,
) -> V:
    """Get value or insert default if missing."""
    if key in dict:
        return dict[key]
    dict[key] = default
    return default
```

### Generic Wrapper Patterns

Create wrappers that forward trait requirements.

**When:** Building generic containers or decorators

**Do:**

```mojo
# Modern Mojo requires Self.T syntax and explicit traits
struct Container[T: Movable & ImplicitlyDestructible & ImplicitlyCopyable](
    ImplicitlyDestructible
):
    var value: Self.T

    fn __init__(out self, var value: Self.T):
        self.value = value^

    fn get(self) -> Self.T:
        return self.value

# Wrapper that forwards Writable
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

### Traits with Default Methods

```mojo
trait Describable:
    fn name(self) -> String: ...  # Required — implementor must provide

    fn describe(self) -> String:  # Default — can be overridden
        return "I am " + self.name()
```

> **Note:** Default methods can call other trait methods. Implementors override defaults by providing their own implementation.

### Conditional Conformance

Add methods conditionally based on type parameter traits.

**When:** Some operations only make sense for certain types

**Do:**

```mojo
struct Pair[T: Movable & ImplicitlyDestructible & Equatable](
    ImplicitlyDestructible
):
    var first: Self.T
    var second: Self.T

    fn __init__(out self, var first: Self.T, var second: Self.T):
        self.first = first^
        self.second = second^

    # __eq__ available because T: Equatable
    fn __eq__(self, other: Self) -> Bool:
        return self.first == other.first and self.second == other.second
```

**Key traits for generic types:**

| Trait | Purpose | When Needed |
|-------|---------|-------------|
| `Movable` | Value can be moved | Almost always |
| `ImplicitlyDestructible` | Auto-cleanup | Struct contains generic field |
| `ImplicitlyCopyable` | Return by value | Methods return `Self.T` |
| `Equatable` | `==` / `!=` operators | Comparison methods |
| `Writable` | Print support | Output methods |

### Runtime Trait Access

Check trait conformance and downcast at runtime when needed.

**When:** Generic code that can optionally use trait methods

**Do:**

> **Note:** `_constrained_conforms_to` and `trait_downcast` are internal APIs (prefixed with `_` or not publicly documented) and are not intended for general use. They are shown here for illustrative purposes only. Prefer using trait bounds in function signatures to statically enforce conformance.

```mojo
# nocompile
fn compare_elements[T: Copyable](a: T, b: T) -> Bool:
    # Check if type conforms to Equatable at runtime
    _constrained_conforms_to[
        conforms_to(T, Equatable),
        Parent=Self,
    ]()

    # Downcast to access trait methods
    ref lhs = trait_downcast[Equatable](a)
    ref rhs = trait_downcast[Equatable](b)
    return lhs == rhs
```

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Type used in collections | Implement Copyable, Movable | [`memory-ownership.md`](memory-ownership.md) |
| Type needs printing | Implement Writable | - |
| Type used as Dict key | Implement Copyable & Hashable & Equatable | [`type-system.md`](type-system.md) |
| Generic function | Use minimal trait bounds | [`fn-design.md`](fn-design.md) |
| Generic container | Require ImplicitlyDestructible | [`memory-ownership.md`](memory-ownership.md) |
| Common trait combos | Define comptime compositions | [`meta-programming.md`](meta-programming.md) |

---

## Quick Reference

- **Trait conformance**: Declare in struct signature: `struct Foo(Trait1, Trait2)`
- **Trait bounds**: Use `[T: Trait1 & Trait2]` for multiple requirements
- **Self.T syntax**: Use inside struct bodies to reference type parameters
- **Minimal bounds**: Only require traits you actually use
- **ImplicitlyDestructible**: Required when struct contains generic fields
- **Writable**: Use Writable for `print()` and string output via `write_to`
- **Trait composition**: Use `comptime MyTraits = Trait1 & Trait2 & Trait3` (or `alias`, both valid)

---

## Version-Specific Features

### v26.1 (Stable): Writable Trait for Output

Implement `Writable` with `write_to` for output. `print()` uses `Writable`.

```mojo
# nocompile
struct Point(Writable):
    var x: Float64
    var y: Float64

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Point(", self.x, ", ", self.y, ")")

# Works with print() and String conversion
var p = Point(1.0, 2.0)
print(p)  # Calls write_to()
var s = String(p)  # Converts to String
```

**With Representable for debug output:**

```mojo
# nocompile
struct User(Writable):
    var name: String
    var email: String

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.name, " <", self.email, ">")

    fn __repr__(self) -> String:
        return "User(name=" + repr(self.name) + ", email=" + repr(self.email) + ")"

var user = User("Alice", "alice@example.com")
print(user)       # Alice <alice@example.com>
print(repr(user)) # User(name="Alice", email="alice@example.com")
```

### v26.1+: Writable Trait

Use `Writable` with `write_to` for more efficient output (no intermediate allocations).

```mojo
# nocompile
struct Point(Writable):
    var x: Float64
    var y: Float64

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Point(", self.x, ", ", self.y, ")")

# Works with print() - no intermediate String created
var p = Point(1.0, 2.0)
print(p)  # Output: Point(1.0, 2.0)
```

**Why Writable over Stringable:**

| Feature | Stringable | Writable |
|---------|------------|----------|
| Intermediate allocations | Yes (creates String) | No (writes directly) |
| Output targets | Only String | Any Writer |
| Composition | String concatenation | Multiple write calls |
| Performance | More allocations | Fewer allocations |
| `Int` | Yes | Yes (changed) |
| `Intable` | Yes | Yes (changed) |
| `ImplicitlyIntable` | Yes | Yes (changed) |

**Converting to String when needed:**

```mojo
# nocompile
fn to_string[T: Writable](value: T) -> String:
    var buffer = String()
    value.write_to(buffer)
    return buffer

var s = to_string(Point(1.0, 2.0))  # "Point(1.0, 2.0)"
```

---

## Traits Mojo Does NOT Have

Understanding what traits are **not** available helps avoid confusion when coming from other languages.

### Not Available in Mojo

| Trait (Other Languages) | Mojo Alternative | Notes |
|-------------------------|------------------|-------|
| `Iterator` (Rust) | `Iterator` trait in `std/iter/` | Uses `fn __next__(mut self) raises StopIteration -> Self.Element` |
| `Display` (Rust) | `Writable` (v26.1+) or `Stringable` | Different naming |
| `Debug` (Rust) | `Representable` | Similar purpose |
| `Default` (Rust) | `Defaultable` | Available in stdlib |
| `Clone` (Rust) | `Copyable` with `__copyinit__` | Mojo uses copy constructors |
| `Drop` (Rust) | `__del__` destructor | Automatic with ASAP destruction |
| `Send` / `Sync` (Rust) | Not available | No trait-based thread safety |
| `Deref` (Rust) | Not available | Use explicit `.value` access |
| `AsRef` / `AsMut` (Rust) | Not available | Use references directly |
| `From` / `Into` (Rust) | `@implicit` constructor | Implicit conversions |
| `TryFrom` / `TryInto` (Rust) | `raises` constructor | Use fallible constructors |
| `Add`, `Sub`, etc. (Rust) | Operator dunder methods | `__add__`, `__sub__`, etc. |
| `PartialEq` / `Eq` (Rust) | `Equatable` | Single trait in Mojo |
| `PartialOrd` / `Ord` (Rust) | `Comparable` | Single trait in Mojo |
| `Serializable` (various) | Not available | Manual serialization |
| `Cloneable` (Java) | `Copyable` | Different mechanism |

### Common Mistakes from Rust Developers

**Using the Iterator trait:**

```mojo
# nocompile
# The stdlib provides an Iterator trait in std/iter/:
#   trait Iterator(ImplicitlyDestructible, Movable):
#       comptime Element: Movable
#       fn __next__(mut self) raises StopIteration -> Self.Element: ...

# Implement __iter__ and __next__ methods
struct MyIter[T: Movable & ImplicitlyDestructible]:
    var data: List[T]
    var index: Int

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) raises StopIteration -> T:
        if self.index >= len(self.data):
            raise StopIteration()
        var result = self.data[self.index]
        self.index += 1
        return result
```

**Expecting Send/Sync for thread safety:**

```mojo
# WRONG: No Send/Sync traits
# fn spawn_thread[T: Send](value: T):  # Not available

# CORRECT: Use explicit synchronization primitives
# Thread safety is handled via other mechanisms (not trait-based)
```

**Expecting Deref for smart pointers:**

```mojo
# nocompile
# WRONG: No Deref trait
# struct SmartPtr[T](Deref):  # Not available

# CORRECT: Use explicit access methods
struct SmartPtr[T: Movable & ImplicitlyDestructible]:
    var ptr: UnsafePointer[T]

    fn get(self) -> ref [self.ptr] T:
        """Explicit dereference method."""
        return self.ptr[]
```

### Trait Composition and Inheritance

Mojo supports both trait composition with `&` and trait inheritance (refinement):

```mojo
# Works: combining traits with & (e.g. for Dict keys)
comptime DictKey = Copyable & Hashable & Equatable

# Works: using composed traits as bounds
fn process[T: Copyable & Hashable & Equatable](value: T): ...

# SUPPORTED: trait inheritance (refinement)
# The stdlib uses this extensively:
#   trait Copyable(Movable)                           -- Copyable refines Movable
#   trait Comparable(Equatable)                       -- Comparable refines Equatable
#   trait ImplicitlyCopyable(Copyable, ImplicitlyDestructible)
#   trait Iterator(ImplicitlyDestructible, Movable)

# Custom trait refinement works the same way:
trait MyTrait(Copyable):
    fn custom_method(self): ...

# Types conforming to MyTrait must also satisfy Copyable
fn process[T: MyTrait](value: T): ...  # T is guaranteed Copyable
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `type does not implement trait` | Missing required method | Implement all methods required by trait |
| `trait bound not satisfied` | Generic param missing trait | Add trait to constraint: `T: Movable & MyTrait` |
| `cannot compose traits with &` | Old syntax or conflicting traits | Use `T: Trait1 & Trait2`; ensure no method conflicts |
| `default method not found` | Missing `pass` in trait body | Use `pass` for empty default or `...` for required (v26.1+) |
| `Dict key requires hash and equality` | Missing methods for Dict/Set | Implement both `__hash__` and `__eq__` (Copyable & Hashable & Equatable) |
| `Copyable requires Movable` | Only implementing Copyable | In v26.1+, Copyable refines Movable; implement both |

---

## Related Patterns

- [`type-system.md`](type-system.md) — Dict key requirements and basic type patterns
- [`type-simd.md`](type-simd.md) — Register-passable and trivial types
- [`memory-ownership.md`](memory-ownership.md) — Copyable/Movable implementation

---

## References

- [Mojo Traits Documentation](https://docs.modular.com/mojo/manual/traits/)
- [Mojo Parameters (Generics)](https://docs.modular.com/mojo/manual/parameters/)
