---
title: Struct Design Patterns
description: Comprehensive patterns for designing Mojo structs including initialization, encapsulation, composition, operators, and iterators
impact: HIGH
category: struct
tags: [struct, initialization, encapsulation, composition, operators, iterators, generics]
error_patterns:
  - "struct .* has no member"
  - "cannot use .* as mutable"
  - "missing __init__"
  - "field .* not initialized"
  - "Self.T vs T"
scenarios:
  - "Create simple data struct"
  - "Add iterator to collection"
  - "Implement operator overloading"
  - "Use @fieldwise_init decorator"
  - "Organize struct code properly"
  - "Implement model configuration struct"
  - "Create model layer struct with weights"
consolidates:
  - struct-fieldwise-init.md
  - struct-encapsulation.md
  - struct-code-organization.md
  - struct-immutable-default.md
  - struct-composition.md
  - struct-operator-overloading.md
  - struct-self-type.md
  - struct-iterator-patterns.md
---

# Struct Design Patterns

**Category:** struct | **Impact:** HIGH

This pattern consolidates all struct design best practices in Mojo, covering initialization decorators, encapsulation strategies, code organization, mutability defaults, composition over inheritance, operator overloading, generic type parameters, and iterator implementation. Following these patterns ensures maintainable, performant, and idiomatic Mojo code.

---

## Core Concepts

### @fieldwise_init for Simple Data Structs

The `@fieldwise_init` decorator automatically generates a constructor that initializes all fields, reducing boilerplate and potential errors.

**Pattern:**

```mojo
# nocompile
@fieldwise_init
struct Config:
    var host: String
    var port: Int
    var timeout: Float64
    var debug: Bool

# Constructor is automatically generated
var config = Config("localhost", 8080, 30.0, True)

@fieldwise_init
struct Point3D:
    var x: Float64
    var y: Float64
    var z: Float64

var point = Point3D(1.0, 2.0, 3.0)
```

**When to use `@fieldwise_init`:**

- Simple data structs (DTOs, records)
- Types where all fields should be set at construction
- No complex initialization logic needed

**When NOT to use:**

- Types requiring validation in constructor
- Types with computed or derived fields
- Types needing partial initialization

**Combining with other functionality:**

```mojo
@fieldwise_init
struct Rectangle(Writable):
    var width: Float64
    var height: Float64

    # Additional methods work alongside generated __init__
    fn area(self) -> Float64:
        return self.width * self.height

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Rectangle(", self.width, "x", self.height, ")")
```

> **Note**: Use `@fieldwise_init` with explicit trait conformances like `Copyable`, `Movable`. The `@value` decorator has been removed - use `@fieldwise_init` with explicit traits instead.

### Encapsulation: Private Fields with Accessor Methods

Encapsulation allows you to change internal representation without breaking client code. It also enables validation and computed properties.

**Pattern:**

```mojo
# nocompile
struct Circle:
    var _radius: Float64  # Private by convention (underscore prefix)

    fn __init__(out self, radius: Float64) raises:
        if radius < 0:
            raise "radius must be non-negative"
        self._radius = radius

    @staticmethod
    fn from_diameter(diameter: Float64) raises -> Self:
        return Self(diameter / 2.0)

    fn radius(self) -> Float64:
        return self._radius

    fn set_radius(mut self, value: Float64) raises:
        if value < 0:
            raise "radius must be non-negative"
        self._radius = value

    fn diameter(self) -> Float64:
        return self._radius * 2.0

    fn area(self) -> Float64:
        return 3.14159 * self._radius * self._radius

# Client code uses methods
var c = Circle(5.0)
var r = c.radius()     # Can change implementation
var d = c.diameter()   # Computed property
# c.set_radius(-1.0)   # Raises error - validation enforced
```

**Benefits of encapsulation:**

- Change internal representation without breaking clients
- Add validation logic
- Support computed/derived properties
- Clearer API boundaries

**When direct field access is OK:**

- Simple data transfer objects (DTOs)
- Performance-critical inner loops (with `@fieldwise_init`)
- Internal/private types

### Self.T for Generic Type Parameters

In modern Mojo, generic type parameters must be referenced as `Self.T` inside struct bodies, not just `T`. This makes the parameter scope explicit.

**Pattern:**

```mojo
# nocompile
struct Box[T: Movable & ImplicitlyDestructible](ImplicitlyDestructible):
    var value: Self.T  # Use Self.T, not T

    fn __init__(out self, var value: Self.T):
        self.value = value^

    fn get(self) -> Self.T where Self.T: ImplicitlyCopyable:
        return self.value
```

**Multiple type parameters:**

```mojo
# nocompile
struct Pair[K: Movable & ImplicitlyDestructible, V: Movable & ImplicitlyDestructible](
    ImplicitlyDestructible
):
    var key: Self.K      # Use Self.K
    var value: Self.V    # Use Self.V

    fn __init__(out self, var key: Self.K, var value: Self.V):
        self.key = key^
        self.value = value^

    fn get_key(self) -> Self.K where Self.K: ImplicitlyCopyable:
        return self.key

    fn get_value(self) -> Self.V where Self.V: ImplicitlyCopyable:
        return self.value
```

**Functions vs Structs:**

```mojo
# nocompile
# Functions: Can use unqualified T
fn identity[T: Movable](var value: T) -> T:
    return value^  # OK in functions

# Structs: Must use Self.T
struct Wrapper[T: Movable & ImplicitlyDestructible](ImplicitlyDestructible):
    var inner: Self.T  # Required: Self.T
```

---

## Common Patterns

### Standard Code Organization

The official Modular style guide prescribes a specific section header format for organizing struct members.

**Do:**

```mojo
# nocompile
# ===-----------------------------------------------------------------------===#
# MyStruct
# ===-----------------------------------------------------------------------===#


struct MyStruct(Sized, Stringable):
    """Description of the struct and its purpose."""

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime factor = 5

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var field: Int

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        self.field = 0

    fn __init__(out self, *, deinit take: Self):
        self.field = take.field

    fn __init__(out self, *, copy: Self):
        self.field = copy.field

    fn __del__(deinit self):
        pass

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    fn from_value(value: Int) -> Self:
        var result = Self()
        result.field = value
        return result

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __getitem__(self, index: Int) -> Int:
        return self.field

    fn __add__(self, other: Self) -> Self:
        return Self.from_value(self.field + other.field)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __len__(self) -> Int:
        return 1

    fn __str__(self) -> String:
        return String(self.field)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn do_something(self):
        pass
```

**Don't:**

```mojo
struct MyStruct:
    var field: Int
    fn do_something(self): pass
    fn __init__(out self): self.field = 0
    comptime factor = 5
    fn __len__(self) -> Int: return 1
```

**Section order:**

1. **Aliases** - Type aliases and compile-time constants
2. **Fields** - Instance variables (`var` declarations)
3. **Life cycle methods** - `__init__`, `__moveinit__`, `__copyinit__`, `__del__`
4. **Factory methods** - `@staticmethod` constructors returning `Self`
5. **Operator dunders** - `__getitem__`, `__setitem__`, `__add__`, `__iter__`, etc.
6. **Trait implementations** - `__len__`, `__str__`, `__bool__`, `__hash__`, etc.
7. **Methods** - Regular instance methods

### Immutability by Default

Methods should only take `mut self` when they actually need to modify the struct. Immutable methods enable better compiler optimizations and clearer API contracts.

**Do:**

```mojo
struct Counter:
    var value: Int

    fn __init__(out self, initial: Int = 0):
        self.value = initial

    fn get(self) -> Int:  # Immutable - clearly doesn't modify
        return self.value

    fn is_zero(self) -> Bool:  # Read-only
        return self.value == 0

    fn to_string(self) -> String:  # Read-only
        return String(self.value)

    fn increment(mut self):  # Mutable - clearly modifies
        self.value += 1

    fn add(mut self, amount: Int):  # Mutable - modifies state
        self.value += amount

    fn incremented(self) -> Self:  # Returns new value, doesn't modify
        return Self(self.value + 1)
```

**Don't:**

```mojo
struct Counter:
    var value: Int

    fn get(mut self) -> Int:  # Why mut? This doesn't modify anything
        return self.value

    fn is_zero(mut self) -> Bool:  # Unnecessarily mutable
        return self.value == 0
```

**Pattern: Immutable operations return new values**

```mojo
# nocompile
struct Vector2D:
    var x: Float64
    var y: Float64

    # Immutable - returns new vector
    fn scaled(self, factor: Float64) -> Self:
        return Self(self.x * factor, self.y * factor)

    fn normalized(self) -> Self:
        var len = sqrt(self.x*self.x + self.y*self.y)
        return Self(self.x / len, self.y / len)

    # Mutable - modifies in place
    fn scale(mut self, factor: Float64):
        self.x *= factor
        self.y *= factor

    fn normalize(mut self):
        var len = sqrt(self.x*self.x + self.y*self.y)
        self.x /= len
        self.y /= len
```

### Composition Over Inheritance

Mojo uses traits rather than class inheritance. Compose behavior by embedding structs and implementing traits for a more flexible design.

**Do:**

```mojo
# Define behavior contracts with traits
trait Drawable:
    fn draw(self): ...

trait Updatable:
    fn update(mut self, dt: Float64): ...

# Compose with embedded structs
struct Transform:
    var x: Float64
    var y: Float64
    var rotation: Float64

    fn __init__(out self, x: Float64 = 0, y: Float64 = 0, rotation: Float64 = 0):
        self.x = x
        self.y = y
        self.rotation = rotation

struct Velocity:
    var dx: Float64
    var dy: Float64

    fn __init__(out self, dx: Float64 = 0, dy: Float64 = 0):
        self.dx = dx
        self.dy = dy

# Compose structs and implement relevant traits
struct Sprite(Drawable, Updatable):
    var transform: Transform   # Composition
    var velocity: Velocity     # Composition
    var texture_id: Int

    fn __init__(out self, x: Float64, y: Float64, texture_id: Int):
        self.transform = Transform(x, y)
        self.velocity = Velocity()
        self.texture_id = texture_id

    fn draw(self):
        print("Drawing sprite", self.texture_id, "at",
              self.transform.x, self.transform.y)

    fn update(mut self, dt: Float64):
        self.transform.x += self.velocity.dx * dt
        self.transform.y += self.velocity.dy * dt
```

**Use traits for polymorphism:**

```mojo
# nocompile
fn draw_all[T: Drawable](items: List[T]):
    for item in items:
        item.draw()

fn update_all[T: Updatable](mut items: List[T], dt: Float64):
    for i in range(len(items)):
        items[i].update(dt)
```

### Complete Operator Overloading

When implementing operators, provide the complete set: binary, reverse-binary, and in-place variants.

**Do:**

```mojo
# nocompile
struct Complex:
    var re: Float64
    var im: Float64

    # Binary operator
    fn __add__(self, rhs: Self) -> Self:
        return Self(self.re + rhs.re, self.im + rhs.im)

    # Reverse binary (called when left operand doesn't support op)
    fn __radd__(self, lhs: Float64) -> Self:
        return Self(self.re + lhs, self.im)

    # In-place operator
    fn __iadd__(mut self, rhs: Self):
        self = self + rhs

# All work now
var result1 = c1 + c2       # __add__
var result2 = 2.0 + c1      # __radd__
c1 += c2                    # __iadd__
```

**Complete arithmetic operator set:**

```mojo
# nocompile
struct Vector:
    var x: Float64
    var y: Float64

    # Addition
    fn __add__(self, rhs: Self) -> Self: ...
    fn __radd__(self, lhs: Float64) -> Self: ...
    fn __iadd__(mut self, rhs: Self): ...

    # Subtraction
    fn __sub__(self, rhs: Self) -> Self: ...
    fn __rsub__(self, lhs: Float64) -> Self: ...
    fn __isub__(mut self, rhs: Self): ...

    # Multiplication (element-wise and scalar)
    fn __mul__(self, rhs: Self) -> Self: ...
    fn __mul__(self, rhs: Float64) -> Self: ...  # Scalar multiply
    fn __rmul__(self, lhs: Float64) -> Self: ...
    fn __imul__(mut self, rhs: Float64): ...

    # Division
    fn __truediv__(self, rhs: Float64) -> Self: ...
    fn __rtruediv__(self, lhs: Float64) -> Self: ...
    fn __itruediv__(mut self, rhs: Float64): ...

    # Negation (unary)
    fn __neg__(self) -> Self:
        return Self(-self.x, -self.y)

    # Positive (unary, usually returns self)
    fn __pos__(self) -> Self:
        return self
```

**Comparison operators:**

```mojo
# nocompile
struct Point(Equatable, Comparable):
    var x: Int
    var y: Int

    # Equality (required for Equatable)
    fn __eq__(self, other: Self) -> Bool:
        return self.x == other.x and self.y == other.y

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    # Ordering (required for Comparable)
    fn __lt__(self, other: Self) -> Bool:
        # Lexicographic ordering
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y

    fn __le__(self, other: Self) -> Bool:
        return self == other or self < other

    fn __gt__(self, other: Self) -> Bool:
        return other < self

    fn __ge__(self, other: Self) -> Bool:
        return other <= self
```

**Indexing operators:**

```mojo
# nocompile
struct Matrix:
    var data: UnsafePointer[Float64]
    var rows: Int
    var cols: Int

    # Single index (returns row)
    fn __getitem__(ref self, row: Int) -> ref [self] Float64:
        return self.data[row * self.cols]

    # Multi-index (row, col)
    fn __getitem__(ref self, row: Int, col: Int) -> ref [self] Float64:
        return self.data[row * self.cols + col]

    # Setitem for mutation
    fn __setitem__(mut self, row: Int, col: Int, value: Float64):
        self.data[row * self.cols + col] = value

    # Contains check
    fn __contains__(self, value: Float64) -> Bool:
        for i in range(self.rows * self.cols):
            if self.data[i] == value:
                return True
        return False
```

**Boolean context and type conversion:**

```mojo
# nocompile
struct Optional[T: Movable](Boolable):
    var _value: T
    var _has_value: Bool

    # Boolean conversion (for if statements)
    fn __bool__(self) -> Bool:
        return self._has_value

# Usage
var opt = Optional[Int](42)
if opt:  # Calls __bool__
    print("Has value")

struct Fraction:
    var num: Int
    var den: Int

    # Convert to Float64
    fn __float__(self) -> Float64:
        return Float64(self.num) / Float64(self.den)

    # Convert to Int (truncates)
    fn __int__(self) -> Int:
        return self.num // self.den

    # Implicit conversion (use sparingly)
    @implicit
    fn __init__(out self, value: Int):
        self.num = value
        self.den = 1
```

### Iterator Implementation

Mojo's for-loop works with types that implement `__iter__()` returning an iterator with `__next__()`.

#### Understanding `origin` and `origin_of()`

In Mojo's iterator pattern, `origin` is a compile-time type parameter that tracks memory ownership. It ensures references returned by the iterator remain valid for the lifetime of the collection.

- **`Origin`**: A type representing where a reference "comes from" (its lifetime/ownership source)
- **`origin_of(expr)`**: Extracts the origin from an expression at compile time
- **Purpose**: Prevents iterators from outliving the collection they iterate over

```mojo
# nocompile
# The origin parameter links iterator references to the collection's lifetime
struct MyListIter[T: Movable, origin: Origin]:
    #                        ^^^^^^^^^^^^^^
    # origin captures "this iterator's references come from a MyList"

fn __iter__(ref self) -> MyListIter[T, origin_of(self)]:
    #                                  ^^^^^^^^^^^^^^^
    # origin_of(self) = "references come from THIS specific list instance"
    return MyListIter(self.data, self.size)
```

#### Simple Iterator Example

Here's a minimal iterator for a range-like type:

```mojo
struct CountUp:
    """Counts from 0 to limit-1."""
    var limit: Int

    fn __init__(out self, limit: Int):
        self.limit = limit

    fn __iter__(self) -> CountUpIter:
        return CountUpIter(0, self.limit)

struct CountUpIter:
    var current: Int
    var limit: Int

    fn __init__(out self, start: Int, limit: Int):
        self.current = start
        self.limit = limit

    fn __len__(self) -> Int:
        return self.limit - self.current

    fn __has_next__(self) -> Bool:
        return self.current < self.limit

    fn __next__(mut self) -> Int:
        var result = self.current
        self.current += 1
        return result

fn main():
    for i in CountUp(5):
        print(i)  # Prints: 0, 1, 2, 3, 4
```

**Do:**

```mojo
# nocompile
struct MyList[T: Movable]:
    var data: UnsafePointer[T]
    var size: Int

    fn __iter__(ref self) -> _MyListIter[T, origin_of(self)]:
        return _MyListIter(self.data, self.size)

    fn __len__(self) -> Int:
        return self.size

struct _MyListIter[T: Movable, origin: Origin]:
    var ptr: UnsafePointer[T]
    var remaining: Int

    fn __init__(out self, ptr: UnsafePointer[T], size: Int):
        self.ptr = ptr
        self.remaining = size

    fn __len__(self) -> Int:
        return self.remaining

    fn __next__(mut self) -> ref [origin] T:
        var item_ptr = self.ptr
        self.ptr = self.ptr + 1
        self.remaining -= 1
        return item_ptr[]

# Now works naturally
var list = MyList[Int](...)
for item in list:
    print(item)  # Clean!
```

**Don't:**

```mojo
# nocompile
struct MyList[T: Movable]:
    var data: UnsafePointer[T]
    var size: Int

# Users must manually index
var list = MyList[Int](...)
for i in range(len(list)):
    print(list[i])  # Awkward
```

**Mutable iteration pattern:**

```mojo
# nocompile
struct MutableListIter[T: Movable, origin: Origin, is_mutable: Bool = False]:
    """Iterator that can optionally allow mutation.

    Parameters:
        T: Element type
        origin: Lifetime of the source collection
        is_mutable: True for mutable iteration, False for read-only
    """
    var ptr: UnsafePointer[T]
    var remaining: Int

    fn __init__(out self, ptr: UnsafePointer[T], size: Int):
        self.ptr = ptr
        self.remaining = size

    fn __len__(self) -> Int:
        return self.remaining

    fn __next__(mut self) -> ref [origin] T:
        var item_ptr = self.ptr
        self.ptr = self.ptr + 1
        self.remaining -= 1
        return item_ptr[]

# Usage - mutable iteration modifies elements in place
for mut item in list:
    item += 1  # Modifies in place
```

**Reversed iteration:**

```mojo
struct MyList[T: Movable]:
    # ... other methods ...

    fn __reversed__(ref self) -> _MyListRevIter[T, origin_of(self)]:
        return _MyListRevIter(self.data + self.size - 1, self.size)

struct _MyListRevIter[T: Movable, origin: Origin]:
    var ptr: UnsafePointer[T]
    var remaining: Int

    fn __next__(mut self) -> ref [origin] T:
        var item_ptr = self.ptr
        self.ptr = self.ptr - 1  # Move backwards
        self.remaining -= 1
        return item_ptr[]

# Usage
for item in reversed(list):
    print(item)
```

**Key requirements:**

- `__iter__()` returns iterator object
- Iterator has `__next__()` returning next item (may `raises StopIteration` per the stdlib `Iterator` trait)
- Iterator optionally has `__len__()` returning remaining items and `__has_next__()` returning `Bool`

**IMPORTANT: Mojo Iterator trait (stdlib):**

The stdlib defines a formal `Iterator` trait that uses typed raises with `StopIteration`:

```mojo
# nocompile
# From stdlib std/iter/__init__.mojo:
trait Iterator(ImplicitlyDestructible, Movable):
    comptime Element: Movable

    fn __next__(mut self) raises StopIteration -> Self.Element:
        """Returns the next element, or raises StopIteration when exhausted."""
        ...
```

Simple iterators can also use `__len__()` + `__has_next__()` for length-aware iteration:

```mojo
# Complete iterator example (simple approach)
struct CountdownIter:
    var remaining: Int

    fn __len__(self) -> Int:
        return self.remaining

    fn __has_next__(self) -> Bool:
        return self.remaining > 0

    fn __next__(mut self) -> Int:
        self.remaining -= 1
        return self.remaining + 1
```

Both patterns work with Mojo's `for` loop. The `Iterator` trait with `raises StopIteration` is the canonical stdlib approach.

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Simple data struct with all fields | Use `@fieldwise_init` | [`fn-design.md`](fn-design.md) |
| Fields need validation | Manual `__init__` with checks | - |
| Public API struct | Use section headers for organization | - |
| Generic struct with type parameter | Use `Self.T` syntax | [`memory-ownership.md`](memory-ownership.md) |
| Need iteration support | Implement `__iter__` + iterator with `__next__` | - |
| Numeric type | Implement complete operator set | - |
| Shared behavior across types | Use traits + composition | - |
| Read-only method | Use `self` (immutable default) | - |
| Modifying method | Use `mut self` | - |

---

## Quick Reference

- **@fieldwise_init**: Auto-generates constructor for all fields
- **Underscore prefix**: Convention for private fields (`_field`)
- **Self.T**: Required syntax for generic type parameters in structs
- **Section headers**: Use `# ===---...---===` format for organization
- **Immutable default**: Methods use `self`, only `mut self` when modifying
- **Complete operators**: Implement `__add__`, `__radd__`, `__iadd__` together
- **Iterator protocol**: `__iter__()` + `__next__()` (stdlib `Iterator` trait uses `raises StopIteration`)

---

## Version-Specific Features

### @fieldwise_init Decorator (Current Best Practice)

Use `@fieldwise_init` with explicit trait conformances. The `@value` decorator was removed.

```mojo
# nocompile
@fieldwise_init
struct Config(Copyable, Movable):
    var host: String
    var port: Int
    var timeout: Float64

# Only __init__ is generated - traits provide copy/move
var config = Config("localhost", 8080, 30.0)
var config_copy = config  # Uses Copyable trait
```

**Register-passable types:**

| Version | Syntax | Notes |
|---------|--------|-------|
| **v26.1** | `@register_passable("trivial")` | Still works but deprecated in v26.2 |
| **v26.2+** | `TrivialRegisterType` trait | Preferred replacement |

```mojo
# nocompile
# Preferred: Use TrivialRegisterType trait
@fieldwise_init
struct Point(TrivialRegisterType, Copyable):
    var x: Float64
    var y: Float64

# Legacy (deprecated in v26.2, will be removed):
@register_passable("trivial")
struct LegacyPoint:
    var x: Float64
    var y: Float64

    fn __init__(out self, x: Float64, y: Float64):
        self.x = x
        self.y = y
```

### v26.2+ (Nightly): @align Decorator

Control struct memory alignment for cache and SIMD optimization.

```mojo
# nocompile
from sys import align_of

@align(64)
struct CacheAligned:
    var data: Int

fn main():
    print(align_of[CacheAligned]())  # Prints 64
```

**Common alignment use cases:**

| Use Case | Alignment | Reason |
|----------|-----------|--------|
| Prevent false sharing | 64 bytes | Match CPU cache line size |
| AVX-256 SIMD | 32 bytes | Match vector register width |
| AVX-512 SIMD | 64 bytes | Match vector register width |
| GPU buffers | 256+ bytes | Meet device memory requirements |

```mojo
# nocompile
# Cache line alignment for thread safety
@align(64)
struct CacheLinePadded:
    var counter: Int
    # Each instance on its own cache line

# SIMD alignment for AVX-512
@align(64)
struct SimdBuffer:
    var data: SIMD[DType.float32, 16]
```

**Alignment rules:**

- N must be power of 2 (1, 2, 4, 8, 16, 32, 64...)
- Cannot reduce below natural alignment
- Actual = max(specified, natural)

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `does not implement trait Copyable` | Missing `__copyinit__` method | Add `__copyinit__` or use `@fieldwise_init` decorator |
| `struct has no __init__ method` | Missing initializer | Add `fn __init__(out self, ...)` or use `@fieldwise_init` |
| `abandoned without being explicitly destroyed` | Generic field missing ImplicitlyDestructible | Add `ImplicitlyDestructible` to trait bounds |
| `@value decorator not found` | Using removed decorator | Replace `@value` with `@fieldwise_init` + explicit trait conformance |
| `field 'x' not initialized` | `__init__` doesn't set all fields | Initialize all fields in `__init__` body |
| `cannot use Self.T in method` | Wrong parametric syntax | Use `Self.T` for type access, `self.field` for value access |

---

## Related Patterns

- [`fn-design.md`](fn-design.md) — Function design patterns for methods
- [`memory-ownership.md`](memory-ownership.md) — Lifecycle methods and ownership

---

## References

- [Mojo Structs Documentation](https://docs.modular.com/mojo/manual/structs)
- [Mojo Traits Documentation](https://docs.modular.com/mojo/manual/traits)
- [Mojo Decorators - fieldwise_init](https://docs.modular.com/mojo/manual/decorators/fieldwise-init)
- [Modular Style Guide](https://github.com/modular/modular/blob/main/mojo/stdlib/docs/style-guide.md)
