---
title: Memory Safety Patterns
description: Preventing dangling references, origin tracking, safe pointer types, and allocation strategies in Mojo
impact: CRITICAL
category: memory
tags: [safety, dangling-references, origin, pointers, allocation]
error_patterns:
  - "dangling reference"
  - "origin mismatch"
  - "lifetime of .* does not outlive"
  - "reference to local variable"
  - "use after free"
scenarios:
  - "Return reference from function safely"
  - "Store reference in struct"
  - "Fix dangling reference error"
  - "Choose between pointer types"
consolidates:
  - memory-no-dangling-refs.md
  - memory-pointee-lifecycle.md
  - memory-origin-tracking.md
  - memory-origin-casting.md
  - memory-safe-pointers.md
---

# Memory Safety Patterns

**Category:** memory | **Impact:** CRITICAL

This pattern covers Mojo's core memory safety mechanisms: preventing dangling references, tracking pointer origins, choosing the right pointer types, and allocation strategies. Mastering these prevents undefined behavior, use-after-free bugs, and data corruption.

> **Note:** For Span, MaybeUninitialized, and collection destructor patterns, see [`memory-collections.md`](memory-collections.md).

---

## Core Concepts

### Dangling References Prevention

Returning a reference to a local variable creates a dangling reference when the function returns. The referenced memory is deallocated, leading to undefined behavior.

**Anti-pattern (dangling reference):**

```mojo
# nocompile
fn get_first(items: List[Int]) -> ref [_] Int:
    var local = items[0]  # local variable
    return local  # ERROR: local is destroyed when function returns

fn create_and_return() -> ref [_] String:
    var s = String("hello")
    return s  # ERROR: s is destroyed, reference dangles
```

**Pattern (return by value):**

```mojo
fn get_first_value(items: List[Int]) -> Int:
    return items[0]  # Return by value - safe copy

fn create_string() -> String:
    var s = String("hello")
    return s^  # Transfer ownership out - safe
```

**Pattern (reference tied to input's lifetime):**

```mojo
fn get_first_ref(items: List[Int]) -> ref [items] Int:
    # Reference lifetime is tied to 'items' parameter
    # Safe because items outlives the returned reference
    return items[0]

fn get_longest(a: String, b: String) -> ref [a, b] String:
    # Reference lifetime tied to both inputs
    if len(a) > len(b):
        return a
    return b
```

**Rules for safe references:**

- Only return references to data that outlives the function call
- Use lifetime parameters `ref [param]` to express dependencies
- When in doubt, return by value or transfer ownership

### Origin Tracking for UnsafePointer

The origin system tracks where pointers come from, enabling the compiler to catch dangling pointer bugs. Always specify origins explicitly rather than relying on defaults.

**Anti-pattern (implicit origins):**

```mojo
# nocompile
struct Container:
    # BAD: No origin specified - uses default which may not be correct
    var data: UnsafePointer[Int]

    fn get_ptr(self) -> UnsafePointer[Int]:
        # BAD: Returning pointer without origin tracking
        return self.data
```

**Pattern (explicit origin tracking):**

```mojo
# Origin parameter syntax changed in nightly v26.2+

struct Container:
    # GOOD: Explicit origin specification
    var data: UnsafePointer[mut=True, type=Int, origin=MutAnyOrigin]

    # GOOD: Return pointer with tracked origin
    fn unsafe_ptr[
        mut: Bool,
        origin: Origin[mut=mut]
    ](ref [origin] self) -> UnsafePointer[Int, origin]:
        return self.data.mut_cast[mut]().unsafe_origin_cast[origin]()
```

**Origin casting patterns from stdlib:**

```mojo
# nocompile
# From ArcPointer.unsafe_ptr():
fn unsafe_ptr[
    mut: Bool,
    origin: Origin[mut=mut]
](ref [origin] self) -> UnsafePointer[Self.T, origin]:
    return (
        UnsafePointer(to=self._inner[].payload)
        .mut_cast[mut]()
        .unsafe_origin_cast[origin]()
    )

# Mutability casting
ptr.mut_cast[True]()   # Make mutable
ptr.mut_cast[False]()  # Make immutable

# Origin casting (use sparingly)
ptr.unsafe_origin_cast[new_origin]()
```

### Origin Types and ASAP Destruction

| Origin Type | Use Case | ASAP Destruction |
|-------------|----------|------------------|
| `MutExternalOrigin` | Owned heap allocations, FFI | Yes |
| `ImmutExternalOrigin` | Read-only external memory | Yes |
| `MutAnyOrigin` | Type-erased callbacks, opaque pointers | **No** |
| `ImmutAnyOrigin` | Static constants, type-erased reads | **No** |
| `origin_of(self)` | Derived pointers tied to object lifetime | Yes |

**Warning:** Using `MutAnyOrigin` disables Mojo's "As Soon As Possible" (ASAP) destruction optimization.

---

## Safe Pointer Type Hierarchy

Mojo provides four pointer types with different safety guarantees. Use the safest option that meets your needs.

| Type | Ownership | Use Case |
|------|-----------|----------|
| `Pointer` | Non-owning | Safe reference to single initialized value |
| `OwnedPointer` | Exclusive | Single owner, automatic cleanup |
| `ArcPointer` | Shared | Multiple owners, reference counted |
| `UnsafePointer` | Manual | Low-level, uninitialized memory |

**Pattern (using OwnedPointer for exclusive ownership):**

```mojo
from memory import OwnedPointer

struct Container:
    var data: OwnedPointer[Int]

    fn __init__(out self, value: Int):
        self.data = OwnedPointer(value)  # Automatic allocation

    # __del__ not needed - OwnedPointer cleans up automatically
    # move ctor handled by OwnedPointer's move semantics

    fn get(self) -> Int:
        return self.data[]
```

**Pattern (using ArcPointer for shared ownership):**

```mojo
from memory import ArcPointer

fn share_data():
    var shared = ArcPointer(42)
    var copy1 = shared  # Increments reference count
    var copy2 = shared  # Increments again
    print(shared[])  # 42
    # Data freed when all copies are destroyed
```

### When to Use Each Pointer Type

| Use Case | Pointer Type |
|----------|--------------|
| Reference to single initialized value | `Pointer` |
| Exclusive ownership with automatic cleanup | `OwnedPointer` |
| Shared ownership (thread-safe) | `ArcPointer` |
| High-performance collections with SIMD | `UnsafePointer` |
| Interfacing with C/C++ libraries | `UnsafePointer` |
| Managing uninitialized memory explicitly | `UnsafePointer` |

---

## UnsafePointer Lifecycle Management

### Pointee Lifecycle

When using `UnsafePointer`, you must manually manage the lifecycle of values at pointer locations. Memory from `alloc()` is uninitialized; you must initialize before reading and destroy before freeing.

**Anti-pattern (memory leak and uninitialized read):**

```mojo
# nocompile
fn bad_container() -> UnsafePointer[String]:
    var ptr = alloc[String](10)
    ptr[0] = String("hello")  # Writing to uninitialized memory!
    ptr.free()  # Leaks the String - never destroyed
    return ptr  # Returns freed pointer
```

**Pattern (proper lifecycle management):**

```mojo
fn good_container():
    var ptr = alloc[String](10)

    # Initialize with init_pointee_move (transfers ownership)
    (ptr + 0).init_pointee_move(String("hello"))

    # Or initialize with init_pointee_copy (copies value)
    var source = String("world")
    (ptr + 1).init_pointee_copy(source)

    # Use the values
    print(ptr[0], ptr[1])

    # Destroy before freeing
    (ptr + 0).destroy_pointee()
    (ptr + 1).destroy_pointee()

    # Now safe to free
    ptr.free()
```

**Initialization patterns:**

```mojo
# 1. Move initialization (transfers ownership, original consumed)
fn init_pointee_move[T: Movable](self: UnsafePointer[T], var value: T):
    __get_address_as_uninit_lvalue(self.address) = value^

# 2. Copy initialization (copies value, original unchanged)
fn init_pointee_copy[T: Copyable](self: UnsafePointer[T], value: T):
    __get_address_as_uninit_lvalue(self.address) = value.copy()

# 3. Move from another pointer location
fn init_pointee_move_from[T: Movable](self: UnsafePointer[T], src: UnsafePointer[T]):
    __get_address_as_uninit_lvalue(self.address) = __get_address_as_owned_value(src.address)
```

**Take vs Destroy patterns:**

```mojo
# nocompile
# take_pointee: Move value out, leaving memory uninitialized
fn pop_element[T: Movable](ptr: UnsafePointer[T], index: Int) -> T:
    return (ptr + index).take_pointee()

# destroy_pointee: Destroy value in-place (more efficient, no move)
fn clear_elements[T: ImplicitlyDestructible](ptr: UnsafePointer[T], count: Int):
    for i in range(count):
        (ptr + i).destroy_pointee()
```

### UnsafePointer Struct Field Syntax

When using `UnsafePointer` as a struct field, you MUST specify the full type with named parameters:

```mojo
# nocompile
from memory import UnsafePointer

# INCORRECT - will not compile:
struct BadContainer:
    var data: UnsafePointer[Float32]  # ERROR: 'UnsafePointer' failed to infer parameter 'mut'

# CORRECT - full type specification:
struct GoodContainer:
    var data: UnsafePointer[mut=True, type=Float32, origin=MutAnyOrigin]
```

**Best Practice (type aliases):**

```mojo
# Define type aliases for common pointer types
comptime Float32Ptr = UnsafePointer[mut=True, type=Float32, origin=MutAnyOrigin]
comptime Int64Ptr = UnsafePointer[mut=True, type=Int64, origin=MutAnyOrigin]

struct Container:
    var data: Float32Ptr  # Clean and readable
```

---

## Stack vs Heap Allocation

Use this decision tree to choose the right allocation strategy:

```
Is the size known at compile time?
├─ No → Use heap allocation (alloc/List)
└─ Yes → Is it small (<= ~1KB)?
         ├─ Yes → Is it inside a recursive function?
         │        ├─ Yes → Consider heap allocation
         │        └─ No → Use stack allocation (stack_allocation/InlineArray)
         └─ No → Use heap allocation
```

### Stack Allocation with `stack_allocation`

```mojo
from memory import stack_allocation

# Small, compile-time known size: use stack
fn process_small_data():
    var buffer = stack_allocation[256, Float32]()  # 1 KB - safe
    for i in range(256):
        buffer[i] = Float32(i)
    # No free() needed - automatically cleaned up

# With custom alignment (for SIMD)
fn simd_process():
    var buffer = stack_allocation[16, Float32, alignment=64]()  # AVX-512 aligned
    var vec = buffer.load[width=16]()
```

### Heap Allocation with `alloc()`

```mojo
from memory import alloc  # v26.1+

# Large or dynamic size: use heap
fn process_large_data(size: Int):
    var buffer = alloc[Float32](size)  # v26.1+ syntax
    for i in range(size):
        buffer[i] = Float32(i)
    buffer.free()  # Must explicitly free heap memory

# Fixed size but large: heap is safer
fn process_large_fixed_data():
    var buffer = alloc[Float32](1_000_000)  # 4 MB - heap
    # ... use buffer
    buffer.free()
```

### GPU Shared Memory

```mojo
from gpu.memory import AddressSpace

fn reduction_kernel[dtype: DType](...):
    comptime BLOCK_SIZE = 256
    var shared_data = stack_allocation[
        BLOCK_SIZE,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,  # GPU shared memory
    ]()
    # ...
```

**Size guidelines:**

| Context | Recommended Max Stack |
|---------|----------------------|
| Normal function | ~1 KB |
| Leaf function | ~4 KB |
| GPU shared memory | 48-164 KB |
| Recursive function | Minimal or none |

**Common pitfalls:**

- Large arrays on stack in recursive functions → stack overflow
- Forgetting `free()` on heap allocations → memory leak
- Returning pointers to stack memory → dangling pointer

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Return data from function | Return by value or transfer ownership (`^`) | Dangling refs |
| Non-owning view of array | Use `Span[T, origin]` | memory-collections.md |
| Single exclusive owner | Use `OwnedPointer` | Safe pointers |
| Shared ownership | Use `ArcPointer` | memory-refcounting.md |
| Low-level memory management | Use `UnsafePointer` with proper lifecycle | Pointee lifecycle |
| Deferred initialization | Use `UnsafeMaybeUninit` | memory-collections.md |
| Track pointer lifetime | Use explicit origin parameters | Origin tracking |
| Collection with owned elements | Implement proper `__del__` | memory-collections.md |

---

## Quick Reference

- **Dangling reference**: Reference to memory that has been deallocated
- **Origin**: Compile-time tracking of where pointers come from
- **ASAP destruction**: Mojo destroys values as soon as possible (disabled by `MutAnyOrigin`)
- **`init_pointee_move`**: Initialize uninitialized memory by moving value in
- **`destroy_pointee`**: Destroy value in-place without moving
- **`take_pointee`**: Move value out, leaving memory uninitialized

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `dangling reference` | Reference outlives its source | Use `ref` returns or `Span` with proper origin |
| `origin mismatch` | Different origins in same expression | Ensure consistent origin annotations |
| `cannot borrow as mutable` | Aliasing rules violated | Restructure to avoid overlapping borrows |
| `use after free` | Memory freed while reference active | Track origins, use Span for safe slicing |
| `invalid pointer` | NULL or uninitialized pointer | Always check before dereference |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **Heap allocation** | `from memory import alloc; alloc[T](n)` | Same syntax both versions |
| **Type aliases** | `comptime` | `alias` is deprecated; use `comptime` |
| **Origin types** | `MutAnyOrigin, ImmutAnyOrigin` | Prelude symbols — no import needed |
| **OwnedPointer** | Available | Stable |
| **ArcPointer** | Available | Stable |
| `MutAnyOrigin` | `MutAnyOrigin` (v26.1+) | `MutableAnyOrigin` was renamed |
| `origin_of()` | `origin_of()` (v26.1+) | `__origin_of()` was renamed |

**Example (v26.1+):**

```mojo
from memory import UnsafePointer, alloc

# alias is deprecated; use comptime
comptime Float32Ptr = UnsafePointer[mut=True, type=Float32, origin=MutAnyOrigin]

fn allocate_buffer(size: Int) -> Float32Ptr:
    var ptr = alloc[Float32](size)
    return ptr
```

---

### Mutation During Iteration

**Never modify a collection while iterating over it.** This causes undefined behavior.

```mojo
# ❌ WRONG: Modifying list during iteration
# nocompile
for i in range(len(items)):
    if should_remove(items[i]):
        items.pop(i)  # Shifts indices — skips elements or crashes
```

```mojo
# nocompile
# ✅ CORRECT: Build new list or iterate in reverse
# Option 1: Filter into new list
var kept = List[Int]()
for item in items:
    if not should_remove(item[]):
        kept.append(item[])

# Option 2: Remove in reverse (indices stay valid)
for i in range(len(items) - 1, -1, -1):
    if should_remove(items[i]):
        _ = items.pop(i)
```

---

## Related Patterns

- [`memory-collections.md`](memory-collections.md) — Span, MaybeUninitialized, collection destructors
- [`memory-ownership.md`](memory-ownership.md) — Ownership transfer, borrowing, lifecycle methods
- [`memory-refcounting.md`](memory-refcounting.md) — Reference counting implementation

---

## References

- [Mojo Ownership Documentation](https://docs.modular.com/mojo/manual/values/ownership)
- [Mojo Pointers Documentation](https://docs.modular.com/mojo/manual/pointers/)
- [Mojo Memory Module](https://github.com/modular/modular/blob/main/mojo/stdlib/std/memory/)
