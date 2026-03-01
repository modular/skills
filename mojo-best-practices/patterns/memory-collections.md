---
title: Memory Collections Patterns
description: Span usage, MaybeUninitialized, collection destructors, and bulk memory operations in Mojo
impact: CRITICAL
category: memory
tags: [span, uninitialized, collection, destructor, bulk-operations]
error_patterns:
  - "use after free"
  - "double free"
  - "memory leak"
  - "uninitialized memory"
  - "buffer overflow"
scenarios:
  - "Use Span for non-owning view"
  - "Handle uninitialized memory"
  - "Implement collection destructor"
  - "Bulk copy or fill memory"
consolidates:
  - memory-maybe-uninitialized.md
  - memory-span-non-owning.md
  - memory-span-operations.md
  - memory-collection-destructor.md
---

# Memory Collections Patterns

**Category:** memory | **Impact:** CRITICAL

This pattern covers `Span` for zero-copy views over contiguous memory, `UnsafeMaybeUninit` for deferred initialization, and proper collection destructor implementation. These are essential for building high-performance containers.

> **Note:** For core memory safety (dangling references, origin tracking, pointer types), see [`memory-safety.md`](memory-safety.md).

---

> **Warning:** `List[T]` requires `T: Copyable`, not just `Movable`. Move-only types cannot be stored in `List`. For move-only types, use `UnsafePointer`-based storage with `UnsafeMaybeUninit` (see below).

## UnsafeMaybeUninit Patterns

`UnsafeMaybeUninit` provides a memory location that may or may not be initialized, enabling deferred initialization patterns essential for implementing containers.

### The Lifecycle: init_from() -> unsafe_assume_init_ref() -> unsafe_assume_init_destroy()

```mojo
# nocompile
from memory import UnsafeMaybeUninit

fn lifecycle_example():
    # 1. Create uninitialized storage (no constructor called)
    var maybe = UnsafeMaybeUninit[String]()

    # 2. Initialize with init_from() - takes ownership of value
    maybe.init_from(String("hello"))

    # 3. Access the initialized value with unsafe_assume_init_ref()
    print(maybe.unsafe_assume_init_ref())  # "hello"

    # 4. CRITICAL: Destroy before going out of scope
    maybe.unsafe_assume_init_destroy()

    # Note: The destructor is a no-op - you MUST call unsafe_assume_init_destroy()
```

**Anti-pattern (reading before writing):**

```mojo
# nocompile
fn undefined_behavior():
    var maybe = UnsafeMaybeUninit[Int]()

    # UNDEFINED BEHAVIOR: Reading uninitialized memory!
    var value = maybe.unsafe_assume_init_ref()  # Garbage data!
```

**Anti-pattern (missing destroy):**

```mojo
# nocompile
fn memory_leak():
    var maybe = UnsafeMaybeUninit[String]()
    maybe.init_from(String("hello"))

    # MEMORY LEAK: The String's destructor is never called!
    # UnsafeMaybeUninit's __del__ is a no-op by design.
```

**Key API methods:**

| Method | Precondition | Postcondition |
|--------|--------------|---------------|
| `init_from(value^)` | self uninitialized | self initialized |
| `unsafe_assume_init_ref()` | self initialized | Returns reference |
| `unsafe_assume_init_destroy()` | self initialized | self uninitialized |
| `unsafe_assume_init_take()` | self initialized | Returns owned value, self uninitialized |
| `unsafe_ptr()` | none | Returns pointer (always safe to call) |

**Container implementation pattern:**

```mojo
# nocompile
struct Container[T: Copyable & Movable & ImplicitlyDestructible, size: Int]:
    var _storage: InlineArray[UnsafeMaybeUninit[Self.T], Self.size]
    var _count: Int  # Track how many elements are initialized

    fn __init__(out self):
        self._storage = InlineArray[UnsafeMaybeUninit[Self.T], Self.size](
            uninitialized=True
        )
        self._count = 0

    fn add(mut self, var value: Self.T):
        debug_assert(self._count < Self.size, "Container full")
        self._storage[self._count].init_from(value^)
        self._count += 1

    fn __del__(deinit self):
        # CRITICAL: Only destroy initialized elements
        for i in range(self._count):
            self._storage[i].unsafe_assume_init_destroy()
```

---

## Span Usage Patterns

### Non-Owning Contiguous Data Views

`Span[T, origin]` provides a non-owning view over contiguous memory. It tracks lifetimes via the origin system and enables zero-copy operations.

**Anti-pattern (copying data unnecessarily):**

```mojo
fn process_data(data: List[Float32]) -> Float32:
    # Creates a copy of the list
    var total: Float32 = 0.0
    for i in range(len(data)):
        total += data[i]
    return total
```

**Pattern (non-owning view):**

```mojo
# nocompile
fn process_data(data: Span[Float32, _]) -> Float32:
    # Span provides read-only view without copying
    var total: Float32 = 0.0
    for i in range(len(data)):
        total += data[i]
    return total

# Usage - automatic conversion from List
var my_list = List[Float32]()
my_list.append(1.0)
my_list.append(2.0)
my_list.append(3.0)
var result = process_data(Span(my_list))
```

### Span Bulk Operations

Use `fill()` and `copy_from()` for bulk operations instead of element-by-element loops.

**Don't (manual loop):**
```mojo
# nocompile
fn initialize_buffer(data: Span[Float32, _]):
    for i in range(len(data)):
        data[i] = 0.0
```

**Do (bulk operation):**
```mojo
# nocompile
fn initialize_buffer(data: Span[Float32, _]):
    data.fill(0.0)  # Clear intent, single operation

fn copy_data(dest: Span[Int, _], src: Span[Int, _]):
    dest.copy_from(src)  # Validates equal lengths, element-wise copy
```

**fill() for initialization:**
```mojo
var buffer = List[Int](unsafe_uninit_length=100)
var span = Span(buffer)
span.fill(0)            # All 100 elements set to 0
span[10:20].fill(42)    # Elements 10-19 set to 42
```

**copy_from() for element-wise copying:**
```mojo
var source = [1, 2, 3, 4, 5]
var dest = [0, 0, 0, 0, 0]

Span(dest).copy_from(Span(source))  # dest = [1, 2, 3, 4, 5]

# Partial copy - sizes must match!
Span(dest)[1:4].copy_from(Span(source)[0:3])  # dest = [0, 1, 2, 3, 0]
```

**Safety:** `copy_from()` asserts spans have equal length at runtime.

### Span Slicing and Subspan

```mojo
# nocompile
# Create from List
var span = Span(list)

# Create from pointer and length (unsafe)
var span = Span(ptr=data_ptr, length=count)

# Slice operations (zero-copy)
var sub = span[2:5]  # Subspan via slice
var sub = span.unsafe_subspan(offset=2, length=3)

# Fill mutable spans
span.fill(0.0)  # Fill all elements

# Copy between spans
dest_span.copy_from(src_span)
```

**Subspan pattern:**

```mojo
# nocompile
fn process_chunks(data: Span[Float32, _], chunk_size: Int):
    var offset = 0
    while offset + chunk_size <= len(data):
        var chunk = data.unsafe_subspan(offset=offset, length=chunk_size)
        process_chunk(chunk)
        offset += chunk_size
    # Handle remainder
    if offset < len(data):
        var remainder = data[offset:]
        process_chunk(remainder)
```

---

## Advanced Span Patterns

### Span Mutability

Span mutability is controlled by the `mut` parameter and the origin it's created from.

**Creating immutable Span from List:**

```mojo
var list: List[Int] = [1, 2, 3, 4, 5]

# Immutable span - read-only access
var immutable_span = Span(list)  # Defaults to immutable
print(immutable_span[0])  # OK: reading
# immutable_span[0] = 10  # ERROR: cannot mutate through immutable span
```

**Creating mutable Span:**

```mojo
var list: List[Int] = [1, 2, 3, 4, 5]

# Mutable span - requires mutable reference to source
var mutable_span = Span(list)  # Mutability is inferred from the reference
mutable_span[0] = 100  # OK: mutation allowed
mutable_span.fill(0)   # OK: bulk mutation
```

**Function parameter patterns:**

```mojo
# nocompile
# Read-only processing - immutable span
fn sum_values(data: Span[Int, _]) -> Int:
    var total = 0
    for i in range(len(data)):
        total += data[i]
    return total

# In-place modification - mutable span
fn double_values(data: Span[mut=True, Int, _]):
    for i in range(len(data)):
        data[i] *= 2

# Usage
var numbers: List[Int] = [1, 2, 3]
var total = sum_values(Span(numbers))       # Immutable: OK
double_values(Span(numbers))               # Mutable: inferred from context
```

**Mutability rules:**

| Source | Span Creation | Mutability |
|--------|---------------|------------|
| `Span(list)` | Immutable reference | Read-only |
| `Span(list)` (with `mut` binding) | Mutable reference | Read-write |
| `Span(ptr=ptr, length=n)` | From pointer | Inherits from pointer's mutability |

### Span Iteration

**Direct iteration with `for item in span`:**

```mojo
var list: List[Int] = [10, 20, 30, 40, 50]
var span = Span(list)

# Iterate over elements (yields values directly)
for item in span:
    print(item)

# With mutable span for modification
var mutable_span = Span(list)
for i in range(len(mutable_span)):
    mutable_span[i] *= 2  # Double each element in-place
```

**Index-based iteration with `range(len(span))`:**

```mojo
# nocompile
var span = Span(list)

# Traditional index loop - useful when you need the index
for i in range(len(span)):
    print("Index", i, "value", span[i])

# Useful for algorithms needing index access
for i in range(len(span)):
    if i > 0 and span[i] < span[i - 1]:
        print("Decreasing at index", i)
```

**Chunk iteration pattern:**

```mojo
# nocompile
fn process_in_chunks[T: Copyable](data: Span[T, _], chunk_size: Int):
    var i = 0
    while i < len(data):
        var end = min(i + chunk_size, len(data))
        var chunk = data[i:end]
        # Process chunk...
        for item in chunk:
            print(item)
        i += chunk_size
```

### Bounds Checking

Span provides bounds-checked access by default, with unsafe alternatives for performance-critical code.

**Default bounds-checked access with `[]` operator:**

```mojo
var list: List[Int] = [1, 2, 3]
var span = Span(list)

# Safe access - bounds checked
var value = span[0]   # OK: returns 1
var last = span[2]    # OK: returns 3

# Out-of-bounds access behavior:
# var oob = span[10]  # Runtime error: index out of bounds
```

**Unsafe unchecked access with `unsafe_get`:**

```mojo
# nocompile
# When you've verified bounds externally, use unsafe_get for performance
fn sum_verified_range[T: Copyable](span: Span[T, _], start: Int, end: Int) -> T:
    debug_assert(start >= 0 and end <= len(span), "Invalid range")
    var total: T = T()
    for i in range(start, end):
        # Skip bounds check - we verified above
        total += span.unsafe_get(i)
    return total
```

**Bounds checking comparison:**

| Method | Bounds Check | Use Case |
|--------|--------------|----------|
| `span[i]` | Yes (runtime) | Default, safe access |
| `span.unsafe_get(i)` | No | Performance-critical inner loops |
| `span[start:end]` | Yes | Slicing with bounds validation |
| `span.unsafe_subspan(offset, length)` | No | Pre-validated subspan extraction |

### Empty Span Handling

Properly handling empty spans prevents runtime errors and undefined behavior.

**Checking for empty:**

```mojo
# nocompile
var span = Span(some_list)

# Check using len()
if len(span) == 0:
    print("Span is empty")
    return

# Or use bool conversion (empty = false)
if not span:
    print("Span is empty")
    return

# Proceed with non-empty span operations
var first = span[0]  # Safe: we verified non-empty
```

**Safe patterns for potentially empty spans:**

```mojo
# nocompile
# Pattern 1: Early return
fn process_data(data: Span[Float32, _]) -> Float32:
    if len(data) == 0:
        return 0.0  # Handle empty case explicitly

    var total: Float32 = 0.0
    for i in range(len(data)):
        total += data[i]
    return total / Float32(len(data))

# Pattern 2: Optional result
fn find_max(data: Span[Int, _]) -> Optional[Int]:
    if len(data) == 0:
        return None

    var max_val = data[0]
    for i in range(1, len(data)):
        if data[i] > max_val:
            max_val = data[i]
    return max_val

# Pattern 3: Default value
fn first_or_default[T: Copyable](data: Span[T, _], default: T) -> T:
    if len(data) == 0:
        return default
    return data[0]
```

**Iteration over empty spans is safe:**

```mojo
var empty_list: List[Int] = []
var empty_span = Span(empty_list)

# These are all safe (just do nothing):
for item in empty_span:
    print(item)  # Never executes

for i in range(len(empty_span)):
    print(empty_span[i])  # Never executes

empty_span.fill(42)  # No-op on empty span
```

### Thread Safety

**Read-only sharing is safe:**

```mojo
# SAFE: Multiple threads reading from immutable span
var data: List[Int] = [1, 2, 3, 4, 5]
var span = Span(data)  # Immutable span

# Thread 1: var a = span[0]  # OK
# Thread 2: var b = span[1]  # OK
# All threads reading different or same elements - safe
```

**Partitioned mutable access is safe:**

```mojo
# SAFE: Each thread writes to non-overlapping regions
fn parallel_process(data: Span[mut=True, Int, _], num_threads: Int):
    var chunk_size = len(data) // num_threads

    # Each thread gets exclusive subspan
    for thread_id in range(num_threads):
        var start = thread_id * chunk_size
        var end = start + chunk_size if thread_id < num_threads - 1 else len(data)
        var thread_span = data[start:end]
        # Thread processes only its subspan - no overlap, safe
```

**Thread safety guidelines:**

| Access Pattern | Thread Safety |
|----------------|---------------|
| Multiple readers, immutable span | Safe |
| Single writer, mutable span | Safe |
| Multiple writers, same elements | Unsafe (data race) |
| Multiple writers, disjoint elements | Safe (with care) |
| Reader + writer, same span | Unsafe (data race) |

---

## Collection Destructor Patterns

Collections that own heap-allocated elements must properly destroy those elements before freeing the underlying memory.

**Anti-pattern (freeing memory without destroying elements):**

```mojo
# nocompile
struct BadList[T: Copyable]:
    var _data: UnsafePointer[T, MutExternalOrigin]
    var _len: Int

    fn __del__(deinit self):
        # WRONG: Freeing without destroying elements
        # If T has a destructor (e.g., String, List), those destructors
        # never run, causing memory leaks
        self._data.free()
```

**Pattern (proper element destruction):**

```mojo
# nocompile
# NOTE: The stdlib uses internal APIs (_constrained_conforms_to, downcast) for
# runtime trait checking in its List/InlineArray destructors. User code should
# NOT use these internals. Instead, use compile-time trait bounds in your
# function/struct signatures to ensure elements are destructible.

struct GoodList[T: Copyable & Destructible]:
    """Use trait bounds to guarantee T is destructible at compile time."""
    var _data: UnsafePointer[T, MutExternalOrigin]
    var _len: Int
    var capacity: Int

    fn __del__(deinit self):
        """Destroy all elements in the list and free its memory."""
        # Because T: Destructible is enforced by the trait bound,
        # we can safely call destroy_pointee() on each element.
        for i in range(self._len):  # Only destroy initialized elements
            (self._data + i).destroy_pointee()

        self._data.free()
```

### The `__del__is_trivial` Optimization

Types with trivial destructors (Int, Float64, Bool, etc.) don't need individual destruction calls:

```mojo
# nocompile
# From InlineArray - propagate trivial flag from element type
comptime __del__is_trivial: Bool = downcast[
    Self.ElementType, ImplicitlyDestructible
].__del__is_trivial

fn __del__(deinit self):
    @parameter
    if not TDestructible.__del__is_trivial:
        @parameter
        for idx in range(Self.size):
            var ptr = self.unsafe_ptr() + idx
            ptr.bitcast[TDestructible]().destroy_pointee()
```

### Move Semantics During Destruction

When consuming another container, mark the source as empty after moving elements:

```mojo
# nocompile
fn extend(mut self, var other: List[Self.T, ...]):
    """Extends this list by consuming the elements of other."""
    var other_len = len(other)
    self.reserve(len(self) + other_len)

    var dest_ptr = self._data + self._len
    var src_ptr = other.unsafe_ptr()

    @parameter
    if Self.T.__moveinit__is_trivial:
        memcpy(dest=dest_ptr, src=src_ptr, count=other_len)
    else:
        for _ in range(other_len):
            dest_ptr.init_pointee_move_from(src_ptr)
            src_ptr += 1
            dest_ptr += 1

    self._len += other_len
    # CRITICAL: Mark other as empty so its destructor doesn't
    # destroy the elements we just moved
    other._len = 0
```

### Circular Buffer Destruction (Deque Pattern)

```mojo
# nocompile
fn __del__(deinit self):
    """Destroys all elements in the deque and free its memory."""
    for i in range(len(self)):
        # Calculate physical index from logical position
        var offset = self._physical_index(self._head + i)
        (self._data + offset).destroy_pointee()
    self._data.free()
```

### Linked Structure Destruction

```mojo
# nocompile
fn __del__(deinit self):
    """Clean up the list by freeing all nodes."""
    var curr = self._head
    while curr:
        var next = curr[].next  # Save next pointer before destroying
        curr.destroy_pointee()  # Destroy the node (and its value)
        curr.free()             # Free the node's memory
        curr = next
```

---

## Decision Guide

| Scenario | Approach |
|----------|----------|
| Non-owning view of array | Use `Span[T, origin]` |
| Deferred initialization | Use `UnsafeMaybeUninit` |
| Collection with owned elements | Implement proper `__del__` |
| Bulk fill/copy | Use `span.fill()` and `span.copy_from()` |
| Thread-safe read access | Use immutable Span |

---

## Key Rules for Collection Destructors

1. Destroy elements BEFORE freeing container memory
2. Only destroy initialized elements (track with `_len`, not `capacity`)
3. Use `__del__is_trivial` optimization to skip loops for trivial types
4. After moving elements, mark source container as empty
5. Use `destroy_pointee()` for in-place destruction without moving
6. Handle circular buffers by computing physical indices

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `use after free` | Memory freed while reference active | Track origins, use Span for safe slicing |
| `double free` | Element destroyed twice | Set length to 0 after moving elements |
| `memory leak` | Element destructor not called | Call `destroy_pointee()` before freeing |
| `buffer overflow` | Index out of bounds | Use bounds-checked accessors or Span |
| `uninitialized read` | Reading UnsafeMaybeUninit before init | Always init_from() before unsafe_assume_init_ref() |
| `Dict[key]` raises at runtime | Key not found in Dict | `Dict.__getitem__` raises on missing key. Functions using `dict[key]` need `raises`, or guard with `if key in dict:` first |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status |
|---------|--------|
| Span API (`fill()`, `copy_from()`, slicing) | Stable both versions |
| `UnsafeMaybeUninit` | Stable both versions |
| Collection destructor patterns | Stable both versions |
| `__del__is_trivial` optimization | Stable both versions |

---

## Related Patterns

- [`memory-safety.md`](memory-safety.md) — Core memory safety, origins, pointer types
- [`memory-ownership.md`](memory-ownership.md) — Ownership transfer, borrowing

---

## References

- [Mojo Memory Module](https://github.com/modular/modular/blob/main/mojo/stdlib/std/memory/)
- [Mojo Collections](https://github.com/modular/modular/blob/main/mojo/stdlib/std/collections/)
