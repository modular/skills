---
title: Reference Counting Implementation Patterns
description: Thread-safe reference counting with atomic operations and correct memory ordering for shared ownership in Mojo
impact: HIGH
category: memory
tags: [reference-counting, atomic, thread-safety, arc, shared-ownership]
error_patterns:
  - "double free"
  - "use after free"
  - "memory leak"
  - "data race"
  - "reference count underflow"
  - "Arc"
scenarios:
  - "Implement shared ownership"
  - "Create thread-safe reference counted type"
  - "Fix double-free bug"
  - "Avoid memory leaks with shared data"
consolidates:
  - memory-refcount-impl.md
  - memory-atomic-refcounting.md
---

# Reference Counting Implementation Patterns

**Category:** memory | **Impact:** HIGH

This pattern covers implementing thread-safe reference counting in Mojo: correct atomic memory ordering (MONOTONIC for increment, RELEASE for decrement, ACQUIRE fence when hitting zero), ArcPointer patterns, weak references, and cycle avoidance strategies. Incorrect memory ordering causes data races, use-after-free, double-free, and memory corruption.

---

## Core Concepts

### Atomic Memory Ordering for Reference Counting

The Mojo stdlib's `ArcPointer` demonstrates the proven pattern for thread-safe reference counting. The key insight is using the minimum necessary memory ordering for each operation.

**Memory ordering summary:**

| Operation | Ordering | Reason |
|-----------|----------|--------|
| Increment (copy) | MONOTONIC | Already have valid ref; just need atomicity |
| Decrement | RELEASE | Our writes visible before count decreases |
| Before destroy | ACQUIRE fence | See all writes from all threads |
| Load count | MONOTONIC | Read-only observation, no synchronization |

### Why RELAXED is Dangerous

**Anti-pattern (wrong memory ordering):**

```mojo
# nocompile
from os.atomic import Atomic, Consistency

struct BrokenRefCount[T: Movable & ImplicitlyDestructible]:
    var refcount: Atomic[DType.uint64]
    var payload: T

    fn add_ref(mut self):
        # WRONG: ACQUIRE is unnecessarily strong for increment
        # Wastes CPU cycles on memory barriers
        _ = self.refcount.fetch_add[ordering=Consistency.ACQUIRE](1)

    fn drop_ref(mut self) -> Bool:
        # CRITICAL BUG: RELAXED allows reordering!
        # Another thread may see refcount=0 before seeing our writes to payload
        # This causes use-after-free when the other thread destroys the data
        return self.refcount.fetch_sub[ordering=Consistency.RELAXED](1) == 1
        # Missing ACQUIRE fence - destruction may see stale data!
```

**Why this fails:** Thread A writes to payload, then decrements refcount with RELAXED. Thread B sees refcount hit 0 but hasn't yet seen Thread A's writes to payload (due to reordering). Thread B destroys payload while Thread A's writes are still in flight. Result: corrupted data or use-after-free.

### Correct Memory Ordering

**Pattern (proper ordering from stdlib ArcPointer):**

```mojo
# nocompile
from os.atomic import Atomic, Consistency, fence
from memory import UnsafePointer, alloc

struct _RefCountedInner[T: Movable & ImplicitlyDestructible]:
    """Internal storage combining refcount and payload."""
    var refcount: Atomic[DType.uint64]
    var payload: T

    fn __init__(out self, var value: T):
        """Create with refcount of 1."""
        self.refcount = Atomic(UInt64(1))
        self.payload = value^

    fn add_ref(mut self):
        """Atomically increment the refcount.

        MONOTONIC is sufficient because we already have a valid reference.
        We're copying from an existing reference, so any thread running its
        destructor will not see refcount=0 and will not delete the data.
        """
        _ = self.refcount.fetch_add[ordering=Consistency.MONOTONIC](1)

    fn drop_ref(mut self) -> Bool:
        """Atomically decrement and return True if refcount hits zero.

        RELEASE ensures all our data writes happen-before the decrement.
        Other threads will see our writes before they see refcount decrease.
        """
        if self.refcount.fetch_sub[ordering=Consistency.RELEASE](1) != 1:
            return False

        # ACQUIRE fence synchronizes with all RELEASE decrements above.
        # This ensures we see all writes from all threads before destruction.
        fence[ordering=Consistency.ACQUIRE]()
        return True
```

---

## Complete ArcPointer Implementation

```mojo
# nocompile
@register_passable  # Or use TrivialRegisterType trait in v26.2+
struct RefCounted[T: Movable & ImplicitlyDestructible](ImplicitlyCopyable):
    """Thread-safe reference-counted smart pointer."""

    comptime _inner_type = _RefCountedInner[T]
    var _inner: UnsafePointer[Self._inner_type, MutExternalOrigin]

    fn __init__(out self, var value: T):
        """Construct by moving value into heap allocation."""
        self._inner = alloc[Self._inner_type](1)
        __get_address_as_uninit_lvalue(self._inner.address) = Self._inner_type(value^)

    fn __init__(out self, *, copy: Self):
        """Copy: increment refcount atomically."""
        copy._inner[].add_ref()  # MONOTONIC - we have a valid ref
        self._inner = copy._inner

    @no_inline  # Reduce code bloat from inlining destructor
    fn __del__(deinit self):
        """Decrement refcount; destroy if zero."""
        if self._inner[].drop_ref():  # RELEASE + ACQUIRE fence
            self._inner.destroy_pointee()
            self._inner.free()

    fn __getitem__(ref self) -> ref [self] T:
        """Access the managed value."""
        return self._inner[].payload

    fn count(self) -> UInt64:
        """Return current reference count (for debugging)."""
        # MONOTONIC is fine - just need atomic read, no synchronization
        return self._inner[].refcount.load[ordering=Consistency.MONOTONIC]()
```

---

## Common Patterns

### Reference Cycle Avoidance

Reference counting cannot automatically break cycles. Use these strategies:

**Strategy 1: Weak back-references**

```mojo
# nocompile
struct TreeNode:
    var data: Int
    var children: List[RefCounted[TreeNode]]
    var parent: UnsafePointer[TreeNode, MutExternalOrigin]  # Weak back-reference

    fn __del__(deinit self):
        # Clear children to break potential cycles
        self.children.clear()
```

**Strategy 2: Acyclic data structures**

```mojo
# nocompile
# Use OwnedPointer for tree edges, ArcPointer only for shared leaves
struct Tree:
    var root: OwnedPointer[Node]  # Unique ownership - no cycles possible

struct Node:
    var children: List[OwnedPointer[Node]]  # Each child has single owner
    var shared_data: ArcPointer[SharedState]  # Only share leaf data
```

### Weak Reference Implementation

```mojo
# nocompile
struct WeakRef[T: Movable & ImplicitlyDestructible]:
    """Non-owning reference that doesn't prevent destruction."""
    var _inner: UnsafePointer[_WeakRefInner[T], MutExternalOrigin]

    fn upgrade(self) -> Optional[RefCounted[T]]:
        """Try to get a strong reference. Returns None if already destroyed."""
        if self._inner[].try_add_ref():
            # Successfully incremented refcount before destruction
            return Optional(RefCounted[T]._from_inner(self._inner))
        return None

struct _WeakRefInner[T: Movable & ImplicitlyDestructible]:
    """Shared control block for weak references."""
    var strong_count: Atomic[DType.uint64]
    var weak_count: Atomic[DType.uint64]  # +1 for all strong refs
    var payload: UnsafeMaybeUninitialized[T]

    fn try_add_ref(mut self) -> Bool:
        """Try to increment strong count. Fails if already zero."""
        var current = self.strong_count.load[ordering=Consistency.MONOTONIC]()
        while current != 0:
            var expected = current
            if self.strong_count.compare_exchange[
                success_ordering=Consistency.MONOTONIC,
                failure_ordering=Consistency.MONOTONIC
            ](expected, current + 1):
                return True
            current = expected  # Retry with updated value
        return False

    fn drop_strong(mut self) -> Bool:
        """Decrement strong count. Returns True if payload should be destroyed."""
        if self.strong_count.fetch_sub[ordering=Consistency.RELEASE](1) == 1:
            fence[ordering=Consistency.ACQUIRE]()
            return True
        return False

    fn drop_weak(mut self) -> Bool:
        """Decrement weak count. Returns True if control block should be freed."""
        if self.weak_count.fetch_sub[ordering=Consistency.RELEASE](1) == 1:
            fence[ordering=Consistency.ACQUIRE]()
            return True
        return False
```

### Testing Reference Counted Types

```mojo
# nocompile
from testing import assert_equal, assert_true, assert_false

# Use ObservableDel pattern from stdlib tests
struct ObservableDel:
    """Type that sets a flag when destroyed."""
    var destroyed_flag: UnsafePointer[Bool, MutExternalOrigin]

    fn __del__(deinit self):
        self.destroyed_flag[] = True

def test_refcount_lifecycle():
    var deleted = False
    var flag_ptr = UnsafePointer(to=deleted)

    # Create reference-counted object
    var p1 = RefCounted(ObservableDel(flag_ptr))
    assert_equal(UInt64(1), p1.count())
    assert_false(deleted)

    # Copy increases refcount
    var p2 = p1
    assert_equal(UInt64(2), p1.count())

    # Dropping one copy doesn't destroy
    _ = p2^
    assert_equal(UInt64(1), p1.count())
    assert_false(deleted)

    # Dropping last copy destroys
    _ = p1^
    assert_true(deleted)

def test_no_destruction_while_copies_exist():
    var deleted = False
    var p = RefCounted(ObservableDel(UnsafePointer(to=deleted)))

    # Store in collection
    var vec = List[typeof(p)]()
    vec.append(p)

    # Original destroyed, but copy in vec keeps alive
    _ = p^
    assert_false(deleted)

    # Destroying collection drops last reference
    _ = vec^
    assert_true(deleted)
```

---

## Performance Considerations

1. **Atomic operations are expensive** - 10-100x slower than regular memory access
2. **Use MONOTONIC when possible** - It's the cheapest ordering
3. **Avoid contention** - Cache-line sharing between threads kills performance
4. **Consider thread-local caching** - Batch refcount updates when possible

**Anti-pattern (excessive refcount operations):**

```mojo
# nocompile
# Bad: Every access touches the refcount
fn process_each[T](items: List[RefCounted[T]]):
    for item in items:
        process(item)  # Implicit copy/destroy on each iteration
```

**Pattern (minimize refcount operations):**

```mojo
# nocompile
# Better: Keep reference alive for entire loop
fn process_each[T](items: List[RefCounted[T]]):
    for i in range(len(items)):
        process(items[i][])  # Access payload directly via indexing
```

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Single-threaded shared ownership | Simple integer counter (no atomics) | - |
| Multi-threaded shared ownership | Atomic refcount with MONOTONIC/RELEASE/ACQUIRE | This pattern |
| Need weak references | Implement control block with strong + weak counts | Weak refs section |
| Cycles possible | Use weak back-references or acyclic structure | Cycle avoidance |
| Performance critical | Minimize refcount operations, batch updates | Performance section |
| Already using `ArcPointer` | It handles all this correctly | memory-safe-pointers |

---

## Quick Reference

- **MONOTONIC**: Cheapest ordering, just atomicity, no synchronization
- **RELEASE**: Ensures writes visible before the atomic operation
- **ACQUIRE**: Ensures we see writes from other threads after atomic operation
- **ACQUIRE fence**: Use after decrement hits zero to see all prior writes
- **`fetch_add`**: Returns value before increment
- **`fetch_sub`**: Returns value before decrement (check `== 1` for hitting zero)
- **`compare_exchange`**: Atomic compare-and-swap for lock-free algorithms

---

## Atomic Consistency Levels Summary

| Level | Use Case | Cost |
|-------|----------|------|
| `MONOTONIC` | Counters, statistics, increment/decrement | Lowest |
| `ACQUIRE` | Loading data that another thread released | Medium |
| `RELEASE` | Storing data for another thread to acquire | Medium |
| `ACQ_REL` | Read-modify-write needing both | High |
| `SEQUENTIAL` | Full sequential consistency | Highest |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `memory leak with Arc` | Circular reference | Use weak references or restructure to avoid cycles |
| `use after free with Arc` | Accessing after last strong ref dropped | Ensure Arc outlives all uses; check weak ref upgrade |
| `race condition in refcount` | Wrong memory ordering | Use `RELEASE` for fetch_sub + `ACQUIRE` fence on zero, `MONOTONIC` for fetch_add |
| `double free` | Manual free + Arc destructor | Let Arc handle all deallocation; never call free() manually |
| `atomic ordering too weak` | Using RELAXED/MONOTONIC everywhere | Use RELEASE on decrement, ACQUIRE fence before destroy |
| `ArcPointer null dereference` | Dereferencing dropped Arc | Check `is_valid()` before dereferencing |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **ArcPointer** | `ArcPointer[T]` | Stable |
| **Atomic** | `from os.atomic import Atomic` | Stable |
| **Consistency** | `Consistency.MONOTONIC`, `RELEASE`, `ACQUIRE` | Stable |
| **Heap allocation** | `from memory import alloc; alloc[T](n)` | v26.1+ |
| **ImplicitlyDestructible** | Available | v26.1+ |

**Example (v26.1+):**

```mojo
# nocompile
from os.atomic import Atomic, Consistency, fence
from memory import UnsafePointer, alloc

struct _RefCountedInner[T: Movable & ImplicitlyDestructible]:
    var refcount: Atomic[DType.uint64]
    var payload: T

    fn add_ref(mut self):
        _ = self.refcount.fetch_add[ordering=Consistency.MONOTONIC](1)

    fn drop_ref(mut self) -> Bool:
        if self.refcount.fetch_sub[ordering=Consistency.RELEASE](1) == 1:
            fence[Consistency.ACQUIRE]()
            return True
        return False
```

**Notes:**

- ArcPointer API is stable across versions
- Atomic operations and memory ordering constants are stable
- Reference counting patterns (MONOTONIC increment, RELEASE decrement, ACQUIRE fence) are stable
- Weak reference patterns are stable across versions
- `alloc()` is available in v26.1+ (not nightly-only)

---

## Related Patterns

- [`memory-ownership.md`](memory-ownership.md) — Ownership transfer and lifecycle methods
- [`memory-safety.md`](memory-safety.md) — Safe pointer types including ArcPointer usage

---

## References

- [ArcPointer Implementation](https://github.com/modular/modular/blob/main/mojo/stdlib/std/memory/arc_pointer.mojo)
- [Mojo Atomic Operations](https://github.com/modular/modular/blob/main/mojo/stdlib/std/os/atomic.mojo)
- [C++ Memory Order Reference](https://en.cppreference.com/w/cpp/atomic/memory_order)
