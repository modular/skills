---
title: Mojo Testing Patterns
description: Unit testing patterns including test suites, assertions, lifecycle counters, property-based testing, and GPU test patterns
impact: HIGH
category: test
tags: [testing, unit-test, property-based, lifecycle, assertions]
error_patterns:
  - "test failed"
  - "assertion failed"
  - "assert_equal"
  - "assert_true"
  - "assert_false"
  - "TestSuite"
  - "expected .* but got"
scenarios:
  - "Write unit tests for Mojo code"
  - "Create test suite"
  - "Test lifecycle methods"
  - "Property-based testing"
  - "Fix failing test"
  - "Test GPU kernel correctness"
consolidates:
  - test-suite-patterns.md
  - test-lifecycle-counters.md
  - test-unit-patterns.md
  - test-property-based.md
---

# Mojo Testing Patterns

**Category:** test | **Impact:** HIGH

Comprehensive testing is essential for reliable Mojo code. This pattern covers test organization with TestSuite, unit testing with assertions, lifecycle verification with counter types, and property-based testing for comprehensive coverage.

> **Note:** For performance benchmarking, see [`test-benchmarking.md`](test-benchmarking.md).

---

## Core Concepts

### TestSuite Discovery and Organization

The Mojo testing module provides `TestSuite` for organizing and running tests with automatic discovery.

**Pattern:**

```mojo
from testing import TestSuite, assert_equal, assert_true

# Test functions must start with "test_"
def test_add():
    assert_equal(1 + 1, 2)

def test_subtract():
    assert_equal(5 - 3, 2)

def test_multiply():
    assert_equal(3 * 4, 12)

def main():
    # Automatic discovery of all test_ functions in module
    TestSuite.discover_tests[__functions_in_module()]().run()
```

### Manual Test Registration

**Pattern:**

```mojo
from testing import TestSuite, assert_equal

def my_test_function():
    assert_equal(1 + 1, 2)

def another_test():
    assert_equal("hello".upper(), "HELLO")

def main():
    var suite = TestSuite()
    suite.test[my_test_function]()
    suite.test[another_test]()
    suite^.run()
```

### Test Filtering

**Pattern:**

```mojo
# nocompile
# Skip specific tests
def test_slow_integration():
    # This test takes a long time
    ...

def test_fast_unit():
    ...

def main():
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    suite.skip[test_slow_integration]()  # Skip the slow test
    suite^.run()

# Command line filtering:
# mojo run test.mojo --only test_fast_unit
# mojo run test.mojo --skip test_slow_integration
```

---

## Unit Testing Patterns

### Basic Test Structure

Test functions must be marked `raises` since assertion functions can raise errors.

**Pattern:**

```mojo
from testing import assert_equal, assert_true, assert_false, assert_almost_equal

fn test_basic_addition() raises:
    var result = 2 + 2
    assert_equal(result, 4)

fn test_simd_operations() raises:
    var vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var sum = vec.reduce_add()
    assert_almost_equal(sum, 10.0, atol=1e-6)

fn test_list_operations() raises:
    var items = List[Int]()
    items.append(1)
    items.append(2)
    assert_equal(len(items), 2)
    assert_equal(items[0], 1)

fn main() raises:
    test_basic_addition()
    test_simd_operations()
    test_list_operations()
    print("All tests passed!")
```

### Assertion Functions

| Assertion | Use Case |
|-----------|----------|
| `assert_equal(a, b)` | Exact equality for integers, strings |
| `assert_almost_equal(a, b, atol=1e-6)` | Floating-point comparison with tolerance |
| `assert_true(cond)` | Boolean condition is True |
| `assert_false(cond)` | Boolean condition is False |

### Testing Structs

**Pattern:**

```mojo
from testing import assert_equal, assert_true

@fieldwise_init
struct Counter(Copyable, Movable):
    var count: Int

    fn increment(mut self):
        self.count += 1

    fn get(self) -> Int:
        return self.count

fn test_counter() raises:
    var counter = Counter(count=0)
    assert_equal(counter.get(), 0)

    counter.increment()
    assert_equal(counter.get(), 1)

    counter.increment()
    counter.increment()
    assert_equal(counter.get(), 3)

fn main() raises:
    test_counter()
    print("Counter tests passed!")
```

### Testing Error Conditions

**Pattern:**

```mojo
from testing import assert_true, assert_equal

fn might_fail(value: Int) raises -> Int:
    if value < 0:
        raise Error("Value must be non-negative")
    return value * 2

fn test_error_handling() raises:
    # Test success case
    try:
        var result = might_fail(5)
        assert_equal(result, 10)
    except:
        assert_true(False, "Should not raise for positive value")

    # Test failure case
    var raised = False
    try:
        _ = might_fail(-1)
    except:
        raised = True
    assert_true(raised, "Should raise for negative value")

fn main() raises:
    test_error_handling()
    print("Error handling tests passed!")
```

### Test Fixtures

**Pattern:**

```mojo
from testing import assert_equal

struct TestFixture:
    var data: List[Int]

    fn __init__(out self):
        self.data = List[Int]()
        # Setup: populate with test data
        for i in range(10):
            self.data.append(i * 2)

    fn teardown(mut self):
        # Cleanup: clear data
        self.data.clear()

fn test_with_fixture() raises:
    var fixture = TestFixture()

    # Test using fixture data
    assert_equal(len(fixture.data), 10)
    assert_equal(fixture.data[0], 0)
    assert_equal(fixture.data[5], 10)

    fixture.teardown()
    assert_equal(len(fixture.data), 0)

fn main() raises:
    test_with_fixture()
    print("Fixture tests passed!")
```

---

## Lifecycle Counter Testing

### MoveCounter - Track Move Operations

Use `MoveCounter` to verify that your code performs the expected number of moves.

**Pattern:**

```mojo
# nocompile
struct MoveCounter[T: Copyable & ImplicitlyDestructible](Copyable):
    """Counts moves; helpful for verifying move semantics."""
    var value: T
    var move_count: Int

    @implicit
    fn __init__(out self, var value: T):
        self.value = value^
        self.move_count = 0

    fn __init__(out self, *, deinit take: Self):
        self.value = take.value^
        self.move_count = take.move_count + 1
```

**Usage - verify List doesn't copy unnecessarily:**

```mojo
# nocompile
def test_list_reverse_move_count():
    var list = List[MoveCounter[Int]](capacity=5)
    list.append(MoveCounter(1))
    list.append(MoveCounter(2))
    list.append(MoveCounter(3))
    list.append(MoveCounter(4))
    list.append(MoveCounter(5))

    # Each item moved once into list
    assert_equal(list[0].move_count, 1)

    list.reverse()

    # After reverse:
    # - First 2 elements: temp = a; a = b^; b = temp^ (2 moves each)
    # - Last 2 elements: same (3 moves total: initial + 2 in reverse)
    assert_equal(list[0].move_count, 2)  # Was at end, now at start
    assert_equal(list[4].move_count, 3)  # Was at start, now at end
```

### CopyCounter - Track Copy Operations

**Pattern:**

```mojo
# nocompile
struct CopyCounter[T: ImplicitlyCopyable & Writable & Defaultable = NoneType](
    ImplicitlyCopyable, Writable
):
    """Counts copies; helpful for verifying copy semantics."""
    var value: T
    var copy_count: Int

    fn __init__(out self, var value: T):
        self.value = value
        self.copy_count = 0

    fn __init__(out self, *, copy: Self):
        self.value = copy.value
        self.copy_count = copy.copy_count + 1
```

**Usage - verify function doesn't copy when borrowing:**

```mojo
# nocompile
def test_no_copy_on_borrow():
    var item = CopyCounter[Int](42)
    assert_equal(item.copy_count, 0)

    # Function that borrows (should not copy)
    fn read_value(ref x: CopyCounter[Int]) -> Int:
        return x.value

    var val = read_value(item)
    assert_equal(item.copy_count, 0)  # Still 0 - no copy

    # Function that takes by value (should copy)
    fn take_value(x: CopyCounter[Int]) -> Int:
        return x.value

    val = take_value(item)
    assert_equal(item.copy_count, 0)  # Original unchanged
    # The copy inside take_value had copy_count = 1
```

### DelCounter - Track Destructor Calls

**Pattern:**

```mojo
# nocompile
@fieldwise_init
struct DelCounter[counter_origin: ImmutOrigin](ImplicitlyCopyable, Writable):
    """Counts destructor calls; helpful for verifying cleanup."""
    var counter: UnsafePointer[Int, counter_origin]

    fn __del__(deinit self):
        self.counter.unsafe_mut_cast[True]()[] += 1
```

**Usage - verify destructors run correctly:**

```mojo
# nocompile
def test_list_destructor_count():
    var dtor_count = 0
    var counter_ptr = UnsafePointer(to=dtor_count).as_immutable()

    do:
        var list = List[DelCounter[counter_ptr.origin]]()
        list.append(DelCounter(counter_ptr))
        list.append(DelCounter(counter_ptr))
        list.append(DelCounter(counter_ptr))

        assert_equal(dtor_count, 0)  # Nothing destroyed yet
    # List goes out of scope here

    assert_equal(dtor_count, 3)  # All 3 items destroyed
```

### AbortOnCopy - Catch Unexpected Copies

**Pattern:**

```mojo
# nocompile
@fieldwise_init
struct AbortOnCopy(ImplicitlyCopyable):
    """Type that aborts if copied - for testing move-only code paths."""
    var value: Int

    fn __init__(out self, *, other: Self):
        abort("Unexpected copy of AbortOnCopy!")
```

### Testing Optimal Container Operations

**Pattern:**

```mojo
# nocompile
def test_extend_moves_not_copies():
    """Verify extend uses moves, not copies."""
    var v1 = List[MoveCounter[String]](capacity=5)
    v1.append(MoveCounter("Hello"))
    v1.append(MoveCounter("World"))

    var v2 = List[MoveCounter[String]](capacity=3)
    v2.append(MoveCounter("Foo"))
    v2.append(MoveCounter("Bar"))

    # Extend should move from v2, not copy
    v1.extend(v2^)  # Transfer ownership of v2

    assert_equal(len(v1), 4)
    # Items from v2 should have 2 moves: into v2, then into v1
    assert_equal(v1[2].move_count, 2)
    assert_equal(v1[3].move_count, 2)
```

---

## Property-Based Testing

### Basic Property-Based Testing

Property-based testing generates random inputs to verify that invariants hold for all cases, catching edge cases that example-based tests miss.

**Example-based testing (limited coverage):**

```mojo
# nocompile
def test_list_reverse_examples():
    # Only tests specific cases
    var list1 = [1, 2, 3]
    list1.reverse()
    assert_equal(list1, [3, 2, 1])

    var list2 = [1]
    list2.reverse()
    assert_equal(list2, [1])
    # What about empty lists? Large lists? Negative numbers?
```

**Property-based testing (comprehensive):**

```mojo
# nocompile
from testing.prop import PropTest, PropTestConfig, Rng, Strategy
from testing.prop.strategy import List as ListStrategy, Scalar

def test_list_reverse_property():
    @parameter
    def properties(items: List[Scalar[DType.int32]]):
        var original = items.copy()
        var reversed = items.copy()
        reversed.reverse()

        # Property 1: Length is preserved
        assert_equal(len(original), len(reversed))

        # Property 2: Double reverse equals original
        reversed.reverse()
        for i in range(len(original)):
            assert_equal(original[i], reversed[i])

        # Property 3: First becomes last
        if len(original) > 0:
            assert_equal(original[0], items[len(items) - 1])

    PropTest().test[properties](
        ListStrategy[Scalar[DType.int32]].strategy(
            Scalar[DType.int32].strategy()
        )
    )
```

### Custom Strategies

**Pattern:**

```mojo
# nocompile
from testing.prop import Strategy, Rng

@fieldwise_init
struct PositiveIntStrategy(Movable, Strategy):
    """Generates positive integers only."""
    comptime Value = Int
    var max_value: Int

    fn value(mut self, mut rng: Rng) raises -> Int:
        return Int(rng.next_ui64()) % self.max_value + 1

# Usage
PropTest().test[properties](PositiveIntStrategy(max_value=1000))
```

### Deterministic Tests with Seeds

**Pattern:**

```mojo
# nocompile
def test_deterministic_with_seed():
    @parameter
    def properties(n: Int):
        pass

    # Same seed = same sequence of values
    var config = PropTestConfig(runs=100, seed=12345)

    var results1 = List[Int]()
    var results2 = List[Int]()

    # Recording strategy captures values
    PropTest(config=config.copy()).test[properties](
        RecordingStrategy(UnsafePointer(to=results1))
    )
    PropTest(config=config^).test[properties](
        RecordingStrategy(UnsafePointer(to=results2))
    )

    assert_equal(results1, results2)  # Identical!
```

### PropTestConfig Options

```mojo
# nocompile
var config = PropTestConfig(
    runs=100,        # Number of test iterations (default 100)
    seed=None,       # Random seed for reproducibility (None = random)
    max_size=100,    # Maximum size for generated collections
)
```

### Common Property Patterns

| Property | Description | Example |
|----------|-------------|---------|
| Roundtrip | encode(decode(x)) == x | serialize/deserialize |
| Idempotence | f(f(x)) == f(x) | sort, normalize |
| Commutativity | f(a, b) == f(b, a) | add, multiply |
| Invariant | property holds before/after | length preserved |
| Oracle | compare to known-good impl | CPU vs GPU |

---

## GPU Testing Patterns

### Basic GPU Kernel Testing

Testing GPU kernels requires special consideration for numerical correctness, determinism, and error handling.

**Pattern:**

```mojo
# nocompile
from testing import assert_almost_equal, assert_equal
from gpu.host import DeviceContext
from sys import has_accelerator

fn test_gpu_kernel_correctness() raises:
    """Test GPU kernel against CPU reference."""
    if not has_accelerator():
        print("Skipping GPU test: no accelerator available")
        return

    comptime SIZE: Int = 1024
    var ctx = DeviceContext()

    # Prepare input data
    var h_input = List[Float32](capacity=SIZE)
    for i in range(SIZE):
        h_input.append(Float32(i) * 0.5)

    # Run CPU reference
    var expected = List[Float32](capacity=SIZE)
    for i in range(SIZE):
        expected.append(h_input[i] * 2.0)  # Reference computation

    # Run GPU kernel
    var d_input = ctx.enqueue_create_buffer[DType.float32](SIZE)
    var d_output = ctx.enqueue_create_buffer[DType.float32](SIZE)

    ctx.enqueue_copy(d_input, h_input.unsafe_ptr(), SIZE)
    ctx.enqueue_function[my_gpu_kernel](
        d_input.unsafe_ptr(),
        d_output.unsafe_ptr(),
        SIZE,
        grid_dim=(SIZE // 256,),
        block_dim=(256,)
    )

    var h_output = List[Float32](capacity=SIZE)
    h_output.resize(SIZE)
    ctx.enqueue_copy(h_output.unsafe_ptr(), d_output, SIZE)
    ctx.synchronize()

    # Compare with tolerance for floating-point
    for i in range(SIZE):
        assert_almost_equal(h_output[i], expected[i], atol=1e-5)
```

### GPU Test Fixture

**Pattern:**

```mojo
# nocompile
struct GPUTestFixture:
    """Reusable GPU test fixture with automatic cleanup."""
    var ctx: DeviceContext
    var buffers: List[UnsafePointer[NoneType]]

    fn __init__(out self) raises:
        if not has_accelerator():
            raise Error("No GPU available")
        self.ctx = DeviceContext()
        self.buffers = List[UnsafePointer[NoneType]]()

    fn create_buffer[dtype: DType](mut self, size: Int) raises -> UnsafePointer[Scalar[dtype]]:
        var buf = self.ctx.enqueue_create_buffer[dtype](size)
        return buf.unsafe_ptr()

    fn sync(mut self):
        self.ctx.synchronize()

fn test_with_fixture() raises:
    var fixture = GPUTestFixture()

    var d_a = fixture.create_buffer[DType.float32](1024)
    var d_b = fixture.create_buffer[DType.float32](1024)

    # Test logic here...
    fixture.sync()
```

---

## Decision Guide

| Scenario | Approach |
|----------|----------|
| Basic functionality | Unit tests with assertions |
| Test organization | TestSuite with discovery |
| Move/copy verification | Lifecycle counters |
| Edge case discovery | Property-based testing |
| Destructor verification | DelCounter |
| GPU kernel correctness | GPU test with CPU reference |

---

## When to Apply

### Use TestSuite for

- All unit test files
- Integration tests
- Module-level testing
- CI/CD test suites

### Use Lifecycle Counters for

- Testing container implementations
- Verifying move-only operations
- Debugging unexpected performance (too many copies)
- Ensuring destructors are called correctly

### Use Property-Based Testing for

- Testing data structure invariants
- Verifying mathematical properties
- Encode/decode roundtrip tests
- Finding edge cases in algorithms

---

## Running Tests

```bash
# Run test file
mojo run test_my_module.mojo

# For tests with BLAS/FFI dependencies
# Use mojo run for automatic dynamic linking
mojo run test_blas.mojo

# Run with GPU target
mojo run --target-accelerator=nvidia:sm_80 test_gpu.mojo

# Skip GPU tests when no GPU available
# (Handle in test code with has_accelerator() check)
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `assert_equal type mismatch` | Comparing different types | Ensure both values are same type |
| `test not discovered` | Wrong function signature | Use `fn test_*()` pattern |
| `GPU test hangs` | Missing synchronize | Add `ctx.synchronize()` |
| `MoveCounter shows too many moves` | Unexpected copies | Check container uses moves |
| `property test flaky` | Non-deterministic seed | Set explicit seed |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **TestSuite** | `TestSuite.discover_tests[...]` | Stable |
| **Assertions** | `assert_equal`, `assert_true` | Stable |
| **Constants** | `alias` or `comptime` | Both work in v26.1+ |

**Example (v26.1+):**

```mojo
from testing import TestSuite, assert_equal

comptime MAX_ITEMS = 100

def test_basic():
    assert_equal(1 + 1, 2)

def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
```

---

## Related Patterns

- [`test-benchmarking.md`](test-benchmarking.md) — Performance benchmarking
- [`meta-programming.md`](meta-programming.md) — Parameterized test utilities
- [`memory-ownership.md`](memory-ownership.md) — Lifecycle methods being tested

---

## References

- [Mojo Testing Module](https://github.com/modular/modular/blob/main/mojo/stdlib/std/testing/suite.mojo)
- [Mojo Stdlib Test Utils](https://github.com/modular/modular/blob/main/mojo/stdlib/test/test_utils/types.mojo)
