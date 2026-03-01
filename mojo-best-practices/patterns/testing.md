---
title: Mojo Testing & Benchmarking Patterns
description: Unit testing, property-based testing, lifecycle counters, performance benchmarking with QuickBench, GPU test and benchmark patterns
impact: HIGH
category: test
tags: [testing, unit-test, property-based, lifecycle, assertions, benchmark, performance, timing, quickbench, profiling, gpu, bandwidth]
error_patterns:
  - "test failed"
  - "assertion failed"
  - "assert_equal"
  - "assert_true"
  - "assert_false"
  - "TestSuite"
  - "expected .* but got"
  - "benchmark"
  - "QuickBench"
  - "perf_counter"
  - "timing"
  - "throughput"
scenarios:
  - "Write unit tests for Mojo code"
  - "Create test suite"
  - "Test lifecycle methods"
  - "Property-based testing"
  - "Fix failing test"
  - "Test GPU kernel correctness"
  - "Benchmark function performance"
  - "Compare algorithm implementations"
  - "Measure SIMD speedup"
  - "Create performance regression tests"
  - "Measure throughput metrics"
  - "Benchmark GPU kernel bandwidth"
consolidates:
  - test-suite-patterns.md
  - test-lifecycle-counters.md
  - test-unit-patterns.md
  - test-property-based.md
  - benchmark-patterns.md
  - quickbench-patterns.md
  - perf-testing.md
---
<!-- PATTERN QUICK REF
WHEN: Writing tests, benchmarking, test assertions, measuring performance in Mojo
KEY_TYPES: TestSuite, QuickBench, BenchId, BenchMetric, ThroughputMeasure, MoveCounter, CopyCounter, DelCounter, PropTest, PropTestConfig
SYNTAX:
  - TestSuite.discover_tests[__functions_in_module()]().run()
  - assert_equal(a, b) / assert_almost_equal(a, b, atol=1e-6) / assert_true(cond) / assert_false(cond)
  - QuickBench().run(fn, input, bench_id=BenchId("name"), measures=[ThroughputMeasure(...)])
  - keep(result) / clobber_memory() to prevent dead code elimination
  - perf_counter_ns() returns UInt (not Int)
  - PropTest().test[properties](strategy)
PITFALLS: perf_counter_ns returns UInt not Int; missing keep() causes 0ns measurements; no warmup gives cold-cache results; missing ctx.synchronize() in GPU benchmarks; assert functions can raise so test fns need `raises`
RELATED: debug-debugging, error-handling, perf-vectorization, perf-parallelization, memory-ownership
-->

# Mojo Testing & Benchmarking Patterns

**Category:** test | **Impact:** HIGH

Comprehensive testing and proper benchmarking are essential for reliable, performant Mojo code. This pattern covers test organization with TestSuite, unit testing with assertions, lifecycle verification, property-based testing, and structured benchmarking with QuickBench.

---

## TestSuite Discovery and Organization

The Mojo testing module provides `TestSuite` for organizing and running tests with automatic discovery.

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

```mojo
# nocompile
# Skip specific tests
def test_slow_integration():
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
        self.data.clear()

fn test_with_fixture() raises:
    var fixture = TestFixture()

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

> **Note:** `MoveCounter`, `CopyCounter`, `DelCounter`, and `AbortOnCopy` are test-internal utilities found in `stdlib/test/test_utils/types.mojo`. They are not part of the public standard library API. You can copy these patterns into your own test utilities.

### MoveCounter - Track Move Operations

Use `MoveCounter` to verify that your code performs the expected number of moves.

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

    fn __moveinit__(out self, deinit take: Self):
        self.value = take.value^
        self.move_count = take.move_count + 1

    fn __copyinit__(out self, copy: Self):
        self.value = copy.value
        self.move_count = copy.move_count
```

**Usage — verify List doesn't copy unnecessarily:**

```mojo
# nocompile
def test_list_reverse_move_count():
    var list = List[MoveCounter[Int]](capacity=5)
    list.append(MoveCounter(1))
    list.append(MoveCounter(2))
    list.append(MoveCounter(3))
    list.append(MoveCounter(4))
    list.append(MoveCounter(5))

    assert_equal(list[0].move_count, 1)

    list.reverse()

    assert_equal(list[0].move_count, 2)  # Was at end, now at start
    assert_equal(list[4].move_count, 3)  # Was at start, now at end
```

### CopyCounter - Track Copy Operations

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

    fn __copyinit__(out self, copy: Self):
        self.value = copy.value
        self.copy_count = copy.copy_count + 1
```

**Usage — verify function doesn't copy when borrowing:**

```mojo
# nocompile
def test_no_copy_on_borrow():
    var item = CopyCounter[Int](42)
    assert_equal(item.copy_count, 0)

    fn read_value(ref x: CopyCounter[Int]) -> Int:
        return x.value

    var val = read_value(item)
    assert_equal(item.copy_count, 0)  # Still 0 - no copy

    fn take_value(x: CopyCounter[Int]) -> Int:
        return x.value

    val = take_value(item)
    assert_equal(item.copy_count, 0)  # Original unchanged
```

### DelCounter - Track Destructor Calls

```mojo
# nocompile
@fieldwise_init
struct DelCounter[counter_origin: ImmutOrigin](ImplicitlyCopyable, Writable):
    """Counts destructor calls; helpful for verifying cleanup."""
    var counter: UnsafePointer[Int, counter_origin]

    fn __del__(deinit self):
        self.counter.unsafe_mut_cast[True]()[] += 1
```

**Usage — verify destructors run correctly:**

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

```mojo
# nocompile
@fieldwise_init
struct AbortOnCopy(ImplicitlyCopyable):
    """Type that aborts if copied - for testing move-only code paths."""
    var value: Int

    fn __copyinit__(out self, copy: Self):
        abort("Unexpected copy of AbortOnCopy!")
```

### Testing Optimal Container Operations

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

    v1.extend(v2^)  # Transfer ownership of v2

    assert_equal(len(v1), 4)
    assert_equal(v1[2].move_count, 2)
    assert_equal(v1[3].move_count, 2)
```

---

## Property-Based Testing

Property-based testing generates random inputs to verify invariants hold for all cases, catching edge cases that example-based tests miss.

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
)
```

> **Note:** `PropTestConfig` only accepts `runs` and `seed` parameters. Collection size limits are controlled by the strategy, not the config.

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

### GPU Kernel Correctness Testing

Testing GPU kernels requires comparison against a CPU reference with appropriate tolerance.

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
        expected.append(h_input[i] * 2.0)

    # Run GPU kernel
    var d_input = ctx.enqueue_create_buffer[DType.float32](SIZE)
    var d_output = ctx.enqueue_create_buffer[DType.float32](SIZE)

    ctx.enqueue_copy(d_input, h_input.unsafe_ptr())
    ctx.enqueue_function[my_gpu_kernel](
        d_input.unsafe_ptr(),
        d_output.unsafe_ptr(),
        SIZE,
        grid_dim=(SIZE // 256,),
        block_dim=(256,)
    )

    var h_output = List[Float32](capacity=SIZE)
    h_output.resize(SIZE)
    ctx.enqueue_copy(h_output.unsafe_ptr(), d_output)
    ctx.synchronize()

    for i in range(SIZE):
        assert_almost_equal(h_output[i], expected[i], atol=1e-5)
```

### GPU Test Fixture

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

## Benchmarking

### Basic Benchmark Structure

Use proper methodology with warmup iterations, multiple samples, and appropriate timing. Note: `perf_counter_ns()` returns `UInt`.

```mojo
from time import perf_counter_ns
from benchmark import keep

fn operation_to_benchmark() -> Int:
    var sum = 0
    for i in range(1000):
        sum += i
    return sum

fn benchmark_operation() -> Float64:
    """Benchmark with warmup and multiple iterations."""
    comptime WARMUP_ITERS: Int = 5
    comptime BENCH_ITERS: Int = 100

    # Warmup - don't measure these
    for _ in range(WARMUP_ITERS):
        keep(operation_to_benchmark())

    var total_ns: UInt = 0
    for _ in range(BENCH_ITERS):
        var start = perf_counter_ns()
        var result = operation_to_benchmark()
        keep(result)  # CRITICAL: prevent dead code elimination
        var end = perf_counter_ns()
        total_ns += end - start

    return Float64(total_ns) / Float64(BENCH_ITERS) / 1_000_000.0

fn main():
    var avg_ms = benchmark_operation()
    print("Average time:", avg_ms, "ms")
```

> **Warning:** `_ = fn()` is NOT sufficient to prevent dead code elimination (DCE). The compiler may still optimize away the function call. Always use `keep(result)` from `benchmark` to ensure the computation actually executes.

### Best-of-N Pattern

```mojo
from time import perf_counter_ns

fn operation_to_benchmark() -> Int:
    var sum = 0
    for i in range(1000):
        sum += i
    return sum

fn benchmark_best_of_n[N: Int]() -> UInt:
    """Return the best (minimum) time from N runs."""
    comptime WARMUP: Int = 3

    for _ in range(WARMUP):
        _ = operation_to_benchmark()

    var best_ns: UInt = UInt.MAX
    for _ in range(N):
        var start = perf_counter_ns()
        _ = operation_to_benchmark()
        var elapsed = perf_counter_ns() - start
        if elapsed < best_ns:
            best_ns = elapsed

    return best_ns

fn main():
    var best_ns = benchmark_best_of_n[10]()
    print("Best time:", best_ns, "ns")
```

### Preventing Compiler Optimization

Use `keep()` and `clobber_memory()` to prevent dead code elimination.

```mojo
# nocompile
from benchmark import keep, clobber_memory

fn benchmark_kernel():
    var result = expensive_computation()

    # CRITICAL: Prevent optimizer from eliminating result
    keep(result)

    # CRITICAL: Prevent optimizer from reordering memory ops
    clobber_memory()
```

### Benchmarking SIMD Operations

```mojo
from time import perf_counter_ns
from benchmark import keep

fn benchmark_simd_vs_scalar():
    comptime SIZE: Int = 1024 * 1024
    comptime ITERS: Int = 100

    var data = List[Float32](capacity=SIZE)
    for i in range(SIZE):
        data.append(Float32(i))

    # Scalar benchmark
    var scalar_start = perf_counter_ns()
    for _ in range(ITERS):
        var sum: Float32 = 0.0
        for i in range(SIZE):
            sum += data[i]
        keep(sum)
    var scalar_ns = perf_counter_ns() - scalar_start

    # SIMD benchmark
    var simd_start = perf_counter_ns()
    for _ in range(ITERS):
        var ptr = data.unsafe_ptr()
        var sum = SIMD[DType.float32, 8](0.0)
        for i in range(0, SIZE, 8):
            sum += (ptr + i).load[width=8]()
        keep(sum.reduce_add())
    var simd_ns = perf_counter_ns() - simd_start

    print("Scalar:", scalar_ns / UInt(ITERS), "ns")
    print("SIMD:", simd_ns / UInt(ITERS), "ns")
    print("Speedup:", Float64(scalar_ns) / Float64(simd_ns), "x")
```

---

## QuickBench Structured Benchmarking

### Basic QuickBench Usage

The `QuickBench` API provides structured benchmarking with throughput metrics and proper methodology.

**Incorrect (naive timing):**

```mojo
# nocompile
# BAD: Susceptible to optimizer, no warmup, single sample
from time import perf_counter_ns

def bad_benchmark():
    var start = perf_counter_ns()
    var result = my_function()  # Might be optimized away!
    var elapsed = perf_counter_ns() - start
    print("Time:", elapsed)
```

**Correct (QuickBench):**

```mojo
from benchmark import QuickBench, BenchId, BenchMetric, ThroughputMeasure
from benchmark import keep, clobber_memory

fn my_function(x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return x * x + x

def main():
    var qb = QuickBench()

    var input = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)

    qb.run(
        my_function,
        input,
        bench_id=BenchId("my_function"),
        measures=[
            ThroughputMeasure(BenchMetric.elements, 4),
        ],
    )

    qb.dump_report()
```

### Using the `run[]` Function

```mojo
# nocompile
from benchmark import run

fn my_kernel():
    var data = compute_something()
    keep(data)
    clobber_memory()

def main():
    var report = run[func2=my_kernel](
        min_runtime_secs=0.1,
        max_runtime_secs=1.0,
        max_iters=10000,
    )

    print("Mean time:", report.mean(), "s")
    print("Mean time:", report.mean("ms"), "ms")
    print("Mean time:", report.mean("us"), "us")
    print("Iterations:", report.iters())
    print(report.as_string())
```

### Multiple Benchmark Comparison

```mojo
from benchmark import QuickBench, BenchId, BenchMetric, ThroughputMeasure
import math

@always_inline
fn exp_simd(x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return math.exp(x)

@always_inline
fn tanh_simd(x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return math.tanh(x)

@always_inline
fn manual_exp(x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return 1.0 + x + x*x/2.0 + x*x*x/6.0

def main():
    var qb = QuickBench()
    var input = SIMD[DType.float32, 4](0.5)

    qb.run(exp_simd, input, bench_id=BenchId("math.exp"))
    qb.run(tanh_simd, input, bench_id=BenchId("math.tanh"))
    qb.run(manual_exp, input, bench_id=BenchId("manual_exp"))

    qb.dump_report()
```

### Throughput Metrics

```mojo
# nocompile
measures = [
    ThroughputMeasure(BenchMetric.elements, 1024),    # 1024 elements
    ThroughputMeasure(BenchMetric.bytes, 1024 * 4),   # 4KB
    ThroughputMeasure(BenchMetric.flops, 1024 * 2),   # 2 FLOPS per element
]
```

| Metric | Use Case |
|--------|----------|
| `BenchMetric.elements` | Array processing |
| `BenchMetric.bytes` | Memory bandwidth |
| `BenchMetric.flops` | Compute throughput |

---

## GPU Kernel Benchmarking

Raw execution time is less useful than bandwidth efficiency — the percentage of theoretical peak memory bandwidth achieved.

### Bandwidth Efficiency Formula

```
actual_GB_s = bytes_moved / (time_ns / 1e9) / 1e9
efficiency = actual_GB_s / peak_GB_s * 100
```

### Peak Memory Bandwidth Reference

| GPU | Architecture | Peak BW (GB/s) |
|---|---|---|
| H100 SXM | SM90 | 3350 |
| H100 PCIe | SM90 | 2000 |
| A100 80GB | SM80 | 2000 |
| L40S | SM89 | 864 |
| MI300X | CDNA3 | 5300 |
| Apple M3 Max | Apple GPU | ~400 |

### Efficiency Verdicts

| Efficiency | Assessment | Action |
|---|---|---|
| >60% | Excellent | Memory-bound kernel well-optimized |
| 30-60% | Good | Check coalescing and bank conflicts |
| 10-30% | Needs work | Profile for uncoalesced access, bank conflicts |
| <10% | Compute-bound or broken | Check if compute-bound (expected) or severely uncoalesced |

### Amdahl's Law Warning

A 2.67x kernel speedup sounds impressive, but if the kernel is only 2.3% of total pipeline time, the end-to-end improvement is ~1.5%. Always measure the kernel's contribution to total runtime before optimizing.

### A/B Benchmarking Methodology

- **Warmup:** 5 iterations (discard) — stabilizes GPU clocks and caches
- **Measurement:** 100 iterations
- **Report:** mean, median, p99, and min (min = best achievable)
- Use `keep()` to prevent dead code elimination

### GPU Bandwidth Benchmark Pattern

```mojo
# nocompile

from time import perf_counter_ns
from gpu.host import DeviceContext

fn benchmark_kernel_bw[
    kernel_fn: fn(...) capturing -> None,
](
    ctx: DeviceContext,
    bytes_moved: Int,
    peak_bw_gb_s: Float64,
) raises -> Float64:
    """Benchmark GPU kernel and return bandwidth efficiency percentage."""
    comptime WARMUP: Int = 5
    comptime ITERS: Int = 100

    # Warmup
    for _ in range(WARMUP):
        kernel_fn(...)
        ctx.synchronize()

    # Measure
    var total_ns: UInt = 0
    for _ in range(ITERS):
        var start = perf_counter_ns()
        kernel_fn(...)
        ctx.synchronize()
        total_ns += perf_counter_ns() - start

    var avg_ns = Float64(total_ns) / Float64(ITERS)
    var actual_gb_s = Float64(bytes_moved) / avg_ns  # bytes/ns = GB/s
    var efficiency = actual_gb_s / peak_bw_gb_s * 100.0
    return efficiency
```

**Key difference from CPU benchmarks:** Always call `ctx.synchronize()` after each kernel launch to ensure timing captures actual GPU execution, not just launch overhead.

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
| Quick one-off timing | Basic benchmark with warmup |
| Structured comparison | QuickBench with multiple benchmarks |
| Algorithm comparison | Best-of-N pattern |
| Throughput measurement | QuickBench with ThroughputMeasure |
| CI regression testing | Benchmark report to JSON |
| GPU kernel bandwidth | Bandwidth efficiency benchmark |

---

## Best Practices

| Practice | Rationale |
|----------|-----------|
| **Warmup iterations (3-10)** | Stabilize caches, JIT, memory |
| **Multiple samples (10-100)** | Statistical significance |
| **Report best/avg/worst** | Understand variance |
| **Use perf_counter_ns()** | Nanosecond precision |
| **Prevent dead code elimination** | `keep()` or return results |
| **Isolate the operation** | Exclude setup/teardown from timing |
| **GPU: ctx.synchronize()** | Time actual execution, not launch |

---

## Running Tests

```bash
# Run test file
mojo run test_my_module.mojo

# For tests with BLAS/FFI dependencies
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
| 0ns benchmark measurements | Compiler optimized away code | Use `keep()` or return values |
| Inconsistent benchmark results | No warmup | Add 3-10 warmup iterations |
| Misleading benchmark | Setup included in timing | Time only the operation |

---

## Version-Specific Features (v26.1+)

| Feature | Status | Notes |
|---------|--------|-------|
| **TestSuite** | `TestSuite.discover_tests[...]` | Stable |
| **Assertions** | `assert_equal`, `assert_true` | Stable |
| **Constants** | `alias` or `comptime` | Both work in v26.1+ |
| **perf_counter_ns** | Returns `UInt` | Stable |
| **QuickBench API** | Available | Stable |
| **keep/clobber_memory** | Available | Stable |

---

## Related Patterns

- [`meta-programming.md`](meta-programming.md) — Parameterized test utilities
- [`memory-ownership.md`](memory-ownership.md) — Lifecycle methods being tested
- [`perf-vectorization.md`](perf-vectorization.md) — SIMD code to benchmark
- [`perf-parallelization.md`](perf-parallelization.md) — Parallel code benchmarking

---

## References

- [Mojo Testing Module](https://github.com/modular/modular/blob/main/mojo/stdlib/std/testing/suite.mojo)
- [Mojo Stdlib Test Utils](https://github.com/modular/modular/blob/main/mojo/stdlib/test/test_utils/types.mojo)
- [Mojo Benchmark Module](https://github.com/modular/modular/blob/main/mojo/stdlib/std/benchmark/)
- [QuickBench API](https://github.com/modular/modular/blob/main/mojo/stdlib/std/benchmark/quick_bench.mojo)
