---
title: Python Interoperability
description: Patterns for efficient Python/Mojo integration including boundary minimization, import optimization, type conversion, and error handling
impact: MEDIUM
category: python
tags: [python, interop, boundaries, types, errors, performance]
error_patterns:
  - "Python"
  - "PythonObject"
  - "import"
  - "GIL"
  - "type conversion"
  - "boundary crossing"
  - "interpreter"
scenarios:
  - "Call Python from Mojo"
  - "Minimize boundary crossings"
  - "Convert Python types to Mojo"
  - "Handle Python errors in Mojo"
  - "Import Python modules"
  - "Use Python libraries from Mojo"
consolidates:
  - python-minimize-crossing.md
  - python-import-patterns.md
  - python-type-conversion.md
  - python-error-handling.md
---

# Python Interoperability

**Category:** python | **Impact:** MEDIUM

Comprehensive patterns for efficient Python/Mojo integration. Covers minimizing boundary crossings (10-1000x speedup), import optimization, type conversion at boundaries, and proper error handling. These patterns are essential when leveraging Python's ecosystem from Mojo code.

---

## Core Concepts

### Boundary Crossing Overhead

Each boundary crossing between Mojo and Python has significant overhead. The key principle is: **batch operations to minimize crossings**.

**The Cost:**
- Each Python object access requires interpreter interaction
- Type conversions happen at every crossing
- GIL acquisition/release overhead

### When to Use Python

| Operation | Use Mojo | Use Python |
|-----------|----------|------------|
| Simple math | Yes | No |
| Array operations | Yes (SIMD) | Only if very complex |
| String processing | Yes | For regex/unicode |
| File I/O | Either | For special formats |
| ML inference | Depends | For model loading |
| Visualization | No | Yes (matplotlib) |

---

## Minimizing Boundary Crossings

### Anti-Pattern: Many Boundary Crossings

```mojo
# nocompile
from python import Python

fn slow_stats(data: List[Float64]) -> Tuple[Float64, Float64]:
    var np = Python.import_module("numpy")
    var py_data = Python.evaluate("[]")

    # N boundary crossings to build array
    for item in data:
        py_data.append(item)  # Crossing for each element

    var arr = np.array(py_data)
    var mean = Float64(py=arr.mean())  # Crossing
    var std = Float64(py=arr.std())    # Crossing
    return (mean, std)

fn slow_element_wise(data: List[Float64]) -> List[Float64]:
    var np = Python.import_module("numpy")
    var result = List[Float64](capacity=len(data))

    for item in data:
        # Crossing for EVERY element operation!
        var py_val = Float64(py=np.sqrt(item))
        result.append(py_val)

    return result^
```

### Correct: Compute in Pure Mojo

```mojo
# nocompile
from python import Python

fn fast_stats(data: List[Float64]) -> Tuple[Float64, Float64]:
    # Compute in pure Mojo instead of crossing to Python
    var n = len(data)
    var sum: Float64 = 0.0
    var sum_sq: Float64 = 0.0

    for item in data:
        var val = item
        sum += val
        sum_sq += val * val

    var mean = sum / Float64(n)
    var variance = sum_sq / Float64(n) - mean * mean
    var std = sqrt(variance)  # Use sqrt() function from math

    return (mean, std)

fn fast_element_wise(data: List[Float64]) -> List[Float64]:
    # Pure Mojo - no boundary crossings
    var result = List[Float64](capacity=len(data))
    for item in data:
        result.append(sqrt(item))  # Use sqrt() function from math
    return result^
```

### When Python is Necessary: Batch Operations

```mojo
# nocompile
fn batch_python_operation(data: List[Float64]) -> List[Float64]:
    var np = Python.import_module("numpy")

    # One crossing to send data
    var py_list = Python.evaluate("[]")
    for item in data:
        py_list.append(item)

    # One crossing for the computation (happens in Python)
    var py_result = np.sqrt(np.array(py_list))

    # One crossing to get results back
    var result = List[Float64](capacity=len(data))
    for i in range(len(data)):
        result.append(Float64(py=py_result[i]))

    return result^

    # Total: 3 crossings instead of 2N crossings
```

---

## Import Patterns

### Anti-Pattern: Repeated Imports

```mojo
# nocompile
from python import Python

fn analyze_data(data: PythonObject) -> PythonObject:
    # Import every function call - slow!
    var np = Python.import_module("numpy")
    var pd = Python.import_module("pandas")
    return pd.DataFrame(np.array(data))

fn plot_graph(x: PythonObject, y: PythonObject):
    # Import overhead every time
    var plt = Python.import_module("matplotlib.pyplot")
    plt.plot(x, y)
    plt.show()

fn calculate_stats(data: PythonObject) -> PythonObject:
    var np = Python.import_module("numpy")  # Again!
    return np.mean(data)
```

### Correct: Import Once, Reuse

```mojo
# nocompile
from python import Python, PythonObject

# Module-level storage for imported modules
struct PythonModules:
    var np: PythonObject
    var pd: PythonObject
    var plt: PythonObject
    var _initialized: Bool

    fn __init__(out self):
        self._initialized = False
        # Use PythonObject(None) for uninitialized placeholder
        self.np = PythonObject(None)
        self.pd = PythonObject(None)
        self.plt = PythonObject(None)

    fn ensure_initialized(mut self):
        if not self._initialized:
            self.np = Python.import_module("numpy")
            self.pd = Python.import_module("pandas")
            self.plt = Python.import_module("matplotlib.pyplot")
            self._initialized = True

var modules = PythonModules()

fn analyze_data(data: PythonObject) -> PythonObject:
    modules.ensure_initialized()
    return modules.pd.DataFrame(modules.np.array(data))

fn plot_graph(x: PythonObject, y: PythonObject):
    modules.ensure_initialized()
    modules.plt.plot(x, y)
    modules.plt.show()

fn calculate_stats(data: PythonObject) -> PythonObject:
    modules.ensure_initialized()
    return modules.np.mean(data)
```

### Simpler Pattern: Pass Modules as Parameters

```mojo
# nocompile
from python import Python

fn main():
    # Import once at entry point
    var np = Python.import_module("numpy")
    var pd = Python.import_module("pandas")

    # Pass to functions that need them
    process_with_numpy(np, data)
    analyze_with_pandas(pd, results)

fn process_with_numpy(np: PythonObject, data: PythonObject) -> PythonObject:
    # No import needed - module passed in
    return np.array(data) * 2
```

### Lazy Import for Optional Dependencies

```mojo
# nocompile
fn try_use_optional_feature() raises:
    try:
        var optional_lib = Python.import_module("optional_library")
        # Use the library
    except:
        raise "optional_library not installed. Install with: pip install optional_library"
```

---

## Type Conversion at Boundaries

### Anti-Pattern: Repeated Conversions in Loop

```mojo
# nocompile
from python import Python

fn process_data() -> Float64:
    var np = Python.import_module("numpy")
    var arr = np.random.rand(1000)

    var total: Float64 = 0.0
    for i in range(1000):
        # Conversion happens every iteration!
        total += Float64(py=arr[i])
    return total

fn find_max(py_list: PythonObject) -> Float64:
    var max_val: Float64 = Float64(py=py_list[0])
    for i in range(1, Int(py=len(py_list))):
        var val = Float64(py=py_list[i])  # Repeated conversion
        if val > max_val:
            max_val = val
    return max_val
```

### Correct: Convert Once at Boundary

```mojo
# nocompile
from python import Python

fn process_data() -> Float64:
    var np = Python.import_module("numpy")
    var py_arr = np.random.rand(1000)

    # Convert to Mojo list once at the boundary
    var arr = List[Float64](capacity=1000)
    for i in range(1000):
        arr.append(Float64(py=py_arr[i]))

    # Now use native Mojo operations - no conversion overhead
    var total: Float64 = 0.0
    for item in arr:
        total += item
    return total

fn find_max(py_list: PythonObject) -> Float64:
    # Convert entire list at boundary
    var size = Int(py=len(py_list))
    var data = List[Float64](capacity=size)
    for i in range(size):
        data.append(Float64(py=py_list[i]))

    # Native Mojo search
    var max_val = data[0]
    for i in range(1, len(data)):
        if data[i] > max_val:
            max_val = data[i]
    return max_val
```

### Conversion Helper Functions

```mojo
fn python_list_to_mojo[T: DType](py_list: PythonObject) -> List[Scalar[T]]:
    var size = Int(py=len(py_list))
    var result = List[Scalar[T]](capacity=size)

    for i in range(size):
        @parameter
        if T == DType.float64:
            result.append(Float64(py=py_list[i]))
        elif T == DType.int64:
            result.append(Int(py=py_list[i]))
        # ... other types

    return result^

# Usage
var py_data = Python.evaluate("[1.0, 2.0, 3.0]")
var mojo_data = python_list_to_mojo[DType.float64](py_data)
```

### When Conversion Isn't Needed

- Calling Python functions that return immediately
- One-time operations (not in loops)
- When data stays in Python ecosystem

---

## Error Handling Across Boundaries

### Anti-Pattern: Ignoring Python Errors

```mojo
# nocompile
from python import Python

fn unsafe_python_call():
    var json = Python.import_module("json")
    # Will crash if invalid JSON!
    var result = json.loads("invalid json")

fn unsafe_file_read(path: String) -> PythonObject:
    var builtins = Python.import_module("builtins")
    # Will crash if file doesn't exist!
    return builtins.open(path).read()

fn unsafe_division(a: PythonObject, b: PythonObject) -> PythonObject:
    # Will crash on division by zero!
    return a / b
```

### Correct: Proper Error Handling

```mojo
# nocompile
from python import Python

fn safe_json_parse(text: String) raises -> PythonObject:
    try:
        var json = Python.import_module("json")
        return json.loads(text)
    except e:
        raise "JSON parse error: " + String(e)

fn safe_file_read(path: String) raises -> String:
    try:
        var builtins = Python.import_module("builtins")
        var file = builtins.open(path, "r")
        var content = file.read()
        file.close()
        return String(content)
    except e:
        raise "Failed to read file '" + path + "': " + String(e)

fn safe_division(a: PythonObject, b: PythonObject) raises -> PythonObject:
    try:
        return a / b
    except:
        raise "Division failed - possible division by zero"
```

### Handling Specific Error Types

```mojo
# nocompile
fn robust_python_operation(data: PythonObject) raises -> PythonObject:
    var np = Python.import_module("numpy")

    try:
        # Multiple operations that could fail
        var arr = np.array(data)
        var result = np.mean(arr)
        return result
    except e:
        var error_str = String(e)
        if "could not convert" in error_str:
            raise "Invalid data type: expected numeric values"
        elif "empty" in error_str:
            raise "Cannot compute mean of empty array"
        else:
            raise "Numpy operation failed: " + error_str
```

### Fallback Patterns

```mojo
# nocompile
fn parse_with_fallback(text: String, default: PythonObject) -> PythonObject:
    try:
        var json = Python.import_module("json")
        return json.loads(text)
    except:
        return default  # Return default on any error

fn try_multiple_parsers(text: String) raises -> PythonObject:
    # Try JSON first
    try:
        var json = Python.import_module("json")
        return json.loads(text)
    except:
        pass

    # Fall back to YAML
    try:
        var yaml = Python.import_module("yaml")
        return yaml.safe_load(text)
    except:
        pass

    raise "Could not parse text as JSON or YAML"
```

---

## Common Patterns

### Pattern: Python for I/O, Mojo for Computation

**When:** You need Python libraries for loading data but want fast computation.

**Do:**
```mojo
# nocompile
fn process_image(path: String) raises -> List[Float32]:
    var np = Python.import_module("numpy")
    var PIL = Python.import_module("PIL.Image")

    # Python: Load image (one crossing)
    var img = PIL.open(path)
    var arr = np.array(img)

    # Convert at boundary
    var height = Int(py=arr.shape[0])
    var width = Int(py=arr.shape[1])
    var pixels = List[Float32](capacity=height * width * 3)

    for y in range(height):
        for x in range(width):
            for c in range(3):
                pixels.append(Float64(py=arr[y][x][c]))

    # Mojo: Process pixels (no crossings)
    return process_pixels_fast(pixels)
```

### Pattern: Lazy Python Initialization

**When:** Python is only needed for some code paths.

**Do:**
```mojo
# nocompile
struct LazyPython:
    var _np: PythonObject
    var _initialized: Bool

    fn __init__(out self):
        self._initialized = False
        self._np = PythonObject(None)

    fn numpy(mut self) -> PythonObject:
        if not self._initialized:
            self._np = Python.import_module("numpy")
            self._initialized = True
        return self._np

var py = LazyPython()

fn maybe_use_numpy(use_python: Bool, data: List[Float64]) -> Float64:
    if use_python:
        # Only initialize Python if needed
        var arr = py.numpy().array(data)
        return Float64(py=arr.mean())
    else:
        # Pure Mojo path
        var sum: Float64 = 0.0
        for item in data:
            sum += item
        return sum / Float64(len(data))
```

### Pattern: Bulk Transfer

**When:** Transferring large amounts of data between Python and Mojo.

**Do:**
```mojo
# nocompile
fn transfer_numpy_array(py_arr: PythonObject) -> List[Float64]:
    """Transfer numpy array to Mojo list efficiently."""
    # Get shape once
    var size = Int(py=py_arr.size)

    # Flatten if needed (one Python call)
    var flat = py_arr.flatten()

    # Pre-allocate with exact size
    var result = List[Float64](capacity=size)

    # Single loop transfer
    for i in range(size):
        result.append(Float64(py=flat[i]))

    return result^
```

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Simple math on Python data | Convert to Mojo, compute in Mojo | Type Conversion |
| Need matplotlib/visualization | Keep in Python | Minimize Crossings |
| Loading ML models | Use Python, convert weights once | Bulk Transfer |
| Processing in a loop | Convert at boundary, loop in Mojo | Type Conversion |
| Optional Python features | Lazy import with try/except | Import Patterns |
| Complex string operations | Consider Python regex | Minimize Crossings |

---

## Quick Reference

- **Minimize crossings**: Batch Python operations, convert once at boundaries
- **Import once**: Store imported modules, reuse throughout program
- **Convert early**: Transform Python objects to Mojo types before loops
- **Handle errors**: Wrap Python calls in try/except blocks
- **10-1000x speedup**: Achievable by moving computation to pure Mojo
- **GIL**: Release during long Mojo computations (see [`ffi.md`](ffi.md))

---

## Common Errors

> **List literal gotcha:** Mojo `[1, 2, 3]` creates an `InlineArray`, NOT a Python list. To create a Python list from Mojo, use `Python.evaluate("[1, 2, 3]")` or build it incrementally with `var lst = Python.evaluate("[]"); lst.append(1)`.

| Error | Cause | Fix |
|-------|-------|-----|
| `PythonObject is None` | Failed import or attribute access | Check module exists; use `try/except` around imports |
| `type conversion failed` | Incompatible Python→Mojo type | Use explicit conversion: `Int(py=py_obj)`, `String(py_obj)` |
| `GIL not held` | Calling Python without GIL | Ensure GIL acquired before any Python calls |
| `numpy array not contiguous` | Non-contiguous memory layout | Call `np.ascontiguousarray()` before passing to Mojo |
| `module not found` | Python path not set | Set `PYTHONPATH` or use `sys.path.append()` |
| `slow Python interop` | Too many cross-boundary calls | Batch operations; convert to Mojo types early in loops |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **PythonObject** | `PythonObject` | Stable |
| **Import syntax** | `Python.import_module()` | Stable |
| **Type conversion** | `Int(py=py_obj)`, `String(py_obj)` | Stable |
| **Compile-time constants** | `alias` or `comptime` | Both work in v26.1+ |

**Example (v26.1+):**
```mojo
from python import Python, PythonObject

fn process_data() raises:
    var np = Python.import_module("numpy")

    # Compile-time constant (alias is deprecated; use comptime)
    comptime BATCH_SIZE = 1000

    var arr = np.zeros(BATCH_SIZE)
    var result: Float64 = Float64(py=arr.sum())
```

**Notes:**
- Use `comptime` for compile-time constants (`alias` is deprecated in nightly)
- Python interop APIs are stable across versions
- PythonObject handling and type conversions unchanged
- GIL management patterns remain the same

---

## Related Patterns

- [`ffi.md`](ffi.md) — GIL management and C FFI patterns
- [`perf-vectorization.md`](perf-vectorization.md) — Fast Mojo alternatives to numpy

---

## References

- [Mojo Python Interoperability](https://docs.modular.com/mojo/manual/python/)
