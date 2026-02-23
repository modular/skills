---
title: Error Handling Patterns
description: Comprehensive error handling in Mojo including raises annotation, try/except/finally, context managers, error messages, and debug assertions
impact: HIGH
category: error
tags: [errors, raises, try, except, finally, context-manager, debug_assert, safety]
error_patterns:
  - "unhandled exception"
  - "raises not declared"
  - "missing try block"
  - "resource leak"
  - "uncaught error"
scenarios:
  - "Add error handling to function"
  - "Use context manager for cleanup"
  - "Add debug assertions"
  - "Create custom error type"
  - "Handle errors gracefully"
consolidates:
  - error-raises-annotation.md
  - error-try-except-finally.md
  - error-context-managers.md
  - error-specific-messages.md
  - error-debug-assert.md
---

# Error Handling Patterns

**Category:** error | **Impact:** HIGH

Comprehensive error handling patterns for Mojo, covering compile-time error enforcement with `raises`, structured exception handling with `try/except/finally`, resource management with context managers, descriptive error messages, and development-time assertions with `debug_assert`.

---

## Core Concepts

### The raises Annotation

Explicitly annotate functions that can raise errors with `raises` so callers know to handle them. This provides compile-time enforcement of error handling.

**Key difference between `fn` and `def`:**
- `fn`: Non-raising by default, must declare `raises`
- `def`: Raising by default, errors propagate implicitly

**Pattern:**

```mojo
# nocompile
fn parse_int(s: String) raises -> Int:
    if len(s) == 0:
        raise "empty string cannot be parsed as integer"

    var result: Int = 0
    var negative = False
    var bytes = s.as_bytes()

    for i in range(len(bytes)):
        var byte = bytes[i]
        if i == 0 and byte == ord("-"):
            negative = True
            continue

        var digit = Int(byte) - Int(ord("0"))
        if digit < 0 or digit > 9:
            raise "invalid character at position " + String(i)

        result = result * 10 + digit

    return -result if negative else result

fn divide(a: Float64, b: Float64) raises -> Float64:
    if b == 0:
        raise "division by zero"
    return a / b
```

### Caller Error Handling Options

```mojo
# nocompile
fn main():
    # Option 1: Handle with try-except
    try:
        var num = parse_int("123")
        print(num)
    except e:
        print("Parse error:", e)

    # Option 2: Propagate with raises
    fn process_input(s: String) raises -> Int:
        return parse_int(s) * 2  # Propagates if parse fails

    # Option 3: Use try-except to provide default
    var value: Int = 0
    try:
        value = parse_int("abc")
    except:
        pass  # Keep default value of 0
```

---

## Common Patterns

### try-except-else-finally Structure

**When:** You need fine-grained control over error handling with cleanup.

**Do:**
```mojo
# nocompile
fn safe_divide(a: Float64, b: Float64) -> Optional[Float64]:
    try:
        if b == 0:
            raise "division by zero"
        var result = a / b
    except e:
        # Log or handle the error
        print("Division failed:", e)
        return None
    else:
        # Success path - only runs if no exception
        print("Division successful:", result)
        return result
    finally:
        # Cleanup - always runs
        print("Division operation completed")
```

**When to use each clause:**
- `try`: Code that might raise
- `except`: Handle errors
- `else`: Only runs if no exception was raised
- `finally`: Always runs, even if there's a return or exception

### Resource Cleanup with finally

**When:** Multiple resources need guaranteed cleanup.

**Do:**
```mojo
# nocompile
fn transfer_data(src: String, dst: String) raises:
    var src_file = open(src, "r")
    var dst_file: Optional[File] = None

    try:
        dst_file = open(dst, "w")
        var data = src_file.read()
        dst_file.value().write(data)
    except e:
        print("Transfer failed:", e)
        raise
    finally:
        # Always close files, even on error
        src_file.close()
        if dst_file:
            dst_file.value().close()
```

### Nested try Blocks

**When:** Different error types need different handling at different levels.

**Do:**
```mojo
# nocompile
fn complex_operation() raises -> Result:
    try:
        var config = load_config()  # May raise ConfigError

        try:
            var conn = connect(config)  # May raise ConnectionError
            try:
                return conn.execute()  # May raise ExecutionError
            finally:
                conn.close()
        except e:
            print("Connection failed, using fallback")
            return fallback_operation()

    except e:
        raise "Invalid configuration: " + str(e)
```

---

## Context Managers

Context managers ensure resources are properly cleaned up, even when errors occur. Use the `with` statement for any resource that needs cleanup.

### Basic Context Manager Usage

**Don't:**
```mojo
# nocompile
fn process_file(path: String) raises:
    var file = open(path, "r")
    var content = file.read()
    # If an error occurs here, file may not be closed!
    process(content)
    file.close()  # May never execute if process() raises
```

**Do:**
```mojo
# nocompile
fn process_file(path: String) raises:
    with open(path, "r") as file:
        var content = file.read()
        process(content)
    # file is automatically closed, even if process() raises
```

### Creating Custom Context Managers

```mojo
struct Timer:
    var start_time: Float64
    var name: String

    fn __init__(out self, name: String):
        self.name = name
        self.start_time = 0

    fn __enter__(mut self) -> Self:
        self.start_time = time.now()
        return self

    fn __exit__(mut self):
        var elapsed = time.now() - self.start_time
        print(self.name, "took", elapsed, "seconds")

# Usage
fn benchmark():
    with Timer("computation") as t:
        # ... do work ...
        pass
    # Automatically prints elapsed time
```

### Context Manager with Error Handling

```mojo
# nocompile
struct Transaction:
    var conn: Connection

    fn __enter__(mut self) -> Self:
        self.conn.begin()
        return self

    fn __exit__(mut self):
        self.conn.commit()

    fn __exit__(mut self, error: Error) raises -> Bool:
        self.conn.rollback()
        return False  # Re-raise the error
```

---

## Error Message Best Practices

Error messages should include context about what went wrong, what was expected, and ideally how to fix it.

### Error Message Checklist

1. **What happened**: Describe the error condition
2. **What was expected**: State the valid range/format
3. **Context**: Include relevant values
4. **How to fix** (if possible): Suggest corrective action

**Don't:**
```mojo
# nocompile
fn validate_config(config: Config) raises:
    if config.port < 0:
        raise "invalid"

    if config.timeout <= 0:
        raise "error"

    if len(config.host) == 0:
        raise "bad config"
```

**Do:**
```mojo
# nocompile
fn validate_config(config: Config) raises:
    if config.port < 0 or config.port > 65535:
        raise "Invalid port " + String(config.port) + ": must be between 0 and 65535"

    if config.timeout <= 0:
        raise "Invalid timeout " + String(config.timeout) + "s: must be positive"

    if len(config.host) == 0:
        raise "Host cannot be empty: provide a hostname or IP address"
```

### Position Information for Parsing Errors

```mojo
# nocompile
fn parse_number(text: String, pos: Int) raises -> Tuple[Int, Int]:
    var start = pos
    while pos < len(text) and text[pos].isdigit():
        pos += 1

    if pos == start:
        raise "Expected digit at position " + String(pos) + ", found '" + text[pos] + "'"

    return (int(text[start:pos]), pos)
```

### Custom Error Types

With typed raises (v26.1+), you can define custom error types and use them in `raises` clauses:

```mojo
# nocompile
@fieldwise_init
struct ValidationError(Writable, Stringable):
    """Custom error type for validation failures."""
    var field: String
    var message: String

    fn __str__(self) -> String:
        return "Validation error in '" + self.field + "': " + self.message

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Validation error in '", self.field, "': ", self.message)

# Use with typed raises
fn validate(config: Config) raises ValidationError:
    if len(config.host) == 0:
        raise ValidationError("host", "cannot be empty")
```

---

## debug_assert Usage

`debug_assert` checks conditions during development but is compiled out in release builds. Use it for internal invariant checking that should never fail if the code is correct.

### Basic Usage

**Don't:**
```mojo
# nocompile
fn get_element(ref self, index: Int) -> ref [self] T:
    # This check runs even in release builds - performance cost!
    if index < 0 or index >= len(self):
        abort("Index out of bounds")
    return self._data[index]
```

**Do:**
```mojo
# nocompile
fn get_element(ref self, index: Int) -> ref [self] T:
    # Only checked in debug mode - zero cost in release
    debug_assert[assert_mode="safe"](
        0 <= index < len(self),
        "Index out of bounds: ", index, " not in [0, ", len(self), ")"
    )
    return self._data[index]
```

### debug_assert Modes

```mojo
# nocompile
# "safe" mode (default): enabled in debug builds, disabled in release
debug_assert[assert_mode="safe"](condition, message)

# "warn" mode: prints warning but continues execution
debug_assert[assert_mode="warn"](condition, "Warning: condition failed")

# "none" mode: assertion completely removed (explicit no-op)
debug_assert[assert_mode="none"](condition, message)
```

### Examples by Category

```mojo
# Internal state invariants
fn pop(mut self) -> T:
    debug_assert(len(self) > 0, "pop from empty container")
    self._size -= 1
    return (self._data + self._size).take_pointee()

# Pointer validity
fn process(ptr: UnsafePointer[Int]):
    debug_assert(ptr != UnsafePointer[Int](), "null pointer")
    var value = ptr[]

# Algorithm preconditions
fn binary_search(arr: Span[Int], target: Int) -> Optional[Int]:
    debug_assert(_is_sorted(arr), "binary_search requires sorted input")
    # ... implementation ...

# Memory alignment
fn load_simd[width: Int](ptr: UnsafePointer[Float32]) -> SIMD[DType.float32, width]:
    debug_assert(
        Int(ptr) % (width * 4) == 0,
        "pointer not aligned to ", width * 4, " bytes"
    )
    return ptr.load[width=width]()
```

### Conditional Debug Code

```mojo
# nocompile
from builtin.debug_assert import _is_debug_build

fn complex_operation(data: List[Int]):
    # Expensive validation only in debug
    @parameter
    if _is_debug_build():
        for i in range(len(data)):
            debug_assert(data[i] >= 0, "Negative value at index ", i)

    # Actual operation
    _process_data(data)
```

### Common Mistakes

```mojo
# nocompile
# BAD: Side effects in assertion (removed in release!)
debug_assert(list.pop() > 0, "value must be positive")
#           ^^^^^^^^^^^ This pop() is removed in release!

# GOOD: Side effects outside assertion
var value = list.pop()
debug_assert(value > 0, "value must be positive")

# BAD: Using debug_assert for user input validation
debug_assert(user_input.isdigit(), "must be numeric")
# This check is removed in release - security issue!

# GOOD: Use proper validation for user input
if not user_input.isdigit():
    raise Error("Input must be numeric")
```

---

## Decision Guide

| Scenario | Approach | Notes |
|----------|----------|-------|
| Function can fail with recoverable error | Use `raises` annotation | Compile-time enforcement |
| Need guaranteed cleanup | Use `finally` or context manager | Prevents resource leaks |
| Multiple resources to manage | Use `with` statement | Automatic cleanup |
| Internal invariant checking | Use `debug_assert` | Zero cost in release |
| User input validation | Use `raises` with descriptive message | Never use debug_assert |
| Unrecoverable error | Use `abort()` | Memory corruption, etc. |

---

## Quick Reference

- **`raises`**: Mark functions that can fail; `fn` is non-raising by default, `def` is raising by default
- **`try-except-else-finally`**: Full control over error handling and cleanup
- **Context managers**: Use `with` for automatic resource cleanup; implement `__enter__`/`__exit__`
- **Error messages**: Include what happened, what was expected, context, and how to fix
- **`debug_assert`**: Development-only checks; compiled out in release builds
- **Never put side effects in `debug_assert`**: They are removed in release builds

---

## Version-Specific Features

### v26.1+: Typed Raises

Declare specific error types in function signatures for compile-time safety.

```mojo
# nocompile
fn foo() raises CustomError -> Int:
    if some_condition:
        raise CustomError("something went wrong")
    return 42
```

**How it works:** Raised errors compile into an efficient alternate return value (no stack unwinding).

```mojo
# nocompile
# Compiler generates efficient code similar to:
# struct Result { bool is_error; union { Int error; Float32 value; }; }
fn compute() raises Int -> Float32:
    if invalid:
        raise 42  # Return error path
    return 3.14   # Return success path
```

**Type inference in except blocks:**
```mojo
# nocompile
try:
    print(foo())
except err:  # "err" is automatically typed as CustomError
    print(err)
```

**Generic error propagation:**
```mojo
# nocompile
fn parametric_raise_example[ErrorType: AnyType](
    fp: fn () raises ErrorType
) raises ErrorType:
    fp()  # Propagate error generically
```

**Non-raising inference with Never:**
```mojo
# nocompile
fn doesnt_raise(): pass

# Mojo knows this doesn't raise because doesnt_raise raises Never
parametric_raise_example(doesnt_raise)  # No try block needed!
```

**Context managers with typed throws:**
```mojo
struct MyContextMgr:
    fn __enter__(self): ...
    fn __exit__(self): ...  # Normal exit

    # Error exit - error type is inferred
    fn __exit__[ErrType: AnyType](self, err: ErrType) -> Bool: ...
```

**Benefits:**
- Compile-time type checking
- GPU/embedded safe (no stack unwinding)
- Efficient: works as alternate return value

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `function can raise but caller doesn't handle` | Missing `raises` keyword on caller | Add `raises` to caller signature or wrap in `try/except` |
| `cannot use 'raise' in non-raising function` | Function not marked as `raises` | Add `raises` to function signature: `fn foo() raises` |
| `unhandled error type` | Catching wrong error type | Check error type hierarchy; use `except e:` for all errors |
| `resource leaked on error` | No cleanup on exception path | Use `try/finally` or struct destructor for cleanup |
| `Optional has no value` | Calling `.value()` on empty Optional | Check with `if opt:` before accessing, or use `.or_else(default)` |
| `typed error not raised` (v26.1+) | Wrong error type in `raises` clause | Ensure error type matches declaration: `raises CustomError` |
| `value of type 'Error' cannot be implicitly copied` | Re-raising caught error without `^` | Use `raise e^` to transfer ownership |

### Re-Raising a Caught Error

When you catch an error and want to re-raise it, you must use `^` to transfer
ownership. `Error` does not conform to `ImplicitlyCopyable`.

```mojo
# nocompile
try:
    risky_call()
except e:
    log_error(e)      # Can still read e before transferring
    raise e^           # Transfer ownership — e is consumed here
    # raise e          # WRONG: Error cannot be implicitly copied
```

---

## Related Patterns

- [`debug-debugging.md`](debug-debugging.md) — Debugging numerical accuracy issues
- [`memory-ownership.md`](memory-ownership.md) — Proper resource management in structs

---

## References

- [Mojo Error Handling](https://docs.modular.com/mojo/manual/errors)
- [Mojo Manual](https://docs.modular.com/mojo/manual/)
