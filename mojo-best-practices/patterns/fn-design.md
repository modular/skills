---
title: Function Design Patterns
description: Comprehensive patterns for designing Mojo functions including argument conventions, keyword arguments, overloading, inlining, and target-specific code
impact: HIGH
category: fn
tags: [functions, arguments, ownership, overloading, inlining, performance, target-specific]
error_patterns:
  - "argument convention"
  - "cannot pass .* to"
  - "expected .* argument"
  - "mut"
  - "var"
scenarios:
  - "Design function API"
  - "Choose argument convention"
  - "Use @always_inline for hot path"
  - "Overload function for different types"
consolidates:
  - fn-argument-conventions.md
  - fn-keyword-args.md
  - fn-overloading.md
  - fn-vs-def.md
  - fn-inlining.md
  - fn-no-inline.md
  - fn-target-specific-code.md
---

# Function Design Patterns

**Category:** fn | **Impact:** HIGH

This pattern consolidates all function design best practices in Mojo, covering argument conventions (read, mut, var, ref), keyword-only arguments, function overloading, the distinction between `fn` and `def`, inlining strategies, and target-specific code patterns. Following these patterns ensures performant, maintainable, and idiomatic Mojo functions.

---

## Core Concepts

### Argument Conventions

Mojo provides several argument conventions. Choose based on whether you need to read, modify, or take ownership.

**The four main conventions:**

```mojo
# read (default): Immutable borrow, no copy
fn analyze(data: List[Int]) -> Int:
    # data is borrowed immutably
    # Cannot modify, no copy made
    var sum = 0
    for item in data:
        sum += item
    return sum

# mut: Mutable reference, can modify
fn sort_in_place(mut data: List[Int]):
    # Modifies the caller's list directly
    # Caller sees the changes
    # ... sorting logic ...
    pass

# var: Takes ownership (must transfer or copy)
fn consume_and_process(var data: List[Int]) -> Int:
    # Now owns data, can move/modify freely
    # Caller must use ^ to transfer: func(my_list^)
    sort_in_place(data)
    return data[0]

# ref: Generalized reference (returns reference tied to input lifetime)
fn first(ref items: List[Int]) -> ref [items] Int:
    return items[0]
```

> **Note**: The `owned` keyword is deprecated. Use `var` for ownership transfer arguments, and `deinit` for destructors/move constructors.

> **Critical**: `out` is a reserved keyword in Mojo. Do NOT use `out` as a parameter name. Use `result`, `dst`, `output`, or `target` instead:
> ```mojo
> # WRONG - will not compile:
> fn process(out: Buffer, input: Buffer):  # ERROR: expected argument name
>     pass
>
> # CORRECT:
> fn process(result: Buffer, input: Buffer):
>     pass
> ```

### fn vs def

`fn` provides stricter guarantees enabling better optimization. `def` offers Python-like ergonomics for rapid development and interoperability.

**Key differences:**

| Feature | `fn` | `def` |
|---------|------|-------|
| Error handling | Non-raising by default | Raising by default |
| Arguments | Immutable by default | Mutable by default |
| Optimization | Maximum | Limited |
| Use case | Performance-critical | Scripting, prototyping |

**Pattern:**

```mojo
# Use def for Python-like flexibility
def parse_config(path: String) -> Dict[String, String]:
    # Can raise without explicit declaration
    # Arguments are mutable by default
    var result = Dict[String, String]()
    # ... parsing logic ...
    return result

def quick_script():
    # Rapid development, less ceremony
    print("Hello")
    var data = fetch_data()
    process(data)

# Use fn for performance-critical code
fn compute_hash(data: String) -> UInt64:
    # Must handle all errors or declare 'raises'
    # Arguments are immutable by default
    var hash: UInt64 = 0
    for char in data.codepoints():
        hash = hash * 31 + UInt64(char.to_u32())
    return hash

fn dot_product(a: List[Float64], b: List[Float64]) -> Float64:
    # Compiler can fully optimize this
    var result: Float64 = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# fn with raises: Explicit about error possibility
fn read_file(path: String) raises -> String:
    # Clearly communicates that this can fail
    # Caller must handle or propagate
    pass
```

**Guidelines:**
- Use `fn` in libraries and performance-critical code
- Use `def` for top-level scripts and Python interop
- Use `fn raises` when errors are possible but performance matters

---

## Common Patterns

### Choosing the Right Argument Convention

**When:** Deciding how to pass arguments to functions

| Need | Convention | Example |
|------|------------|---------|
| Read without modifying | `read` (default) | `fn len(s: String) -> Int` |
| Modify caller's value | `mut` | `fn append(mut list: List[T], item: T)` |
| Take ownership | `var` | `fn consume(var data: Buffer)` |
| Return reference | `ref` | `fn get(ref list: List[T]) -> ref T` |

**Do:**
```mojo
# Read-only access
fn print_all(items: List[String]):  # Correct: immutable borrow
    for item in items:
        print(item)

# Modification needed
fn add_item(mut items: List[String], item: String):  # Correct: mutable reference
    items.append(item)
```

**Don't:**
```mojo
# nocompile
# Takes ownership when only reading needed
fn print_all(var items: List[String]):  # Wrong: unnecessarily takes ownership
    for item in items:
        print(item)

# Immutable when modification needed
fn add_item(items: List[String], item: String):  # Wrong: can't modify
    items.append(item)  # Error!
```

### Keyword-Only Arguments for Clarity

**When:** Functions with 3+ parameters, multiple same-type parameters, or boolean flags

**Do:**
```mojo
# nocompile
fn create_user(*, name: String, email: String, age: Int, active: Bool = True):
    pass

fn create_rectangle(*, x: Float64, y: Float64, width: Float64, height: Float64):
    pass

# Must use keywords - self-documenting and safe
create_user(name="Alice", email="alice@example.com", age=25)

# Clear which parameter is which
create_rectangle(x=10.0, y=20.0, width=100.0, height=50.0)
```

**Don't:**
```mojo
# nocompile
fn create_user(name: String, email: String, age: Int, active: Bool):
    pass

fn create_rectangle(x: Float64, y: Float64, width: Float64, height: Float64):
    pass

# Bug: arguments swapped - compiles but wrong!
create_user("alice@example.com", "Alice", 25, True)

# Which is x, which is width?
create_rectangle(10.0, 20.0, 100.0, 50.0)
```

**Mix positional and keyword-only:**

```mojo
# nocompile
fn connect(
    host: String,           # Required positional
    port: Int,              # Required positional
    *,                      # Everything after is keyword-only
    timeout: Float64 = 30.0,
    retries: Int = 3,
    ssl: Bool = True
):
    pass

# First two positional, rest must be keywords
connect("localhost", 8080, timeout=60.0, ssl=False)
```

### Function Overloading for Type-Specific Optimizations

**When:** Need specialized implementations for different types while maintaining consistent API

**Do:**
```mojo
# nocompile
# Generic fallback using traits
fn stringify[T: Stringable](value: T) -> String:
    return String(value)

# Optimized overload for Int (common case)
fn stringify(value: Int) -> String:
    if value == 0:
        return "0"
    # ... optimized int-to-string ...
    return String(value)

# Optimized overload for Bool
fn stringify(value: Bool) -> String:
    if value:
        return "true"
    return "false"
```

**Type-specific optimizations:**

```mojo
# nocompile
# Generic abs using trait
fn abs[T: Comparable](value: T) -> T:
    if value < T():
        return -value
    return value

# Optimized for Int - no comparison needed with bit manipulation
fn abs(value: Int) -> Int:
    var mask = value >> 63  # Sign bit
    return (value ^ mask) - mask

# Optimized for Float64 - use hardware instruction
fn abs(value: Float64) -> Float64:
    return value if value >= 0 else -value
```

**Overloading with different argument counts:**

```mojo
# nocompile
fn create_point() -> Point:
    return Point(0.0, 0.0)

fn create_point(x: Float64, y: Float64) -> Point:
    return Point(x, y)

fn create_point(other: Point) -> Point:
    return Point(other.x, other.y)
```

**Resolution rules:**
1. Exact type match preferred
2. Fewest implicit conversions
3. Non-variadic over variadic
4. Shortest parameter signature

### Inlining Strategies

**When:** Small hot functions need maximum performance

**@always_inline for hot paths:**

```mojo
@always_inline
fn square(x: Float64) -> Float64:
    return x * x

@always_inline
fn cube(x: Float64) -> Float64:
    return x * x * x

fn sum_of_powers(data: List[Float64]) -> Float64:
    var result: Float64 = 0.0
    for item in data:
        # No function call - code is inlined
        result += square(item) + cube(item)
    return result
```

**When to use `@always_inline`:**
- Small functions (1-5 lines)
- Functions called in tight loops
- Simple mathematical operations
- Accessor methods

**Inline variants:**

| Variant | Debug Info | Use Case |
|---------|------------|----------|
| `@always_inline` | Stripped | Hot path functions |
| `@always_inline("builtin")` | Stripped | Core operators, maximum perf |
| `@always_inline("nodebug")` | Preserved | Inlined but debuggable |

```mojo
# nocompile
# Standard: inline and strip debug info
@always_inline
fn fast_add(a: Int, b: Int) -> Int:
    return a + b

# Builtin: for core language operations, maximum optimization
@always_inline("builtin")
fn __add__(self, rhs: Int) -> Int:
    return Int(mlir_value=__mlir_op.`index.add`(
        self._mlir_value, rhs._mlir_value
    ))

# Nodebug: inline but preserve debug info for stack traces
@always_inline("nodebug")
fn __init__[T: Intable](out self, value: T):
    self = value.__int__()
```

**@no_inline for cold paths:**

```mojo
# nocompile
# Error handling - preserve stack traces
@no_inline
fn handle_error(message: String) raises:
    print("Error:", message)
    raise message

# Debugging - easier to set breakpoints
@no_inline
fn debug_checkpoint(name: String):
    print("Checkpoint:", name)

# Reduce binary bloat for large functions called rarely
@no_inline
fn complex_initialization():
    # Large initialization code that shouldn't be duplicated
    pass
```

**When to use `@no_inline`:**
- Error handlers (preserve stack traces)
- Debug helpers (easier to set breakpoints)
- Large init code (reduce binary size)
- Recursive functions (prevent blowup)

**Incorrect (inlining large error handlers):**

```mojo
# Without @no_inline, this gets duplicated everywhere
fn validate(value: Int) raises:
    if value < 0:
        # Large error handling code inlined at every call site
        print("Validation failed")
        print("Value:", value)
        print("Expected: non-negative integer")
        raise "validation error"
```

**Correct (prevent inlining for error path):**

```mojo
@no_inline
fn report_validation_error(value: Int) raises:
    print("Validation failed")
    print("Value:", value)
    print("Expected: non-negative integer")
    raise "validation error"

fn validate(value: Int) raises:
    if value < 0:
        report_validation_error(value)  # Call, not inline
```

### Target-Specific Code

**When:** Writing code that differs by hardware platform (GPU vendors, CPU architectures)

**Do:** Use explicit vendor checks

```mojo
# nocompile
# GOOD: Explicit check for each supported platform
@parameter
if is_nvidia_gpu():
    return "llvm.nvvm.barrier0"
elif is_amd_gpu():
    return "llvm.amdgcn.s.barrier"
else:
    # GOOD: Clear error for unsupported platforms
    return CompilationTarget.unsupported_target_error[
        "barrier intrinsic not supported on this target"
    ]()
```

**Don't:** Use fallthrough else assuming "not NVIDIA means AMD"

```mojo
# nocompile
# BAD: Assumes only two GPU vendors exist
@parameter
if is_nvidia_gpu():
    return "llvm.nvvm.barrier0"
else:
    # BAD: Silently assumes AMD, fails mysteriously on Intel/other
    return "llvm.amdgcn.s.barrier"
```

**Cross-platform fallback pattern:**

```mojo
# nocompile
# OK: Generic fallback that works everywhere
@always_inline("nodebug")
fn prefetch[...](ptr: UnsafePointer):
    @parameter
    if is_nvidia_gpu():
        # NVIDIA-specific prefetch
        inlined_assembly["prefetch.global.L2 [$0];", ...](ptr)
    else:
        # Generic LLVM intrinsic works on all platforms
        llvm_intrinsic["llvm.prefetch", NoneType](ptr, 0, 3, 1)
```

**Complete multi-platform example:**

```mojo
# nocompile
fn get_warp_size() -> Int:
    @parameter
    if is_nvidia_gpu():
        return 32
    elif is_amd_gpu():
        return 64  # AMD wavefront size
    elif is_cpu():
        return 1   # No SIMT on CPU
    else:
        constrained[False, "Unsupported target for get_warp_size()"]()
        return 0

fn shuffle_down[T: DType](val: Scalar[T], offset: Int) -> Scalar[T]:
    @parameter
    if is_nvidia_gpu():
        return __nvidia_shfl_down_sync(0xFFFFFFFF, val, offset)
    elif is_amd_gpu():
        return __amd_ds_swizzle(val, offset)
    else:
        # CPU fallback: no shuffle, return same value
        return val
```

**Checking for specific architectures:**

```mojo
# nocompile
fn use_tensor_cores[T: DType]() -> Bool:
    @parameter
    if is_nvidia_gpu():
        # Check for SM80+ (Ampere and later)
        if is_sm80() or is_sm90() or is_sm100():
            return True
        return False
    elif is_amd_gpu():
        # Check for CDNA architecture
        return is_mi100() or is_mi200() or is_mi300()
    else:
        return False
```

**Error handling patterns:**

```mojo
# nocompile
# Option 1: Compile-time constraint
@parameter
if is_nvidia_gpu():
    ...
elif is_amd_gpu():
    ...
else:
    constrained[False, "This kernel requires GPU"]()

# Option 2: Unsupported target error (from stdlib)
else:
    return CompilationTarget.unsupported_target_error[MyFunction]()

# Option 3: Runtime error (last resort)
else:
    abort("Unsupported target for this operation")
```

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Read-only access to data | Default (immutable borrow) | - |
| Modify caller's data | `mut` convention | - |
| Take ownership of data | `var` convention | [`memory-ownership.md`](memory-ownership.md) |
| Return reference to internal data | `ref` convention | [`memory-safety.md`](memory-safety.md) |
| Many parameters of same type | Keyword-only arguments (`*`) | - |
| Hot path, small function | `@always_inline` | - |
| Error handler, debug code | `@no_inline` | - |
| Performance-critical code | Use `fn` over `def` | - |
| Scripting, prototyping | Use `def` | - |
| GPU-specific code | Explicit vendor checks with `@parameter if` | [`gpu-fundamentals.md`](gpu-fundamentals.md) |

---

## Quick Reference

- **read (default)**: Immutable borrow, no copy - `fn f(data: List[Int])`
- **mut**: Mutable reference - `fn f(mut data: List[Int])`
- **var**: Takes ownership - `fn f(var data: List[Int])` (caller uses `^`)
- **ref**: Returns reference - `fn f(ref list: List[T]) -> ref [list] T`
- **Keyword-only**: After `*` - `fn f(*, name: String, age: Int)`
- **fn vs def**: Use `fn` for performance, `def` for flexibility
- **@always_inline**: Small hot functions (1-5 lines)
- **@no_inline**: Error handlers, debug code, large functions
- **Target-specific**: Always use explicit vendor checks, never fallthrough else

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `cannot pass argument by reference` | Using wrong convention for mutable param | Use `mut` for mutable borrow: `fn foo(mut x: T)` |
| `raises keyword required` | Function can raise but not marked | Add `raises` to signature: `fn foo() raises` |
| `unused parameter` | Parameter declared but never used | Prefix with `_`: `fn foo(_unused: Int)` |
| `@always_inline on recursive function` | Can't inline recursion | Remove `@always_inline`; use iteration or trampolines |
| `type mismatch in overload` | Overloads not distinguished by types | Ensure overloads differ in parameter types, not just names |
| `inout deprecated` | Using old `inout` keyword | Replace `inout` with `mut` for mutable references |
| `owned deprecated` | Using old `owned` keyword | Replace with `var` for ownership transfer, `deinit` for destructors |
| `failed to compile-time evaluate function call` | Default argument calls runtime function (e.g. `getenv`) | Use function overloads; construct the default in the function body |
| `unable to interpret call to unknown external function: getenv` | Default argument evaluates at compile time | Use function overloads instead of default arg calling syscalls |

### Default Arguments and Compile-Time Evaluation

Default argument expressions are evaluated at compile time (for type checking and
parametric instantiation via `TestSuite.discover_tests`). Any default that calls
a runtime function — `getenv`, file I/O, or any `external_call` — will fail.

**❌ WRONG:** Default argument calls `getenv` at compile time
```mojo
# nocompile - Demonstrates anti-pattern
fn connect(url: String, config: TlsConfig = TlsConfig()) raises -> Connection:
    # TlsConfig() calls getenv() — fails at compile time!
    ...
```

**✅ CORRECT:** Use overloads; construct the default value at runtime in the body
```mojo
fn connect(url: String) raises -> Connection:
    return _connect_impl(url, TlsConfig())   # TlsConfig() at runtime — fine

fn connect(url: String, config: TlsConfig) raises -> Connection:
    return _connect_impl(url, config)        # Caller-provided config

fn _connect_impl(url: String, config: TlsConfig) raises -> Connection:
    ...  # Actual implementation
```

**Symptoms:** `note: failed to compile-time evaluate function call`, `unable to interpret call to unknown external function: getenv`

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **Ownership parameter** | `var` (v26.1+) | Use `var` for ownership transfer in function params |
| **Mutable reference** | `mut` (v26.1+) | `inout` is deprecated |
| **Compile-time constants** | `alias` or `comptime` | Both work; compiler warns on `alias` in v26.1+; `comptime` is preferred |
| **Error handling** | `raises` | Unchanged across versions |

**Example (v26.1+):**
```mojo
# nocompile
# Ownership transfer (var keyword, v26.1+)
fn consume(var data: List[Int]) -> Int:
    return data[0]

# Mutable reference (mut keyword, v26.1+)
fn modify(mut data: List[Int]):
    data.append(42)

# Compile-time constant (both work)
comptime MAX_SIZE = 1024
comptime MAX_SIZE_2: Int = 2048
```

**Notes:**
- Use `var` for ownership transfer arguments, `deinit` for destructors/move constructors
- Use `mut` for mutable references
- Both `alias` and `comptime` work for compile-time constants in v26.1+
- `raises` syntax remains unchanged between versions
- Inlining decorators (`@always_inline`, `@no_inline`) are stable across versions

---

## Related Patterns

- [`struct-design.md`](struct-design.md) — Struct design patterns including methods
- [`memory-ownership.md`](memory-ownership.md) — Ownership transfer patterns
- [`gpu-fundamentals.md`](gpu-fundamentals.md) — GPU programming patterns

---

## References

- [Mojo Functions Documentation](https://docs.modular.com/mojo/manual/functions)
- [Mojo Ownership Documentation](https://docs.modular.com/mojo/manual/values/ownership)
- [Mojo Decorators - always_inline](https://docs.modular.com/mojo/manual/decorators/always-inline)
- [Modular Style Guide](https://github.com/modular/modular/blob/main/mojo/stdlib/docs/style-guide.md)
