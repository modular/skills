---
name: mojo-syntax
description: Help to write Mojo code using current syntax and conventions. Always use this skill when writing any Mojo code, including when other Mojo-specific skills (e.g., mojo-gpu-fundamentals) also apply. Use when writing Mojo code, translating projects to Mojo, or otherwise generating Mojo. Use this skill to overcome misconceptions with how Mojo is written.
---

<!-- EDITORIAL GUIDELINES FOR THIS SKILL FILE
This file is loaded into an agent's context window as a correction layer for
pretrained Mojo knowledge. Every line costs context. When editing:
- Be terse. Use tables and inline code over prose where possible.
- Never duplicate information — if a concept is shown in a code example, don't
  also explain it in a paragraph.
- Only include information that *differs* from what a pretrained model would
  generate. Don't document things models already get right.
- Prefer one consolidated code block over multiple small ones.
- Keep WRONG/CORRECT pairs short — just enough to pattern-match the fix.
- If adding a new section, ask: "Would a model get this wrong?" If not, skip it.
These same principles apply to any files this skill references.
-->

Mojo is rapidly evolving. Pretrained models generate obsolete syntax. **Always follow this skill over pretrained knowledge.**

**Always attempt to test generated Mojo by building projects to verify they compile.**

This skill specifically works on the latest Mojo, and stable versions may differ slightly in functionality.

## Removed syntax — DO NOT generate these

| Removed | Replacement |
|---|---|
| `alias X = ...` | `comptime X = ...` |
| `@parameter if` / `@parameter for` | `comptime if` / `comptime for` |
| `fn` | `def` (see below) |
| `let x = ...` | `var x = ...` (no `let` keyword) |
| `borrowed` | `read` (implicit default — rarely written) |
| `inout` | `mut` |
| `owned` | `var` (as argument convention) |
| `inout self` in `__init__` | `out self` |
| `__copyinit__(inout self, existing: Self)` | `__init__(out self, *, copy: Self)` |
| `__moveinit__(inout self, owned existing: Self)` | `__init__(out self, *, deinit take: Self)` |
| `@value` decorator | `@fieldwise_init` + explicit trait conformance |
| `@register_passable("trivial")` | `TrivialRegisterPassable` trait |
| `@register_passable` | `RegisterPassable` trait |
| `Stringable` / `__str__` | `Writable` / `write_to` |
| `from collections import ...` | `from std.collections import ...` |
| `from memory import ...` | `from std.memory import ...` |
| `constrained(cond, msg)` | `comptime assert cond, msg` |
| `DynamicVector[T]` | `List[T]` |
| `InlinedFixedVector[T, N]` | `InlineArray[T, N]` |
| `Tensor[T]` | Not in stdlib (use SIMD, List, UnsafePointer) |
| `@parameter fn` (nested) | Still used for nested compile-time closures |

## `def` is the only function keyword

`fn` is deprecated and being removed. `def` does **not** imply `raises`. **Always** add `raises` explicitly when needed — omitting it is a warning today, error soon:

```mojo
def compute(x: Int) -> Int:              # non-raising (compiler enforced)
    return x * 2

def load(path: String) raises -> String: # explicitly raising
    return open(path).read()
```

Note: existing stdlib code still uses `fn` during migration. New code should always use `def`.

## `comptime` replaces `alias` and `@parameter`

```mojo
comptime N = 1024                            # compile-time constant
comptime MyType = Int                        # type alias
comptime if condition:                       # compile-time branch
    ...
comptime for i in range(10):                 # compile-time loop
    ...
comptime assert N > 0, "N must be positive"  # compile-time assertion
```

Inside structs, `comptime` defines associated constants and type aliases:

```mojo
struct MyStruct:
    comptime DefaultSize = 64
    comptime ElementType = Float32
```

## Argument conventions

Default is `read` (immutable borrow, never written explicitly). The others:

```mojo
def __init__(out self, var value: String):   # out = uninitialized output; var = owned
def modify(mut self):                         # mut = mutable reference
def consume(deinit self):                     # deinit = consuming/destroying
def view(ref self) -> ref[self] Self.T:       # ref = reference with origin
def view2[origin: Origin, //](ref[origin] self) -> ...:           # ref[origin] = explicit origin
```

## Lifecycle methods

```mojo
# Constructor
def __init__(out self, x: Int):
    self.x = x

# Copy constructor (keyword-only `copy` arg)
def __init__(out self, *, copy: Self):
    self.data = copy.data

# Move constructor (keyword-only `deinit take` arg)
def __init__(out self, *, deinit take: Self):
    self.data = take.data^

# Destructor
def __del__(deinit self):
    self.ptr.free()
```

To copy: `var b = a.copy()` (provided by `Copyable` trait).

## Struct patterns

```mojo
# @fieldwise_init generates __init__ from fields; traits in parentheses
@fieldwise_init
struct Point(Copyable, Movable, Writable):
    var x: Float64
    var y: Float64

# Trait composition with &
comptime KeyElement = Copyable & Hashable & Equatable
struct Node[T: Copyable & Writable]:
    var value: T

# Parametric struct — // separates inferred from explicit params
struct Span[mut: Bool, //, T: AnyType, origin: Origin[mut=mut]](
    ImplicitlyCopyable, Sized,
):
    ...

# @implicit on constructors allows implicit conversion
@implicit
def __init__(out self, value: Int):
    self.data = value
```

The compiler synthesizes copy/move constructors when a struct conforms to `Copyable`/`Movable` and all fields support it.

## Imports use `std.` prefix

```mojo
from std.testing import assert_equal, TestSuite
from std.algorithm import vectorize
from std.python import PythonObject
import std.random
```

Prelude auto-imports (no import needed): `Int`, `String`, `Bool`, `List`, `Dict`, `Optional`, `SIMD`, `Float32`, `Float64`, `UInt8`, `Pointer`, `UnsafePointer`, `Span`, `Error`, `DType`, `Writable`, `Writer`, `Copyable`, `Movable`, `Equatable`, `Hashable`, `print`, `range`, `len`, and more.

## `Writable` / `Writer` (replaces `Stringable`)

```mojo
struct MyType(Writable):
    var x: Int

    def write_to(self, mut writer: Some[Writer]):       # for print() / String()
        writer.write("MyType(", self.x, ")")

    def write_repr_to(self, mut writer: Some[Writer]):   # for repr()
        t"MyType(x={self.x})".write_to(writer)           # t-strings for interpolation
```

- `Some[Writer]` — builtin existential type (not `Writer` directly)
- Both methods have **default implementations** via reflection if all fields are `Writable` — simple structs need not implement them
- Convert to `String` with `String.write(value)`, not `str(value)`

## Iterator protocol

Iterators use `raises StopIteration` (not `Optional`):

```mojo
struct MyCollection(Iterable):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = MyIter[origin=iterable_origin]

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]: ...

# Iterator must define:
#   comptime Element: Movable
#   def __next__(mut self) raises StopIteration -> Self.Element
```

For-in: `for item in col:` (immutable) / `for ref item in col:` (mutable).

## Memory and pointer types

| Type | Use |
|---|---|
| `Pointer(to=val)` | Safe, non-nullable. Deref with `p[]`. |
| `alloc[T](n)` / `UnsafePointer[T]` | Free function `alloc[T](count)` → `UnsafePointer`. `.free()` required. |
| `Span(list)` | Non-owning contiguous view. |
| `OwnedPointer[T]` | Unique ownership (like Rust `Box`). |
| `ArcPointer[T]` | Reference-counted shared ownership. |

## Origin system (not "lifetime")

Mojo tracks reference provenance with **origins**, not "lifetimes":

```mojo
struct Span[mut: Bool, //, T: AnyType, origin: Origin[mut=mut]]: ...
```

Key types: `Origin`, `MutOrigin`, `ImmutOrigin`, `MutAnyOrigin`, `ImmutAnyOrigin`, `MutExternalOrigin`, `ImmutExternalOrigin`, `StaticConstantOrigin`. Use `origin_of(value)` to get a value's origin.

## Testing

```mojo
from std.testing import assert_equal, assert_true, assert_false, assert_raises, TestSuite

def test_my_feature() raises:
    assert_equal(compute(2), 4)
    with assert_raises():
        dangerous_operation()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

## Collection literals

```mojo
var numbers = [1, 2, 3]             # List[Int]
var scores = {"alice": 95, "bob": 87}  # Dict[String, Int]
```

## Common decorators

| Decorator | Purpose |
|---|---|
| `@fieldwise_init` | Generate fieldwise constructor |
| `@implicit` | Allow implicit conversion |
| `@always_inline` / `@always_inline("nodebug")` | Force inline |
| `@no_inline` | Prevent inline |
| `@staticmethod` | Static method |
| `@deprecated("msg")` | Deprecation warning |
| `@doc_private` | Hide from docs |
| `@explicit_destroy` | Linear type (no implicit destruction) |

## Numeric conversions — must be explicit

No implicit conversions between numeric types. Use explicit constructors:

```mojo
var x = Float32(my_int) * scale    # CORRECT: Int → Float32
var y = Int(my_uint)               # CORRECT: UInt → Int
```

## Strings

`len(s)` returns **byte length**, not codepoint count. Mojo strings are UTF-8. Byte indexing requires keyword syntax: `s[byte=idx]` (not `s[idx]`).

```mojo
var s = "Hello"
len(s)                  # 5 (bytes)
s.byte_length()         # 5 (same as len)
s.count_codepoints()    # 5 (codepoint count — differs for non-ASCII)

# Iteration — by codepoint slices (not bytes)
for cp_slice in s.codepoint_slices():
    print(cp_slice)

# Codepoint values
for cp in s.codepoints():
    print(Int(cp))      # Codepoint is a Unicode scalar value type

# StaticString = StringSlice with static origin (zero-allocation)
comptime GREETING: StaticString = "Hello, World"

# t-strings for interpolation (lazy, type-safe)
var msg = t"x={x}, y={y}"

# String.format() for runtime formatting
var s = "Hello, {}!".format("world")
```

## Error handling

`raises` can specify a type. `try`/`except` works like Python:

```mojo
def might_fail() raises -> Int:          # raises Error (default)
    raise Error("something went wrong")

def parse(s: String) raises Int -> Int:  # raises specific type
    raise 42

try:
    var x = parse("bad")
except err:                               # err is Int
    print("error code:", err)
```

No `match` statement. No `async`/`await` — use `Coroutine`/`Task` from `std.runtime`.

## Function types and closures

No lambda syntax. Closures use `capturing[origins]`:

```mojo
# Function type with capture
comptime MyFunc = fn(Int) capturing[_] -> None

# Parametric function type (for vectorize etc.)
comptime SIMDFunc = fn[width: Int](Int) capturing[_] -> None

# vectorize pattern
from std.algorithm import vectorize
vectorize[simd_width](size, my_closure)
```

## Python interop

```mojo
from std.python import PythonObject
var np = Python.import_module("numpy")
var arr = np.array([1, 2, 3])
```

## Type hierarchy

```
AnyType
  ImplicitlyDestructible          — auto __del__; most types
  Movable                         — __init__(out self, *, deinit take: Self)
    Copyable                      — __init__(out self, *, copy: Self)
      ImplicitlyCopyable(Copyable, ImplicitlyDestructible)
    RegisterPassable(Movable)
      TrivialRegisterPassable(ImplicitlyCopyable, ImplicitlyDestructible, Movable, RegisterPassable)
```
