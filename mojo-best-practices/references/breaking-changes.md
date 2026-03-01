# Mojo Breaking Changes Reference

> Updated for Mojo v26.2.0.dev nightly and v26.1.0.0.0 stable (2026-01-29)

## Version Compatibility

| Feature | v25.7 (old stable) | v26.1.0.0.0 (current stable) | Nightly (v26.2+) |
|---------|-------------------|------------------------|------------------|
| Constants | `alias` only | `comptime` (preferred), `alias` works | `comptime` preferred |
| GPU imports | `from gpu.id import ...` | `from gpu import ...` (new) | `from gpu import ...` |
| AddressSpace | `_GPUAddressSpace` | `AddressSpace` (unified) | `AddressSpace` |

---

## v26.1.0.0.0 Stable Changes (2026-01-29)

### `comptime` Keyword (Preferred over `alias`)

The `comptime` keyword is now the **preferred syntax** for compile-time constants. `alias` still works but future updates will migrate to `comptime`.

```mojo
# v26.1.0.0.0+ preferred:
comptime PI: Float64 = 3.14159
comptime BUFFER_SIZE = 1024
comptime MyType[T: AnyType] = T

# Still works (backward compatible):
alias PI: Float64 = 3.14159  # Will eventually be deprecated
```

### GPU Import Path Deprecations

The following import paths are **deprecated** in v26.1.0.0.0 (still work with warnings):

| Old (deprecated) | New (recommended) |
|-----------------|-------------------|
| `from gpu.id import block_idx, thread_idx` | `from gpu import block_idx, thread_idx` |
| `from gpu.warp import shuffle_down, lane_id` | `from gpu.primitives.warp import shuffle_down, lane_id` (also available as `from gpu import lane_id`) |
| `from gpu.block import ...` | `from gpu.primitives.block import ...` |
| `from gpu.cluster import cluster_sync` | `from gpu.primitives.cluster import cluster_sync` |
| `from gpu.grid_controls import ...` | `from gpu.primitives.grid_controls import ...` |
| `from gpu.mma import mma` | `from gpu.compute.mma import mma` |
| `from gpu.mma_sm100 import UMMAKind` | `from gpu.compute.arch.mma_nvidia_sm100 import UMMAKind` |
| `from gpu.tcgen05 import tcgen05_alloc` | `from gpu.compute.arch.tcgen05 import tcgen05_alloc` |
| `from gpu.semaphore import Semaphore` | `from gpu.sync.semaphore import Semaphore` |

**Recommended pattern:**
```mojo
# v26.1.0.0.0+ recommended:
from gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer
from gpu.primitives.warp import shuffle_down, lane_id, WARP_SIZE
from gpu.primitives.cluster import cluster_sync
from gpu.compute.mma import mma
```

### AddressSpace Unification

`_GPUAddressSpace` is removed. Use `AddressSpace` directly (part of prelude):

```mojo
# v26.1.0.0.0+: AddressSpace is in prelude, no import needed
from memory import stack_allocation

var shared = stack_allocation[1024, Float32, address_space=AddressSpace.SHARED]()
```

### Parameter Type Changes

| Type | Old | New |
|------|-----|-----|
| `BitSet[size]` | `UInt` parameter | `Int` parameter |
| `String.capacity()` | returns `UInt` | returns `Int` |
| `String.reserve()` | takes `UInt` | takes `Int` |
| `List(length, fill)` | `UInt` length | `Int` length |

---

## Changes in v25.x (Now in Both Stable and Nightly)

| Deprecated/Changed | Replacement |
|------------|-------------|
| `@value` | `@fieldwise_init` + `Copyable, Movable` |
| `owned` | `var` (for ownership), `deinit` (in lifecycle methods) |
| Unqualified `T` | `Self.T` in struct bodies |
| `Stringable` for print | `Writable` with `write_to` |
| `free(ptr)` function | `ptr.free()` method |
| `alloc[T](n)` from `memory.unsafe_pointer` | `UnsafePointer[T].alloc(n)` static method |
| `UnsafePointer[T]` in structs | `UnsafePointer[mut=True, type=T, origin=MutAnyOrigin]` |
| `memcpy(dst, src, n)` | `memcpy(dest=dst, src=src, count=n)` |
| `s[i]` on String | `s.as_bytes()[i]` for byte access |
| `str(value)` | `String(value)` for conversions |
| `math.pow(a, b)` | `from math import pow` (available in current stdlib) |
| `math.sigmoid(x)` | Custom implementation: `1.0 / (1.0 + exp(-x))` |
| `borrowed` keyword | Does not exist - use `read` (immutable) or `mut` (mutable) |
| `ref self` in `__getitem__` | `mut self` for mutable self reference |
| `List[T]` elements | Elements must implement `Copyable` trait to be stored |


## vnightly Changes (v26.2.0.dev2026022717+)

### Deprecations (NOW ACTIVE in Nightly)

| Deprecated | Replacement | Status |
|------------|-------------|--------|
| `@register_passable("trivial")` | `TrivialRegisterPassable` trait | **DEPRECATED** - use trait instead |
| `@register_passable` (non-trivial) | `RegisterPassable` trait | **DEPRECATED** - use trait instead |
| `@parameter if` / `@parameter for` | `comptime if` / `comptime for` | **DEPRECATED** - legacy forms still accepted |
| `__comptime_assert` | `comptime assert` | **DEPRECATED** - use keyword form |
| `String.as_string_slice()` | `StringSlice(str)` constructor | **DEPRECATED** |
| `String.as_string_slice_mut()` | N/A | **REMOVED** - no longer exists |
| `__reversed__()` on String types | `codepoint_slices_reversed()` | **DEPRECATED** - use Unicode-aware method |
| `Stringable` / `Representable` traits | `Writable` trait | **DEPRECATED** - `Tuple`, `Variant`, `Optional` now conform to `Writable` |
| `Int` to `SIMD` implicit conversion | Explicit `SIMD[dtype, 1](int_val)` | **DEPRECATED** |
| `**_` / `*_` in parameter binding | `...` | **REMOVED** - use `...` instead |

### Lifecycle Method Renames (NIGHTLY)

Move and copy constructors have been renamed. Legacy names still compile but are deprecated:

| Old (deprecated) | New (preferred) | Notes |
|-------------------|-----------------|-------|
| `fn __moveinit__(out self, deinit take: Self)` | `fn __init__(out self, *, take: Self)` | Move constructor; accepts `deinit` or `var` for "take" |
| `fn __copyinit__(out self, copy: Self)` | `fn __init__(out self, *, copy: Self)` | Copy constructor |
| `comptime __moveinit__is_trivial: Bool` | `comptime __move_ctor_is_trivial: Bool` | Trivial move flag |
| `comptime __copyinit__is_trivial: Bool` | `comptime __copy_ctor_is_trivial: Bool` | Trivial copy flag |

### Keyword Changes (NIGHTLY)

| Old | New | Notes |
|-----|-----|-------|
| `owned` keyword | **REMOVED entirely** | Use `var` (ownership) or `deinit` (lifecycle) |
| `@parameter if cond:` | `comptime if cond:` | Legacy `@parameter if` still accepted |
| `@parameter for i in range(N):` | `comptime for i in range(N):` | Legacy `@parameter for` still accepted |
| `__comptime_assert(cond, msg)` | `comptime assert cond, msg` | Finalizes previously unstable syntax |

### Trait Changes (NIGHTLY)

- `trait` declarations **no longer auto-inherit** `ImplicitlyDestructible` -- must opt-in explicitly
- `Span[T]` no longer restricted to `Copyable` types

### Module Restructuring (NIGHTLY)

| Old | New | Notes |
|-----|-----|-------|
| `from builtin.math import ...` | `from math import ...` | `builtin.math` merged into `math` |
| `from sys.ffi import ...` | `from ffi import ...` | `ffi` is now a top-level module |

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| Slice literals in subscripts | No `__slice_literal__` argument | Add `__slice_literal__: () = ()` to range constructors | Required to disambiguate slices from collections |
| Math functions constraints | `__comptime_assert` in function body | `where dtype.is_floating_point()` clause | Better error messages at call site |
| `InlineArray` construction | `InlineArray[T, N](1, 2, 3)` | `var a: InlineArray[T, N] = [1, 2, 3]` | Must use bracket literal syntax |
| `String.ljust` / `String.rjust` | `str.ljust(width, fill)` | `str.ascii_ljust(width, fill)` | Renamed for clarity - ASCII-specific padding |
| `StringSlice.ljust` / `StringSlice.rjust` | `slice.ljust(width, fill)` | `slice.ascii_ljust(width, fill)` | Renamed for clarity - ASCII-specific padding |
| `StaticString.ljust` / `StaticString.rjust` | `static.ljust(width, fill)` | `static.ascii_ljust(width, fill)` | Renamed for clarity - ASCII-specific padding |
| `String.resize` behavior | Allowed truncating codepoints | Panics if truncating codepoint | Safety improvement - prevents invalid UTF-8 |
| `String.resize` fill_byte | Accepted values >= 128 | Panics if fill_byte >= 128 | Safety improvement - prevents invalid UTF-8 |
| String/StringSlice subscripting | Panic if index in middle of codepoint | Unconditional panic on invalid UTF-8 index | Use `.as_bytes()[...]` for old behavior |
| `StringSlice[byte=]` subscripting | Returns `String` | Returns `StringSlice` | Consistent with range-based subscripting |
| String/StringSlice byte subscripting | Returns single byte (may be invalid UTF-8) | Returns entire Unicode codepoint | Prevents invalid UTF-8 generation |
| `Dict` implementation | Hash table | Swiss Table | `power_of_two_initial_capacity` renamed to `capacity` |
| `Set.pop()` | FIFO order | LIFO order | Behavior change |
| `Dict.EMPTY` / `Dict.REMOVED` | Comptime aliases | **REMOVED** | No longer available |
| `Int.__truediv__` | Returns `Float64` | Returns `Int` (truncating) | Breaking behavior change |

### New Features

| Feature | Description |
|---------|-------------|
| `codepoint_slices_reversed()` | Added to `String`, `StringSlice`, and `StringLiteral` for Unicode-aware reverse iteration |
| `StringSlice` mutability | Constructor now propagates mutability from source String reference |
| `comptime if` / `comptime for` | Preferred replacements for `@parameter if/for` (legacy still accepted) |
| `comptime assert` | Finalized syntax for compile-time assertions |
| `@align(N)` decorator | Explicit struct alignment control |
| `UnsafeUnion[*Ts]` | C-style untagged unions for FFI interop |
| `uninit_move_n()`, `uninit_copy_n()`, `destroy_n()` | Bulk memory operations in memory module |
| Iterator combinators | `cycle()`, `take_while[]`, `drop_while[]` added to iterators |
| `ConditionalType` | New `utils/type_functions` module with conditional type selection |


## v24.1 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `YY.MAJOR.MINOR` | - | - | Mojo is now bundled with the MAX platform! As such, the M... |
| `vectorize` | - | - | The `vectorize` signatures have changed with the closure ... |
| `unroll` | - | - | The `unroll` signatures have changed with the closure `fu... |
| `capacity` | - | - | The `DynamicVector(capacity: Int)` constructor has been c... |
| `Variant.get[T]()` | - | - | `Variant.get[T]()` now returns a `Reference` to the value... |
| `str.lower()` | `toupper()` | - | The `String` methods `tolower()` and `toupper()` have bee... |
| `raises` | - | - | #1621 - VS Code does not highlight `raises` and `capturin... |
| `SIMD.shift_left` | - | - | #1549 - Fixed an issue when the shift amount is out of ra... |
| `vectorize_unroll` | - | - | `vectorize_unroll` has been removed, and `vectorize` now ... |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `y.__contains__(x)` | Mojo now supports the `x in y` expression as syntax sugar... | |
| `DynamicVector` | `DynamicVector` now supports iteration. Iteration values ... | |
| `DynamicVector` | `DynamicVector` now has `reverse()` and `extend()` methods. | |
| `slice` | The standard library `slice` type has been renamed to `Sl... | |
| `slice` | "Slice" syntax in subscripts is no longer hard coded to t... | |
| `__refitem__()` | The `__refitem__()` accessor method may now return a `Ref... | |
| `AnyPointer.move_into()` | Added `AnyPointer.move_into()` method, for moving a value... | |
| `hex()` | Added built-in `hex()` function, which can be used to for... | |
| `stat()` | The `os` package now contains the `stat()` and `lstat()` ... | |
| `os.path` | A new `os.path` package now allows you to query propertie... | |
| `PathLike` | The `os` package now has a `PathLike` trait. A struct con... | |
| `pathlib.Path` | The `pathlib.Path` now has functions to query properties ... | |
| `listdir()` | The `listdir()` method now exists on `pathlib.Path` and a... | |
| `find()` | The `find()`, `rfind()`, `count()`, and `__contains__()` ... | |
| `breakpoint()` | Breakpoints can now be inserted programmatically within t... | |
| `__origin_of(expr)` | A new magic `__origin_of(expr)` call will yield the lifet... | |
| `__type_of(expr)` | A new magic `__type_of(expr)` call will yield the type of... | |

## v24.2 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `__str__()` | - | - | Structs and other nominal types are now allowed to implic... |
| `**kwargs` | - | - | Mojo now has support for variadic keyword arguments, ofte... |
| `*_` | - | - | The `*_` expression in parameter expressions is now requi... |
| `IntableRaising` | - | - | String types all conform to the `IntableRaising` trait. T... |
| `simd_load()` | - | - | The `simd_load()`, `simd_store()`, `aligned_simd_load()`,... |
| `StaticTuple` | - | - | `StaticTuple` parameter order has changed to `StaticTuple... |
| `elementwise()` | - | - | The signature of the `elementwise()` function has been ch... |
| `PythonObject.__iter__()` | - | - | `PythonObject.__iter__()` now works correctly on more typ... |
| `utils.list` | - | - | `utils.list`, including the `Dim` and `DimList` types, ha... |
| `rand()` | - | - | The `rand()` and `randn()` functions from the `random` pa... |
| `trap()` | - | - | The `trap()` function has been renamed to `abort()`.  It ... |
| `isinf()` | - | - | The `isinf()` and `isfinite()` methods have been moved fr... |
| `mojo.lsp.includeDirs` | - | - | The Mojo LSP server now allow you to specify additional s... |
| `__get_address_as_lvalue` | - | - | The `__get_address_as_lvalue` magic function has been rem... |
| `memcpy` | - | - | The type parameter for the `memcpy` function is now autom... |
| `print_no_newline()` | - | - | As mentioned previously, the `print_no_newline()` functio... |
| `@always_inline` | - | - | #951 - Functions that were both `async` and `@always_inli... |
| `Optional[T].or_else()` | - | - | #1945 - `Optional[T].or_else()` should return `T` instead... |
| `math.copysign` | - | - | #1940 - Constrain `math.copysign` to floating point or in... |
| `print` | - | - | #1838 - Variadic `print` does not work when specifying `e... |
| `SIMD.reduce` | - | - | #1826 - The `SIMD.reduce` methods correctly handle edge c... |
| `List.push_back()` | - | - | `List.push_back()` has been removed.  Please use the `app... |
| `min_or_neginf()` | - | - | The functions `max_or_inf()`, `min_or_neginf()` have been... |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `collections.OptionalReg` | Added a new `collections.OptionalReg` type, a register-pa... | |
| `buffer` | `Buffer`, `NDBuffer`, and friends have moved from the `me... | |

## v24.3 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `initialize_pointee_copy` | - | - | `initialize_pointee_copy` |
| `initialize_pointee_move` | - | - | `initialize_pointee_move` |
| `move_from_pointee()` | - | - | `move_from_pointee()` |
| `move_pointee` | - | - | `move_pointee` |
| `unsafe.bitcast()` | - | - | All of the pointer types received some cleanup to make th... |
| `__source_location()` | - | - | Mojo now allows users to capture the source location of c... |
| `pop(index)` | - | - | `pop(index)` for removing an element at a particular inde... |
| `resize(new_size)` | - | - | `resize(new_size)` for resizing the list without the need... |
| `difference()` | - | - | `difference()` mapping to `-` |
| `difference_update()` | - | - | `difference_update()` mapping to `-=` |
| `intersection_update()` | - | - | `intersection_update()` mapping to `&=` |
| `update()` | - | - | `update()` mapping to `|=` |
| `memory.reference` | - | - | It has moved to the `memory.reference` module instead of ... |
| `offset()` | - | - | Several unsafe methods were removed, including `offset()`... |
| `FileHandle.seek()` | - | - | `FileHandle.seek()` now has a `whence` argument that defa... |
| `FileHandle.read()` | - | - | `FileHandle.read()` can now read straight into a `DTypePo... |
| `sys` | - | - | The `sys` module now contains an `exit()` function that w... |
| `Tensor` | - | - | The constructors for `Tensor` have been changed to be mor... |
| `ord` | - | - | The `ord` and `chr` functions have been improved to accep... |
| `bool(None)` | - | - | `bool(None)` is now implemented. (@zhoujingya) |
| `len()` | - | - | The `len()` function now handles a `range()` specified wi... |
| `testing.assert_equal[SIMD]()` | - | - | The `testing.assert_equal[SIMD]()` function now raises if... |
| `testing.assert_almost_equal()` | - | - | The `testing.assert_almost_equal()` and `math.isclose()` ... |
| `add_with_overflow()` | - | - | `add_with_overflow()` |
| `sub_with_overflow()` | - | - | `sub_with_overflow()` |
| `mul_with_overflow()` | - | - | `mul_with_overflow()` |
| `parallel_memcpy()` | - | - | The `parallel_memcpy()` function has moved from the `buff... |
| `Optional.value()` | - | - | `Optional.value()` now returns a reference instead of a c... |
| `mojo build ./test-dir/program.mojo` | - | - | The behavior of `mojo build` when invoked without an outp... |
| `mojo package` | - | - | The `mojo package` command no longer supports the `-D` fl... |
| `__get_mvalue_as_litref(x)` | - | - | A low-level `__get_mvalue_as_litref(x)` builtin was added... |
| `_properties` | - | - | Properties can now be specified on inline MLIR ops: `_ = ... |
| `__get_lvalue_as_address(x)` | - | - | The `__get_lvalue_as_address(x)` magic function has been ... |
| `DynamicVector[Tuple[Int]]` | - | - | #1609 alias with `DynamicVector[Tuple[Int]]` fails. |
| `main` | - | - | #1987 Defining `main` in a Mojo package is an error, for ... |
| `0__` | - | - | #1913 - `0__` no longer crashes the Mojo parser. |
| `SIMD.reduce()` | - | - | #2068 Fix `SIMD.reduce()` for size_out == 2. (@soraros) |
| `List.pop_back()` | - | - | `List.pop_back()` has been removed.  Use `List.pop()` ins... |
| `SIMD.to_int(value)` | - | - | `SIMD.to_int(value)` has been removed.  Use `int(value)` ... |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `destroy_pointee()` | A new `destroy_pointee()` function runs the destructor on... | |
| `-g` | The `mojo build` and `mojo run` commands now support a `-... | |
| `os.remove()` | Added `os.remove()` and `os.unlink()` for deleting files.... | |

## v24.5 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `mojo test` | - | - | `mojo test` now uses the Mojo compiler for running unit t... |
| `__setitem__()` | - | - | `__setitem__()` now works with variadic argument lists su... |
| `MOJO_PYTHON` | - | - | The environment variable `MOJO_PYTHON` can be pointed to ... |
| `collections` | - | - | The set of automatically imported entities (types, aliase... |
| `builtin` | - | - | Some types from the `builtin` module have been moved to d... |
| `builtin.string` | - | - | The `builtin.string` module has been moved to `collection... |
| `String.format()` | - | - | Added the `String.format()` method. (PR #2771) Supports a... |
| `String.format()` | - | - | `String.format()` now supports conversion flags `!s` and ... |
| `atol()` | - | - | The `atol()` function now correctly supports leading unde... |
| `unsafe_cstr_ptr()` | - | - | Added the `unsafe_cstr_ptr()` method to `String` and `Str... |
| `DTypePointer.int8` | - | - | The `StringRef` constructors from `DTypePointer.int8` hav... |
| `DTypePointer` | - | - | The `DTypePointer` `load()` and `store()` methods have be... |
| `destroy_pointee(p)` | - | - | `destroy_pointee(p)` => `p.destroy_pointee()` |
| `move_from_pointee(p)` | - | - | `move_from_pointee(p)` => `p.take_pointee()` |
| `p.init_pointee_move(value)` | - | - | `initialize_pointee_move(p, value)` => `p.init_pointee_mo... |
| `p.init_pointee_copy(value)` | - | - | `initialize_pointee_copy(p, value)` => `p.init_pointee_co... |
| `p.move_pointee_into(p2)` | - | - | `move_pointee(src=p1, dst=p2)` => `p.move_pointee_into(p2)` |
| `UnsafePointer.offset()` | - | - | The `UnsafePointer.offset()` method is deprecated and wil... |
| `mojo run /tmp/main.mojo` | - | - | `mojo run /tmp/main.mojo` can access `/tmp/mymodule.py` |
| `mojo build main.mojo -o ~/myexe && ~/myexe` | - | - | `mojo build main.mojo -o ~/myexe && ~/myexe` can access `... |
| `Path.home()` | - | - | `Path.home()` has been added to return a path of the user... |
| `os.path.expanduser()` | - | - | `os.path.expanduser()` and `pathlib.Path.exapanduser()` h... |
| `os.path.split()` | - | - | `os.path.split()` has been added for splitting a path int... |
| `os.makedirs()` | - | - | `os.makedirs()` and `os.removedirs()` have been added for... |
| `pwd` | - | - | The `pwd` module has been added for accessing user inform... |
| `NoneType` | - | - | `NoneType` is now a normal standard library type, and not... |
| `c_char` | - | - | Added the `c_char` type alias in `sys.ffi`. |
| `oct()` | - | - | Added the `oct()` builtin function for formatting an inte... |
| `assert_is()` | - | - | Added the `assert_is()` and `assert_is_not()` test functi... |
| `ulp` | - | - | The `ulp` function from `numerics` has been moved to the ... |
| `countl_zero()` | - | - | `countl_zero()` -> `count_leading_zeros()` |
| `countr_zero()` | - | - | `countr_zero()` -> `count_trailing_zeros()` |
| `rank` | - | - | The `rank` argument for `algorithm.elementwise()` is no l... |
| `time.now()` | - | - | The `time.now()` function has been deprecated. Please use... |
| `mojo test` | - | - | `mojo test` now uses the Mojo compiler for running unit t... |
| `mojo test` | - | - | The `mojo test` command now accepts a `--filter` option t... |
| `mojo test` | - | - | The `mojo test` command now supports using the same compi... |
| `mojo test` | - | - | You can now debug unit tests using `mojo test` by passing... |
| `mojo debug --rpc` | - | - | The `mojo debug --rpc` command has been renamed to `mojo ... |
| `__result__` | - | - | The Mojo debugger now hides the artificial function argum... |
| `__copyinit__` | - | - | #1734 - Calling `__copyinit__` on self causes crash. |
| `__setitem__` | - | - | #3142 - [QoI] Confusing `__setitem__` method is failing w... |
| `__setitem__` | - | - | #248 - [Feature] Enable `__setitem__` to take variadic ar... |
| `SIMD.__int__` | - | - | #3065 - Fix incorrect behavior of `SIMD.__int__` on unsig... |
| `tensor` | - | - | The builtin `tensor` module has been removed. Identical f... |
| `StringLiteral.unsafe_uint8_ptr()` | - | - | Removed `StringLiteral.unsafe_uint8_ptr()` and `StringLit... |
| `SIMD.splat(value: Scalar[type])` | - | - | Removed `SIMD.splat(value: Scalar[type])`. Use the constr... |
| `SIMD.{add,mul,sub}_with_overflow()` | - | - | Removed the `SIMD.{add,mul,sub}_with_overflow()` methods. |
| `SIMD.min()` | - | - | Removed the `SIMD.min()` and `SIMD.max()` methods. Identi... |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `TemporaryDirectory` | Added `TemporaryDirectory` in module `tempfile`. (PR 2743) | |
| `NamedTemporaryFile` | Added `NamedTemporaryFile` in module `tempfile`. (PR 2762) | |
| `StaticString` | Added a new `StaticString` type alias. This can be used i... | |
| `strided_load()` | `UnsafePointer` now supports `strided_load()`, `strided_s... | |
| `exclusive: Bool = False` | `UnsafePointer` has a new `exclusive: Bool = False` param... | |
| `Counter` | Added a new `Counter` dictionary-like type, matching most... | |
| `popitem()` | `Dict` now supports `popitem()`, which removes and return... | |
| `Dict.__init__()` | Added a `Dict.__init__()` overload to specify initial cap... | |
| `ImplicitlyBoolable` | Types conforming to `Boolable` (that is, those implementi... | |
| `stable` | `sort()` now supports a `stable` parameter. It can be cal... | |
| `bit_reverse()` | `bit` module now supports `bit_reverse()`, `byte_swap()`,... | |
| `break-on-raise` | The Mojo debugger now supports a `break-on-raise` command... | |

## v24.6 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `Lifetime` | - | - | `Lifetime` and related types in the standard library have... |
| `@implicit` | - | - | Implicit conversions are now opt-in using the `@implicit`... |
| `Deque` | - | - | The standard library has added several new types, includi... |
| `b main` | - | - | The VS Code extension now supports setting data breakpoin... |
| `__init__()` | - | - | The argument convention for the `self` argument in the `_... |
| `@implicit` | - | - | Single argument constructors now require the `@implicit` ... |
| `AnyLifetime` | - | - | The `AnyLifetime` type (useful for declaring origin types... |
| `Origin.type` | - | - | The `Origin.type` alias has been renamed to `_mlir_origin... |
| `ImmutableOrigin` | - | - | `ImmutableOrigin` and `MutableOrigin` are now, respective... |
| `Origin` | - | - | `Origin` struct values are now supported in the origin sp... |
| `__type_of(x)` | - | - | The `__type_of(x)` and `__origin_of(x)` operators are muc... |
| `MutableAnyOrigin` | - | - | The destructor insertion logic in Mojo is now aware that ... |
| `warn` | - | - | `warn`: print assertion errors e.g. for multithreaded tes... |
| `all` | - | - | `all`: turn on all assertions (previously `-D MOJO_ENABLE... |
| `@explicit_destroy` | - | - | Introduced the `@explicit_destroy` annotation, the `__dis... |
| `ArcPointer` | `Arc` | - | `Arc` has been renamed to `ArcPointer`, for consistency w... |
| `ArcPointer` | - | - | `ArcPointer` now implements `Identifiable`, and can be co... |
| `rebind()` | - | - | The `rebind()` standard library function now works with m... |
| `random.shuffle()` | - | - | Introduced the `random.shuffle()` function for randomizin... |
| `Dict.__getitem__()` | - | - | The `Dict.__getitem__()` method now returns a reference i... |
| `Slice.step` | - | - | `Slice.step` is now an `Optional[Int]`, matching the opti... |
| `os.path.expandvars()` | - | - | `os.path.expandvars()`: Expands environment variables in ... |
| `os.path.splitroot()` | - | - | `os.path.splitroot()`: Split a path into drive, root and ... |
| `C_int` | - | - | The aliases for C foreign function interface (FFI) have b... |
| `Float32` | - | - | `Float32` and `Float64` are now printed and converted to ... |
| `StaticIntTuple` | - | - | The `StaticIntTuple` data structure in the `utils` packag... |
| `buildArgs` | - | - | The VS Code Mojo Debugger now has a `buildArgs` JSON debu... |
| `mojo.run.focusOnTerminalAfterLaunch` | - | - | The VS Code extension now has the `mojo.run.focusOnTermin... |
| `mojo.SDK.additionalSDKs` | - | - | The VS Code extension now has the `mojo.SDK.additionalSDK... |
| `b main` | - | - | The Mojo LLDB debugger now supports symbol breakpoints, f... |
| `SIMD[type, simd_width]` | - | - | The Mojo Language Server and generated documentation now ... |
| `@register_passable` | - | - | Generated API documentation now shows the signatures for ... |
| `ETXTBSY` | - | - | The VS Code extension now downloads its private copy of t... |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `Origin.cast_from` | Added `Origin.cast_from` for casting the mutability of an... | |
| `[_]` | Function types now accept an origin set parameter. This p... | |
| `Deque` | Introduced a new `Deque` (double-ended queue) collection ... | |
| `get_type_name` | The `TypeIdentifiable` trait has been removed in favor of... | |
| `TypedPythonObject[Tuple].__getitem__()` | Added `TypedPythonObject[Tuple].__getitem__()` for access... | |
| `Python.add_object()` | Added `Python.add_object()`, to add a named `PythonObject... | |
| `PythonObject.__contains__()` | Added `PythonObject.__contains__()`. (PR #3101) Example u... | |
| `OwnedPointer` | Added a new `OwnedPointer` type as a safe, single-owner, ... | |
| `as_noalias_ptr()` | A new `as_noalias_ptr()` method as been added to `UnsafeP... | |
| `reserve()` | Added a `reserve()` method and new constructor to the `St... | |
| `StringLiteral.get[some_stringable]()` | A new `StringLiteral.get[some_stringable]()` method is av... | |
| `DLHandle.get_symbol()` | Added `DLHandle.get_symbol()`, for getting a pointer to a... | |
| `Configure Build and Run Args` | The VS Code extension now supports a `Configure Build and... | |

## v25.1 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `bool()` | - | - | The `bool()`, `float()`, `int()`, and `str()` functions a... |
| `x.__copyinit__(y)` | - | - | Initializers are now treated as static methods that retur... |
| `bool()` | - | - | The builtin functions for converting values to different ... |
| `Char.is_ascii()` | - | - | `Char` provides methods for categorizing character types,... |
| `UInt32` | - | - | `Char` can be converted to `UInt32` via `Char.to_u32()`. |
| `chr()` | - | - | `chr()` will now abort if given a codepoint value that is... |
| `String.__len__()` | - | - | The `String.__len__()` and `StringSlice.__len__()` method... |
| `String.write()` | - | - | The `String.write()` static method has moved to a `String... |
| `ExplicitlyCopyable` | - | - | The `ExplicitlyCopyable` trait has changed to require a `... |
| `IntLike` | - | - | The `IntLike` trait has been removed and its functionalit... |
| `ImplicitlyIntable` | - | - | The `ImplicitlyIntable` trait has been added, allowing ty... |
| `next_power_of_two()` | `bit_ceil()` | - | `bit_ceil()` has been renamed to `next_power_of_two()`, a... |
| `validate` | - | - | Added a new boolean `validate` parameter to `b64decode()`. |
| `sys.ffi` | - | - | Added more aliases in `sys.ffi` to round out the usual ne... |
| `mblack` | - | - | `mblack` (aka `mojo format`) no longer formats non-Mojo f... |
| `env_get_dtype()` | - | - | The `env_get_dtype()` function has been added to the `sys... |
| `mojo debug --vscode` | - | - | The command `mojo debug --vscode` now sets the current wo... |
| `!lit.ref` | - | - | Issue #3617 - Can't generate the constructors for a type ... |
| `__init__.mojo` | - | - | The Mojo Language Server doesn't crash anymore on empty `... |
| `Tuple.get()` | - | - | Issue #3935 - Confusing OOM error when using `Tuple.get()... |
| `sys.argv()` | - | - | Changed `sys.argv()` to return list of `StringSlice`. |
| `Path()` | - | - | Added explicit `Path()` constructor from `StringSlice`. |
| `rebind[T](tup[i])` | - | - | The `Tuple.get[i, T]()` method has been removed. Please u... |
| `StringableCollectionElement` | `StringableCollectionElement` | `WritableCollectionElement` | `StringableCollectionElement` is deprecated. Use `Writabl... |
| `IntLike` | - | - | The `IntLike` trait has been removed and its functionalit... |
| `Type{field1: 42, field2: 17}` | - | - | The `Type{field1: 42, field2: 17}` syntax for direct init... |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `String()` | Added a `String()` constructor from `Char`. | |
| `StringSlice.from_utf8()` | Added `StringSlice.from_utf8()` factory method, for valid... | |
| `StringSlice.char_length()` | Added `StringSlice.char_length()` method, to pair with th... | |
| `LinkedList` | A new `LinkedList` type has been added to the standard li... | |
| `Optional.copied()` | Added `Optional.copied()` for constructing an owned `Opti... | |
| `Dict.get_ptr()` | Added `Dict.get_ptr()` which returns an `Optional[Pointer... | |
| `List.extend()` | Added new `List.extend()` overloads taking `SIMD` and `Sp... | |
| `SIMD.from_bytes()` | Added `SIMD.from_bytes()` and `SIMD.as_bytes()` to conver... | |

## v25.2 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `DeviceContext.enqueue_function()` | - | - | If you're executing a GPU kernel only once, you can now s... |
| `gpu.shuffle` | - | - | The `gpu.shuffle` module has been renamed to `gpu.warp` t... |
| `Codepoint` | - | - | The standard library has many changes related to strings.... |
| `DType` | - | - | Support has been added for 128- and 256-bit signed and un... |
| `DeviceContext.enqueue_function()` | - | - | You can now skip compiling a GPU kernel first before enqu... |
| `enqueue_copy_to_device()` | - | - | `enqueue_copy_to_device()` |
| `enqueue_copy_from_device()` | - | - | `enqueue_copy_from_device()` |
| `enqueue_copy_device_to_device()` | - | - | `enqueue_copy_device_to_device()` |
| `copy_to_device_sync()` | - | - | `copy_to_device_sync()` |
| `copy_from_device_sync()` | - | - | `copy_from_device_sync()` |
| `copy_device_to_device_sync()` | - | - | `copy_device_to_device_sync()` |
| `gpu.shuffle` | - | - | The `gpu.shuffle` module has been renamed to `gpu.warp` t... |
| `DType` | - | - | The following aliases have been added to the `DType` stru... |
| `Scalar` | - | - | The following `Scalar` aliases for 1-element `SIMD` value... |
| `StringSlice.chars()` | - | - | `StringSlice.chars()` and `String.chars()` to `StringSlic... |
| `StringSlice.char_slices()` | - | - | `StringSlice.char_slices()` and `String.char_slices()` to... |
| `CharsIter` | - | - | `CharsIter` to `CodepointsIter` |
| `Char.unsafe_decode_utf8_char()` | - | - | `Char.unsafe_decode_utf8_char()` to `Codepoint.unsafe_dec... |
| `codepoint_slices()` | - | - | Made the iterator type returned by the string `codepoint_... |
| `center()` | - | - | `center()` |
| `is_ascii_digit()` | - | - | `is_ascii_digit()` |
| `is_ascii_printable()` | - | - | `is_ascii_printable()` |
| `islower()` | - | - | `islower()` |
| `isupper()` | - | - | `isupper()` |
| `ljust()` | - | - | `ljust()` |
| `lower()` | - | - | `lower()` |
| `rjust()` | - | - | `rjust()` |
| `split()` | - | - | `split()` |
| `upper()` | - | - | `upper()` |
| `StringSlice.__getitem__(Slice)` | - | - | `StringSlice.__getitem__(Slice)` now raises an error if t... |
| `StringLiteral.get[value]()` | - | - | The `StringLiteral.get[value]()` method, which converts a... |
| `LinkedList.__iter__()` | - | - | `LinkedList.__iter__()` to create a forward iterator. |
| `LinkedList.__reversed__()` | - | - | `LinkedList.__reversed__()` for a backward iterator. |
| `List.byte_length()` | `List.bytecount()` | - | `List.bytecount()` has been renamed to `List.byte_length(... |
| `InlineArray(unsafe_uninitialized=True)` | - | - | The `InlineArray(unsafe_uninitialized=True)` constructor ... |
| `UnsafePointer.alloc()` | - | - | The `UnsafePointer.alloc()` method has changed to produce... |
| `standard_deviation` | `random.randn()` | - | #3976 The `variance` argument in `random.randn_float64()`... |
| `inout` | - | - | Use of legacy argument conventions like `inout` and the u... |
| `List.size` | - | - | Direct access to `List.size` has been removed. Use the pu... |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `StringSlice.is_codepoint_boundary()` | Added a `StringSlice.is_codepoint_boundary()` method for ... | |
| `IntervalTree` | A new `IntervalTree` data structure has been added to the... | |
| `sys.is_compile_time()` | Added a new `sys.is_compile_time()` function. This enable... | |

## v25.3 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `complex` | - | - | `complex` |
| `gpu` | - | - | `gpu` |
| `logger` | - | - | `logger` |
| `runtime` | - | - | `runtime` |
| `subprocess` | - | - | `subprocess` |
| `layout` | - | - | `layout` |
| `linalg` | - | - | `linalg` |
| `__merge_with__()` | - | - | Mojo can now use user-declared `__merge_with__()` dunder ... |
| `Set` | - | - | `Set` now conforms to the `Copyable` trait so you can sto... |
| `EqualityComparableCollectionElement` | - | - | The following traits have been removed in favor of trait ... |
| `PythonObject.to_float64()` | - | - | The deprecated `PythonObject.to_float64()` method has bee... |
| `debug_assert()` | - | - | `debug_assert()` in AMD GPU kernels now behaves the same ... |
| `constrained[cond, string]()` | - | - | The `constrained[cond, string]()` function now accepts mu... |
| `pathlib.Path.write_text()` | - | - | `pathlib.Path.write_text()` now accepts a `Writable` argu... |
| `Consistency` | - | - | One can now specify the consistency model used in atomic ... |
| `dtype` | `SIMD` | - | The `type` parameter of `SIMD` has been renamed to `dtype`. |
| `is_power_of_two(x)` | - | - | The `is_power_of_two(x)` function in the `bit` package is... |
| `Pointer.address_of(...)` | - | - | The `Pointer.address_of(...)` and `UnsafePointer.address_... |
| `UInt` | - | - | The Mojo compiler is now able to interpret all arithmetic... |
| `__mlir_op` | - | - | The syntax for adding attributes to an `__mlir_op` is now... |
| `sys.is_apple_silicon()` | - | - | #4198 - Apple M4 is not properly detected with `sys.is_ap... |
| `llvm.assume` | - | - | #3662 - Code using `llvm.assume` cannot run at compile time. |
| `count_leading_zeros` | - | - | #4273 - `count_leading_zeros` doesn't work for vectors wi... |
| `IntLiteral` | - | - | #4362 - Function call with `IntLiteral` incorrectly elimi... |
| `StringSlice.replace` | - | - | #4492 - Fix `StringSlice.replace` seg fault. |
| `PythonObject.from_borrowed_ptr()` | - | - | `PythonObject.from_borrowed_ptr()` has been removed in fa... |
| `utils.numerics.ulp` | - | - | `utils.numerics.ulp` has been removed. Use the `ulp()` fu... |
| `Float64` | - | - | The `float` free function. Use the `Float64` constructor ... |
| `DeviceContext` | - | - | Removed deprecated `DeviceContext` methods `copy_sync()` ... |
| `unroll()` | - | - | The `unroll()` utility has been removed. Use the `@parame... |
| `AsBytes` | - | - | The `AsBytes` trait has been removed. |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `String(unsafe_uninit_length=x)` | `String` supports a new `String(unsafe_uninit_length=x)` ... | |
| `Python.list()` | `PythonObject` is no longer implicitly constructible from... | |
| `Pointer` | `Pointer` now has a `get_immutable()` method to return a ... | |
| `pathlib.Path.write_bytes()` | Added `pathlib.Path.write_bytes()` which enables writing ... | |
| `os.path.split_extension()` | Added `os.path.split_extension()` to split a path into it... | |
| `os.path.is_absolute()` | Added `os.path.is_absolute()` to check if a given path is... | |
| `Variant.is_type_supported()` | Added `Variant.is_type_supported()` method. (PR #4057) Ex... | |

## v25.4 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `UInt8` | - | - | Mojo now supports the use of Python-style type patterns w... |
| `__moveinit__()` | - | - | The Mojo compiler will now synthesize `__moveinit__()`, `... |
| `raise` | - | - | `try` and `raise` now work at compile time. |
| `gpu.tcgen05` | - | - | Primitives for working with NVIDIA Blackwell GPUs have be... |
| `sum()` | - | - | Fixed the `sum()` and `prefix_sum()` implementations in t... |
| `CollectionElement` | - | - | The `CollectionElement` trait has been removed. You can r... |
| `PythonObject` | - | - | Since virtually any operation on a `PythonObject` can rai... |
| `KeyElement` | - | - | `KeyElement`. Since Python objects may not be hashable—an... |
| `EqualityComparable` | - | - | `EqualityComparable`. The `PythonObject.__eq__()` and `Py... |
| `Floatable` | - | - | `Floatable`. An explicit, raising constructor is added to... |
| `ConvertibleFromPython` | - | - | The `ConvertibleFromPython` trait is now public. This tra... |
| `PythonObject(alloc=<value>)` | - | - | `PythonObject(alloc=<value>)` is a new constructor that c... |
| `PythonObject.downcast_value_ptr[T]()` | - | - | `PythonObject.downcast_value_ptr[T]()` checks if the obje... |
| `PythonObject.unchecked_downcast_value_ptr[T]()` | - | - | `PythonObject.unchecked_downcast_value_ptr[T]()` uncondit... |
| `TypedPythonObject` | - | - | The `TypedPythonObject` type has been removed. Use `Pytho... |
| `Python.is_type(x, y)` | - | - | The `Python.is_type(x, y)` static method has been removed... |
| `os.abort(messages)` | - | - | `os.abort(messages)` no longer supports a variadic number... |
| `compile` | - | - | The `compile` module now provides the `get_type_name()` f... |
| `mojo build --emit=llvm YourModule.mojo` | - | - | Example usage: `mojo build --emit=llvm YourModule.mojo` |
| `--emit-llvm` | - | - | Removed support for the command line option `--emit-llvm`... |
| `mojo build --emit=asm YourModule.mojo` | - | - | Example usage: `mojo build --emit=asm YourModule.mojo` |
| `mojo doc` | - | - | Added associated alias support for documentation generate... |
| `mojo format` | - | - | Added struct and trait conformance list sorting support t... |
| `math.sqrt` | - | - | #4352 - `math.sqrt` products incorrect results for large ... |
| `UIntN` | - | - | #4677 - `UIntN` Comparison Yields Incorrect Result When F... |
| `Dict.setdefault` | - | - | #4719 - `Dict.setdefault` should not be marked with `rais... |
| `Python.throw_python_exception_if_error_state()` | - | - | `Python.unsafe_get_python_exception()` and `Python.throw_... |
| `VariadicPack.each()` | - | - | `VariadicPack.each()` and `VariadicPack.each_idx()` metho... |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `variadics` | `VariadicList`, `VariadicListMem`, and `VariadicPack` mov... | |
| `Python.str()` | `Stringable`. Instead, the `PythonObject.__str__()` metho... | |
| `def_function()` | A new `def_function()` API was added to `PythonModuleBuil... | |
| `symmetrical` | The `math.isclose()` function now supports both symmetric... | |

## v25.5 Changes

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `_is_amd_rdna3()` | - | - | `_is_amd_rdna3()` |
| `_is_amd_rdna4()` | - | - | `_is_amd_rdna4()` |
| `_is_amd_rdna()` | - | - | `_is_amd_rdna()` |
| `memory.UnsafePointer` | - | - | `memory.UnsafePointer` is now implicitly included in all ... |
| `OwnedKwargsDict[PythonObject]` | - | - | Mojo functions can now natively accept keyword arguments ... |
| `Defaultable` | - | - | Registering initializers that take arguments. Types no lo... |
| `PythonConvertible` | - | - | The `PythonConvertible` trait has been renamed to `Conver... |
| `default_hasher` | - | - | `default_hasher` (AHasher) and `default_comp_time_hasher`... |
| `Hashable` | - | - | Users are now required to implement the method `fn __hash... |
| `List.extend(Span)` | `List.append(Span)` | - | `List.append(Span)` has been renamed to `List.extend(Span... |
| `open()` | - | - | File-related APIs such as `open()`, `FileHandle` and `Fil... |
| `Writer` | - | - | The `Writer` and `Writable` traits. |
| `input()` | - | - | `input()` and `print()` functions. |
| `StringLiteral.strip()` | - | - | `StringLiteral.strip()` family of functions now return a ... |
| `--debug-level=none` | - | - | `-g0`: No debug information (alias for `--debug-level=non... |
| `--debug-level=line-tables` | - | - | `-g1`: Line table debug information (alias for `--debug-l... |
| `--debug-level=full` | - | - | `-g2`: Full debug information (alias for `--debug-level=f... |
| `.value()` | - | - | #4121 - better error message for `.value()` on empty `Opt... |
| `math.exp2` | - | - | #4820 - `math.exp2` picks the wrong implementation for `f... |
| `enqueue_fill` | - | - | #5066 - Correctly fill 64-bit values on AMD in `enqueue_f... |
| `toggle_all` | - | - | #4982 - Add `toggle_all` to `BitSet`. |
| `set_all` | - | - | #5086 - Add `set_all` to `BitSet`. |
| `.modular` | - | - | #5051 - Incorrect `.modular` Directory Location on Linux. |
| `is_x86()` | - | - | `is_x86()` |
| `has_sse4()` | - | - | `has_sse4()` |
| `has_avx()` | - | - | `has_avx()` |
| `has_avx2()` | - | - | `has_avx2()` |
| `has_avx512f()` | - | - | `has_avx512f()` |
| `has_fma()` | - | - | `has_fma()` |
| `has_vnni()` | - | - | `has_vnni()` |
| `has_neon()` | - | - | `has_neon()` |
| `has_neon_int8_dotprod()` | - | - | `has_neon_int8_dotprod()` |
| `has_neon_int8_matmul()` | - | - | `has_neon_int8_matmul()` |
| `UnsafePointer.address_of()` | - | - | `UnsafePointer.address_of()` has been removed.  Use `Unsa... |
| `DType.tensor_float32` | - | - | `DType.tensor_float32` has been removed due to lack of su... |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `Iterator` | Added `Iterator` trait for modeling types that produce a ... | |

## v24.4 Changes

### Breaking Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `String` | - | - | Breaking. Implicit conversion to `String` is now removed ... |

### API Changes

| Change | Old | New | Notes |
|--------|-----|-----|-------|
| `UnsafePointer` | - | - | Continued unification of standard library APIs around the... |
| `Dict` | - | - | Significant performance improvements when inserting into ... |
| `Int32(42)` | - | - | Mojo added support for infer-only parameters. Infer-only ... |
| `@deprecated` | - | - | Mojo now supports adding a `@deprecated` decorator on str... |
| `is_mutable` | - | - | The `is_mutable` parameter of `Reference` and `AnyLifetim... |
| `PATH` | - | - | Mojo will now link to a Python dynamic library based on t... |
| `__TypeOfAllTypes` | `AnyRegType` | - | `AnyRegType` has been renamed to `__TypeOfAllTypes` and M... |
| `repr()` | - | - | Added built-in `repr()` function and `Representable` trai... |
| `Indexer` | - | - | Added the `Indexer` trait to denote types that implement ... |
| `Powable` | - | - | Conforming to the `Powable` trait also means that the typ... |
| `ceildiv()` | - | - | For `ceildiv()`, structs can conform to either the `CeilD... |
| `Ceilable` | - | - | Due to ongoing refactoring, the traits `Ceilable`, `CeilD... |
| `bencher` | - | - | The `bencher` module as part of the `benchmark` package i... |
| `String.split()` | - | - | `String.split()` now defaults to whitespace and has Pytho... |
| `String.strip()` | - | - | `String.strip()`, `lstrip()` and `rstrip()` can now remov... |
| `String` | - | - | `String` now has a `splitlines()` method, which allows sp... |
| `InlineString` | `InlinedString` | - | `InlinedString` has been renamed to `InlineString` to be ... |
| `StringRef` | - | - | `StringRef` now implements `strip()`, which can be used t... |
| `StringRef` | - | - | `StringRef` now implements `startswith()` and `endswith()... |
| `String.unsafe_ptr()` | `String._as_ptr()` | - | Renamed `String._as_ptr()` to `String.unsafe_ptr()`, and ... |
| `StringLiteral.unsafe_ptr()` | `StringLiteral.data()` | - | Renamed `StringLiteral.data()` to `StringLiteral.unsafe_p... |
| `unsafe_ptr()` | `InlineString.as_ptr()` | - | `InlineString.as_ptr()` has been renamed to `unsafe_ptr()... |
| `StringRef.unsafe_ptr()` | - | - | `StringRef.data` is now an `UnsafePointer` (was `DTypePoi... |
| `Slice` | - | - | The `Slice.__len__()` function has been removed and `Slic... |
| `sort()` | - | - | Added a built-in `sort()` function for lists of elements ... |
| `int()` | - | - | `int()` can now take a string and a specified base to par... |
| `bin()` | - | - | Added the `bin()` built-in function to convert integral t... |
| `atof()` | - | - | Added the `atof()` built-in function, which can convert a... |
| `any()` | - | - | You can now use the built-in `any()` and `all()` function... |
| `object` | - | - | `object` now implements all the bitwise operators. (PR #2... |
| `Tuple` | - | - | `ListLiteral` and `Tuple` now only require that element t... |
| `get_null()` | - | - | Removed the `get_null()` method from `UnsafePointer` and ... |
| `unsafe_ptr()` | - | - | Many functions returning a pointer type have been unified... |
| `Tensor.data()` | - | - | The `Tensor.data()` method has been renamed to `unsafe_pt... |
| `List` | - | - | `List` now has an `index()` method that allows you to fin... |
| `List` | - | - | `List` can now be converted to a `String` with a simplifi... |
| `List` | - | - | `List` has a simplified syntax to call the `count()` meth... |
| `List` | - | - | `List` now has an `unsafe_get()` to get the reference to ... |
| `Dict` | - | - | `Dict` now has a simplified conversion to `String` with `... |
| `Dict` | - | - | `Dict` now implements `get(key)` and `get(key, default)` ... |
| `__get_ref(key)` | - | - | Added a temporary `__get_ref(key)` method to `Dict`, allo... |
| `mkdir()` | - | - | The `os` module now provides functionality for adding and... |
| `os.path.getsize()` | - | - | Added the `os.path.getsize()` function, which gives the s... |
| `SIMD.__bool__()` | - | - | `SIMD.__bool__()` is constrained such that it only works ... |
| `SIMD.reduce_or()` | - | - | The `SIMD.reduce_or()` and `SIMD.reduce_and()` methods ar... |
| `ctlz` | - | - | `ctlz` -> `countl_zero` |
| `cttz` | - | - | `cttz` -> `countr_zero` |
| `bit_length` | - | - | `bit_length` -> `bit_width` |
| `ctpop` | - | - | `ctpop` -> `pop_count` |
| `bswap` | - | - | `bswap` -> `byte_swap` |
| `bitreverse` | - | - | `bitreverse` -> `bit_reverse` |
| `math.rotate_bits_left()` | - | - | The `math.rotate_bits_left()` and `math.rotate_bits_right... |
| `is_power_of_2()` | - | - | The `is_power_of_2()` function in the `math` module is no... |
| `abs()` | - | - | The `abs()`, `round()`, `min()`, `max()`, `pow()`, and `d... |
| `math.tgamma()` | - | - | The `math.tgamma()` function has been renamed to `math.ga... |
| `math.gcd()` | - | - | `math.gcd()` now works on negative inputs, and like Pytho... |
| `Coroutine` | - | - | `Coroutine` now requires a lifetime parameter. This param... |
| `InlineArray` | - | - | Added an `InlineArray` type that works on memory-only typ... |
| `base64` | - | - | The `base64` package now includes encoding and decoding s... |
| `unsafe_take()` | `Optional` | - | The `take()` function in `Variant` and `Optional` has bee... |
| `__getitem__()` | `Variant` | - | The `get()` function in `Variant` has been replaced by `_... |
| `algorithm` | - | - | Various functions in the `algorithm` module are now built... |
| `infinity` | - | - | `infinity` and `NaN` are now correctly handled in `testin... |
| `mojo package my-package -o my-dir` | - | - | Invoking `mojo package my-package -o my-dir` on the comma... |
| `--diagnose-missing-doc-strings` | `mojo` | - | The `--warn-missing-doc-strings` flag for `mojo` has been... |
| `@doc_private` | - | - | A new decorator, `@doc_private`, was added that can be us... |
| `@unroll` | - | - | The `@unroll` decorator has been deprecated and removed. ... |
| `round_half_down()` | - | - | `round_half_down()` and `round_half_up()`; these can be t... |
| `add()` | - | - | `add()`, `sub()`, `mul()`, `div()`, `mod()`, `greater()`,... |
| `identity()` | - | - | `identity()` and `reciprocal()`; users can implement thes... |
| `select()` | - | - | `select()`; removed in favor of using `SIMD.select()` dir... |
| `is_even()` | - | - | `is_even()` and `is_odd()`; these can be trivially implem... |
| `rotate_left()` | - | - | `rotate_left()` and `rotate_right()`; the same functional... |
| `math.pow()` | - | - | An overload of `math.pow()` taking an integer parameter e... |
| `align_down_residual()` | - | - | `align_down_residual()`; it can be trivially implemented ... |
| `all_true()` | - | - | `all_true()`, `any_true()`, and `none_true()`; use `SIMD.... |
| `rint()` | - | - | `rint()` and `nearbyint()`; use `round()` or `SIMD.rounde... |
| `math.bit.select()` | - | - | The `math.bit.select()` and `math.bit.bit_and()` function... |
| `math.limit.inf()` | - | - | `math.limit.inf()`: use `utils.numerics.max_or_inf()` |
| `math.limit.neginf()` | - | - | `math.limit.neginf()`: use `utils.numerics.min_or_neg_inf()` |
| `math.limit.max_finite()` | - | - | `math.limit.max_finite()`: use `utils.numerics.max_finite()` |
| `math.limit.min_finite()` | - | - | `math.limit.min_finite()`: use `utils.numerics.min_finite()` |
| `tensor.random` | - | - | The `tensor.random` module has been removed. The same fun... |
| `SIMD` | - | - | The builtin `SIMD` struct no longer conforms to `Indexer`... |
| `FloatLiteral` | - | - | #1787 Fix error when using `//` on `FloatLiteral` in alia... |
| `assert_raises` | - | - | #2692 Fix `assert_raises` to include calling location. |
| `object.print()` | - | - | The method `object.print()` has been removed. Since `obje... |
| `EvaluationMethod` | - | - | The `EvaluationMethod` has been removed from `math.polyno... |

### New Features

| Feature | Description | Example |
|---------|-------------|--------|
| `@parameter for` | A new `@parameter for` mechanism for expressing compile-t... | |
| `ref` | Mojo functions can return an auto-dereferenced reference ... | |
| `@parameter for` | Mojo has introduced `@parameter for`, a new feature for c... | |
| `String.isspace()` | Added `String.isspace()` method conformant with Python's ... | |
| `as_string_slice()` | Added new `as_string_slice()` methods to `String` and `St... | |
| `StringSlice` | Added `StringSlice` initializer from an `UnsafePointer` a... | |
| `as_bytes_slice()` | Added a new `as_bytes_slice()` method to `String` and `St... | |
| `__contains__()` | `Tuple` now supports `__contains__()`. (PR #2709) For exa... | |
| `ImmutableStaticLifetime` | Added new `ImmutableStaticLifetime` and `MutableStaticLif... | |
| `memcpy()` | Added new `memcpy()` overload for `UnsafePointer[Scalar[_... | |
| `__contains__()` | `List()` now supports `__contains__()`, so you can now us... | |
| `fromkeys()` | Added a `fromkeys()` method to `Dict` to return a `Dict` ... | |
| `clear()` | Added a `clear()` method  to `Dict`. (PR 2627) | |
| `reversed()` | `Dict` now supports `reversed()` for its `items()` and `v... | |
| `InlineList` | Added a new `InlineList` type, a stack-allocated list wit... | |
| `Span` | Added a new `Span` type for taking slices of contiguous c... | |
| `os.path.join()` | Added `os.path.join()` function. (PR 2792) | |
| `tempfile` | Added a new `tempfile` module, with `gettempdir()` and `m... | |
| `SIMD.shuffle()` | Added `SIMD.shuffle()` with `IndexList` mask. (PR #2315) | |
| `SIMD.__repr__()` | Added `SIMD.__repr__()` to get the verbose string represe... | |
| `utils.numerics` | The implementation of the following functions have been m... | |
| `--diagnostic-format` | Several `mojo` subcommands now support a `--diagnostic-fo... | |
| `--validate-doc-strings` | A new `--validate-doc-strings` option has been added to `... | |
| `SIMD.clamp()` | `clamp()`; use the new `SIMD.clamp()` method instead. | |
| `SIMD.roundeven()` | `roundeven()`; the new `SIMD.roundeven()` method now prov... | |
| `ceildiv()` | `div_ceil()`; use the new `ceildiv()` function. | |
| `SIMD.reduce_bit_count()` | `reduce_bit_count()`; use the new `SIMD.reduce_bit_count(... | |

## v0.26+ Only Changes

| Deprecated/Changed | Replacement |
|------------|-------------|
| `alias` for constants | `comptime` (e.g., `comptime PI: Float32 = 3.14`) |

## v0.26.2 New Features

| Feature | Syntax | Description |
|---------|--------|-------------|
| **Typed Error Raising** | `fn foo() raises CustomError -> Int` | Functions declare specific error types |
| **@align Decorator** | `@align(64) struct CacheAligned` | Specify minimum alignment for structs |
| **Never Type** | `fn abort() -> Never` | For functions that never return |
| **comptime Expression** | `comptime(layout.size())` | Force compile-time evaluation |
| **Trait ... vs pass** | `fn foo(): ...` vs `fn bar(): pass` | `...` = no default, `pass` = empty default |
| **Fn Type Conversion** | Non-raising → raising implicitly | Implicit conversion between function types |
| **Linear Type Support** | `ImplicitlyDestructible` trait | `AnyType` no longer requires `__del__()` |
| **Copyable refines Movable** | `T: Copyable` | Types requiring `Copyable` don't need to also mention `Movable` |
| **Struct Reflection** | `struct_field_count[T]()` | Compile-time struct introspection |

### Struct Reflection with struct_field_count

Get the number of fields in a struct at compile time:

```mojo
@fieldwise_init
struct Config:
    var host: String
    var port: Int
    var timeout: Float64

fn print_struct_info[T: AnyType]():
    """Print compile-time struct information."""
    comptime num_fields = struct_field_count[T]()
    print("Number of fields:", num_fields)

fn main():
    print_struct_info[Config]()  # Prints: Number of fields: 3
```

**Use cases:**
- Serialization frameworks (JSON, MessagePack)
- Debug printing utilities
- Compile-time validation of struct layouts
- Generic container implementations

## v0.26.2 Breaking Changes

| Deprecated/Changed | Replacement | Notes |
|-------------------|-------------|-------|
| `UnsafePointer[T].alloc(n)` | `from memory import alloc; alloc[T](n)` | Reverted to function-based allocation |
| `__has_next__()` on iterators | `raises StopIteration` on `__next__()` | Iterator trait-based termination |

### Memory Allocation in v0.26.2

The allocation API has changed in v0.26.2. Use the `alloc` function from `memory`:

```mojo
# v26.1.0.0.0 (stable) - static method
var ptr = UnsafePointer[Float32].alloc(count)

# v0.26.2+ (nightly) - function from memory module
from memory import alloc
var ptr = alloc[Float32](count)
```

### Iterator Protocol

Mojo uses the `Iterator` trait with `raises StopIteration` on `__next__()`:

```mojo
# nocompile
# The Iterator trait (from iter module):
# trait Iterator(ImplicitlyDestructible, Movable):
#     comptime Element: Movable
#     fn __next__(mut self) raises StopIteration -> Self.Element

# Example implementation:
struct MyIter(Iterator):
    comptime Element = Int
    var data: List[Int]
    var index: Int

    fn __next__(mut self) raises StopIteration -> Int:
        if self.index >= len(self.data):
            raise StopIteration()
        var val = self.data[self.index]
        self.index += 1
        return val
```

**How for-loops work in Mojo:**
1. Call `__iter__()` to get an `Iterator`
2. Call `__next__()` on the iterator
3. If `__next__()` returns a value, bind it and execute the loop body
4. If `__next__()` raises `StopIteration`, the loop terminates

## v0.26.2 Type System Changes

### Copyable Refines Movable

In v0.26.2+, `Copyable` automatically implies `Movable`. You no longer need to specify both:

```mojo
# v26.1.0.0.0 (stable)
@fieldwise_init
struct Data(Copyable, Movable):
    var value: Int

# v0.26.2+ (nightly) - simplified
@fieldwise_init
struct Data(Copyable):
    var value: Int
```

### ImplicitlyDestructible Trait

In v0.26.2+, `AnyType` no longer requires `__del__()`. Use `ImplicitlyDestructible` for types that need automatic cleanup:

```mojo
# Types that own resources should still implement __del__()
# but simple value types no longer need it
struct SimpleValue(ImplicitlyDestructible):
    var x: Int
    var y: Int
```

### Implicit Function Type Conversion

Non-raising functions can now be implicitly converted to raising function types:

```mojo
fn safe_op() -> Int:
    return 42

fn takes_raising(f: fn() raises -> Int):
    try:
        _ = f()
    except:
        pass

fn main():
    takes_raising(safe_op)  # OK in v0.26.2+ (implicit conversion)
```

## Nightly Deprecations (v26.2.0.dev2026012905+)

> **IMPORTANT:** These deprecations are NOW ACTIVE in nightly builds (v26.2.0.dev2026012905+).
> Stable (v26.1.0.0.0) still uses the old APIs without warnings.
>
> **How to verify:** Run `mojo build your_file.mojo` - deprecation warnings confirm your version.

### Breaking Changes

| Change | Description |
|--------|-------------|
| **StringSlice constructor** | `StringSlice(str)` now propagates mutability from source String reference |
| **String.resize safety** | Will panic if truncating a codepoint (previously created invalid UTF-8) |
| **String.as_string_slice_mut()** | **REMOVED** - no replacement, use mutable StringSlice directly |

### Active Deprecations

> These deprecations are **NOW ACTIVE** in nightly. You will see compiler warnings.

| Deprecated | Replacement | Notes |
|------------|-------------|-------|
| `@register_passable("trivial")` | `TrivialRegisterPassable` trait | Fully replaced -- 0 occurrences of old decorator in stdlib |
| `String.as_string_slice()` | `StringSlice(str)` constructor | Use constructor instead of method |
| `__reversed__()` on strings | `codepoint_slices_reversed()` | Unicode-aware reverse iteration |
| `String.ljust` / `String.rjust` | `String.ascii_ljust` / `String.ascii_rjust` | Renamed for clarity |

### TrivialRegisterPassable Migration

> **Current status:** The `@register_passable("trivial")` decorator has been fully replaced by the `TrivialRegisterPassable` trait. There are 0 occurrences of the old decorator in the current stdlib.

**Old syntax (replaced):**
```mojo
# nocompile - deprecated
@register_passable("trivial")
struct Point:
    var x: Float32
    var y: Float32
```

**New syntax:**
```mojo
struct Point(TrivialRegisterPassable, Copyable):
    var x: Float32
    var y: Float32
```

**Size guidelines for TrivialRegisterPassable:**
- Ideal: 2-4 machine words (16-32 bytes on 64-bit)
- Maximum recommended: ~48 bytes (6 machine words)
- Larger types: Use normal structs with `Copyable` + `Movable`

**When to use TrivialRegisterPassable:**
- Small fixed-size types with no heap allocations
- Types that should be passed in registers (no pointer indirection)
- Types with no lifecycle requirements (no `__del__` needed)

### API Renames

| Old Name | New Name |
|----------|----------|
| `String.ljust` | `String.ascii_ljust` |
| `String.rjust` | `String.ascii_rjust` |
| `StringSlice.ljust` | `StringSlice.ascii_ljust` |
| `StringSlice.rjust` | `StringSlice.ascii_rjust` |
| `StaticString.ljust` | `StaticString.ascii_ljust` |
| `StaticString.rjust` | `StaticString.ascii_rjust` |

## Critical Syntax Rules

### Reserved Keywords

The following are reserved keywords and cannot be used as parameter or variable names:

| Keyword | Usage | Alternative Names |
|---------|-------|-------------------|
| `out` | Output parameter convention | `result`, `output`, `dst` |
| `match` | Pattern matching (planned) | `matches`, `matched`, `match_result` |
| `in` | Membership testing | `input`, `data`, `value` |

**`out` is reserved** - Cannot be used as a parameter name:
```mojo
# WRONG: fn process(out: Buffer) - will not compile
# CORRECT: fn process(result: Buffer)

# WRONG in custom ops/kernels:
# fn my_kernel(out: Tensor, input: Tensor)  # ERROR
# CORRECT:
fn my_kernel(output: Tensor, input: Tensor)  # OK
```

**`match` is reserved** - Cannot be used as a variable name:
```mojo
# WRONG: var match = regex_search(pattern, text)
# CORRECT: var match_result = regex_search(pattern, text)
```

### No Global Variables

Module-level `var` declarations are NOT supported:
```mojo
# WRONG: var global_counter: Int = 0
# CORRECT: Encapsulate state in structs and pass explicitly
```

### UnsafePointer in Struct Fields

Must use full type specification with imports:
```mojo
from builtin.type_aliases import MutAnyOrigin

struct Buffer:
    var data: UnsafePointer[mut=True, type=Float32, origin=MutAnyOrigin]
```

### memcpy Count = Elements, NOT Bytes

```mojo
# WRONG: 4x overflow for Float32
memcpy(dest=dst, src=src, count=1024 * 4)
# CORRECT:
memcpy(dest=dst, src=src, count=1024)
```
