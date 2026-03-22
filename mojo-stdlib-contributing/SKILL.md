---
name: mojo-stdlib-contributing
description: Patterns and pitfalls for contributing to the Mojo standard library. Use when modifying code under mojo/stdlib/, writing tests or benchmarks for stdlib, or preparing PRs to the modular/modular repository. Distilled from 30+ reviewed PRs.
---

<!-- EDITORIAL GUIDELINES FOR THIS SKILL FILE
Distilled from real reviewer feedback on 30+ PRs to the Mojo stdlib.
Every entry reflects an actual rejection or correction. Only include
patterns that are non-obvious or that a model would get wrong. -->

## Process — before writing code

- **New APIs require a GitHub issue first.** Do not add methods to existing types (String, List, Deque, etc.) without prior consensus. Python parity alone is not justification. Open the issue, then create a draft PR linking it.
- **Keep PRs minimal and focused.** One logical change per PR. Never mix unrelated changes (e.g. don't add `@always_inline` to Set/Dict in a Deque fix PR). Benchmark utilities and their usage belong in separate PRs.
- **Always create draft PRs.** Never mark a PR as ready for review yourself. Before the human marks it as ready, ask them to review the diff, PR description, and benchmark results carefully. Maintainer review cycles are expensive -- catching issues before requesting review avoids wasted rounds.
- **Branch from `upstream/main`.** Never work on `main` directly. Each PR branch must contain only commits for that specific change.

## Assertion semantics — critical

The `assert` statement desugars to `debug_assert()` with `assert_mode="none"`. The default `ASSERT_MODE` is `"safe"` (from `get_defined_string["ASSERT", "safe"]()`). This means:

| Form | Runs when `ASSERT_MODE="safe"` (default) | Runs when `-D ASSERT=all` |
|---|---|---|
| `debug_assert[assert_mode="safe"](...)` | Yes | Yes |
| `debug_assert(...)` (default `assert_mode="none"`) | No | Yes |
| `assert cond, msg` (desugars to above) | No | Yes |

**Do NOT downgrade `debug_assert[assert_mode="safe"]` to `debug_assert` or `assert`.** These are intentional safety invariants. The maintainers want aggressive checking on operations like `byte=` indexing. Users who need to bypass safety should use the existing unsafe escape hatches (e.g. `.as_bytes().unsafe_get(idx)` for byte access).

## Optimizations — verify codegen and benchmark before submitting

**Every optimization PR must include both IR evidence and benchmark results.**

### Verify codegen with `compile_info`

Use `std.compile.compile_info` to inspect the generated IR *before* and *after* your change. If the IR is identical, the optimization is a no-op regardless of what the source looks like.

```mojo
from std.compile import compile_info

# Define the function to inspect
def my_function(x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return x + x

# Inspect optimized LLVM IR
comptime info = compile_info[my_function, emission_kind="llvm-opt"]()
print(info)  # prints optimized IR

# Check for specific instructions
assert "fadd" in compile_info[my_function, emission_kind="llvm-opt"]()

# Write IR to file for detailed comparison
compile_info[my_function, emission_kind="llvm-opt"]().write_text("after.ll")
```

Supported `emission_kind` values:

| Kind | Output |
|---|---|
| `"asm"` | Assembly (default) |
| `"llvm"` | Unoptimized LLVM IR |
| `"llvm-opt"` | Optimized LLVM IR (use this to compare) |
| `"llvm-bitcode"` | LLVM bitcode |

**Workflow for optimization PRs:**
1. Write a small test that calls `compile_info` on the function before your change. Save the IR.
2. Apply your change.
3. Compare IR. If identical, the optimization does nothing. Do not submit.
4. If IR differs, run benchmarks to confirm the improvement is measurable.

### Common pitfalls

- **Do not add manual fast paths that LLVM already optimizes.** For example, `1 if b < 128 else _utf8_first_byte_sequence_length(b)` produces identical IR to just calling `_utf8_first_byte_sequence_length(b)` because LLVM sees through the branch.
- **Measure before optimizing compile-time-evaluated paths.** Format strings may be evaluated at compile time, so runtime SIMD scanning provides no benefit.
- **Verify alignment claims with data.** Cache-line alignment (e.g. 64-byte `@align`) requires benchmark evidence. Do not apply speculatively.
- **Include before/after comparisons in the PR description.** Show the benchmark numbers and, if relevant, the IR diff.

## Code design

- **Reuse existing stdlib primitives.** For SIMD byte search, use `_memchr`/`_memchr_impl` rather than writing a new loop. If a new algorithm is better, update the existing primitive so all callers benefit. Use `clamp`, `Int64.__xor__`, etc. instead of reimplementing.
- **Implement fast paths on the lowest-level type** (`Span`), not higher-level types (`List`). Then delegate: `List.__contains__` calls `Span.__contains__`.
- **Generalize type constraints.** Don't hard-code a single type (e.g. `Byte`). Use trait conformance like `TrivialRegisterPassable` to cover all applicable types.
- **Use move semantics (`^`)** when transferring ownership, not `.copy()`.
- **Use `uninit_copy_n`, `uninit_move_n`, `destroy_n`** for bulk operations on trivial types instead of manual loops.
- **Prefer lazy evaluation (iterators) over eager allocation** for collection operations like slicing. E.g., `deque[0:1:2]` should return an iterator, not allocate a new `Deque`.
- **Check if traits synthesize default methods** before writing explicit implementations. E.g., `Equatable` may already synthesize `__ne__` from `__eq__`.
- **Lift constants into traits** when they vary across implementations (e.g., `MAX_NAME_SIZE` into a `_DirentLike` trait).
- **Keep t-string expressions simple.** Assign complex sub-expressions to local variables before interpolating.
- **Do not add redundant logic** just for Python API parity if Mojo already has a better primitive.
- **Question two-pass patterns.** A "collect then delete" approach with an extra allocation may be slower than a single-pass approach. Always benchmark to confirm.
- **Ensure PR description matches reality.** If `discard()` internally uses try/except, don't claim "avoids exception overhead".

## Benchmarks

```mojo
# Correct benchmark pattern:
@parameter
def bench_something(mut b: Bencher) raises:
    var data = setup_data()
    @always_inline
    def call_fn() unified {read}:
        var result = black_box(data).some_operation(black_box(arg))
        keep(result)
    b.iter(call_fn)
```

- **Always wrap inputs with `black_box()` and results with `keep()`** to prevent the compiler from optimizing away the benchmark.
- **Do not include construction inside the hot loop.** Setup goes before `call_fn`. Use `iter_with_setup` for destructive benchmarks.
- **Data must match the described scenario.** If the docstring says "element in the middle", verify it is actually there.
- **Use `comptime` instead of `@parameter`** for compile-time tuples/loops in benchmark `main()`. (`@parameter` is still used as a function decorator on benchmark functions themselves.)
- **Setup functions passed to `iter_with_setup` must not raise.**
- **Provide benchmarks for performance PRs**, covering both small and large inputs.

## Testing

- **Add unicode test cases** for any string/byte operation.
- **Cover both hit and miss paths** in search/contains tests.
- **Add edge cases:** `start > end`, out-of-bounds indices, empty input, exact-width (no padding).
- **Use `debug_assert`** for internal preconditions on values that must be non-negative by invariant.
- **Tests for `Writable` must use `check_write_to`**, not `String(x)` or `repr(x)`.

## SIMD / memory safety

- **Validate pointer arithmetic** before low-level memory access. Clamp/normalize `[start, end]` into `[0, len]` before `memcmp`.
- **Check for negative `pos_in_block`** when computing leading/trailing zeros in reverse SIMD scans.
- **Verify `vectorized_end`** accounts for full SIMD block width to prevent OOB reads.
- **SIMD test comments must be platform-agnostic.** Write "exercises both SIMD path and scalar tail" instead of "one full 64-byte SIMD block + scalar tail".
- **Consider endianness** when working with bit-level SIMD operations (e.g. `count_trailing_zeros` on packed bitmasks). Implementations may need a branch for big-endian targets.

## Writable trait patterns

```mojo
# Correct signature (not generic [W: Writer]):
def write_to(self, mut writer: Some[Writer]):
    writer.write_string("literal")  # not writer.write("literal")
    writer.write(self.field)        # non-literals use .write()

def write_repr_to(self, mut writer: Some[Writer]):
    writer.write("TypeName.", self)  # include type name for enums
```

- **`write_to` and `write_repr_to` must not allocate.** No `repr()`, `String(x)`, or heap-allocating calls.
- **Use `FormatStruct`** from `format._utils` for consistent repr formatting of structured types.
- **Add an explicit `else` fallback** in enum-style `write_to`.
- **`Stringable` and `Representable` are deprecated.** Only implement `Writable`. Do not add `__str__` or `__repr__`.

## Changelog

- **Do NOT add changelog entries for NFC/implementation-detail changes.** Only user-facing behavior changes belong in the changelog. Internal optimization, refactoring, or performance improvements that don't change the public API are not changelisted.

## Build and lint

```bash
# Build stdlib
./bazelw build //mojo/stdlib/std

# Run specific tests
./bazelw test //mojo/stdlib/test/collections/string:test_string_slice.mojo.test

# Run all stdlib tests
./bazelw test mojo/stdlib/test/...

# Format check (run before pushing)
./bazelw run format
```

- **Always run `./bazelw run format` before pushing.** The CI lint check will reject unformatted code.
- **Sign commits** with `git commit -s`.
- **Include `Assisted-by: AI`** as a trailer in every PR description per `AI_TOOL_POLICY.md`.
