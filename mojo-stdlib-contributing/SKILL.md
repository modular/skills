---
name: mojo-stdlib-contributing
description: Patterns and pitfalls for contributing to the Mojo standard library. Use when modifying code under mojo/stdlib/, writing tests or benchmarks for stdlib, or preparing PRs to the modular/modular repository. Distilled from 30+ reviewed PRs.
---

<!-- EDITORIAL GUIDELINES FOR THIS SKILL FILE
This file is loaded into an agent's context window as a correction layer for
pretrained contribution knowledge. Every line costs context. When editing:
- Be terse. Use tables and inline code over prose where possible.
- Never duplicate information from the mojo-syntax skill.
- Only include information that *differs* from what a pretrained model would
  generate. Don't document things models already get right.
- Prefer one consolidated code block over multiple small ones.
- If adding a new section, ask: "Would a model get this wrong?" If not, skip it.
These same principles apply to any files this skill references.
-->

Contributing to the Mojo stdlib has non-obvious patterns that differ from typical open-source projects. **Always follow this skill to avoid common rejection reasons.**

## Process

- **New APIs require a GitHub issue first.** Do not implement new methods on existing types without prior consensus. Open the issue, then create a draft PR linking it.
- **Keep PRs focused.** One logical change per PR. Benchmark utilities and their usage belong in separate PRs.
- **Always create draft PRs.** Never mark as ready yourself. Ask the human to review the diff, description, and benchmarks before requesting maintainer review.
- **No changelog entries for internal changes.** Only user-facing behavior changes belong in the changelog.

## Assertion semantics

The default `ASSERT_MODE` is `"safe"` (from `get_defined_string["ASSERT", "safe"]()`).

| Form | Runs by default (`"safe"`) | Runs with `-D ASSERT=all` |
|---|---|---|
| `debug_assert[assert_mode="safe"](...)` | Yes | Yes |
| `debug_assert(...)` / `assert cond, msg` | No | Yes |

**Do NOT downgrade `debug_assert[assert_mode="safe"]`.** These are intentional safety invariants, not performance bugs.

## Optimizations

**Every optimization PR must include both IR evidence and benchmark results.**

Use `std.compile.compile_info` to compare generated IR before and after your change:

```mojo
from std.compile import compile_info

def my_function(x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return x + x

# Inspect optimized LLVM IR ("llvm" for unoptimized, "asm" for assembly)
comptime info = compile_info[my_function, emission_kind="llvm-opt"]()
print(info)

# Pattern-match on IR content
assert "fadd" in compile_info[my_function, emission_kind="llvm-opt"]()
```

**Workflow:** save IR before your change, apply change, compare. If IR is identical, the optimization is a no-op. Do not submit. If IR differs, run benchmarks to confirm measurable improvement.

- **Do not add manual fast paths that LLVM already optimizes.** E.g., `1 if b < 128 else _utf8_first_byte_sequence_length(b)` produces identical IR to just calling the function.
- **Verify alignment claims with benchmarks.** Cache-line alignment requires evidence.
- **Include before/after benchmark numbers in the PR description.**

## Code design

- **Reuse existing stdlib primitives.** Use `_memchr`/`_memchr_impl`, `clamp`, etc. rather than reimplementing. If a new algorithm is better, update the existing primitive.
- **Implement fast paths on `Span`**, not `List`. Then delegate upward.
- **Generalize type constraints.** Use trait conformance (e.g. `TrivialRegisterPassable`) rather than hard-coding a single type.
- **Use move semantics (`^`)**, not `.copy()`. Use `uninit_copy_n`, `uninit_move_n`, `destroy_n` for bulk operations.
- **Question two-pass patterns.** A "collect then delete" approach may be slower than single-pass. Benchmark to confirm.

## Benchmarks

```mojo
# CORRECT
@parameter
def bench_something(mut b: Bencher) raises:
    var data = setup_data()  # setup outside hot loop
    @always_inline
    def call_fn() unified {read}:
        var result = black_box(data).some_operation(black_box(arg))
        keep(result)
    b.iter(call_fn)
```

- **Always `black_box()` inputs and `keep()` results.** Otherwise the compiler may optimize away the benchmark.
- **Setup goes outside `call_fn`.** Use `iter_with_setup` for destructive benchmarks. Setup must not raise.
- **Data must match the described scenario.** If the docstring says "element in the middle", verify it.
- **Provide benchmarks for performance PRs**, covering both small and large inputs.

## Testing

- **Add unicode test cases** for any string/byte operation.
- **Cover both hit and miss paths** in search/contains tests.
- **Add edge cases:** `start > end`, out-of-bounds indices, empty input, exact-width.
- **Tests for `Writable` must use `check_write_to`**, not `String(x)` or `repr(x)`.

## SIMD / memory safety

- **Validate pointer arithmetic** before low-level memory access. Clamp `[start, end]` into `[0, len]` before `memcmp`.
- **Check for negative `pos_in_block`** in reverse SIMD scans with `count_leading_zeros`.
- **Verify `vectorized_end`** accounts for full SIMD block width to prevent OOB reads.
- **SIMD test comments must be platform-agnostic.** Write "exercises both SIMD path and scalar tail" instead of assuming 64-byte width.

## Writable trait (stdlib-specific additions to mojo-syntax)

The `mojo-syntax` skill covers basic `Writable` patterns. Additional stdlib-contributing rules:

- **`write_to` and `write_repr_to` must not allocate.** No `repr()`, `String(x)`, or heap-allocating calls inside these methods.
- **Use `FormatStruct`** from `format._utils` for consistent repr formatting of structured types.
- **Add an explicit `else` fallback** in enum-style `write_to`.
