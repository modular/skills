---
name: closure_migration
description: >-
  Migrates Mojo code off legacy parametric closures (`capturing[_]`,
  `@__parameter` / `@parameter`, `api[fn](args)`) onto value-taking unified
  closures (`api(args, fn)` with `{imm}` / `{mut}` / `{var}` / named capture
  lists). Use when removing parametric overloads, fixing "capturing thin"
  conversion errors, rewriting nested launch/callback closures, or migrating
  any API that took a comptime function parameter.
---

# Closure migration

Migrate callers **before** deleting parametric overloads. Prefer value-taking
APIs with unified closures. Pair with `mojo-syntax` (and
`mojo-gpu-fundamentals` for GPU launch code).

## Forbidden

**Hard ban — do not under any circumstance** add `@__parameter` / `@parameter`
to a nested closure to persist, introduce, or paper over a legacy closure.
Not as a migration bridge, not to satisfy a still-capturing API, not to
“borrow imm”, not behind a thin `*_value` wrapper. That is forbidden.

| Do not | Do instead |
|--------|------------|
| `@__parameter` / `@parameter` on nested defs | Unified `def … {imm}:` / `{mut x, imm}:` / named captures |
| `@__parameter` body + `*_value` forwarder | Make the body unified; pass it as a value |
| `@__parameter` “imm borrow” helper | File-scope / normal function with an imm parameter |
| `@__copy_capture` on a nested closure | Capture list (`{imm}`, `{mut x, imm}`, …) |
| Keep `@__parameter` because an API is still `capturing[_]` / comptime `fn` | Migrate or widen that API to value-taking; do not paper over it |

If a callee still only accepts a comptime `capturing[_]` function parameter,
use a nested `def … capturing -> T` **without** `@__parameter` when that is
what the type requires, or **change that API** (or leave the call site
unmigrated). Never put `@__parameter` on the caller.

For a still-`capturing` epilogue next to unified launch closures: capture an
**imm parameter** (e.g. `residual_buf: DeviceBuffer[…]`) directly. Do **not**
leave a live `var` LayoutTensor/pointer of that origin in the same scope —
that mut+imm aliases once unified children are formed.

## Target shapes

| Legacy | Preferred |
|--------|-----------|
| `api[fn](a, b)` | `api(a, fn, b)` |
| Comptime param `fn: def(...) raises capturing[_] -> None` | `FuncType: def(...) raises -> None` + runtime `ref func: FuncType` (or `func: FuncType`) |
| Nested `@__parameter` / `@parameter` def | Unified `def ...(…) raises {imm}:` / `{mut buf, imm}:` / named captures |

`@__parameter` nested defs type as `capturing thin` and do **not** convert to
a unified `FuncType`. Call rewrite alone is not enough — change how the
closure is declared (and never re-add `@__parameter`).

## Checklist

1. Inventory: `rg 'api_name\[' --glob '*.mojo'` and `rg '@__parameter|@parameter' --glob '*.mojo'`
2. Rewrite calls: `api[fn](a, b)` → `api(a, fn, b)`
3. On every nested closure in scope: drop `@__parameter` / `@parameter` /
   `@__copy_capture`; add a capture list
4. If a callee still needs a comptime capturing param → migrate that API first
   (or leave the call site unmigrated); **do not** keep `@__parameter` on the
   caller
5. Delete parametric overloads only after callers typecheck
6. Update skills/docs that still teach the legacy path
7. Typecheck: `mojo build --emit llvm <file> -o /tmp/x.ll` (filters Metal noise)
8. Self-check: `rg '@__parameter|@parameter' --glob '<touched>.mojo'` → zero on
   nested closures you own

## Capture choice

| Default / symptom | Choice |
|-------------------|--------|
| Read-only use of outer state | `{imm}` (capture-all) — **not** if the body calls `offset_ptr` or builds a mut `TileTensor` (see freeze note below) |
| Mutates some outer state; also reads `Int` / other register-passable values | `{mut buf, imm}` — **not** capture-all `{mut}` |
| Mutates several outer names | `{mut a, mut b, imm}` |
| Needs ownership / move | `{var}` or named `var x` |
| Named precision only | `{mut count}`, `{imm buf, imm shape}` |
| No free runtime captures | `{}` |
| `Could not infer capture convention` | Add `{imm}` / `{mut name, imm}` / named list |
| `expression must be mutable in assignment` on a capture | That name needs `mut` |
| `register passible value … can not be captured by 'mut'` | Capture-all `{mut}` pulled in an `Int` (etc.) — use `{mut buf, imm}` |
| `.mut … is 'False' but … is 'True'` on `.unsafe_ptr()` / `TileTensor` | Buffer captured `{imm}` — `{mut out_buf, imm}`; do **not** paper over with `unsafe_mut_cast` |
| `'lit.call' op callee expected call argument #0` on `offset_ptr` / similar | `{imm}` froze the buffer used as `self` — `{mut cb_a, mut cb_b, …, imm}` |
| `cannot bind an RValue to a reference` on `bench_func` | `kernel_launch` nested inside `bench_func` with `@__copy_capture` — define `kernel_launch` at outer function scope and remove `@__copy_capture` from `bench_func` |
| `expected ':' in function definition` at `raises {…}` | Capture list on an `@__parameter` def — strip `@__parameter` and keep the list |
| `aliasing values passed immutably…mutably` / note names `origin_of(buf)` | Closure struct would hold fields that alias the same origin, one mut and one imm — see **Aliasing** below |
| `cannot capture … not copyable` / not a parameter reference | `{imm}` if possible; else `{var}` / named `var x`; never `@__parameter` |

Do **not** use capture-all `{mut}` when the closure also mentions register-passable
outer values (`Int`, indices, lengths). Mix an explicit `mut` name with a
trailing default `imm` (`{mut tt_in, imm}`). At most one bare convention
(`imm` / `mut` / `var`) may appear as the default for unlisted captures.

`{imm}` **freezes** captured buffers. That is wrong whenever the body:

- writes an output (`TileTensor` / `.unsafe_ptr()` where `mut=True` is required), or
- calls a method that internally does `self._buf.unsafe_ptr()` then
  `unsafe_mut_cast[True]()` — `CacheBustingBuffer.offset_ptr` is the usual
  case. The method may be declared `self` (imm); the captured field still
  fails to match (`lit.call` argument #0). Mut-capture every such buffer
  (`{mut cb_a, mut cb_b, mut cb_c, mut cb_a_scales, mut cb_b_scales, imm}`).

Do **not** “fix” this with extra `unsafe_mut_cast` (the method already has
that) or by restoring `@__parameter`. Only skip `mut` when that would create
an aliasing pair with another captured field of the same origin (see below).

### Aliasing

When several captured / nested values reference the **same origin** and one of
them is mutable, the closure struct would contain aliasing fields. Prefer, in
order:

1. **Make the mutable use immutable** if the API allows (read-only
   `DeviceBuffer`, `as_immut()` views, drop `{mut …}`). Prefer this also when
   an outer `var buf` stays mutable while a nested unified closure imm-captures
   a view of `buf`: call a **normal function** (file-scope or otherwise
   non-capturing) that takes `buf: DeviceBuffer[…]` as an imm parameter, and
   form the launch there. `enqueue_memset` takes an imm `DeviceBuffer`, so
   memset alone does not require `{mut buf, …}`.
2. **Pass the mutable origin as a closure argument** (`mut buf: …` in the
   parameter list) so it is not a captured field. When the value-taking API's
   `FuncType` is fixed, use a thin `{mut buf, imm}` adapter that only forwards
   `buf` into the all-imm body. Prefer widening the API when practical.
3. Do **not** “fix” aliasing with `.as_unsafe_any_origin()` (or similar). That
   erases the lifetime tracker and is highly discouraged.
4. Do **not** “fix” anything with `@__parameter`.

## Safety rules

- Key bulk edits off the **value-argument name**, not every def with that name
- Do not bulk-replace capture lists (destroys `{mut count}` etc.)
- `name[i] =` inside `with … as name` is a local, not an outer `mut` capture
- Zero `@__parameter` / `@parameter` on nested closures in migrated code

## More detail

Full step-by-step process, error catalog, and verification:
[process.md](process.md).
