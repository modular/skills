# Legacy → unified closure migration process

General playbook for moving Mojo off parametric `capturing[_]` /
`@__parameter` closures onto value-taking unified closures. Applies to any
API shaped like `api[fn](…)` with a comptime function parameter — benchmarks,
timers, callbacks, higher-order kernels, etc.

## Forbidden

**Hard ban — do not under any circumstance** add `@__parameter` /
`@parameter` (or `@__copy_capture`) to a nested closure to persist, introduce,
or paper over a legacy closure — not as a bridge, not to satisfy a
still-capturing API, not for an “imm borrow”. Forbidden.

If a callee still only accepts a comptime `capturing[_]` function parameter,
prefer migrating that API. Until then, a nested `def … capturing` **without**
`@__parameter` is allowed when the type requires it. Do not put
`@__parameter` on the caller.

## Why a call rewrite is not enough

`@__parameter` (and legacy `@parameter`) nested defs type as:

```text
def(...) raises capturing thin -> None
```

Value-taking overloads expect a unified closure type, e.g.:

```text
FuncType: def(...) raises -> None
```

Those do not convert. After `api[fn](a, b)` → `api(a, fn, b)` you must also
change how `fn` is declared (drop `@__parameter`, add a capture list). Typical
error if you skip that:

```text
candidate not viable: value passed to 'func' cannot be converted from
'def(...) raises capturing thin -> None' to 'FuncType'
```

## Step 1 — Inventory

```bash
rg 'api_name\[' --glob '*.mojo'          # parametric calls
rg 'api_name\(' --glob '*.mojo'          # already value-taking
rg '@__parameter|@parameter|@__copy_capture' --glob '*.mojo'
```

## Step 2 — Mechanical call rewrite

```text
api[NAME](arg0, arg1)  →  api(arg0, NAME, arg1)
```

Argument order follows the value-taking overload (often “receiver / state,
closure, then context”). Confirm against the API definition before bulk edit.

```bash
rg 'api_name\[\w+\]\(' --glob '*.mojo'   # expect no parametric calls
# `def api_name[` type-parameter syntax on the definition itself is fine
```

## Step 3 — Migrate nested closures

For each nested def passed as a **runtime** closure argument (or otherwise
owned by this migration):

1. Remove `@__parameter` / `@parameter` / `@__copy_capture`
2. Keep other decorators (`@always_inline`, …)
3. Add a capture list:
   - `{imm}` — read-only outer state. **Not** the default if the body calls
     `offset_ptr` or builds a mut `TileTensor` / kernel output — those freeze
     under `{imm}` (see Step 4)
   - `{mut buf, imm}` — mutate `buf`, capture everything else as `imm`
     (required when the body also uses register-passable values like `Int`)
   - `{mut a, mut b, imm}` — several mutated names (incl. every
     `CacheBustingBuffer` passed to `offset_ptr`)
   - `{var}` / named — ownership or precise conventions
   - `{}` — only when there are no free runtime captures
   - **Never** capture-all `{mut}` if the closure also reads `Int` / indices /
     lengths — those are register-passable and cannot be `mut`-captured

If the nearest use is still a **comptime** `something[NAME](` and that API has
not been migrated yet, **stop and migrate that API** (or leave the whole call
site for a follow-up). Do not keep `@__parameter` on `NAME`.

## Step 4 — Fix by error class

```bash
source ./utils/start-modular.sh
mojo build --emit llvm path/to/file.mojo -o /tmp/chk.ll 2>&1 | grep ': error:'
```

| Error | Cause | Fix |
|-------|--------|-----|
| `capturing thin` → `FuncType` | Value arg still `@__parameter` | Strip decorator; add `{imm}` or `{mut name, imm}` |
| `Could not infer capture convention` | Free vars without a capture list | Add `{imm}` / `{mut name, imm}` / named |
| `expression must be mutable in assignment` | Capture is `imm` but body mutates it | Give that name `mut` |
| `register passible value … can not be captured by 'mut'` | Capture-all `{mut}` included an `Int` (etc.) | `{mut buf, imm}` — bare `imm` is the default for the rest |
| `.mut … is 'False' but … is 'True'` on `.unsafe_ptr()` / `TileTensor` | Buffer captured `{imm}` | `{mut out_device, imm}` — not `unsafe_mut_cast`, not `@__parameter` |
| `'lit.call' op callee expected call argument #0` on `offset_ptr` (or similar) | `{imm}` froze `CacheBustingBuffer` / `DeviceBuffer` used as `self`; `offset_ptr` does `unsafe_ptr()` + `unsafe_mut_cast[True]()` internally | `{mut cb_a, mut cb_b, mut cb_c, …, imm}` for every buffer passed to `offset_ptr` or used as kernel output |
| `cannot bind an RValue to a reference` on `bench_func` | `kernel_launch` nested inside `bench_func` with `@__copy_capture` | Define `kernel_launch` at outer function scope; drop `@__copy_capture` on `bench_func` |
| `expected ':' in function definition` at `raises {…}` | Capture list left on a still-`@__parameter` def | Strip `@__parameter`; keep the list |
| `aliasing values passed immutably…mutably` / note `origin_of(buf)` | Closure would capture mut+imm fields that alias the same origin | **(1)** Prefer making the mutable use immutable (incl. file-scope imm borrow of `buf`). **(2)** Explicitly mark the backing buffer `mut` in the capture list (`{mut buf, imm}`) if a captured helper mutates it. **(3)** Else pass the mutable origin as a **parameter** (`mut buf: …`). Do **not** use `.as_unsafe_any_origin()` or `@__parameter` |
| `cannot capture … not copyable` / not a parameter reference | Capture-all over a bad type | `{imm}` / `{var}` / named; never `@__parameter` |
| Counters / mut locals broken after bulk `{imm}` | Capture list overwritten | Restore `{mut name}` |

`{imm}` freezes captured buffers. A launch that only *looks* read-only still
needs `{mut buf, imm}` when it builds a mut `TileTensor` or calls
`CacheBustingBuffer.offset_ptr` (or any method that internally
`unsafe_ptr()` + `unsafe_mut_cast[True]()`). Mut-capture every such buffer;
do not paper over with extra casts or `@__parameter`.

Cache-busting launches that rewrite `tt_in` / `in_bufs` / `in_tensors` and
also read lengths or `ctx_idx` should use `{mut tt_in, imm}`, not `{mut}` —
but only when those rewrites do not alias other imm-captured views of the
same storage (if they would, prefer imm-only or pass the mut root as an arg).

### Fixing mut+imm origin aliasing

```mojo
# Bad: outer `var buf` is still mutable while a unified closure imm-captures
# a view of the same origin.
var buf = ctx.enqueue_create_buffer[dtype](n)
var view = TileTensor(buf.unsafe_ptr(), shape)
def call_fn(ctx: DeviceContext, cache_iter: Int) raises {imm}:
    kernel(..., view, ...)

# Preferred (1): form launch under an imm borrow of buf (normal function).
def run_with_imm_buf(
    mut b: Bench,
    buf: DeviceBuffer[dtype],
    ...
) raises:
    var view = TileTensor(buf.unsafe_ptr(), shape)
    def call_fn(ctx: DeviceContext, cache_iter: Int) raises {imm}:
        kernel(..., view, ...)
    api(..., call_fn, ...)

run_with_imm_buf(b, buf, ...)

# Preferred (2): mutable origin is a parameter, not a captured field.
# Fixed FuncType APIs use a thin adapter that only forwards the mut root.
def call_fn(
    ctx: DeviceContext,
    cache_iter: Int,
    mut bufs: List[DeviceBuffer[dtype]],
) raises {imm}:
    var out = TileTensor(bufs[i].unsafe_ptr(), shape)
    kernel(out, ...)

def call_fn_adapt(ctx: DeviceContext, cache_iter: Int) raises {mut bufs, imm}:
    call_fn(ctx, cache_iter, bufs)
```

Do not “solve” aliasing with `.as_unsafe_any_origin()` or `@__parameter`.

## Step 5 — Delete parametric API overloads

Only after callers typecheck. Remove overloads of the form:

```mojo
def api[
    fn: def(...) raises capturing[_] -> None
](...):
    ...
```

Keep value-taking overloads:

```mojo
def api[
    FuncType: def(...) raises -> None,
](..., ref func: FuncType, ...):
    ...
```

Unused parametric siblings can stay if they are out of scope for the change.

## Step 6 — Teaching surfaces

Update any skill or doc that still shows `@__parameter` nested closures for
the migrated API. Point examples at value-taking + capture lists
(`mojo-syntax`, domain skills).

## Step 7 — Verify

- API unit / integration tests for the value-taking path
- Spot `mojo build --emit llvm` on representative callers
- `rg 'api_name\[\w+\]\(' --glob '*.mojo'` clean for the migrated API
- `rg '@__parameter|@parameter' --glob '<touched>.mojo'` → no nested closures
  you own
- Ignore host-only backend noise (Metal / wrong-arch instantiation) when
  judging migration regressions

## Bulk-edit tips

- Transform in layers: calls first, then captures, then API deletion
- Match the value-argument identifier at the call site; do not globally edit
  every `def` with that name
- Never blindly rewrite all `{…}` capture lists in a file
- Prefer `{mut name, imm}` when a scan shows `name[…] =` on a **non-local**
  outer name and the body also uses register-passable values; never bare
  capture-all `{mut}` in that case
- `with buf.map_to_host() as host: host[i] = …` mutates a local binding — keep
  `{imm}`; do not list `host` in the capture list

## Worked instance (optional grounding)

One completed instance of this process was migrating
`bencher_iter_custom[fn](b, ctx)` → `bencher_iter_custom(b, fn, ctx)` in
`max/mojo/max/benchmark/bencher.mojo` and its callers. Useful references after
that change:

| Role | Path |
|------|------|
| Value-taking API | `max/mojo/max/benchmark/bencher.mojo` |
| Named `{mut …}` captures | `max/mojo/test/benchmark/test_bencher_iter_custom.mojo` |
| Clean `{var}` style | `max/kernels/benchmarks/gpu/layout/bench_tile_io_copy.mojo` |
| Mixed comptime + value in one file | `max/kernels/benchmarks/gpu/bench_launch.mojo`, `…/bench_stencil.mojo` |
| `{mut}` for in-place cache-bust | `max/kernels/benchmarks/gpu/comm/bench_allgather.mojo` (`{mut tt_in, imm}`) |
| `{mut}` for kernel output under `{imm}` freeze | `max/kernels/benchmarks/gpu/nn/bench_concat.mojo` (`{mut output_device, imm}`) |
| `{mut}` for `offset_ptr` `self` | `max/kernels/benchmarks/gpu/linalg/bench_block_scaled_matmul.mojo` (`{mut cb_a, mut cb_b, mut cb_c, mut cb_a_scales, mut cb_b_scales, imm}`) |
| Unified launch + still-`capturing` residual epilogue | `Kernels/benchmarks/gpu/linalg/bench_matmul_reducescatter.mojo` — imm `residual_buf` param; nested `def … capturing` loads via `residual_buf.unsafe_ptr()` (no live `var` LayoutTensor/ptr; no `@__parameter`) |

Treat those as examples of the general rules above, not as the scope of this
skill.
