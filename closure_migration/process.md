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
   - `{var x}` — each name that was `@__copy_capture(x)` (copies into
     storage). Required for `Int` / other register-passable locals; `{imm}`
     of `var n = …` aliases (`origin_of(n)`)
   - `{imm}` — read-only outer state. **Not** the default if the body calls
     `offset_ptr` or builds a mut `TileTensor` / kernel output — those freeze
     under `{imm}` (see Step 4)
   - `{mut buf, imm}` — mutate `buf`, capture everything else as `imm`
     (required when the body also uses register-passable values like `Int`)
   - `{mut a, mut b, imm}` — several mutated names (incl. every
     `CacheBustingBuffer` passed to `offset_ptr`)
   - `{var}` capture-all — only when every capture should be owned
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
| `expression must be mutable in assignment` | Capture is `imm` but body mutates it | Give that name `mut`, or declare temporary view arrays locally inside the closure if rebuilt per iteration |
| `register passible value … can not be captured by 'mut'` | Capture-all `{mut}` included an `Int` (etc.) | `{mut buf, imm}` — bare `imm` is the default for the rest |
| `.mut … is 'False' but … is 'True'` on `.unsafe_ptr()` / `TileTensor` | Buffer captured `{imm}` | `{mut out_device, imm}` — not `unsafe_mut_cast`, not `@__parameter` |
| `'lit.call' op callee expected call argument #0` on `offset_ptr` (or similar) | `{imm}` froze `CacheBustingBuffer` / `DeviceBuffer` used as `self`; `offset_ptr` does `unsafe_ptr()` + `unsafe_mut_cast[True]()` internally | `{mut cb_a, mut cb_b, mut cb_c, …, imm}` for every buffer passed to `offset_ptr` or used as kernel output |
| `cannot bind an RValue to a reference` on `bench_func` | `kernel_launch` nested inside `bench_func` with `@__copy_capture` | Define `kernel_launch` at outer function scope; drop `@__copy_capture` on `bench_func` |
| `expected ':' in function definition` at `raises {…}` | Capture list left on a still-`@__parameter` def | Strip `@__parameter`; keep the list |
| `aliasing values passed immutably…mutably` / note `origin_of(buf)` | Closure would capture mut+imm fields that alias the same origin | **Aliasing** below: (1) capture as read, (2) disassemble, (3) pass mut as an argument. Do **not** use `.as_unsafe_any_origin()` or `@__parameter` |
| `aliasing … origin_of(n)` on an `Int` local | `@__copy_capture(n)` became an imm ref to `var n` | `{var n}` (copy). Do not delete `n` / rewire to another `Int` unless that is smaller |
| `aliasing values passed mutably to 'x' … and passed mutably to 'y'` | Two mut captures of the same origin (view + backing buffer, or a value plus a nested helper that also captures it) | Same **Aliasing** order |
| `'lit.call' op invalid symbol use` / `origin<false>` vs `origin<true>` on `bench_function[fn]` | `@__parameter` capturing-thin `bench_fn` captured unified `{imm}` children; capturing-thin origin params are mut | Drop `@__parameter`; `bench_function(fn, id, …)` value-taking |
| `cannot capture … not copyable` / not a parameter reference | Capture-all over a bad type | `{imm}` / `{var}` / named; never `@__parameter` |
| Counters / mut locals broken after bulk `{imm}` | Capture list overwritten | Restore `{mut name}` |
| Memset / launch aliases with a wrapper whose ptr is already `MutUntrackedOrigin` | Second `DeviceBuffer` over `EPLocalSyncCounters.ptr` / `offset_ptr` (origin already erased) | Pass the original `DeviceBuffer` as an imm argument. `enqueue_memset` takes imm. Do **not** wrap the untracked ptr |

`{imm}` freezes captured buffers. A launch that only *looks* read-only still
needs `{mut buf, imm}` when it builds a mut `TileTensor` or calls
`CacheBustingBuffer.offset_ptr` (or any method that internally
`unsafe_ptr()` + `unsafe_mut_cast[True]()`). Mut-capture every such buffer;
do not paper over with extra casts or `@__parameter`.

Cache-busting launches that rewrite `tt_in` / `in_bufs` / `in_tensors` and
also read lengths or `ctx_idx` should use `{mut tt_in, imm}`, not `{mut}` —
but only when those rewrites do not alias other imm-captured views of the
same storage (if they would, prefer imm-only or pass the mut root as an arg).

### Fixing origin aliasing

Two captures that alias the same memory are allowed only when both are
read-only. `@__copy_capture(n)` on an `Int` becomes `{var n}` (copy). `{imm}`
of `var n = …` aliases `origin_of(n)` once a unified sibling embeds it — do
not delete `n` just to dodge that.

Otherwise, in order:

1. **Capture as read.** Drop `{mut …}` / pass an imm parameter if the body
   does not mutate that origin.
2. **Disassemble.** If the compiler overlaps two uses because they capture a
   whole object but the mutation does not touch the other use's memory, hoist
   the fields you need (finer-grained origin tags) instead of the parent.
3. **Pass mut as an argument** so it is not a captured field. When
   `FuncType` is fixed, a thin `{mut buf, imm}` adapter forwards `buf`.

```mojo
# Bad: outer `var buf` is still mutable while a unified closure imm-captures
# a view of the same origin.
var buf = ctx.enqueue_create_buffer[dtype](n)
var view = TileTensor(buf.unsafe_ptr(), shape)
def call_fn(ctx: DeviceContext, cache_iter: Int) raises {imm}:
    kernel(..., view, ...)

# (1) Capture as read: form launch under an imm borrow of buf.
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

# (2) Disassemble: do not capture whole `config` next to a mut `capturing`
# use of it — hoist the Ints the helper actually reads.
var rank_counts = Array[Int, ngpus](uninitialized=True)
comptime for i in range(ngpus):
    rank_counts[i] = config.rank_units(i)

# (3) Lift only a capture that must be mut and still aliases another
# use — pass it as an argument. Read-only captures stay captures.
def rebuild(cache_iter: Int, mut shards: Array[TileType, ngpus]) {imm}:
    shards[j] = make_shard(cache_iter, rank_counts[j])

def call_fn(
    ctx: DeviceContext,
    cache_iter: Int,
    mut bufs: List[DeviceBuffer[dtype]],
) raises {imm}:
    var out = TileTensor(bufs[i].unsafe_ptr(), shape)
    kernel(out, ...)

def call_fn_adapt(ctx: DeviceContext, cache_iter: Int) raises {mut bufs, imm}:
    call_fn(ctx, cache_iter, bufs)

# Also (3): memset the original buffer. Do not wrap an already-untracked
# ptr (`EPLocalSyncCounters.ptr`) in a second DeviceBuffer.
def clean_up(
    ctx: DeviceContext,
    atomic_counter: DeviceBuffer[DType.int32],
) raises {}:
    ctx.enqueue_memset(atomic_counter, 0)
```

Do not “solve” aliasing by erasing the origin: `.as_unsafe_any_origin()`,
`unsafe_origin_cast[MutUntrackedOrigin]` / `MutAnyOrigin` on the original
buffer, or `DeviceBuffer(ctx, some.ptr, …, owning=False)` over a wrapper that
already origin-cast. Do not use `@__parameter`.

### Stay NFC

A migration changes capture lists and call shape (`api[fn](…)` → `api(…, fn,
…)`). It does not change when memory is allocated, which layout constructor
runs, or what the timed `call_fn` does on the host.

- Locals that lived in a `comptime if` arm stay there: buffers, host copies,
  `TileTensor`s, `row_major[M, N]()` layouts. Do not hoist them to function
  scope with a size-1 placeholder on the other arms so a higher closure can
  capture them. Define the unified `def` in the same arm.
- Do not replace a comptime layout (`row_major[M_2D, out_channels]()`) with
  `row_major(Coord(IndexList[…]))`. The `Coord` form is dynamic and can break
  kernels that read `static_shape`.
- Host objects built once (`a_shape`, `c_full_shape`) stay in the helper body
  and are `{imm}`-captured. Rebuilding them inside the timed `call_fn` is extra
  measured work.
- Lift a capture to an argument **only** for origin exclusivity (or the `{}`
  helper around an imm `DeviceBuffer` param). Do not thread extra values as
  arguments just to move a closure.

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
- Diff the timed `call_fn` body and allocation sites against the pre-migration
  file: only capture lists and `api[fn](…)` → `api(…, fn, …)` should change.
  No new `.as_unsafe_any_origin()` / `unsafe_origin_cast` /
  `DeviceBuffer(…, some.ptr, owning=False)` used to dodge aliasing
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
- Do not hoist `comptime if` arm allocations so a closure at a higher scope can
  see them; keep the closure in the arm
- Do not wrap `MutUntrackedOrigin` pointers to paper over aliasing; pass the
  original `DeviceBuffer`

## Worked instance (optional grounding)

One completed instance of this process was migrating
`bencher_iter_custom[fn](b, ctx)` → `bencher_iter_custom(b, fn, ctx)` in
`max/mojo/max/benchmark/bencher.mojo` and its callers across benchmarks and GPU tests. Useful references after
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
| Unified launch + residual epilogue | `Kernels/benchmarks/gpu/linalg/bench_matmul_reducescatter.mojo` — imm `residual_buf` param; host `LayoutTensor` from `unsafe_ptr()`. `{var residual_lt}` does not convert to `elementwise_compute_lambda_type` (`capturing thin`); until that API is migrated, keep `@__copy_capture` on `def … capturing` (no `@__parameter`). Never call `DeviceBuffer` methods inside the GPU epilogue. Build `a_shape` in the helper body, not inside timed `call_fn` |
| Imm `DeviceBuffer` arg instead of untracked wrap | `max/kernels/benchmarks/gpu/comm/bench_ep_combine.mojo` (`clean_up(ctx, atomic_counter)` / `enqueue_memset` on the original buffer) |
| Unified closure stays in the `comptime if` arm | `max/kernels/benchmarks/gpu/nn/bench_conv2d.mojo` (AMD buffers / `output_2d_layout` stay in the `amd_4wave` arm) |

Treat those as examples of the general rules above, not as the scope of this
skill.
