# Idiomatic Mojo

The main `mojo-syntax` skill makes code compile; these rules make it pass
review. Before submitting or reviewing Mojo, scan the diff for every WRONG
pattern below. The first five sections are the ones reviewers flag most
often in agent-written code.

## 1. Use existing helpers; never open-code them

Search `std.math`, `std.gpu`, `nn/`, `linalg/`, and the TileTensor API
before writing index math, reductions, reshapes, or activations. The
existing code a few lines away usually already uses the helper.

```mojo
# WRONG                                  # CORRECT
(M + BM - 1) // BM                       ceildiv(M, BM)          # std.math
(x + a - 1) // a * a                     align_up(x, a)          # std.math
var s = b // H; var h = b % H            var s, h = divmod(b, H)
1 / (1 + exp(-x))                        sigmoid(x)              # nn.activations
manual shuffle-reduce loop               warp.sum(val)           # std.gpu warp
hard-coded 32 lanes                      WARP_SIZE
scalar tap loop + cast per element       (v.load[w](i) * wts).reduce_add(), cast after reduce
manual ", " string-building loop         ", ".join(items)
```

`ceildiv` works on `Int` and on any `SIMD` scalar (`UInt32`, ...).
Prefer `vectorize` over hand-written scalar loops, and call LLVM/NVVM
intrinsics rather than inline assembly.

Vectorized pointer loads and stores default to the scalar alignment, which
can split them into scalar accesses. Pass the real alignment; the address
must actually satisfy it, or the access faults:

```mojo
# WRONG
var v = ptr.unsafe_load[width=8](off)
# CORRECT
comptime align = align_of[SIMD[dtype, 8]]()
var v = ptr.unsafe_load[width=8, alignment=align](off)
```

## 2. Don't cast values that already have the right type

`thread_idx`, `block_idx`, `block_dim`, `grid_dim` fields return `Int`.
Don't wrap them in `Int(...)`, and don't copy that pattern from older
code. Keep a value in its natural type; a downcast followed by an upcast
is a review flag.

```mojo
# WRONG                                           # CORRECT
var token = Int(block_idx.x)                      var token = block_idx.x
var tile_m = UInt32(block_idx.y)  ...  Int(tile_m)   var tile_m = block_idx.y
```

Cast only at the boundary where the other type is actually required
(for example, comparing against a `UInt32` offset).

## 3. Let parameters be inferred

Omit parameter lists that the arguments already determine. Don't add a
`rank`/`dtype`/`layout` parameter that the argument type carries.

```mojo
# WRONG
Self.run_at_tile[c_layout, a_layout, b_layout, c_engine, a_engine, b_engine](c, a, b, m, n)
epilogue[dtype, width, alignment=alignment](idx, val)
def f[rank: Int, axis: Int](shape: IndexList[rank]): ...
x: TileTensor[DType.float32, layout, origin, Engine=engine]

# CORRECT
Self.run_at_tile(c, a, b, m, n)
epilogue[alignment=alignment](idx, val)
def f[axis: Int](shape: IndexList): ...
x: TileTensor[DType.float32, ...]        # `...` always goes last
def g[dtype: DType, //, filter_len: Int](...)   # infer-only params before `//`
```

Check the parameter's default before dropping it — removing an explicit
`[2]` where the default is `1` changes behavior.

A default is not a constraint: with `b_type: DType = a_type`, a check
comparing the two never fires for callers that omit `b_type`.

## 4. Build coordinates from tuples, not `IndexList`/`Index`

Tuples convert implicitly to coordinate types. A single-element tuple is
`(x,)`. Don't create an `IndexList` or `Index` only to pass it on.

```mojo
# WRONG                                   # CORRECT
Index(m, n)                               (m, n)
f(Index(tokens))                          f((tokens,))
t.load[width=1](IndexList[2](i, k))       t.load((i, k))
Coord(IndexList[2](a, b))                 Coord(a, b)
Coord(index_list[0], ..., index_list[N])  Coord(index_list)
Coord(Idx[1], Idx[5])                     coord[1, 5]
t.load((Idx[5], Idx[3]))                  t.load(coord[5, 3])
```

If a mixed-type tuple fails to unify (e.g. `Tuple[Int, UInt32]`), cast the
odd element; don't fall back to `IndexList`. For fully static shapes and
sizes, prefer the `coord` comptime helper.

## 5. `comptime` for compile-time-known work

```mojo
# WRONG                                   # CORRECT
for i in range(CHUNK):                    comptime for i in range(CHUNK):
nested comptime for x3                    comptime for a, b, c in product(range(X), range(Y), range(Z)):  # std.itertools
for i in range(rank - 1, -1, -1):         comptime for i in reversed(range(rank)):
var scale = 1.0 / sqrt(Float32(D))        comptime scale = Float32(1.0) / sqrt(Float32(D))
if ctx.api() == "metal":  (runtime)       comptime if ...:   # target/vendor checks are always comptime
```

Don't unroll a loop that doesn't benefit — reviewers ask "do you need to
unroll here?". If a comptime `Float32` math call fails to fold on GPU,
file an issue rather than silently changing the dtype.

Keep comptime dimensions static all the way into layouts. Dispatch
heuristics read `static_shape`; passing a comptime value as a runtime
`Int` erases it and silently routes to a slower path.

```mojo
# WRONG: N is comptime, but c.static_shape[1] == -1
var c = TileTensor(ptr, row_major((M, N)))
# CORRECT
var c = TileTensor(ptr, row_major((M, Idx[N])))
```

## 6. Don't `rebind` unless the types already match

`rebind` is a compile-time promise that two types are identical after
elaboration, not a cast. Try deleting every `rebind` you add or touch; keep
it only if the code stops building without it. To reinterpret a pointer, use
`ptr.unsafe_bitcast[T]()`; to convert a `SIMD` value, use
`x.cast[DType]()`.

```mojo
# WRONG: rebinds a type back to itself
p.init_pointee_copy(rebind[StrideType](StrideType()))
# CORRECT
p.init_pointee_copy(StrideType())

# WRONG: bridges different TileTensor layouts; the stored Coord shapes the
# ABI, so row_major[4, 4]() and row_major((Idx[4], 4)) differ in size
rebind[TileTensor[dtype, LayoutB, origin]](tile_a)
# CORRECT: build the tile with the layout the consumer expects

# WRONG: spell out params, then rebind so inference works
# CORRECT: pass tuples/values directly and let parameters infer
```

- After a migration, check whether old rebinds are still needed.
- Legitimate: `rebind[IndexList[rank]](x.canonicalize())` (valid only
  after `canonicalize()`), `SIMD` width/dtype reconciliation inside
  epilogue lambdas, and return-boundary nominal mismatches.
- Use an existing coercion helper (e.g. `_coerce_dynamic[T]`) rather
  than repeating `rebind[T](Scalar[...](...))`.
- For a constraint the compiler can't prove, use a `where` clause or
  `comptime assert`, not a rebind that hides the mismatch. A `where`
  clause pushes the check up to the caller (caught by the LSP, but generic
  callers must restate it); a `comptime assert` pushes it down into the
  body (still caught, only at elaboration). Use `where` for APIs called
  mostly with concrete types (`UInt8(1) in (1, 2, 3)` requiring
  `T in Self.Ts`), and `comptime assert` for APIs meant for generic
  contexts, where a `where` clause would force every caller to repeat the
  constraint.
- More than one rebind per function, or one you can't justify in a
  sentence, is a red flag.

The same applies to origins: never cast to an any-origin
(`.as_unsafe_any_origin()`, `unsafe_origin_cast[MutAnyOrigin]`) to make
types line up (see §7), and use `TileTensor.as_imm()` rather than
overriding `mut=False`.

## 7. Leave origins unbound; mark read-only data immutable

An argument's origin is `_` unless something forces a concrete one. `_`
infers the caller's origin, so lifetimes and exclusivity stay checked.
Never write `MutAnyOrigin`, `ImmutAnyOrigin`, or their `UnsafeAnyOrigin`
spellings in new code, and never add `.as_unsafe_any_origin()` at a call
site to satisfy one; fix the signature instead. Inputs that are never
written take immutable origins so the signature shows which arguments are
outputs.

```mojo
# WRONG                                        # CORRECT
src: Pointer[Float32, ImmAnyOrigin]            src: ImmPointer[Float32, _]
dst: Pointer[Float32, MutAnyOrigin]            dst: MutPointer[Float32, _]
x: TileTensor[dtype, L, ImmAnyOrigin].         x: ImmTileTensor[dtype, L, _]
f(p.as_unsafe_any_origin())                    f(p)
```

GPU kernel entry points are the exception: `enqueue_function` can't
infer `_`, so they take `MutUntrackedOrigin` / `ImmUntrackedOrigin`.
Helpers the kernel calls still take `_`.

A caller holding mutable data doesn't make the parameter mutable. A
mutable value converts implicitly to an immutable parameter with a
generic origin (`ImmTileTensor[dtype, L, _]`); with a concrete origin
such as `ImmUntrackedOrigin`, the caller passes `.as_imm()`. Never widen a
read-only parameter to `mut=True` to match a caller.

Loads from immutable data are invariant by default; don't pass
`invariant=True` or other no-op arguments. Prefer the named type prefixes
to describe the mutability of the underlying data such as `ImmPointer`,
`MutPointer`, `ImmTileTensor`, and `MutTileTensor` over the `mut=True`
or `mut=False` prefix.

## 8. TileTensor over raw pointers

Kernels use TileTensor, coordinate indexing, and slice syntax instead of
flat pointer offsets. Direct pointer access was a workaround for the
performance cliffs of the earlier tensor types (they carried dead bytes);
a fully static TileTensor has the same ABI as a pointer, so index the
tensor directly.

```mojo
# WRONG                                   # CORRECT
out.ptr[row * N + col] = v                out[row, col] = v
src.slice(...)                            src[:h, :w]
```

Don't index raw pointers or compute offsets from strides in kernels.
Some accelerators don't expose raw pointers (their tensors are backed by
a memref), so `ptr[offset]` and hand-written `i * stride + j` math don't
port to them. Let TileTensor own the layout: index it by coordinate, and
use `tile`, `vectorize`, and `distribute` for sub-views instead of
pointer arithmetic.

```mojo
# WRONG
var p = t.unsafe_ptr()
p[row * row_stride + col] = v
var tile_ptr = p + (tm * BM) * N + tn * BN
# CORRECT
t[row, col] = v
var tile = t.tile[BM, BN](tm, tn)
```

Don't use `TileTensor.ptr`. When you do need the pointer, call
`TileTensor.unsafe_ptr()`, and only when you know the tensor is backed by
a pointer (not, for example, a memref).

Remaining `raw_load`/`raw_store` uses need a tracking issue.

## 9. Simplify control flow; remove duplication

- Return early instead of nesting `if/else`; no `else` after `return`.
- Use `clamp(x, lo, hi)` over nested min/max calls, rather than
  `min(max(x, lo), hi)` or `min(x, max(lo, hi))`.
- Iterate collections directly (`for d in dtypes:`), not by index.
- Start accumulators at their identity (`accum = 0`) instead of peeling
  the first iteration.
- Derive lists from existing ones (`float_dtypes.join(integer_dtypes)`).
- Lift repeated blocks into a helper or `comptime` alias.
- Return a `Tuple` from a helper rather than making callers do `+1, +2`
  offset arithmetic; pack scalar kernel args into one `InlineArray` and
  buffer instead of several buffers or fake `Pointer`s.
- Prefer t-strings over string concatenation. For example:

  ```mojo
  # WRONG
  "expected " + String(expected_dtype) + " but got " + String(actual_dtype)
  # CORRECT
  t"expected {expected_dtype} but got {actual_dtype}"
  ```

## 10. Pattern matching: `match` over `elif` chains

`match` (and `comptime match`, `if case`, `while case`) are stabilized. When
several cases dispatch on one subject, a `match` replaces the `elif` chain.
When the arms poke at unrelated expressions, keep `if`/`elif`. Two safety
conditions before rewriting: `match` evaluates its subject **once** and each
case sees that snapshot, so a ladder whose arms re-evaluate or mutate the
subject between tests is not equivalent; and a `comptime match` is the form
for parameter-driven specialization — a runtime `match` on parameters
compiles both sides of every arm.

```mojo
# WRONG: elif ladder over one subject
if dtype == .float32:
    return 1
elif dtype == .float16 or dtype == .bfloat16:
    return 2
else:
    return 3

# CORRECT
match dtype:
    case .float32:
        return 1
    case .float16 | .bfloat16:
        return 2
    case _:
        return 3
```

The subject is evaluated once; cases are tried in order. For a parameter
subject, `comptime match` guarantees compile-time dispatch and binds names
as parameters (no `var`/`ref` there — they're meaningless and rejected).
Two hard rules for the comptime form: the subject must be a compile-time
value (a parameter or comptime expression — `comptime match` on a runtime
argument is "cannot use a dynamic value in 'match' subject"; use a runtime
`match` there), and guards must be compile-time evaluable too.

```mojo
comptime match (m, n, k):
    case (16, 16, 16):
        ...                      # hand-tuned 16³ path
    case (mt, nt, kt) if mt * nt <= 256:
        ...                      # bound sizes usable as parameters
    case _:
        ...
```

The most common kernel shape is a `dtype` (parameter) or enum-value
(`X.tag`) dispatch. The mechanical rewrite — `.NAME` arms for a `DType`
parameter, `== value` arms for an `Int` tag:

```mojo
# WRONG                                          # CORRECT
comptime if dtype == .float32:                    comptime match dtype:
    path_f32()                                        case .float32:
elif dtype == .float16 or dtype == .bfloat16:              path_f32()
    path_f16()                                    case .float16 | .bfloat16:
else:                                                  path_f16()
    comptime assert False, "unsupported"          case _:
                                                      comptime assert False, (
                                                          t"unsupported {dtype}"
                                                      )
```

### Pattern forms

```mojo
match x:
    case 0 | 1 | 2:              # literals + or-patterns
    case == expected:            # compare against a value/expression; see below
    case == (expected + 1):
    case n if n < 0:             # guard; never counts toward exhaustiveness
    case (0, y):                 # tuples, nested arbitrarily
    case Point(x=x, y=0):        # struct fields by name
    case {x=0, y=0}:             # subject-typed fields by name
    case == {1, offset=2}:       # compare against an initializer-list value
    case (0, var x) | (var x, 1):# or-pattern: same names bound on every
    case .Some(0):               #   alternative (payload-less cases bind none)
    case .Some(ref v):           # ref into a mutable payload writes back
    case var y:                  # owned copy (mutable); bare name is imm
    case 0 as n:                 # bind the whole subject after testing it
    case _:                      # wildcard; irrefutable, closes the match
```

### `== expr` — when an arm needs it

Use the arm's native spelling when it compares on its own; add `==` only
where the grammar requires it. Compiler-verified matrix:

| Arm                                   | Compares alone? | Notes |
|---------------------------------------|-----------------|-------|
| `case 1:` / `case "a":` / `case True:` | yes | `== 1:` **warns "redundant"** |
| `case -1:` (negative literal)          | no — error      | write `case == -1:` |
| `case {1, 2}:` (initializer list)      | no — error      | name the subject fields, such as `case {x=1, y=2}:` |
| `case .Some(0):` / `case DType.int32:` / `case _Ops.BARRIER.value:` | yes | enum tags and any dotted path compare; no `==` |
| `case LOAD_DRAM:` (bare name)          | **binds**       | even a comptime alias — silently shadows, later arms warn unreachable; write `case == LOAD_DRAM:` |
| `case e + 1:` / `case a[i]:` / calls  | no — error      | write `case == (e + 1):` |
| `case == None:`                       | —               | `None` isn't a plain literal; `==` required |

The rule: **a dotted path or enum case compares; a bare name binds; everything
else needs `==`.** It recurses into payloads (`.Some(expected)` binds;
`.Some(== expected)` compares).

```mojo
# WRONG: binds x to a new name W; doesn't compare
match x:
    case W:
    case _:

# CORRECT — == only where required
match x:
    case == W:                    # bare name: == required
    case _Ops.BARRIER.value:      # dotted path: no ==
    case == (expected + 1):       # expression: == required
    case _:
```

Types are comptime values, so `comptime match T` with `== Type` arms is the
"type switch". No `is` — `==` is type identity in Mojo:

```mojo
comptime match type_of(x):       # builtin is `type_of`; `typeof` doesn't parse
    case == Float32 | == Float64:
        ...
    case _:
        ...

# For a single type question, comptime if reads better:
comptime if type_of(x) == Float32:
    ...
```

You can't hoist `==` alternatives into a shared set (`== FPTypes` compares
against the set value) or put predicates inside patterns — spell each `== T`
out, and move trait questions to a guard on the *dtype* value
(`case dt if dt.is_floating_point():` — a bound *type* has no such method)
or a plain `comptime if`.

### Enum-disguise structs: conform to `EnumLike`

When a match (or would-be elif ladder) dispatches on a struct whose logical
cases are `comptime Self(...)` constants — a struct acting as an enum —
conform it to `EnumLike` and use `.CASE` arms: discriminant dispatch instead
of the struct's `__eq__`, plus exhaustive-match diagnostics. Contract members:

|Member|Contract|
|---|---|
|`_enum_case_names`|`ParameterList.of[...].values`: constant names in case-index order.|
|`_enum_case_types`|`TypeList` values in the same order; `NoneType` for payload-less cases.|
|`_get_enum_discriminant(self) -> Int`|Map the backing value to its case index.|
|`_unsafe_get_enum_payload[id: Int](ref self)`|Return `ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[id]`; payload-less enums `comptime assert False`.|

If unnamed backing values can exist (e.g. a `-1` "unsupported" sentinel),
set `comptime _enum_is_exhaustive = False` and keep `case _`. For
string-backed structs, map strings to case indices in
`_get_enum_discriminant` (a runtime `match self.name:` is fine);
`_enum_case_names` lists the **constant names**, not the backing strings.
In-tree conformances: `TuningGroup`, `FlashAttentionAlgorithm`, `GEMMKind`.
Recent conversions of the same shape: `UMMAKind`, `MaskName` (see the
kernel PRs applying this rule).

```mojo
# WRONG — Kind is a struct with FIRST/SECOND constants but no EnumLike
if kind == Kind.FIRST:
    return 10
elif kind == Kind.SECOND:
    return 20
else:
    return -1

# CORRECT — unnamed backing values remain possible
@fieldwise_init
struct Kind(EnumLike, TrivialRegisterPassable):
    var value: Int32
    comptime FIRST = Self(0)
    comptime SECOND = Self(1)
    comptime _enum_is_exhaustive = False
    comptime _enum_case_names = ParameterList.of[
        "FIRST".value, "SECOND".value
    ].values
    comptime _enum_case_types = TypeList.splat[2, NoneType].values

    def _get_enum_discriminant(self) -> Int:
        return Int(self.value)

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[
        Trait=AnyType, Self._enum_case_types
    ]()[id]:
        comptime assert False, "Kind has no payload"

def classify(kind: Kind) -> Int:
    match kind:
        case .FIRST:
            return 10
        case .SECOND:
            return 20
        case _:
            return -1
```

### Exhaustiveness and warnings

The compiler warns on non-exhaustive `EnumLike` subjects (incl. `Bool` and
tuples of those), on duplicate cases, and on unreachable arms. Guards never
contribute coverage, so a `case _ if cond:` is **not** a catch-all. In
kernels, prefer an exhaustive match with a `case _:` that `comptime assert`s,
so a new enum value fails at build time, not in production:

```mojo
comptime match dtype:
    case .float32 | .float64:
        ...
    case _:
        comptime assert False, t"unsupported dtype {dtype}"
```

Once every case of a runtime match is covered, a trailing `case _:` warns
unreachable — drop it. Open enums (`_enum_is_exhaustive = False`) still need
`case _`; covering the known cases doesn't close an open set.

The coverage model (what counts as "covered"):

| Subject / pattern                            | Coverage behavior |
|----------------------------------------------|-------------------|
| `case _` or bare binding                     | irrefutable — closes the whole space |
| `case X if cond:` (any guard)                | never counts toward coverage |
| `case .Some(0):`                             | doesn't cover `Some` (payload refined) |
| `case .Some(_):`                             | covers `Some` |
| `case (True, True):` on `Bool × Bool`        | covers the one combination; `(False, False)` next to it still leaves two cells open — coverage is per-combination, not per-field |
| open enum (`_enum_is_exhaustive = False`)     | known cases never close the set; `case _` stays reachable |
| `Int`/`String` literals                      | duplicates warn, but no exhaustivity requirement |

"No warning" is not proof of exhaustive or non-redundant dispatch: refined
payload duplicates and nested open-type literals aren't diagnosed, and
literal identity is spelling-based (`4` vs `0x4` are different keys).

### `if case` / `while case` / `or_else`

Pick by shape, in order:

- **Fallback to a default** → `opt.or_else(default)`. One expression, no
  branch: `var global_rank = my_rank.or_else(Int(ctx.device().id))`. Works on
  runtime and comptime Optionals alike. This replaces the
  `opt.value() if opt else default` ternary and the two-line
  default-then-overwrite dance.
- **Branch on presence** (use the payload only when it exists, other
  control flow otherwise) → `if case .Some(x) = opt:`. The pattern is the
  guard AND the binding.
- **Dispatch on the payload's shape** (multiple cases, or the None side
  carries its own logic) → `match opt: case .Some(x): ... case .None: ...`.
- Presence guaranteed by construction (an enclosing condition already
  proves it, and absence is a caller bug)? Keep `.value()` — it aborts
  loudly instead of silently skipping work.
```mojo
# WRONG: guard-then-unwrap
var global_rank = my_rank.value() if my_rank else Int(ctx.device().id)

# CORRECT: fallback is one expression
var global_rank = my_rank.or_else(Int(ctx.device().id))
```

For the branch-on-presence shape, `if case pattern = value:` evaluates the
value once, tests one pattern, and binds names for that block only; failure
falls to the next `elif`/`else`. The pattern is both the guard and the binding,
so it unwraps an `Optional` without the `is not None` / `Bool(opt)` guard +
`.value()` dance:

```mojo
# WRONG: guard-then-unwrap
if sink_weight is not None:
    sink_logit = sink_weight.value()

var stored = get_global_or_null(key)
if not stored or Int(stored.value()) != Int(record):
    raise Error("missing")

# CORRECT: the pattern IS the guard; the payload binds inline
if case .Some(w) = sink_weight:
    sink_logit = w

if case .Some(s) = get_global_or_null(key):
    if Int(s) != Int(record):
        raise Error("missing")

while case .Some(item) = get_next():   # consume while it yields
    process(item)
```

`if case` composes with the else-arm discipline: an unwrap in one branch of
a larger conditional chain stays readable as
`if case .Some(x) = opt: ... elif <other cond>: ...`. Prefer it even when
the guard is `comptime` (`comptime if case .Some(fn) = wrapper:`).

Both compose with `comptime` — parameters bind as parameters:

```mojo
comptime if case == Int = T:
    ...
elif case == String = T:
    ...
else:
    ...
```

An or-pattern binding must bind the same name compatibly on every
alternative; `ref` binds a reference to whichever element matched, and it
keeps the subject's mutability:

```mojo
match point:
    case (2, ref x) | (ref x, 3):
        use(x)                  # x references point[1] or point[0]
    case _:
        ...
```

### Review checks

- A chain of 2+ `elif x == ...` arms on one subject becomes a `match` —
  unless arms re-evaluate or mutate the subject between tests; `match`
  snapshots the subject once.
- `elif` chains over unrelated expressions stay `if`/`elif`. Parameter
  subjects take `comptime match`.
- A bare name in a pattern binds; if a comparison was meant, use `== name`
  (enum cases excepted: `case Color.red:` / `case .red:`).
- Guarded cases don't count toward exhaustiveness; missing *coverage* warns
  (a covered match's trailing `case _:` warns unreachable instead).
- `ref` bindings need a memory origin — `case ref y:` on a register-passable
  subject is an error; use it inside tuples or mutable enum payloads.

## 11. Naming

- No vendor or device class in generic APIs (`CompilationTarget(MI355X)`,
  not a `gpu_`-prefixed variant).
- Acronyms in capitals: `TMem`, not `Tmem`.
- Don't shadow builtins (`min_value`, not `min`).
- Descriptive over cryptic (`tA`, `tR` get flagged).
- Always declare locals with `var`.

## 12. API design

- `Optional[T]` instead of sentinel defaults.
- One generic entry point (`is[NVIDIA_GPU]()`) over a growing set of
  `is_xyz()` methods.
- Take `StringSlice`/`Span`/`Iterable` rather than owning types.
- Never accept an argument and silently ignore it on some targets.
- Keep advanced, low-level functions out of module exports.
- For `call_location()`, use a thin `@inline(.always)` wrapper that forwards
  the location to a `@inline(.never)` `_impl`.
- Use named results with `rebind[type_of(out)](x)` instead of restating
  a long return type (only when the rebind is otherwise required).

## 13. Invariants and hardware facts

- Query constants (`WARP_SIZE`, device limits); don't hard-code `32` or
  `128`.
- State shape and config assumptions with `comptime assert` /
  `debug_assert`; don't remove existing bounds checks.
- In tests, use `NaN`/`Inf` sentinels, not arbitrary magic numbers, and
  use prime and tile-straddling lengths (`13`, `63`/`64`/`65`) so the
  masking and tail code runs.
- An out-of-bounds write fix needs a guard band after the output, filled
  with a value the kernel can't produce and checked afterwards; comparing
  in-range elements can't see the overrun.
- Device graph capture replays the host's choices, so host code must not
  pick a kernel or size a grid from per-request values or from device
  data copied back to the host, and must handle `M == 0` warmup launches.
- Declare `MAX_THREADS_PER_BLOCK_METADATA` to match the real block size.

## 14. Write portable kernels; gate only what's vendor-specific

Specializing a kernel for one vendor is fine when it uses that vendor's
hardware (TMA, WGMMA, tcgen05 on NVIDIA; MFMA, `lgkmcnt` waits on AMD;
inline asm). A kernel built only from generic primitives (`thread_idx`,
`block_dim`, `barrier`, `warp.sum`, `WARP_SIZE`, shared memory, TileTensor)
isn't specific to the vendor it was written on, so don't gate it or its
test to that vendor by default. This applies in both directions: a kernel
written and tuned on AMD is no more AMD-only than one written on NVIDIA is
NVIDIA-only.

```bzl
# WRONG: generic kernel, gated to whichever vendor it was written on
gpu_constraints = ["//:nvidia_gpu"],
gpu_constraints = ["//:amd_gpu"],
# CORRECT: any GPU (add an Apple `incompatible` select only if it fails there)
gpu_constraints = ["//:has_gpu"],
```

- Before adding a vendor gate, run the kernel on the other vendor too (an
  AMD MI300X/MI355X or NVIDIA H100/B200 dev box, or let the other CI lane
  run it). If it also has a host path, try dropping the GPU constraint
  entirely so it runs on every accelerator.
- When reviewing, if a vendor-gated kernel uses no vendor intrinsics, ask
  the author to try enabling it for all GPUs, or all accelerators.
- Keep the vendor-specific part small: put the fast path behind
  `comptime if is_nvidia_gpu()` / `is_amd_gpu()` and keep a generic
  fallback, rather than gating the whole kernel.
- Don't gate generic optimizations on hardware. Vectorized loads and
  stores, `vectorize`, unrolling, and wider per-thread tiles usually help
  every vendor, so apply them unconditionally and size them with
  `simd_width_of` instead of putting them behind `is_nvidia_gpu()` or
  `is_amd_gpu()`. Gate one only if a benchmark shows it regresses a vendor.
- If it really fails on another vendor, gate it with a `TODO(<ticket>)`
  naming the failure (a missing intrinsic or a numerics gap), not just
  "NVIDIA only" or "AMD only".
- Don't hard-code one vendor's facts (warp size 32 vs 64, shared memory
  size) in generic code; see §13.

## 15. Hygiene

- Every temporary hack or API abuse gets a `TODO(<ticket>)` linking a
  filed issue.
- Keep comments short: a line or two on *why*, never a paragraph. Don't
  explain what the code plainly shows; delete comments that add nothing
  (`# (TileTensor-based)`).
- Don't commit agent status or design notes next to source.
- Changelog entries describe the end state, not intermediate steps.
