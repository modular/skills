---
name: mojo-optimizations
description: Performance optimization patterns for Mojo code. Use this skill in addition to mojo-syntax when writing or refactoring Mojo code that needs to be fast, when profiling shows a hot path, when the user mentions benchmarks, regressions, latency, throughput, "make it faster", or when porting performance-sensitive code (parsers, matchers, numeric loops, byte-level scanners) to Mojo. Use to overcome misconceptions about where Mojo spends cycles and which idioms compile to tight code.
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

Apply these patterns **on top of `mojo-syntax`** — they assume you already know
modern Mojo syntax. These patterns compound: any one of them in isolation
gives a small win, but chaining several on the same hot path is where
order-of-magnitude speedups come from.

**Measure before and after every change.** Never guess.

## The profile-driven workflow

Use `std.benchmark` — **never** hand-roll a `perf_counter_ns` harness.
`Bench`/`Bencher` handles warmup, auto-calibration, repetitions, and
statistical summary for you. The stdlib `mojo/stdlib/benchmarks/` tree is the
reference — copy-paste from there when starting a new file.

```mojo
from std.benchmark import (
    Bench, BenchConfig, Bencher, BenchId,
    BenchMetric, ThroughputMeasure,
    keep, black_box,
)

# One benchmark = an @parameter def that takes `mut b: Bencher`.
# The inner @always_inline @parameter closure is what gets measured.
@parameter
def bench_parse_json(mut b: Bencher) raises:
    var source = load_fixture("large.json")     # setup OUTSIDE the timed body
    @always_inline
    @parameter
    def call_fn() raises:
        for _ in range(1000):                   # batch to amortize per-iter overhead
            var r = parse_json(source)
            keep(r)                             # prevent dead-code elimination
    b.iter[call_fn]()
    keep(Bool(source))                          # keep the setup alive past the loop

# Parametric benchmarks use compile-time params; the harness calls them
# once per specialization.
@parameter
def bench_insert[size: Int](mut b: Bencher) raises:
    var items = make_dict[size]()
    @always_inline
    @parameter
    def call_fn() raises:
        for k in range(size, size + 10):
            items[k] = k
    b.iter[call_fn]()
    keep(Bool(items))

def main() raises:
    var m = Bench(BenchConfig(num_repetitions=5))
    m.bench_function[bench_parse_json](BenchId("parse_json"))
    comptime for size in (10, 100, 1_000, 10_000):
        m.bench_function[bench_insert[size]](
            BenchId(String("insert[", size, "]"))
        )
    print(m)                                   # prints the results table
```

Key rules the harness enforces, so you don't:

- **Setup outside `call_fn`.** Anything built inside the inner closure is
  rebuilt on every iteration. Load fixtures, allocate buffers, pre-compile
  before `b.iter[...]`.
- **`keep(value)` every result.** Without it the optimizer deletes the call
  you're trying to measure. `keep(Bool(container))` at the end of the outer
  function keeps setup alive past the timed region. `black_box(x)` is the
  stronger sibling — use it on inputs you want to force through memory.
- **Batch inside `call_fn`** (e.g., `for _ in range(1000)`) when a single
  call is sub-microsecond. The harness auto-calibrates, but batching further
  reduces timer overhead for very fast ops.
- **`num_repetitions > 1`** when you care about stability; the harness
  reports min/mean/max across repetitions.
- **Throughput units**: pass `ThroughputMeasure` so results are normalized to
  GElems/s, GB/s, or GFLOPS/s instead of raw time. Use `bench_with_input` to
  pipe a fixture to a parametric bench fn:

```mojo
m.bench_with_input[InputT, bench_fn](
    BenchId("atof", filename),
    input_data,
    [
        ThroughputMeasure(BenchMetric.elements, len(input_data)),
        ThroughputMeasure(BenchMetric.bytes, total_bytes),
    ],
)
```

- **`iter_custom`** is the escape hatch when the thing you're measuring needs
  its own context (e.g., GPU dispatch). Pass a closure taking an iteration
  count and returning elapsed ns. Use this only when `iter[...]` can't
  express the setup.

`BenchConfig` defaults are sensible: `num_warmup_iters=10`,
`max_runtime_secs=1.0`, `max_iters=1_000`. Override `num_repetitions` (for
stability) and `max_runtime_secs` (for precision) first; leave the rest alone
unless you've measured a reason.

File layout: name benchmark files `bench_<thing>.mojo`, mirror the source
tree, and put them under a `benchmarks/` directory. Stdlib tooling keys on
the `bench_` prefix.

**Common mistakes a pretrained model will make**: hand-rolling
`perf_counter_ns` loops; omitting `keep(...)` so the compiler deletes the
work; constructing inputs inside `call_fn`; reporting a single run instead of
`num_repetitions>1`; forgetting `@parameter` on the bench function or
`@always_inline @parameter` on the inner closure; using `Bench()` without
printing `print(m)` at the end (nothing renders otherwise).

## Inlining hot-path trampolines

Wrapper methods that just forward to an inner engine are "trampolines". If a
small input (e.g., a 16-byte `StringSlice`) flows through 4 trampoline levels
without `@always_inline`, LLVM can't fold the call chain and the fast path pays
3-4 call-frame costs per invocation. Mark **every level** of the dispatch chain
`@always_inline`:

```mojo
# JsonParser -> Tokenizer -> ByteScanner : all @always_inline
struct Tokenizer:
    @always_inline
    def peek(self) -> UInt8:
        return self.scanner_ptr[].peek()

    @always_inline
    def next_token(mut self) -> Token:
        return self.scanner_ptr[].next_token()
```

- One link missing `@always_inline` breaks the fold — check the whole chain.
- `@always_inline("nodebug")` additionally strips debug info for tiny helpers
  (accessors, byte reads) so they don't clutter stack traces.
- Use `@no_inline` on **cold** paths inside a hot function to keep the hot loop
  small and reduce I-cache pressure (error handlers, first-time-setup paths).
- Don't blindly inline large functions — `@always_inline` on a 200-line
  function bloats call sites. The rule is: inline thin forwarders and small
  leaf helpers, not big functions.

## Unsafe pointer access in inner loops

Know what `__getitem__` actually costs — it differs by type. All three go
through `normalize_index`, but with different assert modes:

| Type                  | Default-release check                              | Extra per-access work |
|-----------------------|----------------------------------------------------|-----------------------|
| `List[T][i]`          | **None** (`assert_mode="none"`, compiled out)      | Negative-index normalization branch |
| `Span[T][i]`          | **None** (`assert_mode="none"`, compiled out)      | Negative-index normalization branch |
| `StringSlice[byte=i]` | **Bounds check + UTF-8 start-byte assert** (`"safe"`) | Negative-index normalization branch |

The global `ASSERT` mode defaults to `safe`. Under `-D ASSERT=all` or a
debug build, **all three** types emit bounds checks on every access. So
whether `unsafe_ptr()` actually saves a branch depends on the build mode.

What `unsafe_ptr()` reliably buys you, even in default release:

1. Skips the negative-index normalization branch (a conditional on every
   access for signed index types).
2. Removes the `StringSlice[byte=i]` bounds check + UTF-8 assert.
3. Enables more aggressive loop optimization — with no possible trap, LLVM
   can vectorize, unroll, and hoist more freely.
4. Makes `-D ASSERT=all` debug builds as fast as release on the hot path.

```mojo
# Safe but slow in -D ASSERT=all, and still branches per iter on sign check
for i in range(len(states)):
    if states[i].is_active:
        ...

# Pointer-based hot loop — no normalization, no possible trap
var states_ptr = states.unsafe_ptr()
var n = len(states)
for i in range(n):
    if states_ptr[i].is_active:
        ...
```

For `List` specifically, `list.unsafe_get(idx)` is a safer middle ground —
it asserts in debug (`assert_mode` default) but still avoids negative-index
handling. Use `unsafe_ptr()` only when you also want the raw-pointer loop
form (e.g., to feed `+ offset` arithmetic or SIMD loads).

**Separately, audit the loop for checks that are always true for the input
type**: `uint8_val >= 0` is always true; `char_code < 256` is always true
for `UInt8`-typed inputs. Such dead conditions still cost instructions —
delete them.

## Pre-allocate collections; lazy-allocate when zero is common

| Situation                             | Pattern                                    |
|---------------------------------------|--------------------------------------------|
| Known upper bound N                   | `List[T](capacity=N)`                      |
| Known lower bound, growable           | `var xs = List[T](); xs.reserve(estimate)` |
| Zero is the common case (`findall`)   | Lazy: don't allocate until first append    |
| Fixed compile-time size               | `InlineArray[T, N]` (stack, no heap)       |
| Small hot container, size never grows | `SIMD[DType.uint8, 128]` as a dense bitset |

```mojo
# Known bound — pre-size at construction
var elements = List[Node](capacity=len(tokens))

# Lazy container — defer allocation until first insert
struct LazyList[T: Copyable & Movable]:
    var _data: UnsafePointer[T, MutAnyOrigin]
    var _len: Int
    var _capacity: Int

    def __init__(out self):
        self._data = UnsafePointer[T, MutAnyOrigin]()
        self._len = 0
        self._capacity = 0

    def append(mut self, value: T):
        if self._capacity == 0:
            self._realloc(8)       # first-use reservation
        elif self._len == self._capacity:
            self._realloc(self._capacity * 2)
        (self._data + self._len).init_pointee_move(value)
        self._len += 1
```

**Pitfall**: `List` grows by doubling. A loop that appends 1000 items triggers
~10 reallocations + memcpys if you start from zero. Pre-sizing is usually a
1.2-2x win on append-heavy code.

## Struct layout for cache efficiency

Small, trivially-copyable structs pass in registers and avoid memcpy. Aim to
keep hot-iterated structs under a cache line (64 bytes), ideally 32 bytes.

```mojo
struct Token(Copyable, Movable, TrivialRegisterPassable):
    comptime __copy_ctor_is_trivial = True   # LLVM elides the copy entirely
    var kind: Int                             # 8
    var start: Int                            # 8
    var length: Int                           # 8
    var source_ptr: UnsafePointer[Byte, ImmutAnyOrigin]   # 8
    # Total: 32 bytes — fits in 4 registers, no stack ops on copy.
```

- Store `UnsafePointer[Byte]` + offsets, not full `StringSlice` fields, when
  every byte of the struct matters. (A `StringSlice` is ~16 bytes — two pointer
  payloads — which doubles the struct.)
- `TrivialRegisterPassable` requires all fields to be trivially copyable;
  adding a `String` or `List` field silently drops the trait.
- `@fieldwise_init` synthesises the constructor without you typing field
  assignments — use it on any plain data struct.

## Pass views, not owned strings

`String` is heap-allocated; copying or building one from a literal allocates.
Public APIs that only read text should take a `StringSlice` (view) instead.
Define a module alias so every layer agrees on the same type:

```mojo
comptime ImmSlice = StringSlice[ImmutAnyOrigin]

def tokenize(source: ImmSlice) raises -> List[Token]:
    ...                                       # callers pass literals, zero alloc
```

`Span[Byte]` plays the same role for raw byte views. Prefer `Span[Byte]` over
`(ptr: UnsafePointer[Byte], len: Int)` parameter pairs — it's the same two
machine words but carries provenance and can't go out of sync.

## Avoid copies in loops: `ref` over `var`

Iterating a container of large structs (records, tree nodes, tokens) with
`var x = container[i]` **copies** on every step. Use `ref` to alias:

```mojo
# WRONG — full struct copy per iteration
for i in range(len(rows)):
    var row = rows[i]          # copy
    process(row)

# CORRECT — zero-copy reference
for i in range(len(rows)):
    ref row = rows[i]          # alias
    process(row)

# Also for local bindings to nested fields:
ref table = table_ptr[]        # instead of `var table = table_ptr[]`
```

`^` transfer: when you *do* want ownership but won't use the source again, use
`value^` to move instead of copy: `ptr.init_pointee_move(value^)`.

## Lazy-initialized global caches

For precomputed lookup tables, interned symbol caches, or any expensive
build-once value, use `_Global` from `std.ffi`. It guarantees single-shot
lazy initialization behind a pointer you can mutate through:

```mojo
from std.ffi import _Global
from std.os import abort

comptime SymbolTable = Dict[Int, InternedSymbol]
comptime _SYMBOL_TABLE = _Global["SymbolTable", _init_symbol_table]

def _init_symbol_table() -> SymbolTable:
    return SymbolTable()

def _get_symbol_table() -> UnsafePointer[SymbolTable, MutAnyOrigin]:
    try:
        return _SYMBOL_TABLE.get_or_create_ptr()
    except e:
        abort[prefix="ERROR:"](String(e))

@always_inline
def intern(id: Int) raises -> InternedSymbol:
    var table_ptr = _get_symbol_table()
    ref table = table_ptr[]
    if id not in table:
        table[id] = InternedSymbol(id)
    return table[id]
```

- The string key in `_Global["SymbolTable", ...]` is the **global identity** —
  must be unique per cache across the whole program.
- Return `UnsafePointer` for interior mutability even when the caller is
  `read`-self: this is how you cache through a trait method that demands
  immutable `self`.

## Heap-allocated fields: use `init_pointee_move`, not assignment

`alloc[T](1)` returns **uninitialized** memory. Assigning `ptr[] = T(...)`
invokes *move-assignment* into that uninitialized storage, which runs
destructor logic on garbage — a classic flaky double-free at process exit:

```mojo
# WRONG — move-assign into uninitialized memory, undefined behavior
self._arena_ptr = alloc[Arena](1)
self._arena_ptr[] = Arena(capacity^)     # UB

# CORRECT — construct in place
self._arena_ptr = alloc[Arena](1)
self._arena_ptr.init_pointee_move(Arena(capacity^))
```

Same rule applies in `__copyinit__` when copying heap-owned fields:

```mojo
def __copyinit__(out self, copy: Self):
    self._ptr = alloc[Self.T](1)
    self._ptr.init_pointee_move(copy._ptr[].copy())
```

A `__del__` that calls `.free()` on the pointer is mandatory to pair with
these.

## Cache by hash, not by string

`Dict[String, V]` forces callers to allocate a `String` just to check cache
membership. Hash the slice instead — cache hits become zero-allocation.
Works for any "expensive value keyed by a string": parsed configs, compiled
templates, resolved file paths, interned symbols, SQL plans.

```mojo
# Generic pattern: expensive T built from a string key, memoized.
comptime Cache = Dict[UInt64, Entry]

def get_or_build(key: StringSlice, mut cache: Cache) raises -> Entry:
    var h = hash(key)
    if h in cache:
        ref hit = cache[h]
        if hit.source == key:               # collision guard: byte-compare
            return hit.copy()
    var built = build(String(key))          # allocate the String once, on miss
    cache[h] = built
    return built
```

The collision guard is mandatory. 64-bit hash collisions are astronomically
rare but not zero, and silently returning the wrong value under a collided
key is the worst possible failure mode. On mismatch, fall through to a fresh
build.

## Comptime specialization for fast-path code

`comptime if`/`comptime for` generate **distinct code per specialization** and
have zero runtime cost. Use them to pick between SIMD widths, architectures, or
unrolled loop bodies:

```mojo
comptime SIMD_WIDTH = simd_width_of[DType.uint8]()

def match_chunk[size: Int](self, chunk: SIMD[DType.uint8, size]) -> SIMD[DType.bool, size]:
    comptime if size == 16:
        # Fast path: one pshufb pair
        var lo = self.low_lut._dynamic_shuffle(chunk & 0x0F)
        var hi = self.high_lut._dynamic_shuffle((chunk >> 4) & 0x0F)
        return (lo & hi) != 0
    else:
        # Generic path: process in 16-byte sub-chunks
        var result = SIMD[DType.bool, size](False)
        comptime for offset in range(0, size, 16):
            ...
        return result
```

Also precompute lookup tables as `comptime` so they end up as `.rodata`, not
generated at runtime:

```mojo
comptime DIGIT_LUT = _build_digit_lut()  # runs at compile time
```

## SIMD byte scanning: nibble-based lookup

For byte-level membership tests — JSON whitespace (`\t \n \r ' '`), CSV
delimiters, URL-safe characters, digit ranges, any fixed byte set — the naive
256-entry lookup table is too big for `_dynamic_shuffle`. Decompose each byte
into two **nibbles** (4-bit halves) and use two 16-entry tables — this fits
exactly in a single `pshufb`/`vpshufb` instruction:

```mojo
# Two 16-entry tables, precomputed at compile time
var lo = low_nibble_lut._dynamic_shuffle(chunk & 0x0F)
var hi = high_nibble_lut._dynamic_shuffle((chunk >> 4) & 0x0F)
var matches: SIMD[DType.bool, 16] = (lo & hi) != 0
# matches[i] is True iff chunk[i] is in the byte set
```

Nibble lookup is typically **20-100x faster** than per-byte scalar dispatch on
hot paths. For *contiguous* ranges (e.g., `a`-`z`, `0`-`9`), an even cheaper
path exists:

```mojo
# Range check via unsigned subtract — no lookup table needed
var offset = chunk - SIMD[DType.uint8, 32](range_start)
var matches = offset <= SIMD[DType.uint8, 32](range_end - range_start)
```

Record `range_start`/`range_end` on the scanner struct at construction time so
the hot path can pick this fast path. Non-contiguous sets fall back to the
nibble tables.

## Prefilters: cheap scan before the expensive path

Running the full processing pipeline per byte is the wrong granularity.
Extract a *cheap* signal — a required literal substring, a lead-byte set,
a fixed prefix — and use the fastest scan primitive to locate candidate
positions first. The expensive logic only runs where the prefilter says
"maybe". This applies to parsers, validators, search engines, log scanners,
packet decoders, etc.

| Prefilter          | Scan primitive                | Example                                |
|--------------------|-------------------------------|----------------------------------------|
| Required literal   | `StringSlice.find`            | Scan for `"error"` before parsing line |
| Last occurrence    | `String.rfind` (single pass)  | Find last `/` to extract filename      |
| Lead-byte set      | SIMD equality sweep + bitmask | Scan for `{`, `[`, `"` to find JSON values |
| Byte range/class   | Nibble SIMD scan              | Skip to first digit before number parse|

**Critical anti-pattern**: using repeated forward `find` to locate the *last*
occurrence is O(N x occurrences). Use `rfind` for a single reverse O(N) pass:

```mojo
# WRONG — O(N * k)
var pos = 0
while True:
    var next = text.find(delimiter, pos)
    if next == -1: break
    pos = next + 1
var last = pos - 1

# CORRECT — O(N)
var last = text.rfind(delimiter)
```

In practice, switching from repeated-`find` to `rfind` for a "find last
suffix" operation has delivered **8-40x speedups** on multi-KB inputs.

## Numeric accumulation over string concat

When parsing numbers or building values byte by byte, accumulate into an `Int`
directly — don't build a `String` and parse at the end:

```mojo
# WRONG — N allocations for an N-digit number
var num_str = String("")
while is_digit(text[i]):
    num_str += String(chr(text[i]))
    i += 1
var num = Int(num_str)

# CORRECT — zero allocation
var num = 0
while is_digit(text[i]):
    num = num * 10 + (Int(text[i]) - Int(ord("0")))
    i += 1
```

Same pattern applies to checksum accumulation, hash building, and any
"consume-bytes-and-fold" logic.

## Fast-path dispatch by input shape

At construction time (not execution time), classify the workload and record
which optimized path can run. Simple cases should bypass the general engine
entirely — the analysis cost is paid once and amortized across every call:

```mojo
struct QueryPlan:
    var _is_pk_lookup: Bool         # equality on primary key
    var _is_simple_scan: Bool       # single-table, no joins
    var _executor: GeneralExecutor  # general fallback

    def __init__(out self, query: Query):
        self._is_pk_lookup = _has_pk_equality(query)
        self._is_simple_scan = _is_single_table(query)
        self._executor = plan_general(query)

    @always_inline
    def execute(self, db: Database) -> ResultSet:
        if self._is_pk_lookup:
            return _pk_index_lookup(db, self._executor.key)   # O(1) path
        if self._is_simple_scan:
            return _sequential_scan(db, self._executor.table)
        return self._executor.run(db)
```

Keep the classification in `__init__`; keep the hot path branchy but cheap.
The same pattern applies to parsers (literal vs. complex grammar), formatters
(fixed-width vs. general), and serializers (flat struct vs. nested).

## Unlikely-branch hoisting

When a fast path dominates, put the check **first** and **return immediately**,
so the fallback isn't inlined into the hot prologue:

```mojo
@always_inline
def validate(self, input: ImmSlice) -> Bool:
    if len(input) == 0:            # cold edge case
        return self._empty_valid
    if self._exact_mode:           # hot path, short-circuits
        return _byte_eq(input, self._expected)
    return self._general.check(input)    # general path, not inlined hot
```

Pair with `@no_inline` on the general-path helper if it's large, so the inlined
caller stays small.

## What NOT to optimize

- **Don't** replace clear code with micro-optimizations before profiling. All
  of the above earned their place by moving a measured hot path; applied
  elsewhere they're just noise.
- **Don't** pre-allocate containers you'll use once or twice — the `List`
  default growth strategy is already fine for small cases.
- **Don't** blanket-`@always_inline` functions over ~50 lines; you'll bloat
  every caller and slow compile times. Inline thin forwarders, not whole
  engines.
- **Don't** `unsafe_ptr()` outside hot loops — you lose bounds checks without
  payoff, and debugging the next segfault costs far more than the saved
  microseconds.
- **Don't** cache aggressively without measuring cache hit rate. An
  infrequently-hit cache wastes memory and adds a branch.

The single most reliable workflow: profile, pick one pattern from this list,
apply it, re-benchmark. Commit that delta. Repeat. Stop when the hot path is
no longer hot.
