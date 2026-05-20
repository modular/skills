---
name: mojo-stdlib-review
description: >
  Red-team adversarial review of Mojo stdlib PRs. Spawns parallel agents that deep-read every
  changed file, extract factual claims, verify each against the actual stdlib source of truth and
  the rules in `mojo-stdlib-contributing`, and write findings to `.specs/` by default ("local
  mode") so the author can self-review before flipping a draft PR to ready. Pass "in post mode" /
  "post comments" / "post to PR" in the arguments to post review comments on the PR via
  `gh pr review` instead. Use when the user mentions "adversarial review", "red team review",
  "deep review", "fact-check PR", or wants thorough verification of a Mojo stdlib PR before
  requesting maintainer review. Distilled from 30+ reviewed PRs to `modular/modular`.
user-invocable: true
argument-hint: "[PR numbers or URLs, space-separated] [\"in post mode\"]"
---

Red-team adversarial review of Mojo stdlib PRs: $ARGUMENTS

## Overview

This skill performs adversarial (red-team) reviews of pull requests against the Mojo standard
library. Unlike friendly reviews that skim for obvious issues, adversarial reviews actively try
to prove every claim wrong by cross-referencing the actual stdlib source, the docstrings of
sibling APIs, the trait definitions, the existing tests/benchmarks, and the rules in
[`mojo-stdlib-contributing`](../mojo-stdlib-contributing/SKILL.md).

The skill spawns one `general-purpose` agent per PR in parallel. Each agent:

1. Reads the full PR diff
2. Loads `mojo-stdlib-contributing` rules and the local stdlib source as ground truth
3. Extracts every verifiable claim and every removed line
4. Verifies each claim against the source of truth
5. Double-checks findings before reporting
6. Writes findings to a local file for the author to auto-fix (default), or posts review
   comments on the PR (opt-in via "in post mode")

## Output modes

Default is **local mode**: write findings to `.specs/stdlib-review-<PR>.md` and return the path
plus per-category counts to the caller. Do NOT post PR comments. This is the canonical
"self-review gate" used by stdlib contributors before flipping a draft PR to ready-for-review.
It runs on the local Claude Code subscription and avoids spamming the PR with comments that the
author will resolve before any maintainer sees them.

If `$ARGUMENTS` contains "post", "post mode", "post comments", "post to PR", or equivalent
phrasing, switch to **post mode**: findings are posted on the PR via `gh pr review --comment`.
Use this when reviewing someone else's PR rather than self-reviewing your own draft.

Phrases like "in local mode" / "local fix" / "no post" also explicitly request the default
local mode and are accepted (no-op).

## Phase 1: Parse Input

### Step 1: Extract PR identifiers

Parse `$ARGUMENTS` to extract PR numbers or URLs. Supported formats:

- **PR numbers**: `6276 6510 6530`
- **PR URLs**: `https://github.com/modular/modular/pull/6276`
- **Mixed**: `6276 https://github.com/modular/modular/pull/6510`
- **No arguments**: Ask the user for PR numbers.

Normalize all inputs to PR numbers against the `modular/modular` repo.

### Step 2: Fetch PR metadata

For each PR, fetch basic info:

```bash
gh pr view $PR_NUMBER --repo modular/modular \
  --json number,title,state,headRefName,changedFiles,additions,deletions,files,isDraft,body
```

Present a summary table and ask for confirmation:

| # | PR | Title | Files | +/- | Draft? |
|---|-----|-------|-------|-----|--------|
| 1 | #6276 | Add `take()`/`skip()` iterator adapters | 5 | +180/-0 | yes |

### Step 3: Scope guard

This skill targets PRs that touch the Mojo stdlib. From the `files` list returned by
`gh pr view`, verify at least one path matches `mojo/stdlib/std/**`, `mojo/stdlib/test/**`,
`mojo/stdlib/benchmarks/**`, `mojo/stdlib/docs/**`, `mojo/docs/nightly-changelog.md`, or
`mojo/docs/code/**`. If none match, warn the user and ask whether to proceed (a non-stdlib PR is
better served by the general `adversarial-review` skill).

Do NOT do any further "domain detection". This skill has a single domain: Mojo stdlib.

---

## Phase 2: Spawn Review Agents

### Step 4: Launch agents in parallel

For each PR, spawn a `general-purpose` agent using the Agent tool with `run_in_background: true`.

**All agents MUST be launched in a single message** to maximize parallelism.

Use `stdlib-review-{pr_number}` as the agent `description` (e.g., `"stdlib-review-6276"`).
Capture the returned agent ID from each Agent tool call.

Before spawning, choose **one** emit block (local mode default OR post mode) based on
`$ARGUMENTS` and inline it as `{EMIT_SECTION}` so the agent receives a single unambiguous
instruction. Default is local mode; switch to post mode only if `$ARGUMENTS` explicitly opts
in via "post", "post mode", "post comments", "post to PR", or equivalent phrasing.

### Agent Prompt Template

Fill in `{PR_NUMBER}`, `{PR_TITLE}`, and `{EMIT_SECTION}`.

````
You are doing an ADVERSARIAL REVIEW of Mojo stdlib PR #{PR_NUMBER}
({PR_TITLE}): https://github.com/modular/modular/pull/{PR_NUMBER}

## What is an adversarial review?

You are NOT a friendly reviewer. You are a red-team reviewer whose job is to find every
factual error, behavioral inconsistency, missing test, undocumented edge case, and
optimization that is actually a no-op. You actively try to prove claims wrong by checking
the actual stdlib source.

## Ground truth, in priority order

1. The Mojo stdlib source under `mojo/stdlib/std/**` (read the actual files; do not rely on
   pretrained knowledge of older Mojo syntax).
2. `mojo-stdlib-contributing/SKILL.md` (recurring rejection reasons; treat its rules as if
   they were a maintainer's standing review checklist).
3. `mojo/CONTRIBUTING.md`, `mojo/stdlib/docs/style-guide.md`, `mojo/stdlib/docs/development.md`,
   `mojo/stdlib/docs/docstring-style-guide.md`, `AI_TOOL_POLICY.md`.
4. Sibling APIs in the same module (for consistency/symmetry checks).
5. CPython documentation, only when the PR explicitly claims Python equivalence.

Use Grep, Glob, and Read extensively. Do NOT skip Step 3 below.

## Methodology -- follow ALL steps

### Step 1: Read the full PR

```bash
gh pr diff {PR_NUMBER} --repo modular/modular
gh pr view {PR_NUMBER} --repo modular/modular --json title,body,commits,labels,isDraft
```

Read the full diff carefully. Note the PR description, the linked GitHub issue (if any), and
whether the PR is a draft.

### Step 2: Extract every verifiable claim

For each changed file, extract claims of every category below. Each table row's "Verify
against" column is where the agent must look to confirm or refute the claim.

#### A. Process & scope (sources: PR body, commits, `mojo/CONTRIBUTING.md`, `AI_TOOL_POLICY.md`)

| Claim type | Verify against |
|---|---|
| Adds a new public API on an existing type | `gh issue list --repo modular/modular --search "<api name>"` -- a GH issue MUST exist before new APIs |
| PR is marked draft | `isDraft` field -- new-API PRs should stay draft until consensus |
| `Assisted-by: AI` present in every commit and the PR body | `gh pr view ... --json commits,body` |
| One logical change per PR | The set of files touched -- flag PRs that mix unrelated concerns (e.g. `ArcPointer` hashable + new `String.capitalize`) |
| Changelog entry exists / does not exist | `mojo/docs/nightly-changelog.md` -- entries are required for user-facing changes and FORBIDDEN for NFC / implementation-detail PRs (e.g. DRY refactors) |
| Changelog scope matches PR scope | The diff -- flag entries that document changes not in the PR, or omit user-facing additions present in the PR |

#### B. API design & symmetry (sources: sibling methods, CPython docs when claimed)

| Claim type | Verify against |
|---|---|
| "Equivalent to Python's `math.X`" / "matches Python" | CPython behavior. Check positional-only (`/`), keyword-only (`*`), exception types, edge cases (e.g. `k > n` in `math.perm`). |
| New method's error type matches sibling methods | The sibling implementations (e.g. `tell()` vs `seek()` on `FileHandle` -- both should raise `Error("...")`, not bare strings) |
| New method duplicates existing logic | Sibling methods -- e.g. `tell()` could be `seek(0, SEEK_CUR)`; new ordering dunders could delegate to `_compare()` |
| Takes `Int` for an integer arg | Consider whether `Scalar[dtype]` is the right generic; flag `Int`-only when other integer dtypes are natural |
| Adds `Hashable` without `Equatable` | Dict/Set key element traits in `std/collections/dict.mojo` -- both are required |

#### C. Trait conformance (sources: trait definitions in `std/builtin/`)

| Claim type | Verify against |
|---|---|
| Conforms to `Comparable` | `std/builtin/comparable.mojo` -- requires `def __lt__`; defaults provide `__le__/__gt__/__ge__`; `Comparable` extends `Equatable` so don't list both |
| Required dunders implemented as `fn` | Trait requires `def`; `fn` may fail to satisfy conformance |
| Duplicates default trait methods | Drop redundant overrides unless there's a measured reason |
| Generic constrained-conformance dunder (e.g. `Tuple.__lt__` over `AllComparable`) | Existing pattern: use `_constrained_conforms_to_*` + `trait_downcast`; do not re-implement `_compare`-style logic |

#### D. Docstring accuracy (sources: the implementation directly below)

| Claim type | Verify against |
|---|---|
| "ASCII-only, non-ASCII passed through unchanged" | The implementation -- if it calls `self.lower()`, it Unicode-lowercases and the claim is wrong |
| "Undefined for negative inputs" | If the body asserts, the docstring must say "asserts", not "undefined" |
| "Equivalent to Python's X" with `k > n` etc. | CPython's actual behavior + the implementation |
| Args section reflects positional/keyword constraints | The signature (`/`, `*`) |
| `@no_inline` annotation on `write_repr_to` | Sibling `write_repr_to` implementations in `std/benchmark/bencher.mojo` etc. |
| Verbose / redundant docstring or inline comment | The implementation -- flag docstrings that restate the signature, multi-paragraph summaries, or comments explaining WHAT (instead of WHY) when the code is self-evident. Stdlib style is one-line summaries plus Args/Returns/Raises only when non-obvious; inline comments only when the WHY isn't visible in the code. |

#### E. Tests (sources: existing tests, the implementation)

| Claim type | Verify against |
|---|---|
| Asserts `hash(a) != hash(b)` for unequal values | Hash contract: only `a == b => hash(a) == hash(b)` is required. Inequality assertions are flaky by definition and forbidden. |
| Tests for string/byte operations cover Unicode | Look for emoji / multi-byte test cases; flag if missing |
| Search/contains tests cover BOTH hit and miss paths | The diff -- both required |
| Edge cases: empty input, exact-width input, `start > end`, OOB indices | The diff |
| `String("...").method()` on a temporary | Bind to a `var` first when the method returns an aliasing slice (dangling-slice risk) |
| Asserts on `assert_equal` failure message wording for a type that just gained `Writable` | Overload selection -- the format may now legitimately be `String(x)` instead of `repr(x)` |
| Tests for `Writable` use `String(x)` / `repr(x)` instead of `check_write_to` | Flag and recommend `check_write_to` |
| Tests for runtime aborts using `assert_raises` | Aborts in Mojo are not catchable -- abort tests must be FileCheck (`_FILECHECK_TESTS=True`, `_EXPECT_CRASH=True`, `# CHECK:` directives). See `test_list_getitem_invalid_index_int.mojo`. |
| New public API without negative tests (overflow, wrong arity, wrong arg type) | Demand them |
| Pack-forwarding / variadic tests cover empty pack | The diff -- length-0 case is often missed |
| Compile-fail tests for `constrained_*` error messages | Done via `not %mojo ... \| FileCheck` |

#### F. Iterator design (sources: existing iterators in `std/itertools/`)

| Claim type | Verify against |
|---|---|
| State mutation in `__next__` before the inner `next()` succeeds | The inner iterator can raise `StopIteration`; state must change AFTER successful element consumption |
| Public `take(it, n)` / `skip(it, n)` accept negative `n` silently | `repeat()` asserts non-negative; new adapters should match (assert OR document) |
| `bounds()` can return a negative number | Clamp to 0 |
| Inline `next(self._inner)` passed straight into `rebind_var[downcast[...]](...)` | Pattern elsewhere: `var elem = next(...)` then discard `elem^` |
| `_TakeIterator` / `_SkipIterator` conformance to `IterableOwned` | Other adapters conform; consider adding an owned overload |

#### G. Benchmarks (sources: `std/benchmark/`, `mojo-stdlib-contributing` benchmark section)

| Claim type | Verify against |
|---|---|
| Inputs and results in the hot loop are not `black_box()`/`keep()` | Required, or the compiler may optimize the bench away |
| Uses `@parameter`-style closure | New code should use unified closures: `def call_fn() raises {mut f}:` |
| `keep(x)` used to silence unused warnings | Forbidden -- use `_ = value^` |
| Setup inside the hot loop | Setup goes outside `call_fn`; use `iter_with_setup` for destructive benches |
| Bench data doesn't match the docstring/comment | Read setup carefully (e.g. "element in the middle" must actually be in the middle) |
| Performance PR with no benchmarks, or only one input size | Demand small AND large input coverage |
| Bench utilities and their usage land in the same PR | Should be split into two PRs |

#### H. Optimizations (sources: `compile_info` IR diff + benchmark numbers)

| Claim type | Verify against |
|---|---|
| "Adds a fast path" / "Avoids a branch" without IR diff | Demand `compile_info[..., emission_kind="llvm-opt"]` evidence in the PR description |
| Manual ASCII fast path before `_utf8_first_byte_sequence_length(b)` (or similar) | LLVM already optimizes this -- the IR is identical OR the "fast path" produces WORSE codegen via `is_zero_poison=false` |
| Cache-line / alignment claim | Demand benchmark numbers, not speculation |
| Performance PR without before/after benchmark numbers in the body | Required |
| "Adds branch to skip computation" where the skip is cheap | Often slower than the branchless version; demand IR + bench |

#### I. Code reuse / DRY (sources: `std/format/_utils`, existing primitives)

| Claim type | Verify against |
|---|---|
| Hand-rolls struct repr with manual `", "` separators | Use `FormatStruct(writer, "Name").params(Named(...)).fields(Named(...), Named(...))` |
| `write_to` / `write_repr_to` allocates via `repr()` or `String(x)` | Forbidden -- these methods must not allocate |
| Reimplements byte scan, clamp, etc. | Reuse `_memchr` / `_memchr_impl` / `clamp` / `simd_width_of[DType.bool]()` |
| Bulk copy/move/destroy in trivial-type code paths | Use `uninit_copy_n`, `uninit_move_n`, `destroy_n` |
| New fast path on `List` (not `Span`) | Implement on `Span`, then have `List` delegate upward |
| Two write/repr methods duplicate field-formatting logic | Factor into a helper, or both call the same `FormatStruct` chain |

#### J. Memory / lifetime / move semantics (sources: surrounding code patterns)

| Claim type | Verify against |
|---|---|
| `.copy()` in performance-sensitive paths | Use move `^` |
| `Dict[...]` parameter taken by value when only iterating | Take by `ref` / borrowed; by-value either errors or defeats the optimization |
| Test creates `String("...")` and calls a slice-returning method on the temporary | Bind to `var s = String("...")` first |
| ARC/Weak test with `as_any_origin` | This extends lifetimes via `AnyOrigins`; destruction-order assertions may be vacuous. Restructure to a single shared counter with a concrete origin. |
| Pre-condition assert lives on `__len__` (or another read accessor) | Move it to where the invariant is *established* (e.g. capacity assignment), not where it's read |

#### K. Asserts & safety (sources: `mojo-stdlib-contributing` assertion section, style-guide)

| Claim type | Verify against |
|---|---|
| Downgrades `debug_assert[assert_mode="safe"]` to default / removes it | Forbidden -- these are intentional safety invariants. Users who need to bypass should use `.as_bytes().unsafe_get(idx)` instead. |
| Adds `debug_assert(...)` (default `assert_mode="none"`) expecting it to run in release | Default does NOT run in release; only `-D ASSERT=all` enables it |
| Pointer arithmetic with unclamped `[start, end]` | Clamp into `[0, len]` before `memcmp`/SIMD load |
| Reverse SIMD scan without negative `pos_in_block` check | Use `count_leading_zeros` and handle the no-match case |
| `vectorized_end` doesn't account for full SIMD block width | OOB read risk; verify |

#### L. Style / patterns (sources: `style-guide.md`, surrounding code)

| Claim type | Verify against |
|---|---|
| Uses `fn` for new stdlib code | Prefer `def` (per recurring reviewer feedback) |
| Uses `@parameter` in new code | Prefer `comptime` |
| Imports from deep submodule path (e.g. `std.benchmark.benchmark.Batch`) | Use stable re-export (`from std.benchmark import Batch`) |
| Section header formatting inconsistent with the rest of the file | Match existing style exactly (spacing of `# ===---===#`) |
| Error message constructed with `+ String(...)` concatenation | Use t-strings (`t"... {expr} ..."`) -- avoids intermediate allocations |
| SIMD test comment hard-codes a width (e.g. "64-byte SIMD") | Make platform-agnostic: "exercises both SIMD path and scalar tail" |

#### M. String byte indexing (source: `StringSlice.__getitem__` signature)

| Claim type | Verify against |
|---|---|
| Slice via `self[:idx]` / `self[idx + n:]` on a `StringSlice[byte=...]` | Required: `self[byte = :idx]` / `self[byte = idx + n :]` (`byte` is a required keyword) |

#### N. Renames / key references (use Grep)

| Claim type | Verify against |
|---|---|
| Renames a dict key, env var, string identifier, or constant | Grep the OLD name across the whole repo. Every read AND write site must be updated; silent default lookups otherwise. |

### Step 3: Verify EVERY claim against source of truth

For each claim from Step 2, find and read the actual source file. Do NOT trust the PR's
self-description. Use Grep + Read.

### Step 3b: Analyze removed code for downstream side effects

**MANDATORY**: For every removed/replaced line:

1. Identify all effects of the removed code -- side effects, return value usage, state changes.
2. Trace the downstream code path WITHOUT the removed call. Check every branch.
3. If the PR (or a comment) claims "another code path handles this": READ that path. Verify it
   calls the same function under the same conditions, with no intervening flag that re-routes.
4. Flag ordering dependencies: if the removed call ran BEFORE a state change but the
   "replacement" runs AFTER, they take different paths.
5. "Pre-existing behavior" is NOT an excuse -- the removed call may have been the only place
   that action happened for this scenario.

### Step 4: Check completeness

- Missing test cases (Unicode, empty, edge widths, abort paths)
- Missing changelog entry for a user-facing change (or unwanted entry for NFC)
- Public API without docstring example
- New API without GH issue link

### Step 5: Check consistency

- Internal: does the docstring match the implementation? does the PR description match the diff?
- External: does the new method's error type, signature shape, and naming match its siblings?
- Style: does it follow `style-guide.md` and the conventions of the surrounding file?

### Step 6: Double-check before commenting

Before emitting ANY finding:

1. Re-read the specific line in the PR you're about to flag.
2. Re-read the source you're citing as evidence.
3. Confirm the discrepancy is real, not a misunderstanding of Mojo semantics.
4. For optimization claims: if you don't have IR evidence, mark the finding as a *question*,
   not a *factual error*.

False positives destroy credibility. If you're not sure, mark it as a "question" rather than
asserting an error.

### Step 7: Classify findings

- **Critical**: would cause a compile error, runtime abort outside intended assertions, data
  loss, or memory unsafety
- **Factual error**: docstring/PR/comment claim contradicts source of truth
- **Completeness gap**: missing test, missing changelog, missing edge case, undocumented behavior
- **Inconsistency**: violates `mojo-stdlib-contributing` rule, sibling API symmetry, or
  style-guide
- **Question**: design tradeoff worth raising but not a clear defect (naming, scope, "should
  this be `Scalar[dtype]`?")
- **Minor**: typos, formatting, comment style

### Step 8: Emit review

{EMIT_SECTION}

Structure the review body as:

```markdown
## Adversarial Stdlib Review: #{PR_NUMBER}

### Methodology
Verified N claims against source files: [list key stdlib files checked]
Cross-referenced against `mojo-stdlib-contributing/SKILL.md`.

### Issues Found (N total)

#### Critical (N)
1. **[file:line]** -- Description. Source: `path/to/source:line` shows X, but PR claims Y.

#### Factual Errors (N)
1. **[file:line]** -- ...

#### Completeness Gaps (N)
1. **[file]** -- Missing X. Reference: `mojo-stdlib-contributing` section on Y, or sibling at
   `path:line`.

#### Inconsistencies (N)
1. **[file:line]** -- ...

#### Questions (N)
1. **[file:line]** -- ...

#### Minor (N)
1. **[file:line]** -- ...

### Verified Correct
N claims verified accurate. Key sources checked: [list]
```

IMPORTANT:
- Be thorough. Read every changed file. Check every claim.
- Cite the specific source file and line that contradicts the PR.
- Do NOT post findings you haven't double-checked.
- If everything checks out, say so clearly in the "Verified Correct" section.
````

### Emit blocks

Inject exactly one of these as `{EMIT_SECTION}` based on `$ARGUMENTS`:

**Local mode (default):**

```
Do NOT call `gh pr review`. Write the REVIEW_BODY markdown to
`.specs/stdlib-review-{PR_NUMBER}.md` (create the parent directory if needed) and return the
absolute path plus the issue counts (Critical/Factual/Completeness/Inconsistency/Question/Minor)
to the orchestrator. The author will read this file, resolve findings, and re-run this skill to
confirm the PR is clean before flipping the draft to ready-for-review.
```

**Post mode (opt-in via "in post mode"):**

```
Post the review to the PR using a heredoc to avoid shell quoting issues:

\`\`\`bash
gh pr review {PR_NUMBER} --repo modular/modular --comment --body "$(cat <<'EOF'
REVIEW_BODY
EOF
)"
\`\`\`
```

---

## Phase 3: Monitor and Collect Results

### Step 5: Wait for agents

All agents run in background. You will be notified as each completes.

As each agent completes, record its results:

- Number of issues by category (Critical / Factual / Completeness / Inconsistency / Question / Minor)
- Number of claims verified correct
- Whether the review was written to `.specs/` (local mode, default) or posted (post mode)

Report each completion to the user with a brief summary.

### Step 6: Handle failures

If an agent fails (e.g. `gh` auth issue, missing file, hit token budget):

- Report the failure to the user with the error
- Offer to retry the specific PR (spawn a new agent using the same prompt)

---

## Phase 4: Consolidated Report

### Step 7: Scorecard

```markdown
## Adversarial Stdlib Review Scorecard

| # | PR | Title | Total | Crit | Factual | Completeness | Inconsistency | Question | Minor |
|---|-----|-------|-------|------|---------|--------------|---------------|----------|-------|
| 1 | #6276 | ... | N | X | Y | Z | W | Q | M |
| | **Total** | | **N** | **X** | **Y** | **Z** | **W** | **Q** | **M** |

### Top Patterns
- [Recurring issues across PRs]
- [`mojo-stdlib-contributing` rules being violated most often]
- [Suggestions to fold new rules into `mojo-stdlib-contributing`]
```

In **local mode** (default), end with the list of `.specs/stdlib-review-<PR>.md` paths.
In **post mode**, end with: "All review comments have been posted on each PR."

### Step 8: Follow-up options

Ask the user via `AskUserQuestion`:

**Next steps** (header: "Next steps"):

- **Fix locally** (default for local-mode runs): Read each `.specs/stdlib-review-<PR>.md` and
  apply fixes on the local branch. When the fix involves writing or rewriting a docstring or
  inline comment, keep it concise: a one-line summary plus Args/Returns/Raises sections only when
  non-obvious, and inline comments only when the WHY isn't self-evident from the code. Do not
  restate the signature, narrate the implementation, or add multi-paragraph prose.
- **Done**: Reviews are in place, nothing more needed.
- **Re-review**: Run another pass on specific PRs (e.g. after the author pushes fixes).
- **Update `mojo-stdlib-contributing`**: If a recurring pattern surfaced that isn't in the
  contributing skill yet, propose an edit.

---

## Important Notes

- **Single domain**: This skill is Mojo-stdlib-only. Use the general `adversarial-review` skill
  for non-stdlib PRs.
- **Agent type**: Always `general-purpose`. Stdlib expertise comes from the prompt (and from
  reading `mojo-stdlib-contributing` as the agent's first move), not the agent type.
- **Parallelism**: All agents MUST be launched in a single message for maximum parallelism.
- **False-positive prevention**: Step 6 (double-check) is critical. A review that surfaces
  wrong findings wastes the author's time and, in post mode, costs trust and review-bandwidth.
  When in doubt, classify as Question.
- **Scope**: This skill emits reviews; it does NOT modify code. The author (or caller) fixes
  issues based on the findings file (local mode) or PR comments (post mode).
- **Default is local mode**: re-running overwrites `.specs/stdlib-review-<PR>.md` (idempotent).
  In post mode, re-running posts a fresh review comment each time -- warn the user before
  re-reviewing a PR you already reviewed in post mode.
- **Cost**: Each agent reads heavily (PR files + stdlib source + sibling APIs + the
  contributing skill). For N PRs, expect proportional reads. Local mode (default) amortizes
  this: run it once before flipping to ready-for-review and you avoid triggering the paid CI
  reviewer on every push.
