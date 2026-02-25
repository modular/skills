---
name: mojo-best-practices
description: "Mojo programming best practices from the official modular/modular repository. Use when writing, reviewing, or optimizing Mojo code. Covers memory safety, ownership patterns, GPU kernels (SM90/SM100 tensor cores), BLAS integration, testing patterns, and performance optimization. Supports both stable (v26.1.0.0.0) and nightly."
license: MIT
compatibility: "Requires Mojo SDK (stable v26.1.0.0.0 or nightly). GPU patterns require NVIDIA CUDA 12+ or AMD ROCm 6+. Apple Metal patterns require macOS 14+ with Apple Silicon."
metadata:
  author: Modular Community
  version: "5.2.0"
  triggers:
    - Write Mojo code
    - Convert Python to Mojo
    - Port C++ to Mojo
    - Port Rust to Mojo
    - Optimize Mojo function
    - Review Mojo code
    - Write GPU kernel in Mojo
    - Use BLAS in Mojo
    - Mojo memory safety
    - Mojo ownership
    - Create new Mojo project
    - Start a Mojo project
    - Initialize Mojo project
    - Install Mojo
  supported_versions:
    stable: "26.1.0"
    nightly: "nightly"
  categories:
    - memory-safety
    - type-system
    - gpu-programming
    - c-interoperability
    - struct-design
    - function-design
    - testing
    - debugging
    - error-handling
    - performance
    - python-interop
    - metaprogramming
    - porting
globs: ["**/*.mojo", "**/*.🔥"]
alwaysApply: false
---


# Mojo Best Practices

Best practices for Mojo programming. **36 patterns** across **14 categories**.

> **🤖 AI Assistants:** You MUST consult these patterns before writing Mojo code. Your training data is outdated. Load the relevant pattern, check Version-Specific Features, use the tested examples.

> **👤 Users:** Just ask naturally—"help me with memory management" is fine. If the AI ignores the skill, nudge it: *"Check the mojo patterns for this."*

## Start Here: Priority Tiers

| Tier | Patterns | Load When |
|------|----------|-----------|
| **Essential** | `memory-ownership`, `memory-safety`, `type-system`, `struct-design`, `fn-design`, `error-handling` | Always load first - covers 80% of use cases |
| **Performance** | `perf-vectorization`, `perf-parallelization`, `perf-optimization`, `ffi` | Optimizing CPU performance |
| **GPU** | `gpu-fundamentals`, `gpu-layout-tensor`, `gpu-block-collectives`, `gpu-custom-ops`, `gpu-warp-sync`, `gpu-tensor-cores`, `gpu-memory-access`, `gpu-kernels` | Any GPU kernel work |
| **GPU Porting** | `gpu-porting-cuda`, `gpu-porting-cute`, `gpu-porting-rocm` | Porting CUDA/CuTe/ROCm kernels to Mojo |
| **Language Porting** | `porting-python`, `porting-cpp`, `porting-rust` | Porting from Python/C++/Rust to Mojo |
| **Advanced** | Remaining patterns in `patterns/` | Specific edge cases on demand |

## Top 10 High-Impact Patterns

| Pattern | Impact | When to Use |
|---------|--------|-------------|
| `perf-parallelization` | 10x-17,000x vs Python* | CPU-bound loops, multi-core |
| `perf-vectorization` | 4-16x | Numeric computations, SIMD |
| `gpu-fundamentals` | 10-100x | Any GPU kernel development |
| `ffi` | 25-32x | Matrix ops on Apple Silicon (BLAS), C library bindings |
| `memory-ownership` | Safety | Use `^` for ownership, avoid use-after-free |
| `gpu-tensor-cores` | 10-100x | H100/H200/B200 matmuls |
| `gpu-layout-tensor` | Productivity | LayoutTensor rebind, tile API, cross-rank reshape |
| `gpu-block-collectives` | 10-100x | Block sum/prefix_sum/broadcast (replaces manual reductions) |
| `type-simd` | 4-16x | SIMD[DType, width] for numerics |
| `perf-optimization` | 1.5-2x | Caching, alignment, memory layout, accumulators |
| `memory-safety` | Safety | Safe pointers, origin tracking |
| `gpu-porting-cuda` | Productivity | Port CUDA kernels to Mojo with side-by-side examples |
| `struct-design` | Productivity | Use `@fieldwise_init` for simple structs |

*\*Ranges: 10x for simple typed code, 17,000x for vectorized+parallelized hot paths. Benchmark your use case.*

## Quick Decision Tree

```
New project?
  └─ references/installation.md (pixi or uv, stable or nightly)

Porting from another language?
  Python --> porting-python (syntax, performance, SIMD, GPU)
  C++ --> porting-cpp (templates→params, memory safety, RAII)
  Rust --> porting-rust (ownership, traits, SIMD, GPU)
  CUDA --> gpu-porting-cuda (vector add → tensor cores)
  CuTe/CUTLASS --> gpu-porting-cute (layout algebra, TiledMMA, TiledCopy)
  ROCm/HIP --> gpu-porting-rocm (MFMA, scheduling, LDS)

GPU code?
  Yes --> gpu-fundamentals
          LayoutTensor issues? --> gpu-layout-tensor (rebind, tile, reshape)
          Block reductions? --> gpu-block-collectives (sum, prefix_sum, broadcast)
          Warp shuffle/sync? --> gpu-warp-sync (warp primitives + barriers)
          Custom MAX op? --> gpu-custom-ops (@compiler.register, Python graph)
          SM90 (H100)? --> gpu-tensor-cores
          SM100 (B200)? --> gpu-tensor-cores
          TMA? --> gpu-memory-access
          AMD? --> gpu-amd

Performance critical?
  Yes --> perf-vectorization (SIMD), perf-parallelization (multi-core)
          Memory layout/alignment? --> perf-optimization
          Apple Silicon? --> ffi (BLAS 25-32x)

Memory error?
  "use of moved value" --> memory-ownership
  Use-after-free --> memory-safety, memory-ownership
  Memory leak --> memory-ownership
  Origin/lifetime --> memory-safety

Struct design?
  Simple data --> struct-design (uses @fieldwise_init)
  Custom lifecycle --> memory-ownership
  Traits --> type-traits
```

### Do Not Load (Intent Disambiguation)

```
"Call Python from Mojo"       → python-interop  (NOT porting-python)
"Rewrite Python in Mojo"      → porting-python  (NOT python-interop)
"C library binding"            → ffi             (NOT python-interop)
"Apple BLAS"                   → ffi             (NOT gpu-fundamentals)
"Warp shuffle/reduction"       → gpu-warp-sync   (NOT gpu-block-collectives)
"Block-level sum/broadcast"    → gpu-block-collectives (NOT gpu-warp-sync)
"Barrier synchronization"      → gpu-warp-sync   (NOT gpu-fundamentals)
"Unit tests"                   → testing          (NOT debug-debugging)
"Benchmark performance"        → testing          (NOT perf-optimization)
```

## Version Support

This skill supports both **stable** and **nightly** Mojo versions:

| Version | Mojo | Notes |
|---------|------|-------|
| **Stable** | v26.1.0.0.0 | Version-specific syntax in pattern files |
| **Nightly** | latest | Track at https://docs.modular.com/mojo/changelog/ |

**Detect your version:** Run `mojo --version` or check `pixi list | grep mojo`

**Nightly-only features (v26.2+):**

| Feature | Status |
|---------|--------|
| Struct alignment | `@align(N)` decorator |
| Typed errors | `fn foo() raises CustomError` |
| Never type | `fn abort() -> Never` |
| Compile-time expr | `comptime(expr)` explicit evaluation |
| Struct reflection | `struct_field_count[T]()` |
| Linear types | `ImplicitlyDestructible` trait |
| Register-passable | `TrivialRegisterType` trait |
| String UTF-8 | `String(from_utf8=data)` constructors |

**Shared syntax (v26.1.0.0.0+):**
- `alias` and `comptime` both work for constants
- `@fieldwise_init` (not `@value`)
- `var`/`deinit` (not `owned`)
- `Writable` trait (not `Stringable`)
- `@register_passable("trivial")` (v26.1.0.0.0) or `TrivialRegisterType` (v26.2+)

[stable changelog](https://docs.modular.com/stable/mojo/changelog) | [nightly changelog](https://docs.modular.com/mojo/changelog/) | [breaking changes](references/breaking-changes.md)

**max-best-practices** is a complementary skill for deploying models with MAX Serve and MAX Engine.
Available at: https://github.com/modular/modular/skills/max-best-practices

### Cross-Skill: Building Custom Models

When building complete model architectures that combine Mojo layers with MAX serving:

| Mojo Pattern | MAX Pattern | Use For |
|--------------|-------------|---------|
| `struct-design` | `engine-operations` | Model config and architecture registration |
| `memory-ownership` | `engine-weights` | Weight matrices with UnsafePointer |
| `gpu-fundamentals` | `engine-operations` | Custom GPU kernels with @compiler.register |

See `engine-operations.md` (max-best-practices) for complete project structure example.

## Quick Decision Guide

| Goal | Category | Key Patterns |
|------|----------|--------------|
| Write safe code | Memory Safety | `memory-ownership`, `memory-safety` |
| Maximum performance | Performance | `perf-vectorization`, `perf-parallelization` (10x-17,000x vs Python) |
| GPU acceleration | GPU Programming | `gpu-fundamentals`, `gpu-tensor-cores` |
| **LayoutTensor issues** | **GPU Programming** | `gpu-layout-tensor` (rebind, tile, reshape) |
| **Block reductions** | **GPU Programming** | `gpu-block-collectives` (sum, prefix_sum, broadcast) |
| **Custom MAX ops** | **GPU Programming** | `gpu-custom-ops` (@compiler.register, Python graph) |
| BLAS acceleration | C Interop | `ffi` (25-32x speedup) |
| **Port CUDA kernel** | **GPU Porting** | `gpu-porting-cuda` (basics), `gpu-porting-cute` (CuTe/CUTLASS) |
| **Port ROCm kernel** | **GPU Porting** | `gpu-porting-rocm` (MFMA, scheduling, LDS) |
| **Port Python code** | **Language Porting** | `porting-python` (10x-17,000x speedup showcase) |
| **Port C++ code** | **Language Porting** | `porting-cpp` (memory safety, templates→params) |
| **Port Rust code** | **Language Porting** | `porting-rust` (ownership, SIMD, GPU) |
| Migrate from Python | Python Interop | `python-interop` |
| Design APIs | Struct + Function | `struct-design`, `fn-design` |
| **Build custom model** | **Mojo + MAX** | `struct-design`, `memory-ownership` + MAX `engine-operations` |

## Pattern Categories

| Priority | Category | Count | Prefix |
|----------|----------|-------|--------|
| CRITICAL | Memory Safety & Ownership | 4 | `memory-` |
| CRITICAL | Type System | 3 | `type-` |
| CRITICAL | GPU Programming | 11 | `gpu-` |
| HIGH | GPU Porting Guides | 3 | `gpu-porting-` |
| HIGH | Language Porting Guides | 3 | `porting-` |
| CRITICAL | C Interoperability | 1 | `ffi` |
| HIGH | Struct Design | 1 | `struct-` |
| HIGH | Function Design | 1 | `fn-` |
| HIGH | Testing | 1 | `testing` |
| HIGH | Debugging | 1 | `debug-` |
| MEDIUM-HIGH | Error Handling | 1 | `error-` |
| MEDIUM | Performance Optimization | 3 | `perf-` |
| MEDIUM | Python Interoperability | 1 | `python-` |
| LOW | Advanced Metaprogramming | 2 | `meta-` |

*Version-specific features are documented within each pattern file.*

---

## Debugging Memory Issues

If you're experiencing memory errors, use this decision tree to find the relevant patterns:

| Symptom | Likely Cause | Key Patterns |
|---------|--------------|--------------|
| "use of moved value" | Missing ownership transfer | `memory-ownership` |
| Use-after-free | Object destroyed too early | `memory-safety`, `memory-ownership` |
| Memory leak | Missing destructor call | `memory-ownership`, `memory-safety` |
| Double-free | Duplicate destruction | `memory-safety` |
| GPU OOM | Buffer not released | `gpu-fundamentals`, `perf-optimization` |
| Crash in loop | Reference invalidation | `memory-safety` |
| "does not implement Copyable" | Missing trait | `type-traits`, `struct-design` |

**See Also:** `debug-debugging` for GPU memory issues, `error-handling` for automatic cleanup.

## Memory Safety (CRITICAL)

| Pattern | Description |
|---------|-------------|
| `memory-ownership` | Ownership transfer with `^`, borrow vs copy, lifecycle methods |
| `memory-safety` | Dangling references, origin tracking, safe pointers, Span usage |
| `memory-refcounting` | Reference counting implementation, atomic operations |
| `memory-collections` | Collection types, List, Dict, InlineArray usage |

## Type System (CRITICAL)

| Pattern | Description |
|---------|-------------|
| `type-system` | Explicit annotations, optional types, numeric precision |
| `type-simd` | SIMD vectorization, register-passable types |
| `type-traits` | Parametric traits, trait composition, conditional conformance |

## GPU Programming (CRITICAL)

| Pattern | Description |
|---------|-------------|
| `gpu-fundamentals` | Thread hierarchy, memory coalescing, shared memory, kernel restrictions (print, origins) |
| `gpu-layout-tensor` | LayoutTensor API, rebind pattern, tile views, cross-rank reshape |
| `gpu-block-collectives` | Block-level sum, prefix_sum, broadcast (replace manual reductions) |
| `gpu-custom-ops` | Custom op registration, MAX Graph integration, multi-kernel pipelines |
| `gpu-warp-sync` | Warp primitives, shuffle, reduction, barriers, async transactions, async copy |
| `gpu-tensor-cores` | SM90/SM100 patterns, WGMMA, TCGen05 |
| `gpu-memory-access` | TMA loading, prefetch, swizzle, LayoutTensor patterns (element type, rebind, async copy) |
| `gpu-kernels` | Kernel fusion, pipelines, double-buffering, custom ops (@compiler.register) |
| `gpu-structured-kernels` | ScatterGather/RingBuffer/TileOp architecture |
| `gpu-amd` | AMD MFMA shapes, scheduling, waitcnt |
| `gpu-troubleshooting` | Systematic diagnosis of GPU build failures, performance issues, and integration problems |

## GPU Porting Guides (HIGH)

| Pattern | Description |
|---------|-------------|
| `gpu-porting-cuda` | CUDA→Mojo: vector add to tensor core GEMM, side-by-side examples |
| `gpu-porting-cute` | CuTe DSL→Mojo: layout algebra, TiledMMA, TiledCopy, pipelines |
| `gpu-porting-rocm` | ROCm/HIP→Mojo: MFMA, wavefront ops, scheduling, ScatterGather |

## Language Porting Guides (HIGH)

| Pattern | Description |
|---------|-------------|
| `porting-python` | Python 3.13→Mojo: syntax mapping, 10x-17,000x performance, SIMD, GPU, things only Mojo can do |
| `porting-cpp` | C++23→Mojo: templates→parameters, memory safety, RAII, unified CPU+GPU |
| `porting-rust` | Rust 1.84→Mojo: ownership comparison, traits, first-class SIMD, native GPU kernels |

## C Interoperability (CRITICAL)

| Pattern | Description |
|---------|-------------|
| `ffi` | CString safety, libc functions, binary data, Apple BLAS (25-32x), GPU libraries, Python GIL |

## Performance (MEDIUM)

| Pattern | Description |
|---------|-------------|
| `perf-vectorization` | `vectorize` function (4-16x SIMD speedup) |
| `perf-parallelization` | `parallelize` + SIMD (10x-17,000x vs Python*) |
| `perf-optimization` | Caching, lazy loading, accumulators, alignment, layout, prefetch |

## Testing (HIGH)

| Pattern | Description |
|---------|-------------|
| `testing` | Unit testing, integration testing, benchmarking, and performance measurement |

*\*Performance ranges: Lower bound = simple typed loops vs Python. Upper bound = fully vectorized + parallelized hot paths. Actual gains depend on workload, data size, and hardware. Always benchmark your specific use case.*

---

## File Structure

```
mojo-best-practices/
├── SKILL.md               # Entry point (this file) - START HERE
├── metadata.json          # Skill metadata
├── CHANGELOG.md           # Skill version history
├── patterns/              # 36 patterns with version-specific sections
│   ├── memory-*.md        # Memory safety (4 patterns)
│   ├── gpu-*.md           # GPU programming (11 patterns)
│   ├── gpu-porting-*.md   # GPU porting guides (3: CUDA, CuTe, ROCm)
│   ├── porting-*.md       # Language porting guides (3: Python, C++, Rust)
│   ├── type-*.md          # Type system (3 patterns)
│   └── perf-*.md          # Performance (3 patterns)
└── references/            # Detailed reference docs (loaded on-demand)
    ├── FULL_REFERENCE.md  # Complete pattern index (auto-generated)
    ├── ERROR_INDEX.md     # Error message → pattern lookup
    ├── SCENARIOS.md       # Task → pattern mapping
    ├── breaking-changes.md
    ├── installation.md
```

## Local Implementation Notes

When using this skill in a project, agents should collect implementation notes **locally within that project**, not globally. This ensures project-specific learnings stay with the project.

**Where to store notes:**
```
your-project/
├── IMPLEMENTATION_NOTES.md    # Project-specific learnings
├── .cursor/
│   └── rules/                 # Cursor-specific rules (uses "rules" terminology)
└── ...
```

**What to capture:**
- Version-specific workarounds discovered
- Performance optimizations that worked for this codebase
- API quirks encountered
- Build configuration decisions
- Platform-specific adjustments (macOS/Linux/GPU)

**Usage:** Agents should check for and update `IMPLEMENTATION_NOTES.md` in the project root when discovering new patterns or resolving issues.


## Navigation

- **Start here**: This file (SKILL.md) - load first, then drill into patterns
- **Common mistakes?** See [references/GOTCHAS.md](references/GOTCHAS.md) - ❌ wrong → ✅ correct examples
- **Need a specific pattern?** Check `patterns/` directory
- **Got an error?** See [references/ERROR_INDEX.md](references/ERROR_INDEX.md)
- **Task-based lookup?** See [references/SCENARIOS.md](references/SCENARIOS.md)
- **Installation?** See [references/installation.md](references/installation.md)
- **Breaking changes?** See [references/breaking-changes.md](references/breaking-changes.md)
- **Full reference?** See [references/FULL_REFERENCE.md](references/FULL_REFERENCE.md) (complete index)
