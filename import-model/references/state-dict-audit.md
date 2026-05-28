# State-dict coverage audit at load

``nn_model.load_state_dict(strict=False)`` is mandatory when
``tie_word_embeddings=True`` (the HF checkpoint omits ``lm_head.weight``
because it shares with ``embed_tokens.weight``, so strict mode would
reject the load). But ``strict=False`` has a silent failure mode:
**any tensor with a name that doesn't match a real module path is
dropped without warning**.

That's the trapdoor under every weight adapter. A typo in a rename
table, a stale prefix from a previous port revision, a missing
sub-module name — the model still loads, still serves, still emits
real-looking tokens, and only fails at accuracy benchmarks days later.

A typical failure: an adapter renames ``mlp.experts.{j}`` →
``mlp.routed.experts.{j}`` but the graph has MoE at ``self.mlp`` with
no ``routed`` sub-module. Every routed expert tensor is silently dropped
under ``strict=False``. Generation can still look plausible if shared
experts load; accuracy benchmarks surface the gap.

This reference describes a small audit that catches the entire class
of bug at load time, in seconds.

## What the audit checks

Don't check exact tensor suffixes. Different weight-type variants
(BF16, FP8, NVFP4) ship different sidecar tensors per projection:

- BF16: ``X_proj.weight``
- FP8: ``X_proj.weight`` + ``X_proj.weight_scale``
- NVFP4: ``X_proj.weight_packed`` + ``X_proj.weight_scale`` +
  ``X_proj.weight_global_scale`` + ``X_proj.input_global_scale`` +
  ``X_proj.bias``

A suffix-aware check has to enumerate all of them. A
**projection-path** check just asks: "is there *any* tensor at
``layers.{i}.mlp.experts.{j}.gate_proj.*``?" — and that's variant-
agnostic.

The audit indexes the state dict by "everything up to the last dot,"
producing a set of projection paths. Then for each expected projection
(router, every routed expert, every shared expert), it checks
presence. If any projection has zero tensors, raise with a sample of
missing paths.

## Generic implementation

Drop this into your slug's ``model.py``. Call it right after
``nn_model.load_state_dict(...)`` in ``_build_graph``.

```python
def _audit_state_dict_coverage(
    *,
    state_dict: dict,
    num_hidden_layers: int,
    # MoE knobs — set to 0 / None for dense models
    num_routed_experts: int = 0,
    num_shared_experts: int = 0,
    # Attention projections (override if your arch uses non-standard names)
    attn_projections: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    mlp_projections: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
    has_router: bool = True,
) -> None:
    """Verify every expected projection has at least one tensor in state_dict.

    Indexes state_dict keys by projection path (everything up to the last
    dot) so the check is variant-agnostic: BF16's ``weight``, FP8's
    ``weight_scale``, NVFP4's ``weight_packed`` etc. all map to the same
    parent path.

    Raises ``RuntimeError`` listing a sample of missing paths.
    """
    # Index keys by projection FQN.
    projection_paths: set[str] = set()
    for k in state_dict:
        last_dot = k.rfind(".")
        if last_dot > 0:
            projection_paths.add(k[:last_dot])

    missing: list[str] = []
    for layer_idx in range(num_hidden_layers):
        # Attention projections
        for proj in attn_projections:
            p = f"layers.{layer_idx}.self_attn.{proj}"
            if p not in projection_paths:
                missing.append(p)

        if num_routed_experts > 0:
            # Router (gate → MoEGate.gate_score, hence the deeper path)
            if has_router:
                p = f"layers.{layer_idx}.mlp.gate.gate_score"
                if p not in projection_paths:
                    missing.append(p)
            # Routed experts
            for j in range(num_routed_experts):
                for proj in mlp_projections:
                    p = f"layers.{layer_idx}.mlp.experts.{j}.{proj}"
                    if p not in projection_paths:
                        missing.append(p)
            # Shared experts (single MLP per layer)
            if num_shared_experts > 0:
                for proj in mlp_projections:
                    p = f"layers.{layer_idx}.mlp.shared_experts.{proj}"
                    if p not in projection_paths:
                        missing.append(p)
        else:
            # Dense MLP
            for proj in mlp_projections:
                p = f"layers.{layer_idx}.mlp.{proj}"
                if p not in projection_paths:
                    missing.append(p)

    if missing:
        sample = missing[:8]
        raise RuntimeError(
            f"State-dict audit: {len(missing)} expected projection(s) "
            f"missing from state_dict. Likely cause: stale rename in "
            f"weight_adapters.py. First {len(sample)}: {sample}"
        )
```

## What it catches

- **Adapter rename typos.** ``mlp.routed.experts`` when the real path
  is ``mlp.experts`` — fires instantly, lists the missing 4096 paths.
- **Stale prefix.** ``model.text_model.layers`` when only
  ``model.language_model.layers`` is correct — surfaces as every
  attention + MLP missing across all layers.
- **Wrong expert count.** Loading a checkpoint with 64 experts into a
  graph expecting 128 — surfaces experts 64–127 missing.
- **Missing layers.** Adapter dropping layers past N (e.g. a slice
  filter that's off by one) — surfaces the dropped layers.
- **Sub-module name mismatch.** ``mlp.gate`` (bare) vs
  ``mlp.gate.gate_score`` (where MAX's ``MoEGate`` actually puts the
  linear) — surfaces the router as missing.

## What it doesn't catch

- **Renames to a wrong-but-existing path.** If your adapter sends
  ``layers.0.mlp.experts.{j}.gate_proj.weight`` to
  ``layers.0.mlp.shared_experts.gate_proj.weight`` (typo on the
  destination, not the source), the audit sees a tensor at the
  destination and considers shared experts covered. Routed experts
  would still be missing, but the diagnosis is misleading.
- **Tensor dtype / shape mismatches.** The audit verifies *presence*,
  not correctness. A weight at the right path with the wrong dtype
  loads (or fails later in ``_array_from_weight_loader``); the audit
  is silent.
- **Tensors that should not be in the state dict but are.** Extra
  tensors with wrong-but-plausible paths land in ``state_dict`` and
  pass the audit. They get dropped by ``load_state_dict(strict=False)``
  silently — a separate failure mode, not what this audit targets.

For shape / dtype correctness, rely on MAX's own ``_array_from_
weight_loader`` checks (they raise on dtype mismatch — that's how the
``layers.0.self_attn.o_proj.weight`` FP8/bf16 mismatch surfaced
during the FP8 attempt). For extra tensors, the OSS gates run a
separate strict check.

## Where to wire it

Right after ``nn_model.load_state_dict(...)`` in your slug's
``model.py::_build_graph``, **before** you assign
``self.state_dict``:

```python
nn_model.load_state_dict(
    state_dict,
    override_quantization_encoding=True,
    weight_alignment=1,
    strict=(not getattr(text_hf, "tie_word_embeddings", False)),
)
_audit_state_dict_coverage(
    state_dict=state_dict,
    num_hidden_layers=model_config.num_hidden_layers,
    num_routed_experts=getattr(model_config, "num_experts", 0),
    num_shared_experts=getattr(model_config, "num_shared_experts", 0),
)
self.state_dict = nn_model.state_dict()
```

The audit is fast (set arithmetic over ~10K names) and runs once at
graph build. There's no production overhead.

## Reporting style

When the audit passes, log it explicitly so the user sees it:

```python
logger.info(
    "State-dict audit: all %d layers × %d routed experts (+ %d shared) "
    "projection paths present.",
    num_hidden_layers, num_routed_experts, num_shared_experts,
)
```

When it fails, the ``RuntimeError`` message is the entire artifact —
no further work required. The first 8 missing paths usually pattern-
match a single class of bug ("all routed experts missing" → rename
typo; "every attention projection missing across layers 0–31" →
wrong layer-prefix strip).

## Why this isn't covered by existing scripts

``run_oss_gates.py`` does a strict-load test that catches
*extra* tensors (the dual problem). This audit catches *missing*
tensors that strict-load can't see because we have to opt out of
strict mode for tied embeddings. Both checks are needed; they cover
different halves of the failure space.
