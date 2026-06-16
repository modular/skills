# Parallel agent workflow

One lead agent analyzes dumps; helper agents gather orthogonal evidence in
parallel. The coordinator applies fixes after the lead localizes.

Do not dispatch fix-attempt agents before localization. Each mistaken fix costs
a 10 to 25 minute compile cycle.

**No subagent support in your harness?** The protocol survives intact: run the
lead-agent analysis yourself, inline — the numbered steps in the prompt below
are the analysis either way — and do the helper checks sequentially as a
hypothesis calls for them. The parallelism is an optimization; the discipline
(localize with numbers before touching code) is the point, and the
anti-patterns below still apply.

## Lead agent prompt

```text
You are the LEAD debugging agent. Per-layer hidden-state tensors from HF and
MAX on the SAME prompt + SAME weights are on disk. Produce a precise problem
statement with numbers — do not guess or stop at the first hypothesis.

Dumps: HF <path>/hf_layers/*.npy, MAX <path>/max_layers/*.npy
Comparator output: <paste cos_sim table>
Already verified: <list>
Still suspected: <list>

1. Per-token and per-dim cos-sim for each layer pair.
2. For any cliff: (token, dim) of max disagreement; sign and magnitude each side.
3. Token-0 invariant: attn_out[0] = V[0] @ o_proj^T. Compare analytical vs dumps.
4. Hypotheses with file:line citations and predicted tensor signatures.
5. Rank fixes by confidence.

Deliverable: ranked hypotheses with evidence. No speculation without numbers.
```

## Helper agents (parallel with lead)

| Helper         | Task                                                    |
|----------------|---------------------------------------------------------|
| Weight-stats   | Audit cliff-layer weights vs adjacent layers            |
| Code diff      | Broken subsystem vs donor and known-good reference      |
| Kernel inspect | Python wrappers and docstrings for suspect ops          |
| Micro-test     | HF vs MAX on synthetic inputs for one op                |
| Sub-tap prep   | Prepare finer graph taps; do not run until lead decides |

## Anti-patterns

- Fix-attempt agents before lead analysis
- Helpers re-computing what the lead already computes
- Lead asked to implement fixes (localize only)
- Re-spawning lead on identical dumps (send focused follow-up)

## Coordinator checklist

1. Symptom recorded: divergence index, failing layer, or smoke output
2. Comparator built ([comparator-build.md](comparator-build.md))
3. Lead dispatched; helpers only after dumps exist
4. One `max serve` or cold compile at a time per machine
