# Heartbeat / progress reporting protocol

**Purpose**: Standardized JSONL progress on stdout so you can monitor liveness
and detect stalls while an eval runs. Human-readable logs go to stderr.

## JSONL format

Every heartbeat line is a valid JSON object (newline-delimited). Emitted to
**stdout** every `--heartbeat-interval` seconds and at phase transitions
(seed→full, task→task, done).

### Common fields (all heartbeats)

| Field | Type | Description |
|---|---|---|
| `ts` | string (ISO8601 UTC) | Timestamp of emission |
| `phase` | string | One of: `preflight`, `seed`, `full`, `done`, `error` |
| `task` | string | Current task name, or `""` for a global setup error |
| `completed` | int | Examples completed in current phase |
| `total` | int\|null | Total examples, or `null` while a grouped task runs |
| `accuracy` | float\|null | Running accuracy (0.0-1.0) or `null` if not yet computable |
| `eta_sec` | int\|null | Estimated seconds to phase completion |

### Phase-specific fields

**`phase: "preflight"`** — Endpoint capability checks are running.

Preflight heartbeats use an empty `task`, with `completed` and `total` set to
`0`.

**`phase: "seed"`** — Fast-fail seed sample running

```jsonl
{"ts":"2026-07-07T14:23:12Z","phase":"seed","task":"gsm8k","completed":12,"total":24,"accuracy":0.583,"eta_sec":45}
{"ts":"2026-07-07T14:23:42Z","phase":"seed","task":"gsm8k","completed":24,"total":24,"accuracy":0.625,"eta_sec":0,"fast_fail":false,"floor":0.10}
```

- `fast_fail`: boolean — whether fast-fail would trigger at this point
- `floor`: float — the fast-fail floor for this task

**`phase: "full"`** — Full evaluation running

```jsonl
{"ts":"2026-07-07T14:23:43Z","phase":"full","task":"gsm8k","completed":0,"total":200,"accuracy":null,"eta_sec":420}
{"ts":"2026-07-07T14:27:12Z","phase":"full","task":"gsm8k","completed":50,"total":200,"accuracy":0.640,"eta_sec":350}
```

**`phase: "done"`** — Task completed

```jsonl
{"ts":"2026-07-07T14:35:00Z","phase":"done","task":"gsm8k","completed":200,"total":200,"accuracy":0.635,"summary_path":"eval-results/meta-llama_Meta-Llama-3-8B-Instruct/gsm8k_summary.json","status":"met","target":0.50}
```

- `summary_path`: path to written summary JSON
- `status`: `"completed" | "met" | "below-target" | "fast-fail" | "full-eval-skipped"`
- `target`: float or `null` — an explicit `--target`, if provided

The eval script emits `seed`, `full`, `done`, and `error`. It doesn't
self-terminate when an external watcher considers the run stalled.

## Monitoring contract

When tailing stdout:

1. **Liveness** — If no heartbeat for `2 * --heartbeat-interval` seconds,
   treat the run as stalled.
2. **Progress** — Direct evaluation heartbeats can include running accuracy.
   lm-eval heartbeats report liveness; accuracy appears when the task finishes.
   Grouped tasks also include `limit_per_subtask` while their aggregate
   `total` is unknown.
3. **Fast-fail** — On exit code `3`, read the summary JSON for
   `status: "fast-fail"`.
4. **Results** — On completion, read `summary_path` or `eval_summary.json`.

## Example consumer

```python
import json
import sys
import time

HEARTBEAT_INTERVAL = 30
last_heartbeat = time.time()

for line in sys.stdin:
    hb = json.loads(line)
    last_heartbeat = time.time()
    if hb["phase"] == "seed" and hb.get("fast_fail"):
        print(f"⚠️  {hb['task']} seed below floor {hb['floor']:.0%}")
    elif hb["phase"] == "done":
        accuracy = hb.get("accuracy")
        formatted = f"{accuracy:.1%}" if accuracy is not None else "n/a"
        print(f"{hb['task']}: {formatted} ({hb.get('status', 'done')})")

if time.time() - last_heartbeat > 2 * HEARTBEAT_INTERVAL:
    print("Run may be stalled — no recent heartbeat")
```

## Implementation notes

- Stdout is flushed after every heartbeat line.
- Heartbeats emit at phase boundaries even if the interval hasn't elapsed.
- Timestamps use UTC (ISO8601 with `Z` suffix).
