# Benchmark troubleshooting

`max benchmark` is a client driving a live `max serve` endpoint, so most
failures come down to one of two things: the client can't reach the server, or
the workload doesn't match how you started the server. Confirm that
`curl http://localhost:8000/v1/health` returns 200 before you chase anything
else, because a down or still-compiling server explains most zero and garbage
results.

The following table matches each symptom to its cause and fix:

| Symptom | Cause | Fix |
|---|---|---|
| `Connection refused` or can't reach server | Nothing serving, or wrong host or port | Confirm `curl /v1/health` returns 200; check `--host`, `--port`, and `--base-url` against the running server |
| All requests fail, 404, or model not found | `--model` doesn't match the server | Read the served name from `curl /v1/models` and pass it as `--model`; it must equal `--served-model-name` |
| `<name> is not a valid model identifier` | The served name is an alias rather than a Hugging Face ID, so `--tokenizer` defaulted to it and can't load | Pass `--tokenizer <real Hugging Face ID>` explicitly |
| Requests return 400 on `/v1/chat/completions` | The model has no chat template | Use `--endpoint /v1/completions` for base LMs, or serve with `--chat-template` |
| Throughput stays flat as concurrency rises | The server's `--max-batch-size` sits below the sweep, so requests queue instead of batching | Re-serve with `--max-batch-size` at or above the top of the sweep |
| First-token times run wildly high or unstable | You benchmarked during compile or warmup | Wait for `Server ready`, let a few requests through, then benchmark |
| `--collect-gpu-stats` reports nothing | The benchmark isn't on the same box as the server, or the GPU isn't NVIDIA | Run the benchmark on the server's machine; GPU stats are NVIDIA only |

## Reading a failed run

Three patterns account for most bad runs:

- Zero or near-zero throughput almost always means the requests never
  succeeded. Check the server log and confirm `--model` and `--endpoint` first,
  not the workload flags.
- Numbers that swing between runs point at warmup or a shared, loaded box. Fix
  the seed (default `24301`), warm the server with a few requests, and keep the
  workload shape identical between runs. See [flags.md](flags.md) and
  [metrics.md](metrics.md).
- A sweep where every point looks the same means concurrency isn't actually
  rising, because the server's `--max-batch-size` caps it. Raise it and
  re-serve.
