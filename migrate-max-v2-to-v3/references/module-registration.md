# Module registration compatibility

Compatibility snapshot: October 31, 2026. This describes the current repository
implementation, not proposed roles or a guarantee that every released MAX
wheel exposes the same API. Check the installed dispatcher and donor source
before using it. Recheck this reference when changing the registration runtime.

Use this reference when importing a model, migrating its compute to ModuleV3,
or bringing it up for serving. It separates the supported registration
contract from model-specific capabilities that still need validation.

## Choose the integration

Prefer `SupportedArchitecture(modules=..., config=..., ...)` when the model
fits a supported row below. Omit an architecture-specific `pipeline_model`.
The framework generates a pipeline model; “PipelineModel-less” means the
architecture no longer supplies its own subclass. It does not remove the
framework's `PipelineModel` abstraction.

| Task                  | Module backend and registered roles                                       | Current support                                           |
|-----------------------|---------------------------------------------------------------------------|-----------------------------------------------------------|
| Text generation       | ModuleV3 `DECODER`                                                        | Supported.                                                |
| Text generation       | Graph API `DECODER`                                                       | Supported; compute remains ModuleV2.                      |
| Text generation       | ModuleV3 `DECODER` + `VISION`                                             | Supported with the vision contract below.                 |
| Text generation       | Graph API `DECODER` + `VISION`                                            | Unsupported by the graph-API factory.                     |
| Embeddings generation | Graph API `ENCODER`                                                       | Supported; padded or `ragged_batch=True`.                 |
| Embeddings generation | ModuleV3 `ENCODER`                                                        | Unsupported; retain the custom ModuleV3 encoder pipeline. |
| Any                   | Standalone `VISION`, draft/speculator modules, or other role combinations | Unsupported by the registration factories.                |
| Other tasks           | Any backend                                                               | No generic module registration dispatcher.                |

For an unsupported case, state the exact task, backend or feature gap and
retain the appropriate custom integration, or extend shared registration in a
separate change. Do not switch a ModuleV3 encoder back to graph-API compute
just to remove its pipeline subclass. Do not add architecture-local compile
and execute wrappers for a supported contract.

Useful donors are `olmo3/arch.py` for a registered ModuleV3 decoder and
`kimik2_5_modulev3/arch.py` for registered decoder plus vision. Existing custom
pipelines, including ModuleV3 encoders and some VLMs, do not establish that
all their integration glue is required for a new architecture.

## Architecture and framework responsibilities

The architecture supplies its module classes, config, task, tokenizer, weight
adapters and existing deployment metadata. Registration does not replace
checkpoint adaptation, correct weight names or config validation.

The shared runtime initializes and finalizes config, constructs
`module_cls(config)`, builds graph signatures, binds inputs, loads weights,
compiles and executes the module. ModuleV3 construction runs under lazy,
default-dtype and default-device contexts. Do not reproduce those steps in
an architecture `model.py` when using the registered path.

Config follows `ArchConfig.initialize(...)`, including the resolved
`max_seq_len`. An optional
`finalize(hf_config, state_dict, model_output_config)` can return the same
config type or mutate the copied config and return `None`. Keep finalization and
checkpoint adaptation consistent with the compute module's parameter hierarchy.

The decoder boundary uses `DecoderInputs` and `DecoderOutputs` from
`max.pipelines.lib.role_contracts`:

- ModuleV3 implements `forward(DecoderInputs[Tensor]) -> DecoderOutputs`.
- Graph API implements `__call__(DecoderInputs[TensorValue]) -> DecoderOutputs`.
- Preserve the KV tree in `inputs.kv`. Do not reconstruct an architecture's
  former variadic serving ABI at this boundary.

Graph-API encoders instead use `EncoderInputs` from
`max.pipelines.lib.registration_encoder` and return an embeddings
`TensorValue`. The factory supports padded and ragged layouts, without KV.

The framework provides default batching for standard layouts. A custom
batch processor remains appropriate for an architecture-specific layout;
it must produce the fields and structure required by the registered graph.
“No custom pipeline” does not mean “no custom config or batching.”

## Conditional decoder capabilities

A supported role is the starting point, not proof that every deployment
combination works. Validate the model's actual feature matrix.

| Capability               | Registration contract and limits                                                                                                                                                                                                                                                                                                      |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Tensor parallelism       | ModuleV3 constructs compute on the runtime mesh. The module still needs correct sharding; registration does not supply it.                                                                                                                                                                                                            |
| Data parallelism         | Both decoder backends require `supports_data_parallel=True` on the decoder. Currently one replica runs per device, and the DP degree must match the device count. ModuleV3 LoRA plus DP is rejected.                                                                                                                                  |
| Vision                   | ModuleV3 only. The decoder declares `consumes_vision=True`; register a `VISION` module exposing `input_types()`, `embedding_dtype` and `embedding_dim`, and a `VisionBatchProcessor` implementation. The runtime validates the combination, runs vision on the first device and replicates cached embeddings across the decoder mesh. |
| Expert parallelism / MLA | ModuleV3 has shared runtime integration through `DeploymentConfig` and, for EP, `ExpertParallelDeploymentConfig`. Supply the required sizing, initialization and runtime hooks. Do not assume graph-API registration has equivalent EP or MLA input binding.                                                                          |
| LoRA                     | ModuleV3 decoders declare `lora_targets`; the framework binds `LoRAManagerV3`. Check the model's supported projections and deployment combination.                                                                                                                                                                                    |
| Per-request state        | Both decoder backends support `state_store(config, *, max_batch_size, devices)` returning a `RequestStateStore`, with `DecoderInputs.state` and `DecoderOutputs.state`. See the lifecycle below.                                                                                                                                      |

For vision, image-free batches still bind the vision fields using empty
tensors; they do not remove graph arguments. The vision registration is a
static graph contract, not a per-request optional module selection.

A request-state store supplies `input_types()`, `max_dims()`, `prepare()`,
`update()`, `release()` and `release_warmup()`. It owns request-state storage
and buffer pools. Per-step symbolic tensors need dimension bounds. The
batch processor must be compatible with `DecoderBatchProcessor` so the
framework can bind the store. Validate cleanup, mixed batches and warmup;
state support alone does not prove prefix-cache reuse, graph capture or
speculative rollback is correct for a particular stateful model.

## Speculative decoding is a separate integration

**Draft/speculator module registration has not been added.** There is no
architecture `DRAFT` registration contract. Do not register a draft model
under `SPEC_DECODER`: that role names a framework-owned compiled speculative
graph. ModuleV3 registration also reserves `SAMPLER`, `SAMPLER_BITMASK` and
`EP_INIT` for framework graphs.

Existing speculative pipelines, draft architectures, drivers and target
adapters remain the integration path. Some registered graph-API targets expose
a transitional `build_module(context, config_cls)` bridge through their
generated pipeline. That bridge does not register a draft module, and a
registered `DECODER` is not automatically compatible with every speculative
method. Check the actual target/draft pairing and the existing integration API.

## Serving and migration gates

Registration compatibility does not establish export or registry compatibility
for an additional serving stack. Keep export, serving, accuracy, performance and
deployment feature checks as separate gates. Removing an architecture pipeline
subclass must not skip them. If an installed wheel lacks the repository API, use
its matching donors or update the environment before attempting that
integration.

## Sources to check when updating this reference

Paths below are relative to `max/python/max/pipelines/lib/` in the MAX source:

- [arch_lookup.py](https://github.com/modular/modular/blob/main/max/python/max/pipelines/lib/arch_lookup.py):
  `SupportedArchitecture` and generated pipeline selection.
- [registration_graph_api.py](https://github.com/modular/modular/blob/main/max/python/max/pipelines/lib/registration_graph_api.py):
  task/backend dispatcher and graph-API decoder factory.
- [module_registration.py](https://github.com/modular/modular/blob/main/max/python/max/pipelines/lib/module_registration.py):
  ModuleV3 role validation, vision, DP, LoRA, config finalization and state
  binding.
- [registration_encoder.py](https://github.com/modular/modular/blob/main/max/python/max/pipelines/lib/registration_encoder.py):
  graph-API encoder contract and batching layouts.
- [role_contracts.py](https://github.com/modular/modular/blob/main/max/python/max/pipelines/lib/role_contracts.py),
  [request_state.py](https://github.com/modular/modular/blob/main/max/python/max/pipelines/lib/request_state.py)
  and
  [registration_runtime.py](https://github.com/modular/modular/blob/main/max/python/max/pipelines/lib/registration_runtime.py):
  decoder inputs/outputs, state lifecycle and deployment hooks.
- [graph_specs.py](https://github.com/modular/modular/blob/main/max/python/max/pipelines/lib/graph_specs.py):
  role names, including framework graph roles.

Regression coverage lives in `max/tests/tests/pipelines/lib/`, particularly
`test_module_registration.py`, `test_registration_graph_api.py`,
`test_registration_encoder.py` and `test_request_state.py`. Use the installed
source when its version differs from the repository snapshot.
