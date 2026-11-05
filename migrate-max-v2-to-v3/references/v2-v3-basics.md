# ModuleV2 vs ModuleV3 essentials

How a model, a layer, and a distributed layer look in the V2 graph API
(`max.nn.legacy`, `TensorValue`, explicit `Graph`) and in ModuleV3
(`max.nn`, `Tensor`, `F.lazy()` + `compile()`).

## Building and running a model

### ModuleV2

```python
nn_model = DeepseekV3(config)
nn_model.load_state_dict(state_dict)

with Graph("deepseekV3", input_types=input_types) as g:
    outputs = nn_model(*g.inputs)
    g.output(*outputs)

session.load(g, weights_registry=state_dict)
```

### ModuleV3

```python
with F.lazy():
    nn_model = DeepseekV3(config)

nn_model.compile(*input_types, weights=state_dict)
```

These are standalone compilation examples. For architectures using
[module registration](module-registration.md), the shared runtime constructs,
loads, compiles and executes the module. Do not copy these steps into an
architecture-specific pipeline when the registered contract covers the model.

Input types are required in both APIs, and compilation freezes the graph
(`session.load()` or `compile()`). `compile(weights=...)` raises for a
parameter with no state-dict entry (`KeyError`) or with a mismatched shape
or dtype (`ValueError`). A state-dict entry that matches no parameter is
left out without an error.

The differences:

- `F.lazy()` wraps module construction so MAX records the module's
  weight tensors symbolically. The checkpoint weights passed to
  `compile()` replace them, so allocating real tensors up front would
  waste memory.
- `compile(weights=...)` is also the load step, so pass the state dict
  there. Drop the V2 `load_state_dict()` call before graph building.
- A dtype mismatch between a parameter and its loaded tensor raises by
  default. Pass `auto_cast=True` (or `auto_cast_weights_from_env()` from
  `max.pipelines.weights.weight_loading`) to allow safe dtype casts.

## A layer, side by side

### Linear (ModuleV2)

```python
class Linear(Module, Shardable):
    def __init__(self, in_dim, out_dim, dtype, device):
        self.weight = Weight(
            name="weight",
            dtype=dtype,
            shape=(out_dim, in_dim),
            device=device,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return x @ self.weight.T
```

### Linear (ModuleV3)

```python
from max.functional import random
from max.nn import Module
from max.tensor import Tensor


class Linear(Module):
    def __init__(self, in_dim, out_dim):
        self.weight = random.normal([out_dim, in_dim])

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T
```

- ModuleV3 uses default `device` and `dtype` contexts. Move a module
  to one device with `module.to(device)` after construction, or call
  `.to()` on the tensor directly. `Module.to()` rejects a multi-device
  mesh; see the distributed section below.
- There is no `Weight` class. Weights are `Tensor` objects, created
  with `Tensor.zeros()`, `max.functional.random.*`, and similar
  factories.
- `__call__()` becomes `forward()`.
- Constructor args are keyword-only where V2 accepted positions:
  `Embedding(vocab_size, dim=...)` takes its dimension by keyword, and
  `Linear(in_dim, out_dim)` creates a bias by default. Pass
  `bias=False` when the checkpoint ships no bias tensors, or
  `compile(weights=...)` fails its strict key check.

## A distributed layer, side by side

### Distributed Linear (ModuleV2)

```python
class Linear(Module, Shardable):
    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.weight.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        self.weight.sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[Linear]: ...


linear = Linear(...)
linear.sharding_strategy = ShardingStrategy.rowwise(len(devices))
linear_shards = linear.shard(devices)
```

### Distributed Linear (ModuleV3)

```python
from max.nn.common_layers.linear import (
    col_parallel,
    row_parallel,
)
from max.tensor.sharding import DeviceMesh

device_mesh = DeviceMesh(devices=(...), mesh_shape=(n,), axis_names=("tp",))
linear = Linear(...)
linear = row_parallel(linear)
```

`col_parallel()` and `row_parallel()` set a `NamedMapping` on the
layer's weight: which mesh axis each dimension shards over. The model
assigns the mesh before compilation (`kimik2_5_modulev3/model.py` in
`max/pipelines/architectures/`):

```python
llm.mesh = DeviceMesh(tuple(self.devices), (n_devices,), (axis_name,))
```

The model is then constructed inside `default_device(mesh)`, so each
weight is created on the mesh with its placement:

```python
with F.lazy(), default_dtype(dtype), default_device(llm.mesh):
    language_nn = KimiK2_5MoEDecoder(llm, kv_params, ep_batch_manager)
```

- The `devices` list becomes a `DeviceMesh`, and a `"tp"` axis name is a
  prerequisite for the parallel helpers.
- `ShardingStrategy` plus `.shard()` (which returns one module instance
  per device) becomes a placement `Mapping` on the weight tensor. The
  sharding solver redistributes at dispatch time, so there is no
  per-device module list to iterate.

## Eager and compiled use

A ModuleV3 module should work both eagerly and compiled. It takes
`Tensor` where V2 took `TensorValue`, and its ops come from
`max.functional` (`F.gather()`, not `ops.gather()`).

## Imports

### Good imports

```python
from max.dtype import DType

from max import functional as F
from max.tensor import Tensor

from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Sharded,
)

from max.nn import Module
from max.nn.embedding import Embedding
from max.nn.kernels import ...
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.norm import RMSNorm
from max.nn.sequential import ModuleList
```

The KV cache splits across packages: `PagedCacheValues` comes from
`max.nn.common_layers.kv_cache`, while `KVCacheParams`,
`KVCacheParamInterface`, and `MultiKVCacheParams` stay in
`max.nn.kv_cache`. `max.nn.legacy` also exports a `PagedCacheValues` of its
own, the same name resolving to the V2 input type, which lacks
`from_upstream()`. The `max.experimental` path carries the V3 type.

### Double-check these imports

```python
from max.graph import DeviceRef, TensorValue, ops, ...
from max.nn.legacy.layer import LayerList, Module
from max.nn.legacy.embedding import VocabParallelEmbedding
```

Any file that imports layers from `max.nn.legacy` (not `max.nn`)
needs a second look.

### Some max.nn.legacy imports are fine

`max.nn.legacy` is fine when the imported class or method doesn't use
`TensorValue`:

```python
from max.nn.legacy.comm.ep import EPCommBuffers
from max.nn.rope_scaling import DeepseekYarnRopeScalingParams
from max.nn.kv_cache import KVCacheInputs, KVCacheParamInterface
```

## When there is no Tensor equivalent

### One op per `F.functional`

`F.functional()` turns a function over `TensorValue` into one over `Tensor`.
Wrap a single graph op or kernel per call:

```python
from max import functional as F
from max.graph import TensorValue, ops


def _gather(h: TensorValue, indices: TensorValue) -> TensorValue:
    return ops.gather(h, indices, axis=0)


gather_rows = F.functional(_gather)
```

- Arithmetic, slicing and reshaping stay on `Tensor` outside the wrapper,
  where the sharding rules see each op.
- A body that calls two ops hides its intermediate placements. One step the
  rules cannot place turns the whole body into per-device code, and neither
  review nor the IR check can see which step did it. When a computation of
  several ops really must run on one device, `F.call_on_mesh()` is its home,
  not a wider `F.functional` body.
- `TensorValue` appears only in such a body, and in compile input types. A
  helper that takes or returns `TensorValue` for any other reason becomes a
  function of `Tensor`s over `F.*` ops; do not carry a V2 helper's
  `TensorValue` signature into V3 code.

### Code that runs on each device's own shard

Sometimes a computation needs each device's own rows instead of the global
tensor: a data-parallel replica's row offsets address only that replica's
batch. `F.call_on_mesh()` is the way to say that. It runs a function once per
device, hands it that device's shard and that device's `PerShard` entries,
and joins the results under `out_specs`, which claims how they relate:

```python
def gather_last_tokens(h: Tensor, input_row_offsets: Tensor) -> Tensor:
    """Gathers each request's last row from the ragged batch, per shard.

    ``input_row_offsets`` is rebased into each replica's local frame under
    data parallelism, so its values only address that replica's own rows.
    """

    def on_each_device(h: Tensor, offsets: Tensor) -> Tensor:
        # Single-device tensors here: no sharding rule runs.
        return F.gather(h, offsets[1:] - 1, axis=0)

    return F.call_on_mesh(on_each_device, h.mesh, out_specs=h.mapping)(
        h, input_row_offsets
    )
```

- `fn` sees single-device `Tensor`s, so every op inside it takes the
  single-device path: no rule is consulted, nothing reshards, and index
  arithmetic on a sharded operand stays in its own frame. That is why the
  whole computation belongs inside `fn`, not only the op that reads the
  rows.
- `out_specs` claims the result's placement (`h.mapping` here). Without it
  the result is `Unknown`; claim it with `Tensor.rebind_mapping()`.
- No `is_distributed` branch: `call_on_mesh` runs `fn` once on a one-device
  mesh, so one code path covers one device and many.
- Module code, per-device constants and several ops are all fine in `fn`:
  `deepseekV41_modulev3/layers/block.py`'s `_on_each_device` is the
  reference. Its `mesh.num_devices == 1` early return is only a fast path.
- `transfer_to()` cannot move a tensor to or from `Unknown()`; only
  `rebind_mapping()` changes that placement, and it moves no data. The claim
  is trusted: check it against what `fn` actually reads.

For one raw graph op, the shorthand is a rule-less `F.functional(ops.x)`,
which is the same per-device call without the `Tensor` plumbing:

```python
_per_device_gather = F.functional(ops.gather)  # no rule: each device's rows
last = _per_device_gather(h, indices, axis=0).rebind_mapping(h.mapping)
```

Its result is `Unknown`, and any argument computed **outside** it follows the
rules again: on a data-parallel batch `offsets[1:]` is a slice of a sharded
tensor, which raises under the default `auto_reshard(mode="raise")` and
all-gathers into global indices inside `mode="silent"`. Wrap the whole
computation in `call_on_mesh` instead.

An op that must accept a per-device operand can also get an `Unknown` row in
its own rule, in its own PR:

```python
# max/tensor/sharding/rules/shape.py, gather_rule
rows.append(AxisAssignment((Unknown(), R), (Unknown(),)))
rows.append(AxisAssignment((Unknown(), Unknown()), (Unknown(),)))
```

`F.gather`, `slice_tensor` and the elementwise rules name no `Unknown` row
today, so an `Unknown` operand on them raises `ShardingError: ... no
feasible plan`. A row like the above matches only when the operands are
already placed that way, so it changes no call that works today; cover it
with a test under `max/tests/integration/tensor/distributed/rules/`, which
needs no device.

### Everything else

- A missing op used across architectures goes into
  `max/nn/kernels/` in the
  installed `max` package. In the MAX source repository, the file lives
  at `max/python/max/nn/kernels/`.
- A missing layer gets a ModuleV3 version, added to `common_layers` or
  to the architecture's own `<arch>/layers/<layer_name>.py`.
- Other NN components: make the existing class handle both `Tensor` and
  `TensorValue` (`EPBatchManager` is the example), or write a `Tensor`
  wrapper. Wrapping `split_batch_replicated()`:

  ```python
  from max.tensor.sharding import DeviceMapping
  from max.tensor import Tensor
  from max.graph import DeviceRef, TensorValue
  from max.nn._graph_ops.data_parallel import split_batch_replicated


  def split_replicated_batch(
      h: Tensor,
      input_row_offsets: Tensor,
      input_row_offsets_i64: Tensor,
      data_parallel_splits: Tensor,
      mapping: DeviceMapping,
  ) -> tuple[Tensor, Tensor]:
      """Splits a replicated batch into data-parallel shards."""
      devices = mapping.mesh.devices

      h_shards, offsets_shards = split_batch_replicated(
          [DeviceRef.from_device(d) for d in devices],
          [TensorValue(shard) for shard in h.local_shards],
          [TensorValue(shard) for shard in input_row_offsets.local_shards],
          TensorValue(input_row_offsets_i64),
          TensorValue(data_parallel_splits),
      )
      return (
          Tensor.from_shard_values(h_shards, mapping),
          Tensor.from_shard_values(offsets_shards, mapping),
      )
  ```
