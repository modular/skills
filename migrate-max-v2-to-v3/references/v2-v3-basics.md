# ModuleV2 vs ModuleV3 essentials

How a model, a layer, and a distributed layer look in the V2 graph API
(`max.nn`, `TensorValue`, explicit `Graph`) and in ModuleV3
(`max.experimental.nn`, `Tensor`, `F.lazy()` + `compile()`).

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

Input types are required in both APIs, and compilation freezes the graph
(`session.load()` or `compile()`). Weight keys are strict in both:
mismatched keys in the state dict are an error, in `load_state_dict()`
and in `compile(weights=...)` alike.

The differences:

- `F.lazy()` wraps module construction so MAX records the module's
  weight tensors symbolically. The checkpoint weights passed to
  `compile()` replace them, so allocating real tensors up front would
  waste memory.
- `compile(weights=...)` is also the load step: pass the state dict
  there instead of calling `load_state_dict()` before graph building.
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
class Linear(Module):
    def __init__(self, in_dim, out_dim):
        self.weight = random.normal([out_dim, in_dim])

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T
```

- ModuleV3 uses default `device` and `dtype` contexts. Move a module
  with `module.to(device)` after construction, or call `.to()` on the
  tensor directly.
- There is no `Weight` class. Weights are `Tensor` objects, created
  with `Tensor.zeros()`, `max.experimental.random.*`, and friends.
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
from max.experimental.nn.common_layers.linear import (
    col_parallel,
    row_parallel,
)
from max.experimental.sharding import DeviceMesh

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

- The `devices` list becomes a `DeviceMesh`, and a `"tp"` axis name is a
  prerequisite for the parallel helpers.
- `ShardingStrategy` plus `.shard()` (which returns one module instance
  per device) becomes a placement `Mapping` on the weight tensor. The
  sharding solver redistributes at dispatch time, so there is no
  per-device module list to iterate.

## Rule of thumb

A ModuleV3 module should work both eagerly and compiled. It takes
`Tensor` instead of `TensorValue`, and its ops come from
`max.experimental.functional` (`F.gather()`, not `ops.gather()`).

## Imports

### Good imports

```python
from max.dtype import DType

from max.experimental import functional as F
from max.experimental.tensor import Tensor

from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    PlacementMapping,
    Sharded,
)

from max.experimental.nn import Module
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.common_layers.functional_kernels import ...
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import ColumnParallelLinear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
```

The KV cache splits across packages: `PagedCacheValues` comes from
`max.experimental.nn.common_layers.kv_cache`, while `KVCacheParams`,
`KVCacheParamInterface`, and `MultiKVCacheParams` stay in
`max.nn.kv_cache`. `max.nn` also exports a `PagedCacheValues` of its
own, an alias for the V2 input type without `from_upstream()`; the
`max.experimental` path carries the V3 type.

### Double-check these imports

```python
from max.graph import DeviceRef, TensorValue, ops, ...
from max.nn.layer import LayerList, Module
from max.nn.embedding import VocabParallelEmbedding
```

Anything importing layers from `max.nn` instead of `max.experimental.nn`
needs a second look.

### Some max.nn imports are fine

`max.nn` is fine when the imported class or method doesn't use
`TensorValue`:

```python
from max.nn.comm.ep import EPCommBuffers
from max.nn.rotary_embedding import DeepseekYarnRopeScalingParams
from max.nn.kv_cache import KVCacheInputs, KVCacheParamInterface
```

## When there is no Tensor equivalent

- A missing op goes into
  `max/experimental/nn/common_layers/functional_kernels.py` in the
  installed `max` package; in the MAX source repository the file lives
  at `max/python/max/experimental/nn/common_layers/functional_kernels.py`.
- A missing layer gets a ModuleV3 version, added to `common_layers` or
  to the architecture's own `<arch>/layers/<layer_name>.py`.
- Other NN components: make the existing class handle both `Tensor` and
  `TensorValue` (`EPBatchManager` is the example), or write a `Tensor`
  wrapper. Wrapping `split_batch_replicated()`:

  ```python
  from max.nn.data_parallelism import split_batch_replicated


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
