# Tests and CI

When you add `pytest` tests for the ported model (layer tests, module
tests, or model smoke tests under `max/`), minimize the number of
`compile()` calls per test file. A file that recompiles for each
parameter combination, or that contains many independently compiled
modules, can time out in CI.

The patterns below prevent timeouts. Use them together.

## Pattern 1: Compile once via a module-scoped fixture

Compile the module one time in a fixture and reuse it across every test
in the file. Use symbolic dimensions in the input types, so one compiled
module accepts every shape, and combine with `@pytest.mark.parametrize` to
vary inputs without recompiling.

```python
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.tensor.tensor import TensorType

from my_port.layers import MyMLP


@pytest.fixture(scope="module")
def mlp():
    device = Accelerator()
    layer = MyMLP(hidden_size=64, intermediate_size=256)
    layer.to(device)
    return layer.compile(TensorType(DType.bfloat16, ["seq", 64], device))


@pytest.mark.parametrize("seq_len", [1, 7, 128])
def test_forward(seq_len, mlp): ...  # exercise the already-compiled module
```

pytest creates a module-scoped fixture once per test file, so every test
in the file shares the same compiled module.

## Pattern 2: Parallelize different modules with `shard_count`

When a single file must compile different modules (different dtypes,
kernel variants, or distribution shapes), don't split the file manually.
Use Bazel test sharding to spread the work across parallel CI workers:

```bzl
modular_py_test(
    name = "test_attention",
    srcs = ["test_attention.py"],
    shard_count = 3,
)
```

Bazel launches `N` parallel pytest processes. Each runs roughly `1/N`
of the tests. Module-scoped fixtures are per-process, so each shard
compiles only the modules its tests need, and the compiles happen in
parallel.

The `pytest-shard` plugin adds fine-grained markers:

- `@pytest.mark.shard_group("name")` pins every test in the group to
  the same shard so they share a compile.
- `@pytest.mark.unique_shard` gives an expensive test its own shard.

## When to apply

Reach for each pattern in different situations:

- Use fixtures in any test file that exercises a compiled module more
  than once.
- Use sharding in any test file whose total wall time approaches the CI
  timeout, especially when the file contains independently compiled
  modules that can't share a fixture.

Fixtures minimize compiles where modules are
shareable, and sharding parallelizes compiles where they aren't.
