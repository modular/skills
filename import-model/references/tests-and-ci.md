# Tests and CI

When you add `pytest` tests for the ported model — layer tests, graph
tests, or model smoke tests under `max/` — minimize the number of MAX
graph compilations per test file. A file that recompiles for each
parameter combination, or that contains many independent compiled
graphs, will time out in CI.

Two patterns prevent timeouts. Use them together.

## Pattern 1: Compile once via a module-scoped fixture

Compile the graph one time in a fixture and reuse it across every test
in the file. Combine with `@pytest.mark.parametrize` to vary inputs
without recompiling.

```python
import pytest
from max.graph import Graph

@pytest.fixture(scope="module")
def model(session):
    with Graph(...) as g:
        ...
    return session.compile(g)

@pytest.mark.parametrize("input_shape", [...])
def test_forward(input_shape, model):
    ...  # exercise the already-compiled model
```

A module-scoped fixture lives for the lifetime of the test file's
pytest process, so every test in the file shares the same compiled
graph.

## Pattern 2: Parallelize different graphs with `shard_count`

When a single file must compile different graphs — different dtypes,
kernel variants, or distribution shapes — don't split the file manually.
Use Bazel test sharding to spread the work across parallel CI workers:

```python
modular_py_test(
    name = "test_attention",
    srcs = ["test_attention.py"],
    shard_count = 3,
)
```

Bazel launches `N` parallel pytest processes; each runs roughly `1/N`
of the tests. Module-scoped fixtures are per-process, so each shard
compiles only the graphs its tests need, and the compiles happen in
parallel.

The `pytest-shard` plugin adds fine-grained markers:

- `@pytest.mark.shard_group("name")` pins every test in the group to
  the same shard so they share a compile.
- `@pytest.mark.unique_shard` gives an expensive test its own shard.

## When to apply

Reach for each pattern in different situations:

- Use fixtures in any test file that exercises a compiled MAX graph
  more than once.
- Use sharding in any test file whose total wall time approaches the CI
  timeout, especially when the file contains independent compiled
  graphs that can't share a fixture.

The two patterns compose: fixtures minimize compiles where graphs are
shareable; sharding parallelizes compiles where they aren't.
