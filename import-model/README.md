# Import a model into MAX

The `import-model` skill brings a new large language model architecture to MAX
from a Hugging Face model ID. It drives a three-phase workflow—decide and plan,
implement, then verify—while you remain the coordinator and validator at each
checkpoint.

This document describes the procedure in detail. The companion
[`SKILL.md`](SKILL.md) is the terse, mechanical version the agent executes; read
this README when you want the descriptive walkthrough.

## Phase 1: Decide and plan

During the initial phase, the agent gathers information about the target Hugging
Face model and proposes a transition plan:

1. **Inspection**: The agent fetches the model's configuration (`config.json`)
   and reads its parameters, checking whether the architecture class is already
   natively registered in MAX.
2. **Donor selection**: The agent identifies the closest existing MAX
   architecture (the donor template, such as `llama3` or `qwen3`) to use as a
   starting point.
3. **Delta analysis**: The agent compares the Hugging Face modeling code against
   the donor architecture to identify structural differences (deltas) in
   attention mechanisms, norm layers, or activation functions.

The agent presents a written plan listing the chosen donor architecture and the
catalog of structural deltas. Review the plan before authorizing the agent to
write code. Verify that the agent chose the correct donor and identified all
unique layer properties described in the model's paper or Hugging Face model
card.

## Phase 2: Implement

Once you approve the plan, the agent scaffolds the files and writes the Python
implementation:

1. **Scaffolding**: The agent copies the file layout of the donor architecture
   into a new folder named after your model.
2. **Config mapping**: The agent maps Hugging Face config keys to the typed
   configuration classes used by the MAX compile graph.
3. **Graph definition**: The agent modifies the neural network classes
   (`nn.Module`) to implement each architecture difference identified in the
   delta list.
4. **Weight translation**: The agent writes weight adapters that translate
   weight names from the Hugging Face checkpoint to the slots expected by the
   MAX graph.

If the model type is not yet registered in the Hugging Face library, the agent
may need to write a custom config parser. Ensure the agent updates all copied
class docstrings and code comments so they describe your new model rather than
retaining stale references to the donor.

## Phase 3: Verify and validate

After the implementation is complete, the agent runs validation scripts to
confirm correct behavior:

1. **Static analysis**: The agent runs linters (`br lint`) and type checkers
   (`mypy`) to ensure the new code meets MAX codebase standards.
2. **Local serving**: The agent launches the model using `max serve` to check
   that the graph compiles and loads checkpoint weights without orphan keys.
3. **Correctness check**: The agent runs greedy token generation on test prompts
   and compares the output text and token logits against the reference Hugging
   Face model.

Review the generated outputs and verification reports. Because the skill is
continuously improving, it doesn't guarantee model correctness out of the box.
If you observe gibberish or incoherent text in the output, steer the agent to
perform a layer-by-layer divergence hunt: instruct it to inspect intermediate
layer outputs and weights, comparing them against the Hugging Face reference
model until the exact point of divergence is isolated and resolved.
