# Import a model into MAX

The `import-model` skill brings a new large language model architecture to MAX
from a Hugging Face model ID, written in MAX's ModuleV3 API
(`max.nn`). It drives a phased workflow (decide and plan,
implement, then verify), and you stay the coordinator and validator at each
checkpoint.

This document describes the procedure in detail. The companion
[`SKILL.md`](SKILL.md) is the terse, mechanical version the agent executes.

## Phase 1: Decide and plan

During the initial phase, the agent gathers information about the target Hugging
Face model and proposes a port plan:

1. **Inspection**: The agent fetches the model's configuration (`config.json`)
   and reads its parameters, checking whether the architecture class is already
   natively registered in MAX.
2. **Donor selection**: The agent identifies the closest ModuleV3
   architecture in MAX (the donor template, such as `llama3_modulev3` or
   `olmo3`) to use as a starting point.
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

1. **Scaffolding**: The agent writes a skeleton that subclasses the donor
   architecture into a new folder named after your model.
2. **Config mapping**: The agent maps Hugging Face config keys to the typed
   configuration class the model is built from.
3. **Module definition**: The agent modifies the `Module` subclasses from
   `max.nn` to implement each architecture difference identified
   in the delta list.
4. **Weight translation**: The agent writes weight adapters that translate
   weight names from the Hugging Face checkpoint to the parameter names of the
   MAX module.

If the model type isn't yet registered in the Hugging Face `transformers`
library, the agent may need to write a custom config parser. Ensure the agent
updates all copied class docstrings and code comments so they describe your
new model, with no stale references to the donor.

## Phase 3: Verify and validate

After the implementation is complete, the agent runs validation scripts to
confirm correct behavior:

1. **Preflight checks**: The agent runs `run_oss_gates.py` and the local
   smoke checks (import smoke, lazy module build, adapter-to-parameter key
   diff) to catch config and registration errors before serving.
2. **Local serving**: The agent launches the model using `max serve` to check
   that the module compiles and loads checkpoint weights without missing or
   mis-shaped parameters.
3. **Correctness check**: The agent runs greedy token generation on test prompts
   and compares the output text and token logits against the reference Hugging
   Face model.

Review the generated outputs and verification reports. The skill is still
improving, so it doesn't guarantee a correct model on the first run. If the
output is gibberish or incoherent, steer the agent to a layer-by-layer
divergence hunt. Tell it to compare intermediate layer outputs and weights
against the Hugging Face reference model until it finds and fixes the first
point of divergence.
