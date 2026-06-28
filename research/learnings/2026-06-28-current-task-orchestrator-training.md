# Current Task-Level Orchestrator Training

Run context: final task-level training before public repository setup.

## Question

Given the measured data currently available, can we train a task-level
multi-head orchestrator checkpoint worth publishing as the current research
artifact?

## Data

Merged task-level routing dataset:

- `research/datasets/20260628-m5-current-task-66task-routing.jsonl`

Supervised substrate:

- `research/datasets/20260628-m5-current-task-66task-substrate.jsonl`
- 66 examples
- 57 direct workflow labels
- 9 verify-then-fallback labels

Target mix:

- Qwen: 41
- Kimi: 12
- DeepSeek: 9
- GLM: 4

## Model

Trained artifact:

- `research/models/20260628-m5-current-task-66task-multihead.json`

Report:

- `research/evals/20260628-m5-current-task-66task-multihead-report.json`

Training-set metrics:

- target accuracy: 0.6818
- pass@1: 0.8333
- solvable pass@1: 0.9649
- mean latency regret: 1575.7 ms
- workflow accuracy: 0.8939
- abstain accuracy: 0.8636

Leave-one-out metrics:

- target accuracy: 0.5606
- pass@1: 0.7576
- solvable pass@1: 0.8772
- mean latency regret: 2463.5 ms
- workflow accuracy: 0.8636
- abstain accuracy: 0.8636

## Decision

Keep the model as the current task-level orchestrator checkpoint and publish it
with the repo.

Do not promote it as the active policy. Leave-one-out target accuracy and
latency regret remain behind the operational bar set by the probe-gated policy.

## Next

Finish repository publication first. After that, continue task-level training
with more non-Qwen specialist targets before moving to turn-level agentic
training.
