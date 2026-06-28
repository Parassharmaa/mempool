# Qwen Head Held-Out Split

Run tag: `20260628-qwen-heldout-split`

## What Changed

Added a deterministic train/held-out splitter for Qwen logits rows and patched
the trainer to consume prepared row files. This lets Qwen-small head checkpoints
be evaluated on held-out rows instead of only on the rows used for training.

Artifacts:

- `tools/split_qwen_logits_rows.py`
- `tests/test_split_qwen_logits_rows.py`
- `research/datasets/20260628-qwen-small-logits-orchestrator-split-train.jsonl`
- `research/datasets/20260628-qwen-small-logits-orchestrator-split-heldout.jsonl`
- `research/datasets/20260628-qwen-small-logits-orchestrator-split-manifest.json`
- `research/models/20260628-qwen-small-logits-orchestrator-split-smoke/`

## Split

- source rows: 66
- train rows: 53
- held-out rows: 13
- seed: 7
- held-out fraction: 0.2

## One-Epoch Split Smoke

Training:

- rows: 53
- epochs: 1
- loss: 4.2856650959770635

Evaluation:

| Split | Worker Accuracy | Workflow Accuracy | Mean Worker Loss | Mean Workflow Loss |
| --- | ---: | ---: | ---: | ---: |
| train | 0.4717 | 0.5660 | 1.4387 | 0.7223 |
| held-out | 0.3077 | 0.7692 | 1.7572 | 0.5601 |

## Decision

Keep the split machinery and checkpoint as a baseline. The one-epoch Qwen head
is not competitive enough to promote, but the training/evaluation loop now has a
real held-out gate. Next work should run longer frozen-head training and compare
against the linear multi-head router before trying LoRA or backbone updates.
