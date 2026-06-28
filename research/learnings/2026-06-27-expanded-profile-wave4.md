# Expanded-Profile Wave 4

Run tag: `20260627-expanded-profile-wave4`

## What Changed

Wave4 continued expanded-profile BigCodeBench-Hard acquisition until the stable
repeated routing dataset crossed the 50-row M5 data-volume threshold.

Three fresh acquisition slices were run:

- Offset 90: 5 fresh tasks, 40 real outcome rows.
- Offset 115: 3 fresh tasks, 24 real outcome rows.
- Offset 145: 2 fresh tasks, 16 real outcome rows.

All slices used the same top-four Ollama Cloud worker pool with two samples per
worker/task and the expanded evaluator dependency profile:

- `numpy`
- `pandas`
- `matplotlib`
- `requests`
- `sklearn`
- `scipy`
- `seaborn`
- `bs4`

## Result

The run added 5 merge-ready rows:

- `BigCodeBench/752`: DeepSeek target.
- `BigCodeBench/1013`: Kimi target.
- `BigCodeBench/1053`: DeepSeek target.
- `BigCodeBench/969`: Qwen target.
- `BigCodeBench/1124`: Qwen target.

The stable repeated-routing candidate now has 50 rows:

- Dataset: `research/datasets/20260627-expanded-profile-wave4-50task-routing.jsonl`
- Model: `research/models/20260627-expanded-profile-wave4-50task-reward-t0p05-logits-router.json`
- Training report: `research/datasets/20260627-expanded-profile-wave4-50task-reward-t0p05-logits-router-report.json`
- Gate report: `research/evals/results/20260627-expanded-profile-wave4-50task-policy-gate.json`

Target mix at 50 rows:

- Qwen: 30
- DeepSeek: 9
- Kimi: 8
- GLM: 3

## Gate Decision

The 50-task candidate is quarantined, not promoted.

Leave-one-out metrics:

- Target accuracy: `0.620`
- Solvable pass-at-1: `0.872`
- Mean latency regret: `3609.7 ms`

The candidate clears the solvable pass-at-1 floor, but it misses target-accuracy
and latency-regret gates by a wide margin. The active promoted 23-task policy
therefore remains the safer live policy.

The candidate-only M5 readiness audit was written to
`research/programs/small_orchestrator_readiness_wave4_50task_candidate.json`.
It confirms that data volume, worker coverage, the logits head, fallback signal,
and the multi-head action contract are present, but M5 readiness still fails on
target accuracy and latency regret. The active-policy readiness audit remains
separate at `research/programs/small_orchestrator_readiness_active_after_wave4.json`
and correctly reports that the promoted active policy is still the 23-task
policy.

## Interpretation

This is the first candidate dataset that satisfies the M5 data-volume threshold.
That matters even though the policy is quarantined: small-orchestrator training
now has enough stable repeated rows to start as an offline experiment, but not
enough routing quality to replace the active policy.

The main failure mode is no longer missing data volume. It is router
generalization: the logits router over-predicts broad Qwen wins and misses too
many DeepSeek/Kimi/GLM latency targets in leave-one-out.

## Next Step

Keep the active 23-task policy unchanged. Use the 50-task dataset as the first
M5 offline training substrate, then test a small-orchestrator or richer
candidate router against the same policy gate before any promotion. The next
modeling step should focus on reducing latency regret and improving specialist
target accuracy, not on adding more broad-pass Qwen rows.
