# Stable Outcome Source Mining

Added a stable outcome-source miner:

- `src/mempool/outcome_mining.py`
- `tools/mine_stable_outcome_sources.py`
- `tests/test_outcome_mining.py`
- ranking: `research/evals/20260628-stable-outcome-source-ranking.json`

The miner ranks repeated outcome JSONL files by merge-ready routing signal and
separates two useful row types:

- `exclusive_stable_nonqwen_targets`: non-Qwen target rows that are not
  broad-pass rows. These are true specialist/fallback signals.
- `broad_pass_latency_rows`: every worker passed, so the row is mainly a
  latency/cost preference signal.

Top mined sources:

- `research/evals/results/20260627-acquisition-to-50-wave1.jsonl`: 9
  merge-ready rows, 4 exclusive non-Qwen targets, 3 broad-pass latency rows.
  This source was already folded into the 32-task candidate and quarantined:
  target accuracy fell from 0.7826 to 0.625 and latency regret rose from
  501.1 ms to 999.9 ms.
- `research/evals/results/20260628-nonqwen-specialist-rank1-screen.jsonl`: 5
  merge-ready rows, 3 exclusive non-Qwen targets, 2 broad-pass rows. Later top4
  repeat evidence weakened the initial specialist interpretation.
- `research/evals/results/20260628-normal-offset16-contrast-outcomes.jsonl`: 6
  merge-ready rows, 0 exclusive non-Qwen targets, 6 broad-pass latency rows.

Offline candidate test:

- merged active 23-task dataset with the 6 clean broad-pass rows:
  `research/datasets/20260628-normal-offset16-contrast-29task-routing.jsonl`
- selection:
  `research/datasets/20260628-normal-offset16-contrast-29task-profiled-temperature-selection.json`
- profile: `preserve_accuracy`
- decision: `quarantine`

Best candidate result at reward temperature 0.05:

- target accuracy: 0.5517 vs active baseline 0.7826
- pass@1: 0.6897 vs active baseline 0.6957
- solvable pass@1: 0.7692 vs active baseline 0.8
- mean latency regret: 783.2 ms vs active baseline 501.1 ms

Learning:

The existing stable pool is not empty, but the current linear-softmax router
cannot safely absorb it. Even all-pass latency rows distorted leave-one-out
routing enough to miss the preserve-accuracy gate. The next useful step is not
another blind merge; it should be either:

1. a capacity/feature experiment that lets the router condition on task
   structure without overreacting to small batches, or
2. fresh acquisition designed to balance each new specialist/latency row with
   nearby Qwen-safe control rows.

Keep the active 23-task policy unchanged.
