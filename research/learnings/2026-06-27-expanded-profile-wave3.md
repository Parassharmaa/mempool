# Expanded-Profile Wave 3

Run tag: `20260627-expanded-profile-wave3`

## What Changed

- Scanned the next expanded BigCodeBench-Hard window at offset 60 using the
  evaluator dependency profile with `numpy`, `pandas`, `matplotlib`, `requests`,
  `sklearn`, `scipy`, `seaborn`, and `bs4`.
- The scan found 18 canonical-pass tasks in 30 inspected rows.
- Fresh-task selection produced 6 tasks after excluding the active 40-task
  repeated routing candidate and all prior outcome ledgers.
- Ran 48 real worker outcomes: 6 tasks, 4 Ollama Cloud workers, 2 samples per
  worker/task.
- Converted the outcomes into 6 repeated routing records, then filtered out 1
  unstable target row.

## Result

The merge-ready wave3 subset kept 5 records:

- `BigCodeBench/502`
- `BigCodeBench/530`
- `BigCodeBench/532`
- `BigCodeBench/574`
- `BigCodeBench/618`

All five kept records selected `ollama-cloud-qwen3-coder-480b` as the stable
reward target. The dropped row, `BigCodeBench/582`, was solvable but had an
unstable target worker at the 2-sample threshold.

Merging those rows with the 40-task expanded-profile candidate produced:

- Dataset: `research/datasets/20260627-expanded-profile-wave3-45task-routing.jsonl`
- Model: `research/models/20260627-expanded-profile-wave3-45task-reward-t0p05-logits-router.json`
- Gate report: `research/evals/results/20260627-expanded-profile-wave3-45task-policy-gate.json`

The 45-task router was quarantined. It improved solvable pass-at-1 over the
23-task baseline, but missed the promotion gates:

- LOO target accuracy: `0.644`
- LOO solvable pass-at-1: `0.857`
- LOO mean latency regret: `1745.7 ms`

## Interpretation

Wave3 is useful data acquisition, not a policy improvement. It moved the stable
candidate dataset from 40 to 45 rows, leaving 5 more stable rows before the M5
50-row data-volume gate. The new rows are also highly skewed toward Qwen as the
latency winner, so they help volume but do not yet teach the router enough about
specialist routing boundaries.

The active 23-task policy remains the safest promoted policy. The 45-task
candidate should be treated as a training candidate and evidence ledger only.

## Next Step

Continue expanded-profile acquisition until the stable dataset reaches at least
50 rows, but bias selection toward tasks likely to produce non-Qwen specialist
labels or clear latency tradeoffs. If the next offset window repeats the same
Qwen-heavy pattern, add another narrow dependency/domain slice rather than only
adding more broad-pass Qwen rows.
