# Library-Feature Router Refresh

## Result

The library-aware ten-task logits router was promoted to the active policy.

Artifacts:

- model: `research/models/20260627-mixed-winner-10task-library-logits-router.json`
- dataset: `research/datasets/20260627-mixed-winner-10task-routing.jsonl`
- report: `research/datasets/20260627-mixed-winner-10task-library-logits-router-report.json`
- refresh: `research/refreshes/20260627-mixed-winner-10task-library-refresh.json`
- active registry: `research/policies/active_policy.json`

## What Changed

Prompt features now include explicit library indicators, such as `lib_pathlib`,
`lib_tempfile`, and `lib_shutil`, plus additional low-level task keywords for
URL, encoding, hash, socket, and SSL work. This gave the linear logits router a
cleaner signal for broad-pass latency rows versus specialist-win rows.

The first ten-task refresh without these features was quarantined. The
library-aware refresh passed the accuracy and latency-regret gate:

- training target accuracy: 0.80
- training pass@1: 0.90
- training mean latency regret: 518.6 ms
- leave-one-out target accuracy: 0.70
- leave-one-out pass@1: 0.80
- leave-one-out mean latency regret: 919.5 ms

The gate promoted the candidate with warnings because leave-one-out target
accuracy dropped by 0.050 and leave-one-out latency regret increased by
418.4 ms compared with the previous active policy. Both were inside the current
guard bands.

## Held-Out Signal

The held-out broad-pass latency diagnostic on `BigCodeBench/906` and
`BigCodeBench/928` improved from a Kimi-biased route to Qwen on both tasks:

- target accuracy: 1.0
- pass@1: 1.0
- mean latency regret: 0.0 ms

This does not prove generalization. It does show that cheap feature enrichment
can repair a real latency-routing miss before moving to a larger backbone.

## Next Step

Do not jump to a neural or language-model orchestrator solely because this
refresh promoted. The next BigCodeBench step should collect more non-Qwen
specialist wins and held-out broad-pass latency rows, then train a
latency-regret-aware objective. Terminal-Bench should remain the next agentic
harness after this single-step router has stronger held-out evidence.
