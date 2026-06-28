# Probe-Gated Held-Out Validation

Evaluated the frozen probe-gated latency-calibrated policy on measured routing
slices that were disjoint from the 37-task source dataset used to choose the
policy artifact.

Frozen policy:

- `research/policies/20260628-probe-gated-latency-calibrated-policy.json`
- base model:
  `research/models/20260627-mixed-winner-23task-heldout-hard-reward-t0p05-logits-router.json`
- probe gate:
  DeepSeek + Qwen, `all`, min pass rate `1.0`
- calibration:
  latency cost `0.5` per second, probability thresholds `0.0`

Held-out datasets:

| dataset | rows | overlap with policy-source rows |
| --- | ---: | ---: |
| `20260628-normal-offset76-guarded-heldout31-routing.jsonl` | 31 | 0 |
| `20260628-normal-offset236-boundary-heldout38-routing.jsonl` | 38 | 0 |
| `20260628-normal-offset16-contrast-heldout27-routing.jsonl` | 27 | 0 |

Reports:

- `research/evals/20260628-probe-gated-heldout-offset76-report.json`
- `research/evals/20260628-probe-gated-heldout-offset236-report.json`
- `research/evals/20260628-probe-gated-heldout-offset16-report.json`
- summary:
  `research/evals/20260628-probe-gated-heldout-summary.json`

Results:

| slice | policy | pass@1 | target accuracy | mean latency | latency regret |
| --- | --- | ---: | ---: | ---: | ---: |
| offset76 held-out 31 | active logits | 0.9032 | 0.5161 | 6960.2 ms | 1239.6 ms |
| offset76 held-out 31 | probe-gated | 0.9032 | 0.8065 | 5244.8 ms | 208.4 ms |
| offset236 held-out 38 | active logits | 0.8947 | 0.5526 | 6391.2 ms | 526.0 ms |
| offset236 held-out 38 | probe-gated | 0.8947 | 0.8158 | 4759.4 ms | 170.0 ms |
| offset16 held-out 27 | active logits | 0.8889 | 0.5556 | 6928.2 ms | 1017.1 ms |
| offset16 held-out 27 | probe-gated | 0.8889 | 0.8148 | 5210.2 ms | 84.7 ms |

Learning:

The frozen probe-gated policy generalized across three disjoint measured
slices. It preserved active-router pass@1 on every slice, improved target
accuracy by roughly 0.26-0.29, and cut latency regret sharply.

This is stronger evidence than the original 37-task in-slice result. It is
still not a brand-new cloud run in this turn; it reuses already-measured
outcomes that were not part of the policy-source slice. Treat it as held-out
replay validation, not as a fresh external benchmark claim.

Decision:

Keep the frozen policy and promote it as the current conditional-verifier
candidate for baseline comparisons.

Next step:

Run one small live cloud validation batch with this policy frozen, preferably
4-8 tasks selected from a task region not represented in the held-out replay
slices.
