# Live Router Refresh Gate

Run tag: `20260628-live-router-refresh-gate`

## Question

Can the known-positive live repeated slice be merged into the active routing
dataset and produce a refresh candidate that is safe to promote?

## Change

- Added a strongest-worker pass@1 promotion check to
  `tools/policy_refresh_gate.py`.
- Threaded `--min-loo-pass-at-1-vs-strongest` through the temperature selector.
- Merged the active 23-task dataset with the 7-task known-positive live repeated
  validation slice into
  `research/datasets/20260628-live-augmented-30task-routing.jsonl`.
- Swept reward-tempered logits routers over temperatures 0.05, 0.1, 0.2, and
  0.5 with the `preserve_accuracy` profile.

## Result

Keep the gate and artifacts, but do not promote the refreshed router.

The frozen probe-gated policy is still the best operational result on the
30-task merged report:

- active logits router: pass@1 0.7667, target accuracy 0.7333, latency regret
  793.5 ms
- frozen probe-gated policy: pass@1 0.7667, target accuracy 0.8333, latency
  regret 251.4 ms
- strongest/fastest single-worker baseline: pass@1 0.6667, target accuracy
  0.6667, latency regret 576.6 ms

All fresh retrained candidates passed the new strongest-worker pass@1 check, but
all were quarantined because they failed to preserve target accuracy and latency
regret:

- temperature 0.05: LOO target accuracy 0.7000, pass@1 0.7333, regret 687.7 ms
- temperature 0.10: LOO target accuracy 0.7333, pass@1 0.8000, regret 1024.8 ms
- temperature 0.20: LOO target accuracy 0.7000, pass@1 0.8000, regret 1544.4 ms
- temperature 0.50: LOO target accuracy 0.6333, pass@1 0.7667, regret 1619.7 ms

The best pass@1 candidates beat strongest-worker pass@1 0.6667, so the new
guard did not block useful solvability. The blockers were latency-regret and
target-accuracy regression.

## Decision

Keep:

- strongest-worker pass@1 refresh gate
- 30-task live-augmented dataset
- probe-gated baseline report
- quarantined temperature-selection report

Discard as active policy:

- raw live-augmented logits-router retrain

## Next

Train the next candidate with a pass-first objective or conditional/probe-gated
decision structure instead of relying on a raw reward-tempered logits refresh.
The frozen probe-gated policy should be the comparison target for the next
refresh, not only the prior active logits router.
