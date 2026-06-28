# Router Miss Acquisition Plan

## Question

After the 26-task contrast-aware refresh stayed quarantined, which task regions
should the next live data acquisition target?

## Result

Added a reusable router-miss acquisition planner that reads a held-out router
report plus its routing dataset and groups leave-one-out mistakes by:

- target worker
- over-selected predicted worker
- prompt-feature category
- top library signature

Generated plan:

- `research/evals/20260628-router-miss-acquisition-plan.json`

The plan found 8 held-out misses across 26 tasks. The most useful next target
regions are sparse boundaries rather than shallow keyword gaps:

- Kimi target, Qwen over-selected: download/extract ZIP tasks
- GLM target, Qwen over-selected: filesystem ZIP tasks and random/math tasks
- DeepSeek target, Kimi or Qwen over-selected: filesystem/random and
  JSON/CSV/numpy tasks
- Qwen target, Kimi over-selected: random/statistics, pandas/date generation,
  and download/CSV counting tasks

## Decision

Keep the miss-neighborhood planner. It gives the next acquisition run a sharper
target than adding prompt keywords blindly.

The next live run should mine repeated examples near these neighborhoods and
only merge rows that improve leave-one-out routing without increasing latency
regret against the active 23-task policy.
