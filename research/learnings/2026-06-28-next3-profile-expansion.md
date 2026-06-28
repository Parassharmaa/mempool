# Next3 Dependency Profile Expansion

Run tag: `20260628-next3-profile-expansion`

## Question

Can a small isolated dependency-profile expansion unlock fresh
BigCodeBench-Hard tasks that add useful non-Qwen routing signal?

## Dependency Frontier

The next2 dependency-gap report was regenerated from the next2 canonical probe
reports:

- `research/evals/bigcodebench_hard_next2_profile_dependency_gap_report.json`

The next3 profile added four benchmark-only packages to
`.venv-bigcodebench`:

- `statsmodels`
- `opencv-python`
- `lxml`
- `wordcloud`

The resulting profile is recorded in:

- `research/evals/bigcodebench_dependency_profile_next3.txt`
- `research/evals/bigcodebench_hard_next3_dependency_profile_plan.json`

After rescanning the same offset windows, the merged next3 eligible pool had
118 canonical-pass tasks versus 108 in next2. The newly unlocked comparison
found 10 tasks and no previous next2 dropouts:

- `research/evals/bigcodebench_hard_next3_profile_eligible_merged_tasks.json`
- `research/evals/bigcodebench_hard_next3_profile_newly_unlocked_tasks.json`
- `research/evals/bigcodebench_hard_next3_profile_newly_unlocked_report.json`

## Acquisition Result

A non-Qwen specialist-first screen was run over fresh newly unlocked tasks after
excluding prior routing and outcome ledgers:

- tasks: `917`, `214`, `424`
- workers: GLM, DeepSeek, Kimi
- outcomes:
  `research/evals/results/20260628-next3-nonqwen-screen1.jsonl`
- summary:
  `research/evals/results/20260628-next3-nonqwen-screen1-summary.json`

Only `BigCodeBench/917` passed the first screen. GLM passed 1/1; DeepSeek and
Kimi failed 0/1. `214` and `424` were universal failures in this screen.

`917` was then repeated across the top-four pool with two samples per worker:

- outcomes:
  `research/evals/results/20260628-next3-positive-repeat.jsonl`
- summary:
  `research/evals/results/20260628-next3-positive-repeat-summary.json`
- routing row:
  `research/datasets/20260628-next3-positive-repeat-routing.jsonl`

Repeat result for `917`:

- Kimi: 2/2, mean latency 8446 ms, target worker
- GLM: 2/2, mean latency 16880 ms
- DeepSeek: 0/2
- Qwen: 0/2

The active 23-task router predicted Qwen on this row, so the active policy
missed both the target and pass@1:

- `research/evals/results/20260628-next3-active-policy-on-917.json`

## Policy Refresh

The new row was merged with the active 23-task dataset into:

- `research/datasets/20260628-mixed-winner-24task-next3-routing.jsonl`

The temperature sweep and promotion gate wrote:

- `research/datasets/20260628-mixed-winner-24task-next3-temperature-selection.json`
- `research/policies/20260628-mixed-winner-24task-next3-refresh.json`

Decision: quarantine. The best candidate was temperature `0.05`, but its
leave-one-out solvable pass@1 dropped to 0.714, below the 0.800 promotion floor.
The active 23-task policy remains active.

## Decision

Keep the data and the learning, but do not promote the 24-task router.

`917` is a valuable regression slice: Qwen fails, Kimi and GLM pass, and Kimi is
the latency-adjusted target. The policy miss shows the current router still
overroutes some newly unlocked scientific/statistical tasks to Qwen. More
nearby Kimi/GLM-positive rows are needed before this boundary can be safely
absorbed into the active policy.
