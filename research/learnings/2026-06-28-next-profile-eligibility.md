# Next Dependency Profile Eligibility

Run tag: `20260628-next-profile-eligibility`

Goal: expand the isolated BigCodeBench benchmark profile, refresh canonical
eligibility, and mine only newly unlocked rows for routing/fallback signal.

Profile change:

Installed the next planned packages into `.venv-bigcodebench` only:

- `nltk`
- `faker`
- `psutil`
- `flask`

No core project dependency was changed.

Eligibility result:

- Previous top-four profile eligible tasks: `69`
- Next profile eligible tasks over the same 148-row scan windows: `102`
- Newly unlocked eligible tasks: `34`

The realized unlock was larger than the conservative dependency-gap estimate of
10 tasks because some rows became canonical-pass once transitive or combined
test dependencies were available.

Fresh acquisition screen:

Selected six newly unlocked fallback-opportunity candidates using the active
23-task router's uncertainty and non-default second-worker pressure:

- `BigCodeBench/287`
- `BigCodeBench/313`
- `BigCodeBench/826`
- `BigCodeBench/17`
- `BigCodeBench/865`
- `BigCodeBench/655`

One-sample top-four worker screen:

- `BigCodeBench/826`: GLM, DeepSeek, and Qwen passed; Qwen was fastest.
- `BigCodeBench/865`: Kimi passed; GLM, DeepSeek, and Qwen failed.
- `BigCodeBench/17`, `287`, `313`, and `655`: universal failures in this sample.

Decision:

Keep the profile expansion and candidate-selection path. Do not promote a router
from this screen alone because it is single-sample and mostly Qwen-heavy.
Repeat-compare the positive tasks next, especially `BigCodeBench/865`, because
it is a fresh Kimi-specialist candidate from the expanded dependency profile.

Artifacts:

- `research/evals/bigcodebench_dependency_profile_next.txt`
- `research/evals/bigcodebench_hard_next_profile_eligible_merged_tasks.json`
- `research/evals/bigcodebench_hard_next_profile_eligible_merged_report.json`
- `research/evals/bigcodebench_hard_next_profile_newly_unlocked_tasks.json`
- `research/evals/bigcodebench_hard_next_profile_newly_unlocked_report.json`
- `research/evals/20260628-next-profile-fallback-opportunity-tasks.json`
- `research/evals/20260628-next-profile-fallback-opportunity-report.json`
- `research/evals/results/20260628-next-profile-fallback-opportunity-screen1.jsonl`
- `research/evals/results/20260628-next-profile-fallback-opportunity-screen1-summary.json`
- `research/evals/20260628-next-profile-fallback-opportunity-positive-tasks.json`
- `research/datasets/20260628-next-profile-fallback-opportunity-screen1-routing.jsonl`
