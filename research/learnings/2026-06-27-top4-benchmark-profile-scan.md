# Top-Four BigCodeBench Profile Scan

## Question

Does an isolated benchmark-only environment with the four largest missing
BigCodeBench dependencies materially expand the local canonical-pass task pool?

## Result

Yes. Installing `pandas`, `numpy`, `matplotlib`, and `requests` in
`.venv-bigcodebench` increased canonical-pass eligibility from 29 rows to 69
rows over the same 148 scanned BigCodeBench-Hard rows.

Artifacts:

- `research/evals/bigcodebench_dependency_profile_top4.txt`
- `research/evals/bigcodebench_hard_top4_offset44_full_report.json`
- `research/evals/bigcodebench_hard_top4_eligible_merged_tasks.json`
- `research/evals/bigcodebench_hard_top4_eligible_merged_report.json`
- `research/evals/bigcodebench_hard_top4_dependency_gap_report.json`

## Remaining Gap

The top-four profile leaves 72 unique dependency-blocked tasks and 7
non-dependency failures. The next package blockers are:

- `scikit-learn`: 12 tasks
- `scipy`: 11 tasks
- `seaborn`: 8 tasks
- `bs4`: 6 tasks

## Decision

Keep benchmark dependencies isolated from the core project. Use the expanded
69-task top-four eligible pool for the next repeated worker-comparison pass
before installing a second dependency profile or starting the Terminal-Bench
pilot.
