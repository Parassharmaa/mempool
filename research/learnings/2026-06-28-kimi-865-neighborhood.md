# Kimi 865 Neighborhood And Next2 Profile

Run tag: `20260628-kimi-865-neighborhood`

Goal: look for a stronger non-Qwen specialist signal near the partial Kimi row
`BigCodeBench/865`.

865-neighborhood result:

After excluding all existing routing datasets and outcome files, there were no
unscreened candidates left in the newly unlocked profile around `865`. The
partial Kimi signal is real enough to keep as diagnostic evidence, but its
nearby source pool is exhausted.

Dependency-profile follow-up:

Regenerated dependency-gap analysis after the next profile and fixed package
aliasing for missing modules:

- `cgi` -> `legacy-cgi`
- `flask_login` -> `flask-login`
- `flask_mail` -> `flask-mail`
- `Levenshtein` -> `python-Levenshtein`
- `docx` -> `python-docx`

Installed the next2 isolated benchmark profile additions into
`.venv-bigcodebench`:

- `openpyxl`
- `pytz`
- `legacy-cgi`
- `rsa`

Eligibility result:

- Previous next-profile eligible tasks: `102`
- Next2 profile eligible tasks: `108`
- Newly unlocked next2 tasks: `9`
- Previous rows not present in the next2 scan merge: `3`

The missing previous rows are tracked in
`research/evals/bigcodebench_hard_next2_profile_newly_unlocked_report.json` and
should be treated as scan-window variance, not evidence that installing packages
made those rows invalid.

Fresh next2 screen:

Selected four newly unlocked fallback-opportunity candidates:

- `BigCodeBench/199`
- `BigCodeBench/409`
- `BigCodeBench/360`
- `BigCodeBench/274`

One-sample top-four screen:

- `BigCodeBench/199`: GLM, Kimi, and Qwen passed; Qwen was fastest.
- `BigCodeBench/274`, `360`, and `409`: universal failures in this sample.

Merge audit:

- Strict merge readiness blocks the diagnostic dataset because it contains
  all-fail fastest-failure rows.
- Allowing all-fail rows passes, but this should not be used for the promoted
  refresh dataset unless we intentionally train fastest-failure behavior.

Decision:

Keep the profile expansion, package-alias fix, and diagnostic outcomes. Do not
promote a router refresh from this screen. The expanded dependency path is still
useful for source growth, but the next routing-data target should remain stable
non-Qwen specialist wins, not more all-fail or broad-pass Qwen-latency rows.

Artifacts:

- `tools/analyze_bigcodebench_dependency_gaps.py`
- `tests/test_analyze_bigcodebench_dependency_gaps.py`
- `research/evals/bigcodebench_hard_next_profile_dependency_gap_report.json`
- `research/evals/bigcodebench_hard_next2_dependency_profile_plan.json`
- `research/evals/bigcodebench_dependency_profile_next2.txt`
- `research/evals/bigcodebench_hard_next2_profile_eligible_merged_tasks.json`
- `research/evals/bigcodebench_hard_next2_profile_eligible_merged_report.json`
- `research/evals/bigcodebench_hard_next2_profile_newly_unlocked_tasks.json`
- `research/evals/bigcodebench_hard_next2_profile_newly_unlocked_report.json`
- `research/evals/20260628-next2-profile-fallback-opportunity-tasks.json`
- `research/evals/20260628-next2-profile-fallback-opportunity-report.json`
- `research/evals/results/20260628-next2-profile-fallback-opportunity-screen1.jsonl`
- `research/evals/results/20260628-next2-profile-fallback-opportunity-screen1-summary.json`
- `research/datasets/20260628-next2-profile-fallback-opportunity-screen1-routing.jsonl`
- `research/datasets/20260628-next2-profile-fallback-opportunity-screen1-merge-audit.json`
