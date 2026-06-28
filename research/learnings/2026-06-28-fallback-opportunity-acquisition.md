# Fallback Opportunity Acquisition

Run tag: `20260628-fallback-opportunity-acquisition`

Goal: find fresh BigCodeBench tasks likely to produce fallback or specialist
labels for the active 23-task router.

Result:

- The first selector command accidentally used GNU `find -printf`, which is not
  available in the local macOS shell. That left exclusions empty and selected
  already-known rows such as `BigCodeBench/368` and `BigCodeBench/526`.
- Rerunning with portable `find ... -print` plus `awk` excluded existing routing
  datasets and outcome logs correctly.
- With correct exclusions, the current top-four dependency-profile eligible
  pool has `0` fresh candidates. All 69 known eligible rows are already covered
  by routing data or outcome logs.

Decision:

Do not spend more worker calls on the current top-four eligible pool. The next
useful acquisition step is to expand the isolated benchmark dependency profile
and rerun canonical eligibility, then mine from the newly unlocked rows.

Change:

Added `tools/plan_dependency_profile_expansion.py`.

The planner reads a dependency-gap report, skips packages already present in the
benchmark-only profile, normalizes aliases such as `sklearn` to
`scikit-learn`, and writes:

- a JSON plan with projected task unlocks
- a next benchmark-only profile file

Current next-profile plan:

- Current profile: `pandas`, `numpy`, `matplotlib`, `requests`,
  `scikit-learn`, `scipy`, `seaborn`, `bs4`
- Add next: `nltk`, `faker`, `psutil`, `flask`
- Projected fresh dependency-blocked tasks unlocked from the existing scan: `10`

Artifacts:

- `tools/plan_dependency_profile_expansion.py`
- `tests/test_plan_dependency_profile_expansion.py`
- `research/evals/20260628-fallback-opportunity-acquisition-report.json`
- `research/evals/20260628-fallback-opportunity-acquisition-tasks.json`
- `research/evals/bigcodebench_hard_next_dependency_profile_plan.json`
- `research/evals/bigcodebench_dependency_profile_next.txt`
