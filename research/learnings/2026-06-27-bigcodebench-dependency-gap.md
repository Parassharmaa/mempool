# BigCodeBench Dependency Gap

## Result

The current locally eligible BigCodeBench-Hard pool is mostly blocked by missing
Python packages, not by canonical-solution failures.

Artifact:

- `research/evals/bigcodebench_hard_dependency_gap_report.json`

The dependency-gap analyzer scanned the existing canonical probe reports:

- `research/evals/bigcodebench_hard_eligible_report.json`
- `research/evals/bigcodebench_hard_eligible_offset44_report.json`
- `research/evals/bigcodebench_hard_eligible_offset99_report.json`
- `research/evals/bigcodebench_hard_eligible_offset125_report.json`

## Findings

Across 148 scanned rows:

- locally eligible rows: 29
- unique dependency-blocked rows: 117
- non-dependency failures: 2

Top missing packages:

| Package | Blocked tasks |
| --- | ---: |
| pandas | 42 |
| numpy | 23 |
| matplotlib | 15 |
| requests | 9 |

Installing only `pandas` would potentially unlock 42 rows. Installing
`pandas`, `numpy`, `matplotlib`, and `requests` would potentially unlock 89
unique rows before accounting for second-order dependencies or semantic test
failures.

## Recommendation

Create an isolated BigCodeBench dependency profile rather than adding these
packages to the core project dependency list. The core orchestrator should stay
small; benchmark environments can be heavier and versioned separately.

Suggested first profile:

```text
pandas
numpy
matplotlib
requests
```

After creating that environment, rerun the canonical scanner over the same
offset windows and compare:

- newly eligible task count
- dependency failures remaining
- canonical semantic failures
- whether the new eligible rows contain more specialist-worker target diversity

## Next Step

Add a benchmark-environment profile file and a scanner command that can be run
inside that environment. Do not run more cloud worker calls until the expanded
canonical-pass pool is available.
