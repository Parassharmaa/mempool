# Provenance Candidate Rerun

The catalog-candidate regression slice was rerun after adding evaluator
environment provenance and package preflight guards.

Audit result:

- `ready_for_conversion: true`
- 12 rows
- 2 tasks
- 3 workers per task
- 2 samples per worker/task
- 0 package mismatches

Outcome summary:

- `qwen3-coder-next`: 0/4, mean latency 3328 ms.
- `qwen3.5:397b`: 1/4, mean latency 43027 ms.
- `gpt-oss:120b`: 1/4, mean latency 5156 ms.

The guarded routing dataset is valid, but should not be merged into the active
router dataset. `BigCodeBench/526` has only fastest-failure targets among the
new candidates, and `BigCodeBench/763` shows intermittent rather than stable
passes. Keep this as candidate-screening evidence unless a future stability
gate explicitly consumes partial-pass rows.
