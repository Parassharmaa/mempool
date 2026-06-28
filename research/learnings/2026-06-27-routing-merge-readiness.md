# Routing Merge Readiness

Added `tools/audit_routing_merge_readiness.py` to decide whether a structurally
valid routing dataset should be merged into router-refresh training data.

The audit checks:

- all-fail fastest-failure tasks
- solvable tasks whose selected target failed
- unstable target workers below the minimum pass-rate threshold
- routing dataset validation errors

The provenance-backed catalog-candidate routing dataset is valid JSONL, but the
merge audit correctly rejects it:

- `BigCodeBench/526` is all-fail among the new candidates.
- `BigCodeBench/763` selects `gpt-oss:120b` with only 0.5 pass rate.

Keep that dataset as calibration evidence, not active refresh data.
