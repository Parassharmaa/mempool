# Outcome Eligibility Audit

Added `tools/audit_outcome_rows.py` to report whether an outcome JSONL is ready
for routing-dataset conversion.

The audit checks:

- required outcome fields
- evaluator package provenance
- minimum workers per task
- minimum samples per worker/task

Running it on `20260627-catalog-candidate-regression-slices-top4env.jsonl`
with `numpy` and `pandas` requirements correctly reports `ready_for_conversion:
false`, because that run happened before evaluator package provenance was added
to outcome rows. This preserves the result as a historical report while
preventing it from becoming training data under the new guard.
