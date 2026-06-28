# Outcome Record Schema

This is the flat per-task, per-worker record emitted by real worker evaluation.

Required fields:

- `benchmark_id`
- `run_id`
- `timestamp`
- `task_id`
- `task_family`
- `prompt`
- `worker_id`
- `model`
- `workflow_kind`
- `passed`
- `score`
- `failure_mode`
- `latency_ms`
- `cost_usd`
- `reward`
- `evaluator_python`
- `evaluator_python_version`
- `evaluator_required_packages`

These records are not yet training examples. M2 converts them into grouped task
records with worker reward distributions and soft routing targets.

When converting dependency-profile benchmark outcomes into routing datasets,
audit the outcome rows first:

```bash
PYTHONPATH=src python3 tools/audit_outcome_rows.py \
  --input <outcomes.jsonl> \
  --required-evaluator-package numpy \
  --required-evaluator-package pandas \
  --min-workers-per-task 3 \
  --min-samples-per-worker-task 2
```

Then pass the expected evaluator packages to the converter:

```bash
PYTHONPATH=src python3 tools/build_repeated_routing_dataset.py \
  --input <outcomes.jsonl> \
  --output <routing.jsonl> \
  --required-evaluator-package numpy \
  --required-evaluator-package pandas
```

Rows without matching `evaluator_required_packages` are skipped so dependency
environment failures cannot become model-performance labels.
