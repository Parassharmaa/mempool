# Measured Data Strategy

Clarified why `mempool` collects its own routing dataset instead of training
from public benchmark scores alone.

Artifacts:

- `docs/measured_routing_data.md`
- `docs/data_feedback_strategy.md`
- `docs/benchmark_strategy.md`
- `research/datasets/routing_record_schema.md`

Key decision:

- Public benchmark scores are useful for model selection, but not sufficient for
  router training.
- BigCodeBench remains the primary training-data harness because it produces
  cheap single-step task-worker outcomes.
- Terminal-Bench remains held out because it measures multi-step agent and
  scaffold behavior.

The current active dataset has 23 stable routing tasks. Keep collecting repeated
BigCodeBench rows until the 50-task readiness gate is met before training the
small multi-head orchestrator.
