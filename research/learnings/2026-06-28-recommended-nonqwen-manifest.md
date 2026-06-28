# Recommended Non-Qwen Manifest

Run tag: `20260628-recommended-nonqwen-manifest`

## Question

Can the acquisition-source recommendation be converted into a concrete,
auditable worker-run manifest before spending live worker calls?

## Change

Added a manifest materialization path:

- `src/mempool/acquisition_source.py`
- `tools/materialize_recommended_acquisition.py`
- `tests/test_acquisition_source.py`

The manifest builder verifies that selected tasks match the recommended source
report, copies the selected task file, preserves source task metadata, records
the worker pool, and emits the exact run and summary commands.

## Materialized Batch

Generated:

- `research/evals/20260628-nonqwen-pressure-top3-screen-tasks.json`
- `research/evals/20260628-nonqwen-pressure-top3-screen-manifest.json`

The batch contains:

- selected tasks: `6`
- workers: `ollama-cloud-glm-5.2`, `ollama-cloud-deepseek-v4-pro`,
  `ollama-cloud-kimi-k2.7-code`
- repeat count: `1`
- planned calls: `18`
- run id: `20260628-nonqwen-pressure-top3-screen`

The selected tasks all have `environment_risk: 0` and standard-library
dependencies. This is a screening batch for non-Qwen specialist evidence, not a
full repeated comparison or active-policy training dataset.

## Evaluation

Focused test:

- `PYTHONPATH=src python3 -m unittest tests.test_acquisition_source`
- result: `3` tests passed

Full suite:

- `PYTHONPATH=src python3 -m unittest discover -s tests`
- result: `223` tests passed

Research-loop gate:

- `PYTHONPATH=src python3 tools/research_loop.py evaluate --tag 20260628-recommended-nonqwen-manifest`
- result: `pass`
- score: `1.0`

## Decision

Keep the manifest path. The next live step, when allowed, is to run the
manifest command and then summarize, audit, and convert only positive/stable
rows before any router refresh.
