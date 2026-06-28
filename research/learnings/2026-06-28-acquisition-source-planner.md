# Acquisition Source Planner

Run tag: `20260628-acquisition-source-planner`

## Question

After the normal BigCodeBench offset path returned zero rows at offset `676`,
can the research loop choose the next acquisition source from existing evidence
instead of blindly scanning exhausted offsets?

## Change

Added a small source-ranking helper and CLI:

- `src/mempool/acquisition_source.py`
- `tools/plan_next_acquisition_source.py`
- `tests/test_acquisition_source.py`

The planner scores candidate acquisition sources by:

- candidate count
- preselected task count
- exhausted-source detection
- specialist-pressure/source-widening value
- prior quarantine or pass evidence
- whether the step can be inspected without live worker spend

## Current Recommendation

Generated:

- `research/evals/20260628-next-acquisition-source-plan.json`

The ranked recommendation is:

1. `nonqwen-pressure`
2. `contrast-aware`
3. `fresh-54-quarantine`
4. `normal-offset676`

`normal-offset676` is explicitly marked exhausted. The top recommendation is
`nonqwen-pressure` because it has `33` candidates, `6` preselected tasks, and
is aligned with the current need for non-Qwen specialist evidence.

## Evaluation

Focused test:

- `PYTHONPATH=src python3 -m unittest tests.test_acquisition_source`
- result: `2` tests passed

Full suite:

- `PYTHONPATH=src python3 -m unittest discover -s tests`
- result: `222` tests passed

Research-loop gate:

- `PYTHONPATH=src python3 tools/research_loop.py evaluate --tag 20260628-acquisition-source-planner`
- result: `pass`
- score: `1.0`

## Decision

Keep the planner. The next paid or live-worker step should start from the
`nonqwen-pressure` source, not from another normal offset scan.
