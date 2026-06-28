# Outcome-Aware Source Planner

Run tag: `20260628-outcome-aware-source-planner`

## Question

Can source selection incorporate live acquisition outcomes so the loop stops
favoring a source after its first screen produces mostly all-fail rows and no
stable non-Qwen specialist win?

## Change

The acquisition source planner now accepts outcome-summary evidence per source.
It adjusts source score using:

- universal-failure rate
- conversion-candidate count
- whether candidates are non-Qwen wins
- whether repeat evidence failed to produce stable non-Qwen wins

Updated files:

- `src/mempool/acquisition_source.py`
- `tools/plan_next_acquisition_source.py`
- `tests/test_acquisition_source.py`

## Evidence Applied

Attached these summaries to `nonqwen-pressure`:

- `research/evals/results/20260628-nonqwen-pressure-top3-screen-summary.json`
- `research/evals/results/20260628-nonqwen-pressure-positive-repeat-summary.json`

The top-three screen had `4/6` universal failures and two one-shot positives.
The repeat comparison then showed:

- `BigCodeBench-15`: repeat universal failure
- `BigCodeBench-260`: broad-pass Qwen latency target

## New Recommendation

Generated:

- `research/evals/20260628-next-acquisition-source-plan-after-live-screen.json`

The recommendation changed from `nonqwen-pressure` to `contrast-aware`.

Ranking:

1. `contrast-aware`, score `14.0`
2. `nonqwen-pressure`, score `11.3`
3. `fresh-54-quarantine`, score `9.0`
4. `normal-offset676`, exhausted

## Evaluation

Focused test:

- `PYTHONPATH=src python3 -m unittest tests.test_acquisition_source`
- result: `4` tests passed

Full suite:

- `PYTHONPATH=src python3 -m unittest discover -s tests`
- result: `224` tests passed

Research-loop gate:

- `PYTHONPATH=src python3 tools/research_loop.py evaluate --tag 20260628-outcome-aware-source-planner`
- result: `pass`
- score: `1.0`

## Decision

Keep the outcome-aware source planner. The next acquisition step should inspect
or materialize the `contrast-aware` source before spending more calls on the
weakened `nonqwen-pressure` source.
