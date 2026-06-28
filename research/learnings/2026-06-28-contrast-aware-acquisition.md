# Contrast-Aware Acquisition

Run tag: `20260628-contrast-aware-acquisition`

## Question

Can the value-head acquisition selector use recent real-worker outcomes to avoid spending calls on likely all-pass or all-fail tasks?

## Change

Added optional contrast priors to `tools/select_value_head_acquisition_batch.py`.

The selector can now accept `--contrast-outcomes` JSONL files. It builds task-level priors from observed pass/fail rates:

- mixed worker outcomes add `contrast_similarity`;
- uniform all-pass or all-fail outcomes add `uniform_similarity`;
- candidate scores get a contrast bonus and a uniform penalty;
- reports expose `contrast_prior_count`, `contrast_similarity`, and `uniform_similarity`.

Also fixed an acquisition selector invariant: `limit=1` could previously return two tasks because the balancing pass reserved one task per target group before checking the global limit. Group balancing now respects the global limit and only reserves slots for positive-score group candidates.

## Dry Run

Command used the fresh acquisition outcomes as contrast priors and excluded all known outcome rows.

Result:

- `contrast_prior_count`: `6`
- `candidate_count`: `4`
- selected task ids:
  - `bigcodebench-hard-BigCodeBench-5`
  - `bigcodebench-hard-BigCodeBench-0`
  - `bigcodebench-hard-BigCodeBench-8`
  - `bigcodebench-hard-BigCodeBench-2`

The prior surfaced uniform-risk scores, but the offset-0 normal BigCodeBench pool had only four unseen candidates left after strict exclusions. This means the tool can now report contrast risk, but the next useful acquisition pass needs a larger fresh scan before spending more real-worker calls.

## Artifacts

- Updated selector: `tools/select_value_head_acquisition_batch.py`
- Updated tests: `tests/test_select_value_head_acquisition_batch.py`
- Dry-run task output: `research/evals/20260628-contrast-aware-acquisition-tasks.json`
- Dry-run report: `research/evals/20260628-contrast-aware-acquisition-report.json`

## Next Step

Scan the normal BigCodeBench split from offset `16` or later for a larger fresh candidate pool, then rerun the contrast-aware selector with `--contrast-outcomes research/evals/results/20260628-fresh-acquisition-outcomes.jsonl`.
