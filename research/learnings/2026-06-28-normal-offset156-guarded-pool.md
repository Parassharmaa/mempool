# Normal Offset156 Guarded Pool

Run tag: `20260628-normal-offset156-guarded-pool`

## Question

Can another guarded normal BigCodeBench scan add useful contrast to the calibrated multi-head candidate and move it closer to the active-policy gate?

## Scan Result

The normal `bigcode/bigcodebench` scan from offset `156` searched `80` rows and found `10` locally eligible instruct-mode tasks. The next scan offset is `236`.

Eligible tasks:

- `bigcodebench-hard-BigCodeBench-158`
- `bigcodebench-hard-BigCodeBench-172`
- `bigcodebench-hard-BigCodeBench-176`
- `bigcodebench-hard-BigCodeBench-178`
- `bigcodebench-hard-BigCodeBench-192`
- `bigcodebench-hard-BigCodeBench-202`
- `bigcodebench-hard-BigCodeBench-203`
- `bigcodebench-hard-BigCodeBench-205`
- `bigcodebench-hard-BigCodeBench-206`
- `bigcodebench-hard-BigCodeBench-229`

Most rejected rows were blocked by missing local dependency packages such as `pandas`, `numpy`, `matplotlib`, `requests`, `django`, `sklearn`, `scipy`, `nltk`, `PIL`, `geopy`, and `_tkinter`. The first scan attempt hit a transient dataset API `502`, and the retry succeeded.

## Acquisition Result

The selector combined offset `0`, offset `16`, offset `76`, and offset `156` task pools, excluded known routing/outcome rows, used both recent contrast ledgers, and enforced `--max-uniform-similarity 6.0`.

It found `12` candidates and selected six tasks:

- `bigcodebench-hard-BigCodeBench-203`
- `bigcodebench-hard-BigCodeBench-192`
- `bigcodebench-hard-BigCodeBench-158`
- `bigcodebench-hard-BigCodeBench-118`
- `bigcodebench-hard-BigCodeBench-98`
- `bigcodebench-hard-BigCodeBench-176`

## Real-Worker Outcome

The top-four worker pool produced `24` complete outcome rows across `6` tasks. The audit marked the ledger ready for conversion.

Worker pass counts:

- DeepSeek: `2/6`, mean latency `5287.5 ms`
- GLM: `3/6`, mean latency `15268.0 ms`
- Kimi: `2/6`, mean latency `10966.7 ms`
- Qwen: `3/6`, mean latency `4406.7 ms`

Task pass counts:

- `bigcodebench-hard-BigCodeBench-118`: `4/4`
- `bigcodebench-hard-BigCodeBench-158`: `0/4`
- `bigcodebench-hard-BigCodeBench-176`: `1/4`
- `bigcodebench-hard-BigCodeBench-192`: `1/4`
- `bigcodebench-hard-BigCodeBench-203`: `0/4`
- `bigcodebench-hard-BigCodeBench-98`: `4/4`

The batch is useful: it contains two all-pass rows, two all-fail diagnostics, and two single-worker specialist rows.

## Merge Result

The routing conversion produced `6` rows. The merge-readiness filter kept `4` rows and dropped the two all-fail tasks:

- Dropped `bigcodebench-hard-BigCodeBench-158`
- Dropped `bigcodebench-hard-BigCodeBench-203`

The kept rows were merged into the prior 54-task experimental routing dataset, yielding a 58-record routing dataset and substrate.

Substrate target mix:

- DeepSeek: `10`
- GLM: `5`
- Kimi: `9`
- Qwen: `34`

Workflow mix:

- direct: `55`
- verify_then_fallback: `3`

## Raw 58-Task Candidate

The raw latency-regret `0.5` multi-head candidate still failed the 50-task baseline gate on latency regret:

- Candidate LOO target accuracy: `0.603`
- Candidate LOO pass@1: `0.810`
- Candidate LOO solvable pass@1: `0.855`
- Candidate LOO latency regret: `2565.5 ms`

Gate decision: `quarantine`.

Gate reasons:

- latency regret increase `872.0 ms` exceeded the `250.0 ms` limit
- latency regret `2565.5 ms` exceeded the `1800.0 ms` maximum

## Calibrated 58-Task Candidate

The transparent latency-calibrated wrapper again corrected the raw latency-regret failure.

Selected calibration:

- `latency_cost_per_second`: `0.02`
- `min_probability_ratio`: `0.0`
- `min_probability`: `0.0`

Against raw LOO predictions:

- Raw LOO target accuracy: `0.603`
- Calibrated LOO target accuracy: `0.672`
- Raw LOO pass@1: `0.810`
- Calibrated LOO pass@1: `0.810`
- Raw LOO solvable pass@1: `0.855`
- Calibrated LOO solvable pass@1: `0.855`
- Raw LOO latency regret: `2565.5 ms`
- Calibrated LOO latency regret: `500.0 ms`
- Choices changed from the top raw prediction: `5/58`

The calibrated 58-task candidate promoted against the 50-task multi-head baseline:

- Gate: `research/evals/results/20260628-latency-calibrated-router-offset156-w0p5-policy-gate.json`
- Decision: `promote`

But it still quarantined against the actual active-policy baseline:

- Gate: `research/evals/results/20260628-latency-calibrated-router-offset156-vs-active-policy-gate.json`
- Decision: `quarantine`
- Reason: LOO target accuracy drop `0.110` exceeded the allowed `0.080`

## Runtime Check

A simulated active registry for the calibrated 58-task artifact evaluates successfully through the active-policy runtime:

- Registry: `research/policies/20260628-latency-calibrated-router-offset156-simulated-active.json`
- Output: `research/evals/results/20260628-latency-calibrated-router-offset156-simulated-active-eval.json`
- Runtime target accuracy: `0.741`
- Runtime pass@1: `0.879`
- Runtime solvable pass@1: `0.927`
- Runtime latency regret: `455.5 ms`

These are in-sample runtime metrics and should not override the LOO promotion gate.

## Interpretation

Offset156 improved the larger calibrated multi-head lane: it now has `58` tasks, stronger target-worker diversity, excellent calibrated latency regret, and more specialist rows. It is still not ready to replace the active 23-task logits router because its held-out target accuracy remains too far below the active baseline.

The remaining active-promotion gap is no longer latency or runtime support. It is target accuracy on held-out routing decisions.

## Next Step

Continue guarded acquisition, but bias the next selector toward rows that sharpen target-worker boundaries rather than broad all-pass/all-fail rows. The most valuable next rows are likely GLM/DeepSeek/Kimi specialist wins where Qwen is plausible but not best.
