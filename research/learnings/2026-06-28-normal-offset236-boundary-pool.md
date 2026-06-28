# Normal Offset236 Boundary Pool

Run tag: `20260628-normal-offset236-boundary-pool`

## Question

Can another guarded normal BigCodeBench scan add target-boundary rows that improve the calibrated multi-head lane's held-out target accuracy without reopening the latency-regret problem?

## Scan Result

The normal `bigcode/bigcodebench` scan from offset `236` searched `80` rows and found `17` locally eligible instruct-mode tasks. The next scan offset is `316`.

Eligible tasks:

- `bigcodebench-hard-BigCodeBench-254`
- `bigcodebench-hard-BigCodeBench-260`
- `bigcodebench-hard-BigCodeBench-263`
- `bigcodebench-hard-BigCodeBench-265`
- `bigcodebench-hard-BigCodeBench-266`
- `bigcodebench-hard-BigCodeBench-268`
- `bigcodebench-hard-BigCodeBench-271`
- `bigcodebench-hard-BigCodeBench-277`
- `bigcodebench-hard-BigCodeBench-279`
- `bigcodebench-hard-BigCodeBench-281`
- `bigcodebench-hard-BigCodeBench-283`
- `bigcodebench-hard-BigCodeBench-288`
- `bigcodebench-hard-BigCodeBench-297`
- `bigcodebench-hard-BigCodeBench-305`
- `bigcodebench-hard-BigCodeBench-306`
- `bigcodebench-hard-BigCodeBench-310`
- `bigcodebench-hard-BigCodeBench-314`

The rejected rows were mostly blocked by local dependency gaps such as `pandas`, `numpy`, `matplotlib`, `pytz`, `faker`, `mechanize`, `tensorflow`, `nltk`, `sendgrid`, `cv2`, and Python 3.14's missing `cgi`.

## Acquisition Result

The guarded selector used the offset `0`, `16`, `76`, `156`, and `236` task pools, excluded known routing and outcome rows, and used prior contrast ledgers with `--max-uniform-similarity 6.0`.

It found `11` fresh candidates and selected six tasks:

- `bigcodebench-hard-BigCodeBench-314`
- `bigcodebench-hard-BigCodeBench-306`
- `bigcodebench-hard-BigCodeBench-205`
- `bigcodebench-hard-BigCodeBench-178`
- `bigcodebench-hard-BigCodeBench-202`
- `bigcodebench-hard-BigCodeBench-172`

The selected set mixed missed-positive and false-spend-value candidates across network, filesystem, subprocess, and general tasks.

## Real-Worker Outcome

The top-four worker pool produced `24` complete outcome rows across `6` tasks. The audit marked the ledger ready for conversion.

Worker pass counts:

- DeepSeek: `2/6`, mean latency `6035.7 ms`
- GLM: `3/6`, mean latency `6100.8 ms`
- Kimi: `2/6`, mean latency `11134.2 ms`
- Qwen: `3/6`, mean latency `2115.2 ms`

Task pass counts:

- `bigcodebench-hard-BigCodeBench-172`: `4/4`
- `bigcodebench-hard-BigCodeBench-178`: `2/4`
- `bigcodebench-hard-BigCodeBench-202`: `0/4`
- `bigcodebench-hard-BigCodeBench-205`: `4/4`
- `bigcodebench-hard-BigCodeBench-306`: `0/4`
- `bigcodebench-hard-BigCodeBench-314`: `0/4`

The batch produced one pass/fail boundary row: `BigCodeBench-178` passed on GLM and Qwen, failed on DeepSeek and Kimi, and still targeted Qwen by latency-adjusted reward. The rest was split between all-pass latency rows and all-fail diagnostics.

## Merge Result

The routing conversion produced `6` rows. The merge-readiness filter kept `3` rows and dropped the three all-fail tasks:

- Dropped `bigcodebench-hard-BigCodeBench-202`
- Dropped `bigcodebench-hard-BigCodeBench-306`
- Dropped `bigcodebench-hard-BigCodeBench-314`

The kept rows were merged into the prior 58-task experimental routing dataset, yielding a 61-record routing dataset and substrate.

Substrate target mix:

- DeepSeek: `10`
- GLM: `5`
- Kimi: `9`
- Qwen: `37`

Workflow mix:

- direct: `58`
- verify_then_fallback: `3`

## Raw 61-Task Candidate

The raw latency-regret `0.5` multi-head candidate regressed relative to the calibrated 58-task lane:

- LOO target accuracy: `0.623`
- LOO pass@1: `0.820`
- LOO solvable pass@1: `0.862`
- LOO latency regret: `2439.4 ms`

## Calibrated 61-Task Candidate

The transparent latency-calibrated wrapper improved the raw candidate but did not make it promotable.

Selected calibration:

- `latency_cost_per_second`: `0.01`
- `min_probability_ratio`: `0.0`
- `min_probability`: `0.0`

Against raw LOO predictions:

- Raw LOO target accuracy: `0.623`
- Calibrated LOO target accuracy: `0.639`
- Raw LOO pass@1: `0.820`
- Calibrated LOO pass@1: `0.820`
- Raw LOO solvable pass@1: `0.862`
- Calibrated LOO solvable pass@1: `0.862`
- Raw LOO latency regret: `2439.4 ms`
- Calibrated LOO latency regret: `1172.2 ms`
- Choices changed from the top raw prediction: `3/61`

The calibrated 61-task candidate quarantined against the prior 58-task calibrated lane:

- Gate: `research/evals/results/20260628-latency-calibrated-router-offset236-vs-58task-calibrated-gate.json`
- Decision: `quarantine`
- Reason: LOO latency regret increased by `672.2 ms`, above the `250.0 ms` limit
- Warning: target accuracy dropped by `0.033`

It also quarantined against the active-policy registry baseline:

- Gate: `research/evals/results/20260628-latency-calibrated-router-offset236-vs-active-registry-gate.json`
- Decision: `quarantine`
- Reasons:
  - LOO target accuracy drop `0.143` exceeded the allowed `0.080`
  - LOO latency regret increase `671.1 ms` exceeded the allowed `250.0 ms`

A same-dataset active-policy comparison is less strict and should not override the registry gate:

- Active policy on the 61-task routing dataset: target accuracy `0.672`, pass@1 `0.852`, latency regret `1403.0 ms`
- Calibrated candidate on the same dataset: target accuracy `0.639`, pass@1 `0.820`, latency regret `1172.2 ms`

## Interpretation

Offset236 added useful scan coverage and one real boundary row, but the merge-ready rows were all Qwen targets. Adding more Qwen-heavy rows without new GLM, DeepSeek, or Kimi target wins diluted the larger calibrated lane: latency calibration still helped, but the candidate lost target accuracy and carried more regret than the 58-task candidate.

This is a keep as a diagnostic data run, not as a policy refresh. The all-fail rows also show that network/filesystem-looking candidates can still be poor spend unless the selector has stronger solvability priors.

## Next Step

Do not keep broad guarded acquisition as the default next move. Bias the next run toward non-Qwen specialist-positive mining or repeat-confirmed candidate specialist wins, especially GLM/DeepSeek/Kimi rows where Qwen fails or is meaningfully slower. If using the guarded selector again, add a stronger penalty for candidates whose selected top worker is Qwen and whose neighborhood is already Qwen-heavy.
