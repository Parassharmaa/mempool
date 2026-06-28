# Normal Offset76 Guarded Pool

Run tag: `20260628-normal-offset76-guarded-pool`

## Question

Does scanning farther into normal BigCodeBench and filtering candidates that look too similar to prior all-pass/all-fail neighborhoods produce a better acquisition batch?

## Scan Result

The normal `bigcode/bigcodebench` scan from offset `76` searched `80` rows and found `13` locally eligible instruct-mode tasks. The next scan offset is `156`.

Eligible tasks:

- `bigcodebench-hard-BigCodeBench-96`
- `bigcodebench-hard-BigCodeBench-97`
- `bigcodebench-hard-BigCodeBench-98`
- `bigcodebench-hard-BigCodeBench-113`
- `bigcodebench-hard-BigCodeBench-118`
- `bigcodebench-hard-BigCodeBench-125`
- `bigcodebench-hard-BigCodeBench-127`
- `bigcodebench-hard-BigCodeBench-130`
- `bigcodebench-hard-BigCodeBench-131`
- `bigcodebench-hard-BigCodeBench-145`
- `bigcodebench-hard-BigCodeBench-146`
- `bigcodebench-hard-BigCodeBench-147`
- `bigcodebench-hard-BigCodeBench-154`

Most rejected rows failed canonical probing because the local evaluator did not have packages such as `django`, `flask`, `numpy`, `pandas`, `matplotlib`, `scipy`, or `requests`.

## Acquisition Result

The guarded selector combined the offset `0`, offset `16`, and offset `76` task pools, used `12` prior contrast outcomes, and filtered candidates with `--max-uniform-similarity 6.0`.

It found `7` candidates and selected six tasks:

- `bigcodebench-hard-BigCodeBench-146`
- `bigcodebench-hard-BigCodeBench-96`
- `bigcodebench-hard-BigCodeBench-145`
- `bigcodebench-hard-BigCodeBench-5`
- `bigcodebench-hard-BigCodeBench-0`
- `bigcodebench-hard-BigCodeBench-2`

## Real-Worker Outcome

The top-four worker pool produced `24` outcome rows across `6` tasks. Every task has all four worker results, and the audit marked the ledger ready for conversion.

Worker pass counts:

- DeepSeek: `3/6`, mean latency `25411.8 ms`
- GLM: `4/6`, mean latency `24753.3 ms`
- Kimi: `4/6`, mean latency `13589.3 ms`
- Qwen: `4/6`, mean latency `7284.3 ms`

Task pass counts:

- `bigcodebench-hard-BigCodeBench-0`: `4/4`
- `bigcodebench-hard-BigCodeBench-145`: `0/4`
- `bigcodebench-hard-BigCodeBench-146`: `0/4`
- `bigcodebench-hard-BigCodeBench-2`: `3/4`
- `bigcodebench-hard-BigCodeBench-5`: `4/4`
- `bigcodebench-hard-BigCodeBench-96`: `4/4`

This batch is more useful than the offset-16 batch for pass/fail learning. It contains two all-fail diagnostics and one worker-specific contrast task, `BigCodeBench-2`, where DeepSeek failed and the other three workers passed.

## Merge Result

The routing conversion produced `6` rows. The merge-readiness filter kept `4` rows and dropped the two all-fail tasks:

- Dropped `bigcodebench-hard-BigCodeBench-145`
- Dropped `bigcodebench-hard-BigCodeBench-146`

The kept rows were merged with the 50-task base dataset, yielding a 54-record routing dataset and substrate.

Substrate target mix:

- DeepSeek: `9`
- GLM: `4`
- Kimi: `9`
- Qwen: `32`

Workflow mix:

- direct: `51`
- verify_then_fallback: `3`

## Candidate Routers

The latency-regret weight `0.5` candidate improved several held-out metrics over the 50-task baseline but failed the latency gate:

- Candidate LOO target accuracy: `0.611`
- Baseline LOO target accuracy: `0.600`
- Candidate LOO pass@1: `0.815`
- Baseline LOO pass@1: `0.800`
- Candidate LOO solvable pass@1: `0.863`
- Baseline LOO solvable pass@1: `0.851`
- Candidate LOO latency regret: `2592.0 ms`
- Baseline LOO latency regret: `1693.5 ms`

Gate decision: `quarantine`.

Gate reasons:

- latency regret increase `898.5 ms` exceeded the `250 ms` limit
- latency regret `2592.0 ms` exceeded the `1800 ms` maximum

The latency-regret weight `1.0` candidate did not fix the held-out latency problem:

- Candidate LOO target accuracy: `0.593`
- Candidate LOO pass@1: `0.796`
- Candidate LOO solvable pass@1: `0.843`
- Candidate LOO latency regret: `2595.1 ms`

Gate decision: `quarantine`.

## Interpretation

The guarded acquisition strategy worked: it found a mixed-signal batch after the offset-16 all-pass pool. The new data should be kept, but the refreshed 54-task policies should not become active.

The current router is learning solvability faster than latency discipline. Raising the existing latency-regret weight improved neither held-out regret nor accuracy. The next policy improvement likely needs an explicit latency-aware calibration step, a conditional cheap-first fallback/value head, or a routing loss that separates "can solve" from "should spend on this worker now."

The dropped all-fail rows should stay out of winner-routing merges, but they are valuable as a future abstain, verifier, or fallback dataset.

## Artifacts

- Offset-76 tasks: `research/evals/20260628-normal-offset76-eligible-tasks.json`
- Offset-76 scan report: `research/evals/20260628-normal-offset76-eligible-report.json`
- Guarded-selected tasks: `research/evals/20260628-normal-offset76-guarded-selected-tasks.json`
- Guarded-selected report: `research/evals/20260628-normal-offset76-guarded-selected-report.json`
- Real-worker summary: `research/evals/results/20260628-normal-offset76-guarded-outcomes.json`
- Real-worker outcomes: `research/evals/results/20260628-normal-offset76-guarded-outcomes.jsonl`
- Outcome audit: `research/evals/results/20260628-normal-offset76-guarded-outcomes_audit.json`
- Routing rows: `research/datasets/20260628-normal-offset76-guarded-routing.jsonl`
- Merge-ready routing rows: `research/datasets/20260628-normal-offset76-guarded-routing-merge-ready.jsonl`
- 54-task routing dataset: `research/datasets/20260628-normal-offset76-guarded-54task-routing.jsonl`
- 54-task substrate: `research/datasets/20260628-normal-offset76-guarded-54task-substrate.jsonl`
- Latency `0.5` model: `research/models/20260628-normal-offset76-guarded-54task-multihead.json`
- Latency `0.5` report: `research/evals/results/20260628-normal-offset76-guarded-54task-multihead-report.json`
- Latency `0.5` policy gate: `research/evals/results/20260628-normal-offset76-guarded-54task-policy-gate.json`
- Latency `1.0` model: `research/models/20260628-normal-offset76-guarded-54task-latency1p0-multihead.json`
- Latency `1.0` report: `research/evals/results/20260628-normal-offset76-guarded-54task-latency1p0-multihead-report.json`
- Latency `1.0` policy gate: `research/evals/results/20260628-normal-offset76-guarded-54task-latency1p0-policy-gate.json`

## Next Step

Before spending another real-worker batch, scan normal BigCodeBench from offset `156` and run the same guarded selector. In parallel, prototype one small latency-calibrated decision layer that can choose a cheaper passing worker when the primary route has high expected latency regret.
