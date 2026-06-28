# Normal Offset16 Contrast Pool

Run tag: `20260628-normal-offset16-contrast-pool`

## Question

Does scanning farther into normal BigCodeBench and using contrast-aware acquisition produce better real-worker training signal than the first offset-0 pool?

## Scan Result

The normal `bigcode/bigcodebench` scan from offset `16` searched `60` rows and found `8` locally eligible instruct-mode tasks:

- `bigcodebench-hard-BigCodeBench-16`
- `bigcodebench-hard-BigCodeBench-18`
- `bigcodebench-hard-BigCodeBench-19`
- `bigcodebench-hard-BigCodeBench-22`
- `bigcodebench-hard-BigCodeBench-24`
- `bigcodebench-hard-BigCodeBench-25`
- `bigcodebench-hard-BigCodeBench-27`
- `bigcodebench-hard-BigCodeBench-30`

Most rejected rows failed canonical probing because the local evaluator did not have packages such as `pandas`, `numpy`, `sklearn`, `matplotlib`, `requests`, `scipy`, `psutil`, `cryptography`, or `nltk`.

## Acquisition Result

The expanded candidate pool combined offset `0` and offset `16` tasks, then used prior contrast outcomes from `20260628-fresh-acquisition-outcomes`.

Selected six tasks:

- `bigcodebench-hard-BigCodeBench-18`
- `bigcodebench-hard-BigCodeBench-16`
- `bigcodebench-hard-BigCodeBench-24`
- `bigcodebench-hard-BigCodeBench-22`
- `bigcodebench-hard-BigCodeBench-25`
- `bigcodebench-hard-BigCodeBench-30`

## Real-Worker Outcome

The top-four cloud workers all passed every selected task:

- GLM: `6/6`
- DeepSeek: `6/6`
- Kimi: `6/6`
- Qwen: `6/6`

This made the batch useful for latency-preference labels but weak for pass/fail disagreement. The six routing rows were all merge-ready.

Mean latency by worker:

- DeepSeek: `7334.7 ms`
- Qwen: `7986.8 ms`
- GLM: `11363.2 ms`
- Kimi: `21700.7 ms`

## Candidate Router

The six merge-ready rows were merged with the 50-task experimental routing set and exported as a 56-record small-orchestrator substrate.

Target mix:

- DeepSeek: `11`
- GLM: `6`
- Kimi: `9`
- Qwen: `30`

The candidate failed the policy gate and remains quarantined:

- Candidate LOO target accuracy: `0.500`
- Baseline LOO target accuracy: `0.600`
- Candidate LOO pass@1: `0.804`
- Candidate LOO solvable pass@1: `0.849`
- Candidate LOO latency regret: `3107.9 ms`
- Gate reasons:
  - target accuracy below `0.550`
  - target accuracy drop exceeded `0.080`
  - latency regret increase exceeded `250 ms`
  - latency regret exceeded `1800 ms`

## Selector Change

The outcome showed that contrast priors alone were not enough: candidates with high `uniform_similarity` still got selected and produced all-pass rows. The selector now supports `--max-uniform-similarity`, which filters candidates that are too close to observed all-pass or all-fail neighborhoods.

A dry run with both recent outcome ledgers and `--max-uniform-similarity 6.0` filtered out the offset-16 all-pass-like candidates and left only three lower-uniform unseen candidates:

- `bigcodebench-hard-BigCodeBench-5`
- `bigcodebench-hard-BigCodeBench-0`
- `bigcodebench-hard-BigCodeBench-2`

That is not enough to justify another cloud batch yet.

## Artifacts

- Offset-16 tasks: `research/evals/20260628-normal-offset16-eligible-tasks.json`
- Offset-16 scan report: `research/evals/20260628-normal-offset16-eligible-report.json`
- Contrast-selected tasks: `research/evals/20260628-normal-offset16-contrast-selected-tasks.json`
- Contrast-selected report: `research/evals/20260628-normal-offset16-contrast-selected-report.json`
- Real-worker summary: `research/evals/results/20260628-normal-offset16-contrast-outcomes.json`
- Real-worker outcomes: `research/evals/results/20260628-normal-offset16-contrast-outcomes.jsonl`
- Outcome audit: `research/evals/results/20260628-normal-offset16-contrast-outcomes_audit.json`
- Routing rows: `research/datasets/20260628-normal-offset16-contrast-routing.jsonl`
- Merge-ready routing rows: `research/datasets/20260628-normal-offset16-contrast-routing-merge-ready.jsonl`
- 56-task substrate: `research/datasets/20260628-normal-offset16-contrast-56task-substrate.jsonl`
- 56-task model: `research/models/20260628-normal-offset16-contrast-56task-multihead.json`
- 56-task report: `research/evals/results/20260628-normal-offset16-contrast-56task-multihead-report.json`
- 56-task policy gate: `research/evals/results/20260628-normal-offset16-contrast-56task-policy-gate.json`
- Uniform-guard dry run: `research/evals/20260628-normal-offset16-uniform-guard-report.json`

## Next Step

Scan normal BigCodeBench from offset `76` onward, then run the selector with both `--contrast-outcomes` ledgers and `--max-uniform-similarity 6.0`. Do not spend more real-worker calls until the guarded selector has at least six candidates with lower uniform-risk scores.
