# Qwen-Fail Specialist Mining

Run tag: `20260628-qwen-fail-specialist-mining`

## Question

Can the active-router fallback-opportunity selector find fresh tasks where the
active router prefers Qwen but a specialist worker is the better choice?

## Setup

The selector used:

- active registry: `research/policies/active_policy.json`
- task source:
  `research/evals/bigcodebench_hard_next3_profile_eligible_merged_tasks.json`
- seed routing row:
  `research/datasets/20260628-next3-positive-repeat-routing.jsonl`
- seed task: `bigcodebench-hard-BigCodeBench-917`
- preferred second workers:
  - `ollama-cloud-kimi-k2.7-code`
  - `ollama-cloud-glm-5.2`

Artifacts:

- plan:
  `research/evals/20260628-qwen-fail-specialist-mining-report.json`
- tasks:
  `research/evals/20260628-qwen-fail-specialist-mining-tasks.json`
- one-sample top-four outcomes:
  `research/evals/results/20260628-qwen-fail-specialist-mining-screen1.jsonl`
- summary:
  `research/evals/results/20260628-qwen-fail-specialist-mining-screen1-summary.json`

## Selected Tasks

The selector found only five remaining candidates after excluding prior routing
and outcome ledgers. Four were screened:

- `BigCodeBench/1003`: active top Qwen, Kimi second
- `BigCodeBench/34`: active top Qwen, Kimi second
- `BigCodeBench/1015`: active top Kimi, Qwen second
- `BigCodeBench/273`: active top Kimi, Qwen second

## Result

No Qwen-fail specialist row was found.

- `1003`: Qwen, Kimi, and GLM passed; Qwen was fastest.
- `34`: Qwen, Kimi, and DeepSeek passed; Qwen was fastest.
- `1015`: universal failure across the top-four screen.
- `273`: universal failure across the top-four screen.

The positive task extractor wrote:

- `research/evals/20260628-qwen-fail-specialist-mining-positive-tasks.json`

Those positives were not repeated because the first screen already showed Qwen
passing and winning latency on both rows. Repeating them would add more Qwen
latency anchors, not the Qwen-fail evidence this run was designed to find.

## Decision

Do not convert this screen into training data.

Keep the acquisition evidence. The current fallback-opportunity heuristic,
seeded by `917`, still tends to surface Qwen-fast broad-pass rows in the
remaining next3 frontier. The rare boundary we need is more specific:

- Qwen has high prior probability or is top-ranked.
- Kimi/GLM has non-trivial probability.
- The task has features like `917` that cause Qwen to fail rather than merely
lose or win latency.

The next acquisition attempt should either add an explicit Qwen-fail prior from
known misses or expand the eligible frontier again instead of reusing the same
uncertainty heuristic.
