# Non-Qwen Specialist-First Screen

Run tag: `20260628-nonqwen-specialist-first`

## Question

Can a specialist-first acquisition pass find fresh non-Qwen positives before
spending full top-four comparison calls?

## Setup

- Source tasks:
  `research/evals/bigcodebench_hard_next2_profile_newly_unlocked_tasks.json`
- Deduping excluded prior routing datasets and outcome JSONL files.
- Worker pool:
  `research/evals/ollama_cloud_nonqwen_top3_pool.json`
- Workers:
  - `ollama-cloud-glm-5.2`
  - `ollama-cloud-deepseek-v4-pro`
  - `ollama-cloud-kimi-k2.7-code`
- Planned tasks:
  - `bigcodebench-hard-BigCodeBench-1077`
  - `bigcodebench-hard-BigCodeBench-374`

## Result

The screen produced no positive tasks.

- Outcome file:
  `research/evals/results/20260628-nonqwen-specialist-first-screen1.jsonl`
- Summary:
  `research/evals/results/20260628-nonqwen-specialist-first-screen1-summary.json`
- Attempts: 6
- Passed attempts: 0
- Universal-failure tasks: 2
- Positive task file:
  `research/evals/20260628-nonqwen-specialist-first-screen1-positive-tasks.json`

Per-worker pass rates:

- `ollama-cloud-glm-5.2`: 0/2
- `ollama-cloud-deepseek-v4-pro`: 0/2
- `ollama-cloud-kimi-k2.7-code`: 0/2

## Decision

Do not merge this screen into the router training dataset. It contains no
solved examples and no target-worker signal.

Keep the acquisition pattern. Running specialist workers before Qwen avoided a
larger top-four spend and made the no-signal result explicit. The next useful
step is not to repeat these same tasks; expand the benchmark dependency frontier
again or mine a different fresh source where at least one specialist has a
plausible positive region.
