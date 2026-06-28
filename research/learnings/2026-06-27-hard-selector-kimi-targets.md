# Hard Selector Kimi Targets

## Question

Can a harder fresh-task selection strategy find real non-Qwen routing targets
from the expanded top-four BigCodeBench pool?

## Result

Yes. The new `--strategy hard` mode in
`tools/select_fresh_bigcodebench_batch.py` favors higher environment risk,
higher plausibility score, and library/category novelty instead of the
lowest-risk diverse rows.

The first hard batch selected 8 fresh tasks. Non-Qwen mining found three
specialist-positive candidates:

- `BigCodeBench/1004`
- `BigCodeBench/1006`
- `BigCodeBench/760`

Repeat comparison across GLM, DeepSeek, Kimi, and Qwen produced:

- `BigCodeBench/1004`: Kimi, DeepSeek, and GLM 2/2; Qwen 0/2. Kimi is the
  reward target.
- `BigCodeBench/1006`: Kimi 2/2; GLM and DeepSeek 1/2; Qwen 0/2. Kimi is the
  reward target.
- `BigCodeBench/760`: Qwen 2/2 and fastest; specialists are unstable. Qwen is
  the reward target.

This is the first top-four fresh pass that produced repeat-confirmed
Qwen-negative Kimi targets.

## Router Decision

The rows were converted to
`research/datasets/20260627-top4-hard-specialist-repeat-routing.jsonl` and
merged into `research/datasets/20260627-mixed-winner-19task-routing.jsonl`.

The temperature selector still quarantined the 19-task refresh. The best
low-temperature candidate had:

- 0.7368 leave-one-out target accuracy
- 0.7895 leave-one-out pass@1
- 288.3 ms mean leave-one-out latency regret

This misses the 0.75 accuracy gate but improves latency regret substantially.

## Decision

Keep the hard selector and the new Kimi rows. Do not promote the 19-task policy
yet. The next router improvement should add features that distinguish
network/archive/request-style tasks and then re-evaluate the 19-task dataset.
