# Top-Four Fresh Acquisition

## Question

Does the expanded top-four BigCodeBench environment produce fresh routing
signal after excluding every task already present in routing datasets or raw
outcome files?

## Result

Yes, but the first signal is still Qwen-heavy.

`tools/select_fresh_bigcodebench_batch.py` selected 8 diverse fresh tasks from
`research/evals/bigcodebench_hard_top4_eligible_merged_tasks.json`, leaving 40
fresh candidates after exclusions. A one-sample Qwen mining pass solved 3/8:

- `BigCodeBench/123`
- `BigCodeBench/678`
- `BigCodeBench/308`

A two-sample repeat comparison across Qwen, Kimi, GLM, and DeepSeek produced 24
outcomes and 3 routing records. All three rows have Qwen as the reward target:

- `BigCodeBench/123`: Qwen 1/2, all other workers 0/2.
- `BigCodeBench/678`: all workers 2/2, Qwen fastest.
- `BigCodeBench/308`: Qwen, Kimi, and GLM 2/2; Qwen fastest; DeepSeek 1/2.

## Router Decision

Merging these rows with the 11-task diagnostic dataset created
`research/datasets/20260627-mixed-winner-14task-routing.jsonl`. The temperature
selector quarantined every candidate. The best candidates reached:

- 0.7143 leave-one-out target accuracy
- 0.8571 leave-one-out pass@1
- 1334.1 ms mean leave-one-out latency regret

This regresses against the active ten-task reward-tempered policy, so the active
policy should not be promoted.

## Decision

Keep the new rows as measured evidence. For the next acquisition pass, use the
fresh selector but mine first with Kimi, GLM, or DeepSeek over the remaining
fresh top-four tasks to find non-Qwen specialist wins before another refresh.
