# Non-Qwen Specialist Mining

## Question

Can the remaining fresh top-four BigCodeBench pool produce non-Qwen specialist
wins if Qwen is removed from the mining pass?

## Result

Not yet. The non-Qwen mining pool found positives, but repeat comparison showed
they are broad-pass rows where Qwen is much faster.

The mining pass used:

- `ollama-cloud-glm-5.2`
- `ollama-cloud-deepseek-v4-pro`
- `ollama-cloud-kimi-k2.7-code`

It evaluated 8 fresh tasks and found two candidates:

- `BigCodeBench/1085`
- `BigCodeBench/870`

Repeat comparison across GLM, DeepSeek, Kimi, and Qwen showed:

- `BigCodeBench/1085`: all workers 2/2; Qwen mean latency 2314 ms.
- `BigCodeBench/870`: all workers 2/2; Qwen mean latency 2774 ms.

The next fastest workers were much slower, so both rows target Qwen under the
current latency-adjusted reward.

## Router Decision

The two rows were converted into
`research/datasets/20260627-top4-specialist-positive-repeat-routing.jsonl` and
merged into `research/datasets/20260627-mixed-winner-16task-routing.jsonl`.

The temperature selector quarantined every 16-task candidate. The best
temperatures reached:

- 0.75 leave-one-out target accuracy
- 0.875 leave-one-out pass@1
- 1167.3 ms mean leave-one-out latency regret

Accuracy recovered to the gate threshold, but latency regret remained too high.

## Decision

Keep the rows as broad-pass latency evidence, not as a promotion candidate. The
next acquisition pass should continue mining with non-Qwen workers, but should
select harder or more distinctive rows where Qwen is likely to fail, become
unstable, or lose latency after repeated sampling.
