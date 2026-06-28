# Offset-99 Hard Negatives and Offset-125 Refresh Attempt

Run tag: `20260627-specialist-mining-offset99`

## Question

Can the offset-99 tasks that were negative for Qwen and Kimi reveal GLM or
DeepSeek specialist wins, and can fresh offset-125 positives improve the active
router's latency-tie behavior?

## Offset-99 Specialist Mining

Tasks:

- `BigCodeBench/785`
- `BigCodeBench/800`
- `BigCodeBench/857`
- `BigCodeBench/988`

Result over 32 outcomes, two samples per worker/task:

- GLM 5.2: 0/8, mean latency 53883.12 ms
- DeepSeek V4 Pro: 0/8, mean latency 8796.38 ms
- Kimi K2.7 Code: 0/8, mean latency 23605.88 ms
- Qwen3 Coder 480B: 0/8, mean latency 3529.0 ms

No task from this slice was added to the routing dataset. These rows are useful
as hard-negative selection evidence, but training on them would mainly teach
the router to pick the fastest failing worker.

## Harness Fix

Scanning offset 125 exposed a benchmark harness issue: canonical-solution
timeouts crashed the scanner. `SmokeCodeBenchmarkAdapter` now returns structured
`eval_timeout` failures and normalizes timeout stdout/stderr tails to text so
scan reports remain JSON-serializable.

## Offset-125 Candidate Data

The offset-125 scan found five locally eligible tasks. Qwen solved three in a
single mining pass: `BigCodeBench/990`, `BigCodeBench/998`, and
`BigCodeBench/999`.

Repeated comparison showed that `998` was unstable: it failed for every worker
in the repeated run. The stable repeat-positive set is therefore:

- `BigCodeBench/990`
- `BigCodeBench/999`

Both are broad-pass latency targets where Qwen wins:

- `990`: all workers 2/2; Qwen mean latency 2368.0 ms
- `999`: all workers 2/2; Qwen mean latency 3321.5 ms

## Refresh Result

The two stable offset-125 rows were merged with the active eight-task dataset
and trained as a ten-task logits-router candidate.

The refresh gate quarantined the candidate:

- baseline LOO target accuracy: 0.75
- candidate LOO target accuracy: 0.40
- baseline LOO pass@1: 0.75
- candidate LOO pass@1: 0.60

The broader router comparison also showed that the existing active policy
already reaches 8/10 target accuracy and 10/10 pass@1 on the ten-task merged
dataset, while the retrained candidate generalizes poorly.

## Interpretation

Adding more Qwen latency-tie rows alone does not fix latency-tie routing in the
current linear logits model. It can actually wash out the sparse Kimi and GLM
specialist boundaries. The next refresh should either improve features for
latency-sensitive broad-pass tasks or train with an objective that preserves
specialist regions while reducing latency regret.

## Next Step

Keep the active eight-task policy. Use the ten-task dataset as a quarantined
candidate and diagnostic set. Before another promotion attempt, add either:

- more non-Qwen specialist wins, or
- explicit latency-regret features/objective terms for broad-pass tasks.
