# Next Profile Positive Repeat

Run tag: `20260628-next-profile-positive-repeat`

Goal: repeat-compare the two positives from the expanded dependency-profile
screen before deciding whether they are refresh-training rows.

Input tasks:

- `BigCodeBench/826`
- `BigCodeBench/865`

Repeat result:

`BigCodeBench/826` is a stable broad-pass row:

- Qwen: `2/2`, mean latency `2819 ms`
- DeepSeek: `2/2`, mean latency `5843.5 ms`
- GLM: `2/2`, mean latency `7282 ms`
- Kimi: `0/2`

The empirical target is Qwen because it has full pass rate and the lowest
latency among passing workers.

`BigCodeBench/865` is a partial Kimi-specialist row:

- Kimi: `1/2`, mean latency `6532 ms`
- Qwen: `0/2`
- DeepSeek: `0/2`
- GLM: `0/2`

The empirical target is Kimi under the repeated soft-target converter, but the
strict merge gate rejects the row because the target pass rate is only `0.5`.

Merge audit:

- Strict `min_target_pass_rate=1.0`: blocked by unstable target worker on `865`
- Relaxed `min_target_pass_rate=0.5`: passes

Decision:

Keep this as diagnostic routing data and evidence that the expanded profile
produced a fresh Kimi-favored region. Do not merge it into the promoted refresh
dataset under the strict policy yet. Next step should either collect more
samples for `865` or mine nearby newly unlocked sklearn/scipy/nltk rows for a
more stable specialist target.

Artifacts:

- `research/evals/results/20260628-next-profile-positive-repeat.jsonl`
- `research/evals/results/20260628-next-profile-positive-repeat-summary.json`
- `research/datasets/20260628-next-profile-positive-repeat-routing.jsonl`
- `research/datasets/20260628-next-profile-positive-repeat-merge-audit.json`
- `research/datasets/20260628-next-profile-positive-repeat-relaxed-merge-audit.json`
