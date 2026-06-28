# Qwen-Resistant Rank1 Specialist

Run tag: `20260628-qwen-resistant-rank1-specialist`

## Question

Can we improve non-Qwen specialist acquisition by combining three signals:

- specialist rank `1` under the active router
- similarity to measured Qwen-fail rows
- dissimilarity from Qwen-fast anchor rows

The previous specialist-pressure run found many specialist-positive tasks, but Qwen still passed and was fastest on every top-four comparison row.

## Selector Change

`tools/select_qwen_fail_contrast_batch.py` now supports `--max-specialist-rank`.

When set to `1`, candidates are filtered to tasks where GLM, DeepSeek, or Kimi is the highest-probability worker under the active router. The report now includes `specialist_rank` for each candidate.

Tests were added in `tests/test_select_qwen_fail_contrast_batch.py`.

## Seeds

Seeds came from `research/datasets/20260628-normal-offset236-boundary-61task-routing.jsonl`.

Seed counts:

- Qwen-fail task ids: `15`
- Non-Qwen target task ids: `24`
- Qwen-fast anchor task ids: `21`

The Qwen-fast anchors were Qwen-target rows where Qwen passed with latency at or below `3000 ms`.

## Selection

The Qwen-resistant rank-1 selector found `8` candidates and selected six:

- `bigcodebench-hard-BigCodeBench-18`
- `bigcodebench-hard-BigCodeBench-145`
- `bigcodebench-hard-BigCodeBench-158`
- `bigcodebench-hard-BigCodeBench-283`
- `bigcodebench-hard-BigCodeBench-12`
- `bigcodebench-hard-BigCodeBench-229`

The top candidate, `BigCodeBench-18`, had high Qwen-fail similarity (`10.334`) and lower Qwen-anchor similarity (`7.3239`). Kimi was the active router's top worker, with Qwen second.

## Non-Qwen Screen

The non-Qwen specialist screen produced `18` rows across GLM, DeepSeek, and Kimi.

Only one task passed on specialists:

- `BigCodeBench-18`: GLM, DeepSeek, and Kimi all passed in the screen; DeepSeek was fastest at `8079 ms`

The other five selected tasks were universal specialist failures:

- `BigCodeBench-145`
- `BigCodeBench-158`
- `BigCodeBench-283`
- `BigCodeBench-12`
- `BigCodeBench-229`

This is a lower hit rate than the rank-1 specialist-only run, but it is more focused on Qwen-resistant neighborhoods.

## Top-Four Comparison

The single specialist-positive task was compared against the top-four pool:

- GLM: fail, request timeout, `120095 ms`
- DeepSeek: pass, `12743 ms`
- Kimi: pass, `88239 ms`
- Qwen: pass, `11129 ms`

The converted routing row targets Qwen:

- `bigcodebench-hard-BigCodeBench-18`: `ollama-cloud-qwen3-coder-480b`

## Interpretation

The Qwen-resistant selector improved focus: it surfaced a candidate where Qwen was not an obvious fast-anchor neighbor and where all three specialists passed in the first screen.

But the result still did not produce a non-Qwen target. Qwen passed and remained faster than DeepSeek by about `1.6 s`. The margin is much narrower than the broad-pass filesystem rows, but it is still a Qwen target under the current reward.

The key failure mode is now clearer:

- Rank-1 specialist probability finds specialist-solvable rows.
- Qwen-fail contrast finds harder neighborhoods.
- Neither is enough to find specialist-target rows unless Qwen either fails or has a much larger latency penalty.

## Decision

Keep the selector feature and evidence. Do not merge the one Qwen-target row into the refresh dataset.

## Next Step

The next acquisition step should target verified Qwen-fail neighborhoods more aggressively. Two possible paths:

- Require candidates to be close to Qwen-fail seeds and far from Qwen-fast anchors, with a stricter anchor penalty than this run.
- Mine directly from known non-Qwen target families, but run Qwen first as a cheap rejection filter: if Qwen passes under a latency threshold, skip the expensive specialist comparison.

This should reduce spend on rows that are specialist-solvable but still Qwen-targeted.
