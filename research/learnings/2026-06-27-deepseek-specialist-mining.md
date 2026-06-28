# DeepSeek Specialist Mining

## Result

A bounded specialist-mining pass found one verified DeepSeek target, but the
expanded router refresh was quarantined.

Artifacts:

- mining tasks: `research/evals/bigcodebench_hard_qwen_negative_untrained_tasks.json`
- specialist pool: `research/evals/ollama_cloud_glm_deepseek_pool.json`
- mining outcomes: `research/evals/results/20260627-glm-deepseek-qwen-kimi-negative-mining.jsonl`
- repeat outcomes: `research/evals/results/20260627-deepseek-candidate-368-repeat-compare.jsonl`
- new routing row: `research/datasets/20260627-deepseek-368-routing.jsonl`
- merged dataset: `research/datasets/20260627-mixed-winner-11task-routing.jsonl`
- selector report: `research/refreshes/20260627-mixed-winner-11task-temperature-selection.json`

## Mining Signal

The mining set used ten Qwen-negative and Kimi-negative BigCodeBench-Hard tasks
that were not already in the active routing dataset. GLM failed all ten tasks.
DeepSeek passed one task on the first probe:

- `bigcodebench-hard-BigCodeBench-368`

A two-sample repeat comparison across the fast cloud pool confirmed the signal:

- DeepSeek V4 Pro: 2/2
- GLM 5.2: 0/2
- Kimi K2.7 Code: 0/2
- Qwen3 Coder 480B: 0/2

The resulting routing row targets DeepSeek with target probability 0.952036.

## Refresh Decision

Merging the DeepSeek row into the active ten-task dataset produced an
11-task dataset with four target workers: Qwen, Kimi, GLM, and DeepSeek.

The automated reward-temperature selector tested 0.05, 0.10, 0.20, and 0.50.
No candidate passed the current refresh gate. The best candidates, 0.05 and
0.10, reached:

- leave-one-out target accuracy: 0.7273
- leave-one-out pass@1: 0.8182
- leave-one-out mean latency regret: 1047.7 ms

The refresh was quarantined because latency regret exceeded the 1000 ms maximum
and increased by 529.1 ms over the active baseline.

## Failure Mode

The 11-task leave-one-out misses were:

- `BigCodeBench/368`: target DeepSeek, predicted Qwen
- `BigCodeBench/963`: target GLM, predicted Qwen
- `BigCodeBench/999`: target Qwen, predicted Kimi

This is useful evidence rather than a failed data run. One DeepSeek row adds
target diversity, but it is too sparse for the current feature set to learn the
DeepSeek boundary without hurting latency-regret gates.

## Next Step

Keep the DeepSeek row as measured evidence, but do not promote the 11-task
router yet. Mine more filesystem tasks that look like `BigCodeBench/368`,
especially `shutil`/`random`/`os` file-operation tasks, then rerun the
temperature selector once DeepSeek has more than one target row.
