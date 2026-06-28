# Acquisition Wave 1

Executed the acquisition-to-50 wave:

- task file: `research/evals/bigcodebench_hard_acquisition_to_50_wave1_tasks.json`
- summary: `research/evals/results/20260627-acquisition-to-50-wave1.json`
- outcomes: `research/evals/results/20260627-acquisition-to-50-wave1.jsonl`
- raw routing dataset: `research/datasets/20260627-acquisition-to-50-wave1-routing.jsonl`

Outcome audit passed:

- 328 rows
- 41 tasks
- 4 workers per task
- 2 samples per worker/task
- evaluator packages `numpy` and `pandas` present on every row

Raw merge-readiness failed:

- 29 all-fail tasks
- 3 unstable target rows
- 12 solvable tasks

Added a merge filter:

- `src/mempool/routing_merge_filter.py`
- `tools/filter_routing_dataset_for_merge.py`
- `research/datasets/20260627-acquisition-to-50-wave1-merge-ready-routing.jsonl`

The filtered subset kept 9 stable rows:

- Qwen targets: 3
- Kimi targets: 1
- DeepSeek targets: 4
- GLM targets: 1

Merged with the active 23-task dataset to form:

- `research/datasets/20260627-acquisition-to-50-wave1-merged-32task-routing.jsonl`

The 32-task candidate router was quarantined:

- LOO target accuracy: 0.625
- baseline LOO target accuracy: 0.7826
- LOO mean latency regret: 999.9 ms
- baseline LOO mean latency regret: 501.1 ms

Keep the active 23-task policy unchanged. The useful learning is that wave 1
found real stable rows, including GLM/DeepSeek specialist evidence, but the
current linear feature head cannot absorb them safely yet. Next step should be
either more stable specialist/latency rows or a targeted feature/capacity
change before another promotion attempt.
