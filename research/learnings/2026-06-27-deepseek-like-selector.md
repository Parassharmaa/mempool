# DeepSeek-Like Task Selector

## Result

A reusable similar-task selector now ranks BigCodeBench tasks by overlap with a
seed task's libraries, categories, environment risk, and plausibility score.
It can also exclude tasks already present in routing datasets or existing
outcome JSONL files.

Artifacts:

- selector: `tools/select_similar_tasks.py`
- first report: `research/evals/bigcodebench_hard_deepseek_like_remaining_report.json`
- fresh-candidate report: `research/evals/bigcodebench_hard_deepseek_like_fresh_report.json`

## Seed

The seed was `bigcodebench-hard-BigCodeBench-368`, the verified DeepSeek target
involving `shutil`, `random`, and `os` filesystem work.

Without outcome exclusion, the top similar candidates in the known eligible
pool were:

- `BigCodeBench/594`
- `BigCodeBench/720`
- `BigCodeBench/857`
- `BigCodeBench/765`
- `BigCodeBench/906`
- `BigCodeBench/800`

After excluding the active 11-task routing dataset and existing mining/compare
outcomes, the fresh-candidate count was zero.

## Interpretation

The current locally canonical-pass BigCodeBench pool is exhausted for
DeepSeek-like filesystem mining. This does not mean DeepSeek has no more
specialist wins; it means the currently materialized and locally eligible
instruct-mode pool has no untested tasks close to the `BigCodeBench/368`
signature.

## Next Step

Expand the source of eligible tasks before making more cloud calls. Good next
options:

- add a broader dependency profile so more BigCodeBench-Hard tasks are locally
  canonical-pass;
- scan a different benchmark family for filesystem/file-operation tasks;
- start the tiny Terminal-Bench pilot once the single-step router has enough
  data, because terminal/file tasks may expose more DeepSeek-style boundaries.
