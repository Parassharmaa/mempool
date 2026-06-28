# BigCodeBench Mini-Pilot Selector

## Change

Added a mini-pilot selector for the materialized BigCodeBench-Hard smoke set.
It writes:

- `research/evals/bigcodebench_hard_minipilot_tasks.json`
- `research/evals/bigcodebench_hard_minipilot_report.json`

The selector analyzes each task for:

- prompt length
- test length
- suggested libraries
- missing local libraries
- coarse task categories
- environment risk
- canonical-solution eligibility

## Result

With canonical probing enabled, only 3 of the first 10 materialized
BigCodeBench-Hard tasks pass under the current local Python environment:

- `bigcodebench-hard-BigCodeBench-13`
- `bigcodebench-hard-BigCodeBench-15`
- `bigcodebench-hard-BigCodeBench-19`

The selected 3-task mini-pilot is:

- `bigcodebench-hard-BigCodeBench-15`: lowest-risk subprocess/filesystem task
- `bigcodebench-hard-BigCodeBench-19`: lowest-risk filesystem task
- `bigcodebench-hard-BigCodeBench-13`: remaining canonical-pass task, already
  known to be hard for the current local worker pool

## Learning

External benchmark selection has two separate gates:

1. The task must be benchmark-runnable in the current local environment.
2. The task should be informative for comparing workers.

The first 10-task slice failed the first gate for most data-science and web-app
tasks because packages such as `pandas`, `numpy`, `sklearn`, `matplotlib`,
`wordcloud`, `flask`, and `psutil` are not available. Those failures should not
be treated as model failures or routing labels.

## Next Step

Run the 3-task mini-pilot through the resumable real-worker runner, then convert
the results into routing records. After that, decide whether to install a
benchmark dependency profile or continue selecting only standard-library tasks.
