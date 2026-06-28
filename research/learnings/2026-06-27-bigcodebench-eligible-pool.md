# BigCodeBench Eligible Pool Scan

## Change

Added an eligible-pool scanner for BigCodeBench-Hard. It fetches rows from the
dataset API, runs each official canonical solution through the local evaluator,
and writes only tasks whose canonical solution passes in the current Python
environment.

Artifacts:

- `research/evals/bigcodebench_hard_eligible_tasks.json`
- `research/evals/bigcodebench_hard_eligible_report.json`

## Result

The scanner found 8 eligible tasks after probing 44 rows:

- `bigcodebench-hard-BigCodeBench-13`
- `bigcodebench-hard-BigCodeBench-15`
- `bigcodebench-hard-BigCodeBench-19`
- `bigcodebench-hard-BigCodeBench-147`
- `bigcodebench-hard-BigCodeBench-310`
- `bigcodebench-hard-BigCodeBench-324`
- `bigcodebench-hard-BigCodeBench-326`
- `bigcodebench-hard-BigCodeBench-346`

The report records `next_offset = 44`, so future scans can continue from the
next unprobed row instead of repeating the first segment.

## Learning

The locally runnable hard-task pool is heavily shaped by environment
dependencies. Many rejected tasks failed because official solutions imported
packages not installed here, including `pandas`, `numpy`, `matplotlib`,
`sklearn`, `flask`, `requests`, `faker`, `nltk`, and `psutil`. A few also depend
on Python modules such as `cgi` that are unavailable in this runtime.

Without a dependency profile, the fair external benchmark pool is mostly
standard-library and system-oriented tasks: subprocess, filesystem, sockets,
threading, CSV, and archive handling. That is still valuable for routing, but
it should not be mistaken for full BigCodeBench coverage.

## Next Step

Run the expanded 8-task eligible pool through the resumable real-worker runner,
or first create a benchmark dependency profile if we want data-science and web
tasks to enter the external routing dataset.
