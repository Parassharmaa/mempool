# Resumable Real Worker Runs

## Change

The real worker smoke runner now supports resumable runs with compact progress
logging:

- `--resume` loads existing worker/task records from the output summary
- `--progress` prints per-record `run`, `done`, and `skip` events
- `--quiet` suppresses the final full JSON dump for long runs
- summary JSON and outcome JSONL are rewritten after each worker completes

## Result

The resume path was verified without calling any model by copying the existing
expanded Ollama run to `/tmp`, running a one-task resumed pass, and confirming
that all three worker records were skipped and rewritten:

- workers: 3
- records: `[1, 1, 1]`
- solved: `[1, 1, 0]`
- outcomes: 3

The full local unit suite passed after adding regression coverage for record
loading and summary reconstruction.

## Learning

Larger external benchmark pilots need interruption tolerance before they need a
more complex router. The 10-task Ollama run already takes long enough that
losing progress would make experimentation brittle. Resumable real-worker
evaluation is therefore a harness prerequisite for Milestone 4 and for any
30-50 task pilot.

## Next Step

Use the resumable runner while adding the external smoke benchmark loader, so
the next measured dataset can grow from local toy tasks to benchmark-backed
tasks without requiring a single uninterrupted run.
