# Terminal-Bench Metadata Bridge

## What Changed

Added a metadata extraction bridge for future Terminal-Bench 2.1 pilot
selection.

Artifacts:

- `src/mempool/terminal_bench.py`
- `tools/extract_terminal_bench_metadata.py`
- `research/evals/terminal_bench_metadata_example.json`
- `research/evals/terminal_bench_metadata_example_extracted.json`
- `research/evals/terminal_bench_metadata_example_report.json`
- `research/evals/terminal_bench_example_pilot_manifest.json`

## Learning

Terminal-Bench pilot selection needs a safe pre-processing step before task
selection. Real exports or checkouts can contain prompts, verifier details,
oracle material, or raw task files. The bridge now keeps only metadata fields
such as id, category, difficulty, tags, split, version, and source.

For directory-based checkouts, task ids can be derived from task-directory
paths without reading task instructions. This gives us enough information for a
category-diverse pilot manifest while preserving the held-out content boundary.

## Decision

Use the bridge as the first command whenever a local Terminal-Bench export or
checkout is introduced. Keep synthetic example artifacts separate from real
pilot manifests.

## Next Step

Fetch or mount a real Terminal-Bench 2.1 metadata source, run the extractor,
and select the first 3-5 task metadata-only pilot manifest. Run an oracle or
harness sanity check before worker calls.
