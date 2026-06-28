# Guarded Routing Merge

`tools/merge_routing_datasets.py` now has an opt-in `--require-merge-ready`
flag. With the flag, the merge command validates the merged dataset and runs the
merge-readiness audit before writing output.

The provenance-backed catalog-candidate routing dataset is blocked by this
guard because it contains:

- an all-fail fastest-failure task
- an unstable 0.5-pass-rate target

Unguarded merge is still available for historical inspection or explicitly
negative datasets, but router refresh workflows should use the guarded path.
