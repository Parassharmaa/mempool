# Enriched Router Features

Run tag: `20260628-enriched-router-features`

## Question

Can the current linear logits router separate the new Qwen-fail exception rows
by adding general primary-category, category-library, and signal-library
interaction features?

## Change Tested

Temporarily expanded `extract_task_features` with:

- `primary_category_*` one-hot features
- `primary_<category>__lib_<library>` interactions
- `category_<category>__lib_<library>` interactions
- `signal_<signal>__lib_<library>` interactions

The change was intentionally generic: no task IDs, worker IDs, or provider names
were encoded.

## Result

The enriched 25-task candidate was quarantined by the existing refresh gate.
Best candidate was reward temperature `0.05`:

- LOO target accuracy: `0.360`
- LOO pass@1: `0.600`
- LOO solvable pass@1: `0.682`
- LOO mean latency regret: `2977.6 ms`

Active baseline:

- LOO target accuracy: `0.783`
- LOO pass@1: `0.696`
- LOO solvable pass@1: `0.800`
- LOO mean latency regret: `501.1 ms`

Gate reasons:

- target accuracy below minimum `0.700`
- target accuracy drop `0.423` exceeded allowed `0.100`
- latency regret increase `2476.5 ms` exceeded allowed `100.0 ms`

## Interpretation

The interaction features made the head too sparse for the current dataset size.
They exposed useful local structure for `BigCodeBench/917` and
`BigCodeBench/1008`, but leave-one-out generalization collapsed across many
other rows.

Do not promote this feature expansion. The result strengthens the case for one
of two next moves:

1. collect paired counterexamples around each Qwen-fail exception before adding
   higher-cardinality symbolic features;
2. move to a smoother representation, such as embeddings plus a small regularized
   head or a hierarchical fallback head that only activates when the active
   router is uncertain.

## Artifacts

- `research/datasets/20260628-mixed-winner-25task-qwen-fail-contrast-enriched-temperature-selection.json`
- `research/datasets/20260628-mixed-winner-25task-qwen-fail-contrast-enriched-reward-t0p05-logits-router-report.json`
- `research/models/20260628-mixed-winner-25task-qwen-fail-contrast-enriched-reward-t0p05-logits-router.json`
