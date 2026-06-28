# Specialist Solvability Gate

Run tag: `20260628-specialist-solvability-gate`

## Question

Can the Qwen-first acquisition loop avoid universal specialist failures by
adding an empirical specialist-solvability prior before running GLM, DeepSeek,
or Kimi?

The previous Qwen-first rejection filter saved specialist calls on Qwen-fast
rows, but still spent three specialist screens on rows that all specialists
failed. The missing signal was not just "Qwen failed"; it was whether the task
looked like prior specialist successes or like prior universal failures.

## Utility Added

Added `tools/select_solvable_worker_rejections.py`.

The selector takes:

- candidate tasks
- Qwen-first outcome rows
- prior specialist outcome JSONL files
- specialist worker ids

It first selects tasks where the rejected worker failed or passed too slowly.
Then it scores each rejected task with:

- similarity to specialist-positive prior outcomes
- penalty for similarity to universal specialist-failure prior outcomes
- small penalties for environment risk and prompt plausibility risk

Tests were added in `tests/test_select_solvable_worker_rejections.py`.

One scoring bug was caught during the run: negative similarity to a failure seed
was acting as a bonus. Similarity is now clamped at zero, so unrelated failure
seeds are neutral.

## Retrospective Audit

The audit reused the previous five-task Qwen-first batch:

- `BigCodeBench-327`
- `BigCodeBench-322`
- `BigCodeBench-271`
- `BigCodeBench-277`
- `BigCodeBench-328`

The Qwen-first rejection filter had selected three rejected tasks:

- `BigCodeBench-327`
- `BigCodeBench-322`
- `BigCodeBench-271`

Those three were later confirmed as universal specialist failures.

With positive-only prior evidence, the new gate would still select two of the
three rejected tasks:

- `BigCodeBench-322`
- `BigCodeBench-271`

With both specialist-positive and universal-failure prior evidence, all three
rejected tasks scored below the `0` gate threshold and no specialist task was
selected.

## Interpretation

Positive specialist similarity alone is too permissive. Some Qwen-failed tasks
share broad features with specialist-positive rows, especially subprocess and
filesystem tasks, while still being unsolved by all specialists.

The useful acquisition memory is contrastive:

- near specialist-positive rows
- away from universal-failure rows
- only after Qwen has failed or been too slow

This is closer to the desired adaptive-memory shape: measured experience becomes
a local scoring prior that changes which model calls we buy next.

## Decision

Keep the selector and use it before future specialist screens.

Do not merge any training rows from this run; it is a retrospective acquisition
control experiment, not a new outcome collection run.

## Next Step

Run the gate prospectively on a fresh Qwen-first candidate batch. Only spend
specialist calls when:

- Qwen fails or is slow
- the solvability gate score is non-negative
- the task is not already covered by routing or outcome datasets

If the gate selects nothing, scan the next eligible offset or lower the threshold
only after inspecting the ranked scores.
