# Non-Qwen Specialist Pressure

Run tag: `20260628-nonqwen-specialist-pressure`

## Question

Can focused non-Qwen specialist acquisition find GLM, DeepSeek, or Kimi target rows that are more useful than broad guarded acquisition, which recently added mostly Qwen-target evidence?

## Selector Change

The canonical specialist selector now supports `--max-specialist-rank`.

This allows acquisition to require that a specialist worker is ranked first by the active router, instead of merely being second behind Qwen. The motivating failure was the first pressure selection: all six selected tasks had Qwen as the top predicted worker and Kimi or DeepSeek as a close second. The screen found one specialist-positive task, but Qwen was faster in top-four comparison, so the row still targeted Qwen.

Changed files:

- `tools/select_canonical_specialist_batch.py`
- `tests/test_select_canonical_specialist_batch.py`

## Pressure-Only Screen

Initial selector command used specialist pressure without a rank cap.

Selected tasks:

- `bigcodebench-hard-BigCodeBench-15`
- `bigcodebench-hard-BigCodeBench-6`
- `bigcodebench-hard-BigCodeBench-130`
- `bigcodebench-hard-BigCodeBench-131`
- `bigcodebench-hard-BigCodeBench-288`
- `bigcodebench-hard-BigCodeBench-260`

All six had Qwen as top predicted worker and a non-Qwen specialist as second.

The non-Qwen screen produced `18` rows across GLM, DeepSeek, and Kimi. Only `BigCodeBench-260` passed, and it passed on all three specialists. Top-four comparison on that one task showed Qwen also passed and was fastest:

- GLM: pass, `6244 ms`
- DeepSeek: pass, `5673 ms`
- Kimi: pass, `3634 ms`
- Qwen: pass, `2400 ms`

Result: pressure-only selection found a solvable row, but not a non-Qwen target.

## Rank-1 Specialist Screen

The second selector pass used `--max-specialist-rank 1` and selected tasks where Kimi or DeepSeek was the active router's top predicted worker.

Selected tasks:

- `bigcodebench-hard-BigCodeBench-7`
- `bigcodebench-hard-BigCodeBench-206`
- `bigcodebench-hard-BigCodeBench-125`
- `bigcodebench-hard-BigCodeBench-266`
- `bigcodebench-hard-BigCodeBench-265`
- `bigcodebench-hard-BigCodeBench-113`

The non-Qwen screen produced `18` rows. Five of six tasks had at least one specialist pass:

- `BigCodeBench-7`: 2 specialist passes in screen; Kimi fastest at `3718 ms`
- `BigCodeBench-206`: 3 specialist passes; DeepSeek fastest at `5103 ms`
- `BigCodeBench-125`: all specialist fail
- `BigCodeBench-266`: 1 specialist pass; GLM at `7347 ms`
- `BigCodeBench-265`: 3 specialist passes; DeepSeek fastest at `5433 ms`
- `BigCodeBench-113`: 2 specialist passes; Kimi fastest at `7984 ms`

This is a materially better specialist-positive hit rate than the pressure-only screen.

## Top-Four Comparison

The five specialist-positive rank-1 tasks were compared against the top-four pool. Qwen passed all five and was fastest on every task:

- `BigCodeBench-7`: Qwen target, `2032 ms`; all three specialists failed in the comparison sample
- `BigCodeBench-206`: Qwen target, `2597 ms`; all specialists passed, but slower
- `BigCodeBench-266`: Qwen target, `2351 ms`; GLM and DeepSeek passed, Kimi failed
- `BigCodeBench-265`: Qwen target, `2086 ms`; all specialists passed, but slower
- `BigCodeBench-113`: Qwen target, `2366 ms`; all specialists passed, but slower

The converted routing dataset has `5` records and all `5` target `ollama-cloud-qwen3-coder-480b`.

## Interpretation

The rank-1 selector is useful: it raised the specialist-positive screen hit rate from `1/6` to `5/6`. That is worth keeping.

However, specialist-positive is not the same as specialist-target. Qwen remains so fast on these tasks that every top-four comparison row still targets Qwen by latency-adjusted reward. Merging these five rows into the active candidate dataset would likely repeat the dilution pattern from the offset236 run.

The next selector needs one more constraint: prefer tasks where the specialist is predicted top and the task has evidence that Qwen is likely to fail or be materially slower. Existing rank alone is not sufficient.

## Decision

Keep the selector feature and the outcome evidence. Do not merge the five Qwen-target rows into the policy-refresh dataset yet.

## Next Step

Add an acquisition filter or scorer that combines:

- specialist rank `1`
- specialist probability or margin
- prior Qwen-failure neighborhoods
- penalty for broad-pass filesystem rows where Qwen usually wins by latency

Then screen rank-1 candidates through non-Qwen specialists first, but only top-four compare candidates whose profile suggests Qwen may not dominate latency.
