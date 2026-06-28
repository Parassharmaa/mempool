# Prospective Solvability Gate

Run tag: `20260628-prospective-solvability-gate`

## Question

Does the Qwen-first plus specialist-solvability gate work prospectively on a
fresh BigCodeBench eligible slice?

The previous run proved the gate retrospectively. This run tested it on fresh
tasks from offset `359`.

## Fresh Eligible Slice

Scanned `bigcode/bigcodebench` normal tasks from offset `359`.

Result:

- scanned rows: `80`
- eligible tasks: `13`
- next offset: `439`

The slice was mostly filesystem, subprocess, network, and general standard
library tasks. Rejected canonical rows were still dominated by missing packages
such as `pandas`, `numpy`, `matplotlib`, `sklearn`, `cv2`, and related local
environment gaps.

## Candidate Selection

The Qwen-fail contrast selector was run on the fresh offset359 tasks with
rank-1 specialist filtering and prior Qwen-fail/Qwen-fast seed ids from the
offset236 boundary dataset.

It selected two fresh candidates:

- `BigCodeBench-365`: filesystem/json/random/collections, active router
  predicted DeepSeek top with Kimi second
- `BigCodeBench-412`: filesystem/base64/unicodedata/json, active router
  predicted DeepSeek top with Kimi second

## Qwen-First Screen

Qwen was run first:

- `BigCodeBench-365`: failed, `3140 ms`
- `BigCodeBench-412`: passed, `2666 ms`

The Qwen-first filter correctly removed `412` from specialist consideration.

## Solvability Gate And Specialist Spend

Using the initial specialist-solvability gate with:

- specialist-positive prior outcomes
- prior universal-failure outcomes
- min gate score `0`

`BigCodeBench-365` was selected for specialist screening:

- positive similarity: `11.4679`
- universal-failure similarity: `5.4673`
- score: `9.0832`

The specialist screen spent three calls:

- GLM failed, `4362 ms`
- DeepSeek failed, `3546 ms`
- Kimi failed, `3639 ms`

So the selected row was still a universal specialist failure.

## Policy Tightening

The miss showed that subtracting the universal-failure prior is not enough when
a task is also very close to specialist-positive filesystem/json rows.

Added an optional hard cap to `tools/select_solvable_worker_rejections.py`:

- `--max-universal-fail-similarity`

The selector now also reports:

- scored rejected rows
- rows rejected by score
- rows rejected by universal-failure cap

With the no-leakage prior evidence available before the specialist spend, a
cap of `4` would have rejected `BigCodeBench-365`:

- rejected task count: `1`
- rejected by universal-failure cap: `1`
- selected task ids: `[]`

After adding the new specialist failure as memory, the same row scores negative:

- universal-failure similarity: `11.4313`
- score: `-5.8268`

## Interpretation

The Qwen-first stage worked: it avoided specialist spend on a Qwen-fast task.

The first solvability gate improved on raw Qwen failure, but remained too
permissive for ambiguous filesystem/json neighborhoods. A hard cap on
universal-failure similarity is now justified because it would have prevented
this exact spend using only prior evidence.

This is the adaptive-memory loop in miniature:

1. run a bounded acquisition policy
2. measure the miss
3. convert the miss into a stronger local prior
4. make the next model-call decision cheaper and more selective

## Decision

Keep the Qwen-first plus solvability gate, but future prospective runs should use
both:

- `--min-gate-score 0`
- `--max-universal-fail-similarity 4`

Do not merge a routing row from this run. The only new outcome is a
high-confidence universal-failure memory row.

## Next Step

Run the capped gate on the next fresh eligible slice. If it selects no tasks,
scan the next offset instead of lowering the cap immediately.
