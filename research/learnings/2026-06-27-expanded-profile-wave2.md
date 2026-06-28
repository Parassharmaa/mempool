# Expanded Profile Wave 2

## Question

Can the next expanded-profile BigCodeBench offset add enough stable routing rows
to materially close the gap from 38 rows to the 50-row small-orchestrator data
threshold?

## Result

It adds useful evidence, but only two stable rows.

The offset-30 expanded-profile canonical scan found 18 locally runnable
canonical-pass tasks in 30 rows. After excluding every task already present in
the 38-task candidate dataset or any prior raw outcome file, 7 fresh tasks
remained:

- `BigCodeBench/445`
- `BigCodeBench/239`
- `BigCodeBench/302`
- `BigCodeBench/267`
- `BigCodeBench/241`
- `BigCodeBench/341`
- `BigCodeBench/443`

The repeated top-four worker comparison produced 56 outcome rows with the
expanded evaluator package provenance. The outcome audit passed.

Merge filtering kept 2 rows and dropped 5 all-fail rows:

- kept `BigCodeBench/302`
- kept `BigCodeBench/445`
- dropped `BigCodeBench/239`
- dropped `BigCodeBench/241`
- dropped `BigCodeBench/267`
- dropped `BigCodeBench/341`
- dropped `BigCodeBench/443`

Both kept rows are Qwen latency targets. This moves the stable candidate dataset
from 38 rows to 40 rows.

## Router Decision

The 40-task candidate router was quarantined:

- LOO target accuracy: 0.500.
- LOO solvable pass@1: 0.811.
- LOO mean latency regret: 4208 ms.

The active 23-task policy should remain unchanged.

## Decision

Keep the two new stable rows as measured evidence. The expanded dependency
profile remains useful, but offset 30 is failure-heavy under the current
top-four worker pool. Continue acquisition toward 50 rows by scanning another
expanded-profile offset or by adding the next low-risk dependency packages, and
prioritize batches likely to produce non-Qwen specialist wins instead of only
Qwen latency rows.
