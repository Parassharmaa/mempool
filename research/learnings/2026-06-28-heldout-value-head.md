# Held-Out Second-Attempt Value Head

Run tag: `20260628-heldout-value-head`

## Question

Was the learned second-attempt value head being evaluated with a true held-out
action-head loop, or only against leave-one-out worker predictions using a head
trained on all rows?

## Change

Added a true leave-one-out evaluator for the second-attempt value head:

1. hold out one substrate row and its multi-head prediction;
2. train the value head on the remaining rows;
3. select the threshold on the training rows only;
4. evaluate the held-out row;
5. aggregate all held-out rows into one workflow-action report.

The training CLI now keeps the previous deployable in-sample selected-head
evaluation under `in_sample_leave_one_out_predictions` and writes the stricter
held-out action-head result under `leave_one_out`.

## Result

On the 50-task latency-aware substrate:

- in-sample selected value head: pass@1 `0.86`, solvable pass@1 `0.915`,
  latency regret `3340.4 ms`, useful fallbacks `3/7`;
- true held-out value head: pass@1 `0.82`, solvable pass@1 `0.872`,
  latency regret `3109.82 ms`, useful fallbacks `1/6`.

The true held-out report contains 50 folds.

The corrected policy gate quarantined the candidate:

- target accuracy `0.600` below minimum `0.700`;
- target accuracy drop `0.183` exceeded allowed `0.100`;
- latency regret increase `2608.7 ms` exceeded allowed `500.0 ms`;
- absolute latency regret `3109.8 ms` exceeded maximum `750.0 ms`.

## Interpretation

The second-attempt value target is still directionally useful, but the learned
head is not yet reliable enough to promote. The earlier result was optimistic
because it trained the action head on all rows before evaluating the same rows.

This changes the next-step recommendation: do not spend the next loop tuning
thresholds. First increase the number of positive held-out fallback/value labels
or move the workflow head to a smoother shared representation with the worker
router.

## Artifacts

- `src/mempool/second_attempt_value.py`
- `tools/train_second_attempt_value_head.py`
- `research/models/20260628-heldout-value-head.json`
- `research/evals/results/20260628-heldout-value-head-report.json`
- `research/evals/results/20260628-heldout-value-head-policy-gate.json`
