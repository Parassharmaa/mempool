# Preserve-Accuracy Refresh Profile

## Question

Can the stricter repeat-run promotion rule be made reusable instead of relying
on manually remembered threshold flags?

## Result

Added a named refresh profile: `preserve_accuracy`.

The profile derives thresholds from the baseline report:

- minimum LOO target accuracy becomes the baseline target accuracy
- maximum LOO target-accuracy drop becomes `0.0`
- minimum solvable pass@1 becomes the baseline solvable pass@1 when available
- maximum mean latency regret becomes the baseline mean latency regret

This keeps the older `tolerant` profile available for exploratory refreshes,
while making accuracy-preserving refreshes explicit and reproducible.

## Artifact

Re-ran the 24-task router-miss repeat candidate with:

```bash
--promotion-profile preserve_accuracy
```

Output:

- `research/datasets/20260628-router-miss-repeat-24task-profiled-temperature-selection.json`

Decision stayed `quarantine`, with profile metadata attached to the selection
and each candidate refresh.

## Decision

Keep the profile. Future repeat-stabilization or adaptive-memory refreshes
should use `preserve_accuracy` unless the experiment explicitly says it is
testing a latency-for-accuracy tradeoff.
