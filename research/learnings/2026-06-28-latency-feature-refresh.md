# Latency Feature Refresh

## Question

Can a small prompt-feature expansion for random/statistics/math and tabular
filesystem tasks rescue the 26-task contrast-aware refresh candidate?

## Result

No. The feature expansion was tested and reverted because it did not improve
the promotion metric.

The prior 26-task candidate was already quarantined:

- best target accuracy: 0.692 at temperature 0.1
- best pass@1: 0.692 at temperature 0.1
- best low-latency candidate: temperature 0.05 with 0.654 target accuracy and
  800.3 ms mean latency regret

With the added random/statistics/math/tabular features, the clean rerun stayed
quarantined:

- best target accuracy: 0.654
- best pass@1: 0.692 at temperature 0.2
- best low-latency candidate: temperature 0.05 with 0.654 target accuracy and
  825.4 ms mean latency regret

The feature patch worsened the best target-accuracy result and slightly worsened
the best low-latency candidate, so it was reverted.

## Artifacts

- Prior selection: `research/datasets/20260628-contrast-aware-26task-temperature-selection.json`
- Feature-probe selection: `research/datasets/20260628-latency-feature-26task-temperature-selection-with-baseline.json`
- Feature-probe reports: `research/datasets/20260628-latency-feature-26task-reward-t*-logits-router-report.json`

## Decision

Discard the feature change, keep the artifact evidence, and keep the active
policy unchanged.

The useful signal is that the gap is not solved by adding shallow prompt
keywords. The next data step should collect more repeated examples around the
actual miss regions: Kimi-favored download/archive tasks, GLM-favored
filesystem/archive tasks, and Qwen-versus-GLM random/math tasks.
