# Specialist Feature Audit

Run tag: `20260628-specialist-feature-audit`

Goal: improve the 50-task multi-head orchestrator's specialist routing misses,
especially DeepSeek and GLM targets.

Finding:

The miss audit showed most errors still collapse specialist targets back to
Qwen:

- DeepSeek -> Qwen: 9 misses
- GLM -> Qwen/DeepSeek: 3 misses
- Kimi -> Qwen: 5 misses

Tested change:

Added composite task features for patterns visible in the misses, including web
parse, sklearn/dataframe, JSON statistics, download/archive, random filesystem,
and log CSV combinations. Regenerated a 50-task substrate and retrained the
latency-aware multi-head model.

Result:

- Previous latency-aware `w0.5` LOO: target accuracy `0.60`, pass@1 `0.80`,
  solvable pass@1 `0.851`, latency regret `1693.48 ms`.
- Specialist-composite candidate: target accuracy `0.58`, pass@1 `0.80`,
  solvable pass@1 `0.851`, latency regret `2771.90 ms`.

Decision: discard code change.

The composite features were removed from the shared feature extractor after the
experiment because they made the candidate worse. The generated artifacts remain
as evidence.

Next step:

Do not add hand-written composite features from small miss clusters. The better
path is more measured specialist data or a learned representation/feature
selection step that can separate DeepSeek/GLM-positive cases without
overfitting broad Qwen-pass regions.

Artifacts:

- `research/datasets/20260628-m5-specialist-features-substrate-50task.jsonl`
- `research/evals/results/20260628-m5-specialist-features-latency-w0p5-multihead-report.json`
- `research/evals/results/20260628-m5-specialist-features-latency-w0p5-policy-gate.json`
