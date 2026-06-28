# 2026-06-27 Logits Router Leave-One-Out

## Question

Does the mixed-winner logits router show any held-out routing signal, or is it
only memorizing the six training examples?

## Setup

- Dataset: `research/datasets/20260627-mixed-winner-6task-routing.jsonl`
- Model: `research/models/20260627-mixed-winner-6task-logits-router.json`
- Report: `research/datasets/20260627-mixed-winner-6task-logits-router-report.json`
- Evaluation: leave-one-out retraining with the same hyperparameters as the full
  model

## Result

Training-set evaluation remains perfect:

- Target accuracy: 6/6
- pass@1: 6/6
- mean KL: 0.0009576783661455036

Leave-one-out evaluation is no longer perfect but shows useful signal:

- Target accuracy: 5/6
- pass@1: 5/6
- mean KL: 0.6531157944631536

The held-out miss is `BigCodeBench/454`. The LOO model predicts
`ollama-cloud-kimi-k2.7-code`, while the empirical target is
`ollama-cloud-qwen3-coder-480b`.

## Interpretation

This is the first generalization sanity check for the trainable router. The
result is encouraging but fragile: the model can recover most mixed Qwen/Kimi
targets from five examples, but it confuses one Qwen-only filesystem task with
the Kimi-favored region.

The next data target should be more Qwen-only and Kimi-favored filesystem tasks,
so the logits head can learn a boundary instead of leaning on sparse prompt
keywords.
