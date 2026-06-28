# 2026-06-27 Logits Router Prototype

## Question

Can the repeated-sample routing dataset train a small local logits-emitting
orchestrator artifact?

## Setup

- Dataset:
  `research/datasets/20260627-cloud-repeatability-smoke-routing.jsonl`
- Model:
  `research/models/20260627-repeatability-logits-router.json`
- Report:
  `research/datasets/20260627-repeatability-logits-router-report.json`
- Router type: linear softmax head over prompt features
- Workers: DeepSeek V4 Pro, GLM 5.2, Kimi K2.7 Code, Qwen3 Coder 480B
- Features: 45 prompt/task features

## Result

The first raw training attempt was unstable because unscaled prompt-length
features dominated the SGD update. After adding feature scaling inside
`src/mempool/logits_router.py`, the tiny logits head trained cleanly:

- Initial mean KL: 0.23973074583525358
- Final mean KL: 0.000003102083886473775
- Evaluation mean KL: 0.000003066141521569387
- Target accuracy: 2/2
- pass@1: 2/2

The model predicts `ollama-cloud-qwen3-coder-480b` for both repeated-sample
tasks, matching the empirical soft-target argmax.

## Interpretation

This is not yet a generalizing orchestrator. It is a proof that the system can
now move from measured cloud outcomes to repeated-sample soft targets to a local
logits-emitting policy artifact.

The next useful step is not adding a larger model immediately. It is expanding
the repeated-sample dataset so the logits head sees tasks where different
workers win under reliability, latency, or cost-adjusted rewards.
