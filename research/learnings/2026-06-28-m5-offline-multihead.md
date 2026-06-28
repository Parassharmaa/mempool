# M5 Offline Multi-Head Orchestrator

Run tag: `20260628-m5-offline-multihead`

## What Changed

Added a dependency-free local trainer for a small multi-head orchestrator:

- Module: `src/mempool/multi_head_orchestrator.py`
- CLI: `tools/train_multi_head_orchestrator.py`
- Test: `tests/test_multi_head_orchestrator.py`

The trainer consumes the 50-row supervised substrate and learns the M5 action
surface:

- worker softmax head
- workflow-kind softmax head
- verifier-probability sigmoid head
- abstain-probability sigmoid head

This is a local linear bridge model, not the final sub-1B or LoRA backbone. Its
purpose is to validate the training/evaluation path and create a gateable local
orchestrator artifact.

## Artifact

- Substrate: `research/datasets/20260628-m5-small-orchestrator-substrate-50task.jsonl`
- Model: `research/models/20260628-m5-offline-multihead-50task.json`
- Report: `research/evals/results/20260628-m5-offline-multihead-50task-report.json`
- Gate: `research/evals/results/20260628-m5-offline-multihead-50task-policy-gate.json`

## Result

In-sample metrics:

- Worker target accuracy: `0.740`
- Workflow accuracy: `0.940`
- Pass-at-1: `0.900`
- Mean latency regret: `1516.8 ms`

Leave-one-out metrics:

- Worker target accuracy: `0.620`
- Workflow accuracy: `0.940`
- Solvable pass-at-1: `0.872`
- Mean latency regret: `3609.6 ms`

The policy gate quarantined the candidate:

- LOO target accuracy is below the `0.700` promotion floor.
- LOO target accuracy drops too far relative to the active 23-task policy.
- LOO latency regret is far above the `750 ms` ceiling.

## Interpretation

The local trainable multi-head path is now real end to end: substrate ->
training -> artifact -> leave-one-out evaluation -> policy gate.

The model is not promotable. It reproduces the main weakness of the 50-task
logits router: broad Qwen-biased generalization and poor latency-specialist
selection. The workflow head is much easier because the substrate currently has
only 3 verify-then-fallback rows, so high workflow accuracy should not be
overinterpreted.

## Next Step

Do not refresh the active policy. The next improvement should either add
stronger specialist-sensitive features or train a slightly richer local
candidate against the same substrate, then compare against this multi-head
baseline and the active 23-task policy gate.
