# Trainable Local Orchestrator

## Decision

Aim for a small, learnable orchestrator that runs locally and routes over an
OpenAI-compatible worker pool. The orchestrator should optimize workflow choice,
worker choice, verifier use, and abstention under quality, cost, latency, and
policy constraints.

The orchestrator should also be live enough to absorb repeated experience. Raw
memory starts in the ledger, then high-value traces are distilled into training
records and eventually fused into lightweight policy updates.

## First Implementation Path

1. Use Ollama as the first live worker pool.
2. Collect evaluated task-worker outcome records.
3. Train a lightweight classifier/ranker before fine-tuning any language model.
4. Move to MLX/MLX-LM LoRA for Mac-local small-model fine-tuning.
5. Use Unsloth or TRL later when NVIDIA GPU access is available.

## Why This Could Be Novel Enough

The primitives exist elsewhere, but this project can differentiate by combining:

- local trainability
- transparent outcome ledger
- adapter-neutral top/local worker pool
- cost-per-solved-task as a primary reward
- conditional verification and abstention as trained policy actions
- privacy-filtered local agent traces as task-distribution data
- frequent retraining or adapter refresh as the system gathers evidence

## Current Evidence

Ollama is available locally through an OpenAI-compatible endpoint and currently
lists a mixed local/cloud pool including small local models, larger local models,
and stronger cloud-backed models.

The first smoke signal remains green: rule routing matches the strong fixture
pass rate while lowering cost per solved task.
