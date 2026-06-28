# Learned Orchestration Survey

This is a living literature review for learned model routing, multi-agent
coordination, verifier-based workflows, and autonomous research loops.

## Anchor Papers And Systems

### Sakana Fugu Technical Report

- Link: https://arxiv.org/abs/2606.21228
- Mechanism: language-model orchestrators that dynamically choose agentic
  scaffolds over a stronger worker pool.
- Reported variants: a lower-latency single-worker router and a higher-quality
  multi-agent workflow composer.
- Training direction: large-scale fine-tuning, evolutionary algorithms, and
  reinforcement learning.
- Useful idea: treat orchestration itself as a learned capability, not a static
  product rule tree.
- Architectural detail to adapt: the low-latency variant attaches a lightweight
  selection head to a language-model backbone hidden state and emits one logit
  per worker. It dispatches from logits rather than generating routing text.
- Supervised target: run all workers, average measured rewards, convert rewards
  to a soft worker distribution, and train with KL divergence from the measured
  distribution to the predicted routing distribution.
- Open question: how much of the reported gain comes from routing, prompting,
  verifier passes, hidden worker composition, benchmark selection, or worker
  model strength?

### TRINITY: An Evolved LLM Coordinator

- Link: https://arxiv.org/abs/2512.04695
- Mechanism: compact coordinator plus lightweight head assigns roles over
  multiple turns: Thinker, Worker, Verifier.
- Optimization: separable CMA-ES under expensive rollout evaluations.
- Reported strength: strong coding, math, reasoning, and out-of-distribution
  generalization.
- Useful idea: role selection and worker selection can be learned jointly with a
  compact policy.
- Open question: whether evolutionary search remains sample-efficient when the
  worker pool, task distribution, or available tools change frequently.

### Mixture Of Agents

- Link: https://arxiv.org/abs/2406.04692
- Mechanism: layered model collaboration where outputs from one layer become
  context for later layers.
- Useful idea: diversity can be useful even when no single participant dominates.
- Limitation to test: fixed collaboration structures may waste cost on easy
  tasks.

### Smoothie: Label-Free Language Model Routing

- Link: https://arxiv.org/abs/2412.04454
- Mechanism: unsupervised or label-light routing among models.
- Useful idea: routing can be trained when ground-truth labels are scarce.
- Limitation to test: label-free signals may drift under distribution shift.

### MASRouter

- Link: https://arxiv.org/abs/2502.11133
- Mechanism: routing for multi-agent systems rather than only single-model
  selection.
- Useful idea: topology choice is part of routing.
- Limitation to test: topology search may become expensive without strong
  pruning.

### MoMA: Mixture of Models and Agents

- Link: https://arxiv.org/abs/2509.07571
- Mechanism: generalized routing over both direct model calls and agent
  workflows.
- Useful idea: the action space should include both "which model" and "which
  scaffold."
- Limitation to test: profiling datasets can become stale as models improve.

### Autonomous Research Loops

- Representative reference: https://github.com/karpathy/autoresearch
- Mechanism: a bounded loop where an agent edits a small target file, runs a
  fixed-budget experiment, keeps improvements, and records evidence.
- Useful idea: tight constraints make autonomous improvement auditable.
- Limitation to test: single-metric ratchets can overfit or miss interaction
  effects.

## Transferable Design Principles

- The coordinator should choose workflow shape, not only worker identity.
- Verification should be conditional; it is a cost, not a free lunch.
- Logs are training data. Design them before training policies.
- Fixed budgets make experiments comparable, but only within a known hardware
  and task context.
- Independent evaluation sets are necessary because routing policies can overfit
  benchmark distributions.
- Public claims should separate coordinator value from worker-model value.

## Improvement Directions

1. Transparent decision ledger by default.
2. Explicit user constraints over cost, latency, privacy, provider, and workflow
   depth.
3. Calibrated abstention when no available workflow is likely to succeed.
4. Dynamic verifier allocation based on uncertainty and task risk.
5. Open eval harness with task families, not only aggregate leaderboards.
6. Research-loop support that optimizes policies while preserving human review.
