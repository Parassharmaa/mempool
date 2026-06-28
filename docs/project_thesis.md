# Project Thesis

Modern AI systems are no longer just single models answering isolated prompts.
They are increasingly shaped by routing, scaffolding, verification, memory,
tools, and repeated interaction with the environment.

This project explores learned orchestration as a first-class research object. A
coordinator should learn when to answer directly, when to delegate, when to ask
for multiple independent attempts, when to verify, and when to stop.

## Core Hypothesis

A compact coordinator can improve reliability and efficiency by learning
task-specific orchestration policies over a heterogeneous worker pool, provided
that its decisions are trained and evaluated with strong provenance, calibrated
uncertainty, and bounded cost.

The intended worker pool includes the strongest available models and agents, but
the coordinator must not depend on any single provider, brand, or fixed model
list. Its job is to learn how to use whatever high-quality workers are available
under the user's constraints.

## What Should Be Better

- Transparency: every decision should be reconstructable from logs.
- Controllability: users should be able to constrain cost, providers, latency,
  data policy, and allowed workflow shapes.
- Portability: new top-tier or specialist workers should be added by adapter,
  not by rewriting the coordinator.
- Adaptivity: the coordinator should route based on the actual task, not only
  static benchmark categories.
- Evaluation: success should include quality, cost, latency, robustness, and
  graceful degradation.
- Research velocity: autonomous loops should propose and test improvements while
  keeping changes reviewable.

## Early Research Questions

1. Which task features best predict the right orchestration pattern?
2. When does verification help enough to justify its cost?
3. Can the coordinator learn useful policies from weak or noisy preference
   signals?
4. How can the system detect that no available worker is likely to solve a task?
5. How should routing change under hard cost, latency, privacy, or provenance
   constraints?
