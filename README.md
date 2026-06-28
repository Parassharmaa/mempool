# mempool

`mempool` is an open research project for learned model orchestration.

The goal is to build a small, auditable coordinator that can decide how to route,
compose, verify, and improve work across a pool of capable agents or models. The
system should be independent, reproducible, and measurable: every orchestration
decision should leave enough evidence for later analysis.

The worker pool is intentionally open-ended. It should support top-tier hosted
models, strong open-weight models, specialist local models, tool-using agents,
and future systems through the same interface.

## Direction

- Learn routing policies from task outcomes, not fixed hand-written rules.
- Support query-adaptive workflows: direct answer, routed answer, parallel
  comparison, verifier pass, or multi-step decomposition.
- Track quality, latency, cost, uncertainty, and provenance for every run.
- Make frontier-capable workers usable without hard-coding the project around
  any one provider.
- Keep research loops bounded so autonomous experiments produce reviewable diffs
  and comparable metrics.
- Prefer transparent evaluation over benchmark theater.

## Repository Layout

- `src/` - implementation code.
- `tools/` - local scripts for data prep, evaluation, and experiment control.
- `docs/` - project-facing design notes and architecture.
- `research/` - ideas, experiments, evals, and run logs.
- `literature/` - paper notes, surveys, and outside context.
- `external_repos/` - third-party references, kept out of the core project.

## Current Status

The repository now has an end-to-end measured-data loop:

1. Ollama/OpenAI-compatible worker evaluation.
2. BigCodeBench-Hard task materialization and executable evaluation.
3. Repeated outcome JSONL collection.
4. Routing dataset conversion with worker rewards and soft targets.
5. Lightweight logits-router training and promotion gates.
6. Multi-head task-level orchestrator substrate export.
7. A trained local multi-head orchestrator candidate.
8. Adaptive refresh records with quarantine/rollback discipline.

The current trained task-level orchestrator artifact is:

```text
research/models/20260628-m5-current-task-66task-multihead.json
```

Its source substrate is:

```text
research/datasets/20260628-m5-current-task-66task-substrate.jsonl
```

The model is a research checkpoint, not a promoted production policy. The
latest leave-one-out result still shows that specialist target accuracy and
latency regret need improvement before promotion.

For bounded autonomous improvement runs, see
`.agents/skills/research-loop/SKILL.md` and
`research/programs/orchestration_loop.md`.

## Quick Commands

```bash
PYTHONPATH=src python3 -m unittest discover tests
PYTHONPATH=src python3 tools/train_multi_head_orchestrator.py \
  --substrate research/datasets/20260628-m5-current-task-66task-substrate.jsonl \
  --model-output research/models/local-multihead.json \
  --report-output research/evals/local-multihead-report.json
```
