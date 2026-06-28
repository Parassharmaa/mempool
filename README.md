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
8. Qwen-small logits-head orchestrator training plan and rows.
9. Adaptive refresh records with quarantine/rollback discipline.

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

The intended neural orchestrator path is now explicit: use a small Qwen-family
backbone as the fast coordinator representation, attach logits heads for worker
selection, workflow selection, verifier probability, and abstention/fallback,
then train those heads against measured soft routing targets. The current
linear router remains the baseline, not the final architecture.

Prepared Qwen-small artifacts:

```text
research/models/20260628-qwen-small-logits-orchestrator-plan.json
research/datasets/20260628-qwen-small-logits-orchestrator-rows.jsonl
research/models/20260628-qwen-small-logits-orchestrator-smoke/qwen_logits_heads.pt
```

The plan currently reports `can_train_here: false` because this checkout does
not have the ML training stack installed. The current training-readiness audit
also shows that the active Python is `3.14.4`, which may not have stable PyTorch
wheels. Use Python 3.11 or 3.12 for the first local frozen-backbone head run:

```bash
python3.12 -m venv .venv-qwen-train
source .venv-qwen-train/bin/activate
python3 -m pip install -e '.[qwen-train]'
PYTHONPATH=src python3 tools/train_qwen_logits_orchestrator.py \
  --plan-output research/models/local-qwen-logits-plan.json \
  --rows-output research/datasets/local-qwen-logits-rows.jsonl \
  --output-dir research/models/local-qwen-logits-heads \
  --train
```

For a serious run, use GPU or Apple MLX access and keep the backbone frozen for
the first pass. Only try LoRA/backbone updates after the logits heads beat the
linear router on held-out task-level routing.

To check the current machine:

```bash
PYTHONPATH=src python3 tools/audit_qwen_training_readiness.py \
  --output research/models/local-qwen-training-readiness.json
```

The first local smoke checkpoint has been trained in `.venv-qwen-train`. To
prepare Hugging Face upload folders:

```bash
PYTHONPATH=src python3 tools/prepare_hf_release.py
```

After `hf auth login`, publish the dataset and model repos:

```bash
PYTHONPATH=src python3 tools/publish_hf_release.py
```

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
PYTHONPATH=src python3 tools/train_qwen_logits_orchestrator.py \
  --plan-output research/models/local-qwen-logits-plan.json \
  --rows-output research/datasets/local-qwen-logits-rows.jsonl
PYTHONPATH=src python3 tools/audit_qwen_training_readiness.py
```
