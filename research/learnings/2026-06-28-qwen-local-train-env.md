# Qwen Local Training Environment

Run tag: `20260628-qwen-local-train-env`

## What Changed

Created an ignored Python 3.11 training environment at `.venv-qwen-train`,
installed the optional `qwen-train` dependencies, and ran the first real
Qwen-small frozen-head orchestrator smoke.

The active project Python remains 3.14, but the training env uses:

- Python 3.11.14
- PyTorch 2.12.1
- Transformers 5.12.1
- Apple MPS available

## Training Smoke

Command shape:

```bash
PYTHONPATH=src .venv-qwen-train/bin/python tools/train_qwen_logits_orchestrator.py \
  --plan-output research/models/20260628-qwen-small-logits-orchestrator-smoke-plan.json \
  --rows-output research/datasets/20260628-qwen-small-logits-orchestrator-smoke-rows.jsonl \
  --output-dir research/models/20260628-qwen-small-logits-orchestrator-smoke \
  --epochs 1 \
  --batch-size 1 \
  --max-length 512 \
  --train
```

Result:

- trained rows: 66
- epoch count: 1
- loss: 4.53756536136974
- checkpoint: `research/models/20260628-qwen-small-logits-orchestrator-smoke/qwen_logits_heads.pt`
- train report: `research/models/20260628-qwen-small-logits-orchestrator-smoke/train_report.json`
- eval report: `research/models/20260628-qwen-small-logits-orchestrator-smoke/eval_report.json`

The checkpoint stores only the explicit routing heads, not the Qwen base model
weights.

Smoke evaluation on the same 66 rows:

- worker accuracy: 0.3181818181818182
- workflow accuracy: 0.8636363636363636
- mean worker loss: 1.4604794980788773
- mean workflow loss: 0.41937757511605567

This confirms the checkpoint is loadable and evaluable, but the one-epoch head
is weak and should not be promoted.

## Hugging Face Export

Prepared local Hugging Face export folders:

- `research/hf_export/qwen-logits-smoke/dataset`
- `research/hf_export/qwen-logits-smoke/model`

Remote upload is pending Hugging Face authentication. Once `hf auth login` is
complete, run:

```bash
PYTHONPATH=src python3 tools/publish_hf_release.py
```

## Decision

Keep the smoke checkpoint and export tooling. This is the first actual
small-LLM orchestrator training artifact, but it is still a smoke run and not a
promoted policy.
