# Current Status

`mempool` has reached the first trained task-level orchestrator checkpoint.

## Data

- Raw measured outcome rows on disk: 2029 before the latest active-rescue run.
- Unique task ids in measured outcomes: 199 before the latest active-rescue run.
- Clean current task-level substrate: 66 records.
- Current substrate target mix:
  - DeepSeek: 9
  - GLM: 4
  - Kimi: 12
  - Qwen: 41

## Current Task-Level Orchestrator

Artifact:

```text
research/models/20260628-m5-current-task-66task-multihead.json
```

Source substrate:

```text
research/datasets/20260628-m5-current-task-66task-substrate.jsonl
```

Training-set metrics:

- target accuracy: 0.6818
- pass@1: 0.8333
- solvable pass@1: 0.9649
- mean latency regret: 1575.7 ms

Leave-one-out metrics:

- target accuracy: 0.5606
- pass@1: 0.7576
- solvable pass@1: 0.8772
- mean latency regret: 2463.5 ms

Decision: trained and kept as a checkpoint, but not promoted as the active
policy.

## Next

Before turn-level training, improve task-level orchestrator reliability:

- collect more non-Qwen specialist targets
- reduce latency regret against the probe-gated operational reference
- improve leave-one-out target accuracy
- keep all refreshes gated and reversible

Turn-level agentic training should come after this task-level checkpoint is
stronger. The intended design is to predict each agentic turn's worker,
workflow, verifier, stop/repair/switch action, and memory-update decision from
trajectory state.

The turn-level substrate builder is now code-ready as a deferred path:
`tools/build_agentic_turn_substrate.py` converts sanitized trajectory summaries
into per-turn examples, while rejecting raw terminal output. It should remain a
data-contract scaffold until real multi-turn trajectories exist.

The current checkpoint can also be queried locally without retraining:

```bash
PYTHONPATH=src python3 tools/predict_multi_head_orchestrator.py \
  --model research/models/20260628-m5-current-task-66task-multihead.json \
  --prompt "Write Python code that reads files from a directory." \
  --task-family bigcodebench_hard \
  --categories filesystem,text \
  --libraries pathlib
```

The checkpoint can also route and prepare an OpenAI-compatible worker call:

```bash
PYTHONPATH=src python3 tools/run_orchestrated_prompt.py \
  --dry-run \
  --model research/models/20260628-m5-current-task-66task-multihead.json \
  --worker-pool research/evals/ollama_cloud_worker_pool_top4.json \
  --prompt "Write Python code that reads files from a directory." \
  --task-family bigcodebench_hard \
  --categories filesystem,text \
  --libraries pathlib
```

First live orchestrated execution is recorded:

- `research/evals/20260628-live-orchestrated-execution.json`
- `research/evals/20260628-live-orchestrated-execution-outcome.jsonl`

That run selected `ollama-cloud-qwen3-coder-480b`, executed
`qwen3-coder:480b`, and received `Hello from mempool!` in 2057 ms.

The first live prompt-set comparison is also recorded:

- `research/evals/20260628-orchestrated-promptset-comparison.json`
- `research/evals/20260628-orchestrated-promptset-comparison-outcomes.jsonl`

It ran 3 prompts under both the trained-orchestrator policy and a fixed
`ollama-cloud-qwen3-coder-480b` baseline. The orchestrator selected Qwen for all
3 prompts, so this validates the repeatable comparison path but not routing
diversity.

A second live prompt-set comparison targeted measured non-Qwen substrate regions:

- `research/evals/20260628-nonqwen-promptset-comparison-prompts.json`
- `research/evals/20260628-nonqwen-promptset-comparison.json`
- `research/evals/20260628-nonqwen-promptset-comparison-outcomes.jsonl`

This time the orchestrator selected `GLM`, `Kimi`, and `DeepSeek` once each,
matching the selected non-Qwen target regions. The fixed Qwen baseline was still
faster on this small live sample, so the result is routing-diversity evidence,
not a promotion signal.

That same non-Qwen comparison now has evaluator-backed pass/fail scoring:

- `research/evals/20260628-nonqwen-promptset-comparison-evaluation.jsonl`
- `research/evals/20260628-nonqwen-promptset-comparison-evaluation-report.json`

The trained orchestrator solved 1/3 tasks while the fixed
`ollama-cloud-qwen3-coder-480b` baseline solved 0/3. The dependency-aware
evaluation schema also shows that only one row per policy was evaluable in the
current local environment: the orchestrator was 1/1 on evaluable rows and fixed
Qwen was 0/1. The positive row was
`bigcodebench-hard-BigCodeBench-339`, where the orchestrator selected
`ollama-cloud-glm-5.2` and passed all tests. The remaining rows are now marked
as `missing_eval_dependency` because `matplotlib` and `pandas` were
unavailable, so this is an encouraging quality signal but still not a promotion
signal.

To prevent future comparison runs from producing dependency-coupled labels, a
current-environment evaluable task manifest is now recorded:

- `research/evals/20260628-current-env-evaluable-tasks.json`
- `research/evals/20260628-current-env-evaluable-tasks-report.json`

Across the 40-task union used by the non-Qwen comparison source files, 27 tasks
are import-preflight evaluable in the current Python environment and 13 are
excluded because they require unavailable imports such as `bs4`, `matplotlib`,
`numpy`, `pandas`, `requests`, or `sklearn`. Use this manifest, or a pinned
benchmark environment, before converting future live rows into router-training
labels.

The Qwen-small logits-head orchestrator path is now implemented as a source
training path:

- `src/mempool/qwen_logits_orchestrator.py`
- `tools/train_qwen_logits_orchestrator.py`
- `research/models/20260628-qwen-small-logits-orchestrator-plan.json`
- `research/datasets/20260628-qwen-small-logits-orchestrator-rows.jsonl`

The generated plan has 66 task-level substrate records and four worker labels.
It reports `can_train_here: false` because this project environment does not
currently have `torch`, `transformers`, `mlx`, or `mlx_lm` installed. The real
training command is wired, but the first Qwen-small head training run needs the
ML stack installed or GPU/MLX access.

The Qwen training readiness audit is recorded at:

- `research/models/20260628-qwen-training-readiness.json`

It reports macOS arm64 with Python `3.14.4`, no `torch`, no `transformers`, no
`mlx`, and no `mlx_lm`. The recommended local path is a Python 3.11 or 3.12
environment with `.[qwen-train]`; the recommended serious path is GPU or Apple
MLX access.

A clean current-environment live comparison is now recorded:

- `research/evals/20260628-evaluable-live-comparison-prompts.json`
- `research/evals/20260628-evaluable-live-comparison.json`
- `research/evals/20260628-evaluable-live-comparison-outcomes.jsonl`
- `research/evals/20260628-evaluable-live-comparison-evaluation.jsonl`
- `research/evals/20260628-evaluable-live-comparison-evaluation-report.json`

All rows were locally evaluable. The trained orchestrator solved 1/3 tasks while
the fixed `ollama-cloud-qwen3-coder-480b` baseline solved 0/3. The positive was
again `bigcodebench-hard-BigCodeBench-339`, where the router selected
`ollama-cloud-glm-5.2` and passed all tests. This is clean specialist-routing
evidence, but still not a promotion signal because the sample is tiny and the
router's mean latency was 7507 ms versus 2897 ms for fixed Qwen.

The first real Qwen-small frozen-head training smoke has completed in an
isolated Python 3.11 environment:

- `research/models/20260628-qwen-training-readiness-py311.json`
- `research/datasets/20260628-qwen-small-logits-orchestrator-smoke-rows.jsonl`
- `research/models/20260628-qwen-small-logits-orchestrator-smoke-plan.json`
- `research/models/20260628-qwen-small-logits-orchestrator-smoke/train_report.json`
- `research/models/20260628-qwen-small-logits-orchestrator-smoke/qwen_logits_heads.pt`

The smoke trained explicit worker, workflow, verifier, and abstain heads for one
epoch over 66 rows using `Qwen/Qwen2.5-0.5B-Instruct` as the frozen backbone.
The final smoke loss was 4.53756536136974. This is a real training artifact,
but not yet a promoted policy.

The trained heads were reloaded and evaluated on the same 66 rows:

- worker accuracy: 0.3181818181818182
- workflow accuracy: 0.8636363636363636
- mean worker loss: 1.4604794980788773
- mean workflow loss: 0.41937757511605567

The path works end to end, but the one-epoch head is weak and needs longer
training plus held-out evaluation before it can compete with the linear router.

The earlier smoke export was superseded by the v0 release under
`research/hf_export/qwen-logits-v0/`, published to the Hugging Face dataset and
model repos listed below.

A deterministic held-out split for Qwen logits rows now exists:

- `research/datasets/20260628-qwen-small-logits-orchestrator-split-train.jsonl`
- `research/datasets/20260628-qwen-small-logits-orchestrator-split-heldout.jsonl`
- `research/datasets/20260628-qwen-small-logits-orchestrator-split-manifest.json`

The split is 53 train rows and 13 held-out rows. A one-epoch split smoke reached
0.4717 worker accuracy on train and 0.3077 on held-out. This confirms the
held-out gate works, but the Qwen head is not yet competitive.

A 40-epoch Qwen logits-head v0 run has now completed on Lightning AI L40S GPU:

- checkpoint: `research/models/20260628-qwen-small-logits-orchestrator-full-gpu-l40s/qwen_logits_heads.pt`
- full manifest: `research/models/20260628-qwen-small-logits-orchestrator-full-gpu-l40s/full_run_manifest.json`
- train worker accuracy: 0.7358
- train workflow accuracy: 0.9434
- held-out worker accuracy: 0.5385
- held-out workflow accuracy: 0.7692

The checkpoint is published as:

- dataset: `https://huggingface.co/datasets/blazeofchi/mempool-qwen-logits-orchestrator-rows`
- model: `https://huggingface.co/blazeofchi/mempool-qwen-logits-orchestrator-v0`

The first HF dataset upload included auxiliary JSON files next to the row JSONL
files, which made the Hub dataset viewer fail with a schema cast error. The
dataset export now keeps only the train and held-out row JSONL files plus the
dataset card; plan, readiness, eval, and sample-prediction JSON stay in the
model repo.
