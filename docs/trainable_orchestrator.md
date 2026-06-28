# Trainable Orchestrator Roadmap

This document refines the project around a small, learnable, real-time
orchestrator that can run locally and route work across any OpenAI-compatible
worker pool.

## Thesis

The coordinator should be a compact policy model, not a giant answer model. It
should learn when to use a cheap local worker, when to call a stronger model,
when to fan out, when to verify, and when to abstain.

The longer-term goal is live adaptation. The orchestrator should be small and
cheap enough to retrain frequently as new worker models, user patterns, memories,
and evaluation evidence arrive.

The first target is a sub-1B or small local coordinator path. We do not need to
begin with a text-generating planner. A smaller staged path is more measurable:

1. Heuristic router.
2. Trainable classifier or ranking head over task features.
3. Compact logits-head orchestrator trained from worker outcome data.
4. Optional text-generating workflow conductor once the dataset and reward loop
   are strong enough.

The current linear logits router is a baseline and a data-loop sanity check,
not the final orchestrator. The intended next model track is a small local
language-model backbone, such as a Qwen-small family model, with explicit
routing heads attached to its decision hidden state. The model should not
generate worker names as text for the fast path; it should emit action logits.

## Worker Pool

The worker pool should support:

- local models served by Ollama
- cloud-backed models served through Ollama
- any OpenAI-compatible endpoint
- future local inference servers
- verifier workers and tool-using agents

The adapter should expose model identity, latency, cost estimate, availability,
context limits, and observed performance history. The orchestrator should not
hard-code model names in policy logic.

Current local Ollama pool observed on 2026-06-27:

- `LiquidAI/lfm2.5-1.2b-instruct:latest`
- `qwen3:1.7b`
- `qwen3:4b`
- `qwen3:4b-instruct`
- `gemma4:12b-mlx`
- `gemma4:e2b`
- `gemma4:e4b`
- `gemma4:26b`
- `deepseek-v4-flash:cloud`
- `deepseek-v4-pro:cloud`
- `kimi-k2.5:cloud`

## Prior-Art Lessons To Adapt

The useful training pattern from recent learned-orchestration work is:

1. Run all candidate workers on tasks.
2. Score each worker with task-specific rewards.
3. Convert worker scores into soft routing targets.
4. Train the coordinator to emit worker-selection logits that match the soft
   distribution.
5. Improve with end-to-end rollout rewards once the supervised logits-head
   router is stable.

We should adapt the pattern, not copy the product shape. Our differentiators
should be transparency, local trainability, adapter neutrality, and explicit
cost-per-solved-task metrics.

The most important architectural lesson is to avoid making the fast
orchestrator generate text when it only needs to select a worker. The
low-latency orchestrator should compute a hidden state, apply a lightweight head,
and dispatch based on logits. Generative planning should be reserved for
multi-step workflows where the extra latency buys decomposition or verification.

## Dataset Plan

We need four dataset layers.

### Layer 1: Evaluated Benchmark Traces

For each task, run multiple workers and record:

- task id and prompt
- task family and features
- worker id
- output
- executable or verifier result
- cost and latency
- failure mode

This is the cleanest data because it has objective rewards.

### Layer 2: Local Agent Traces

Local Codex or Claude traces may be useful, but only after privacy filtering and
schema extraction. The traces should not be used raw.

Extract:

- task summary
- tool/action sequence shape
- final success/failure if inferable
- latency and iteration count if available
- whether verification happened
- high-level domain label

Avoid storing:

- secrets
- private file contents
- full user prompts unless explicitly approved
- credentials or API responses

### Layer 3: Synthetic Routing Tasks

Generate controlled tasks where the correct routing choice is known by design.
These are useful for bootstrapping but should not dominate training.

### Layer 4: Distilled Memory

Repeated successful traces should be distilled into training records. The aim is
to let useful memory gradually become learned routing behavior through adapters,
fine-tunes, or compact policy updates instead of remaining only as external
retrieval.

## Feedback Signals

Primary reward:

- `1` for solved task, `0` for failed task

Cost-aware reward:

```text
reward = solved - alpha * normalized_cost - beta * normalized_latency
```

Verifier-aware reward:

- reward successful cheap answer
- reward successful verified answer
- penalize unnecessary verifier calls on easy tasks
- penalize unverified failures on risky tasks

Router training label:

```text
target(worker | task) = softmax(worker_reward / temperature)
```

This keeps useful signal when two workers are close rather than forcing a brittle
single winner label.

## Training Path

### Phase 0: Measurement

Use the smoke benchmark and then a 10-task external smoke set to gather outcome
records across the Ollama pool.

### Phase 1: Non-LLM Router

Train a lightweight ranker/classifier from task features to worker choice.
Features can include:

- task family
- prompt length
- code/math/text flags
- required tool/verifier flag
- cheap embedding from a local embedding model
- constraint vector

This gives a fast baseline and a dataset sanity check.

### Phase 2: Small Neural Orchestrator

Train a small model that outputs logits for:

- worker distribution
- workflow kind
- verifier probability
- abstention probability

The first model can be much smaller than 1B parameters. A compact encoder or
small language-model backbone should expose a hidden state at a fixed decision
position; lightweight heads then map that hidden state to action logits. The
training objective should minimize KL divergence between measured soft routing
targets and the model's predicted distribution.

Preferred first implementation:

- use a Qwen-small style instruction backbone when local hardware allows it
- insert a fixed decision token or use the final hidden state after a compact
  routing prompt
- attach heads for worker distribution, workflow kind, verifier probability,
  abstention probability, and later turn-level actions
- train the heads first while freezing the backbone
- then evaluate LoRA/adapters on the backbone only after the heads beat the
  linear router on held-out task-level routing

This keeps the orchestrator close to the desired architecture while preserving a
cheap rollback path. If the Qwen-small head does not beat the linear baseline,
the measured dataset is probably still too small or too noisy.

### Phase 3: LoRA Fine-Tune

For Mac-local training, prefer MLX/MLX-LM LoRA first. For NVIDIA GPU training,
use Unsloth or a standard TRL stack depending on hardware and reward method.

Candidate base models:

- Qwen-small family model for routing hidden states plus logits heads
- local 1-2B instruction model for routing text
- local embedding model plus MLP head for fast classification
- 4B instruction model only if smaller models fail to separate task types

### Phase 4: End-To-End Policy Improvement

Once supervised routing works, optimize against rollout reward:

- benchmark pass/fail
- cost per solved task
- latency
- verifier usage
- abstention quality

Start with offline replay and bandit-style updates before expensive online RL.

### Phase 5: Frequent Refresh

Once updates are cheap and evaluation is reliable, refresh the small router on a
regular cadence. The aspirational target is hourly adaptation, gated by automatic
evaluation and rollback.

## Novelty Target

The goal is not merely to route to the strongest model. The novel target is:

- a local, trainable orchestrator
- explicit cost-quality-latency reward
- transparent decision ledger
- support for top hosted and local workers through the same adapter
- ability to learn from personal agent traces after privacy filtering
- conditional verification and abstention as trained actions
- repeated experience that can be fused into lightweight model updates
