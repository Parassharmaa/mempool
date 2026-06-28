# Architecture Sketch

The first implementation should stay deliberately small.

## Components

- Task: normalized user request plus constraints and evaluation hints.
- Pool: available workers, capabilities, costs, latency estimates, policy
  constraints, and model/provider metadata.
- Coordinator: chooses a workflow for a task.
- Runner: executes the workflow and captures artifacts.
- Verifier: checks candidate outputs against task-specific criteria.
- Synthesizer: produces the final answer or artifact from accepted intermediate
  work.
- Ledger: structured event log for decisions, prompts, outputs, scores, and
  failures.

## Minimal Workflow Types

- direct: one worker answers.
- route: coordinator selects one worker.
- compare: multiple workers answer independently, then a verifier selects.
- decompose: coordinator splits the task into subtasks.
- repair: verifier finds issues and sends targeted fixes.

## First Prototype Contract

The prototype should accept a task JSON object, execute a selected workflow, and
write a JSONL ledger. The ledger is the product's memory and the training data
for future coordinator improvement.

## Trainable Orchestrator Direction

The coordinator should become a small trainable policy. It should not answer the
task itself. Its primary low-latency form should be a logits-emitting
orchestrator:

- encode the normalized task with a compact language-model or encoder backbone
- take a hidden state at a fixed decision position
- apply lightweight heads that emit action logits
- dispatch immediately without autoregressively generating a plan

- workflow kind
- worker distribution
- verifier use
- abstention probability

The first trainable version can still be a lightweight classifier or ranker for
sanity checking, but the project target is a small orchestrator whose worker
selection is produced by learned logits. Text-generating workflow plans are a
later, higher-latency mode for multi-agent workflows, not the default routing
path.

The current linear logits router is therefore a baseline, not the destination.
The next neural path should use a small local language-model backbone, such as a
Qwen-small family model, and attach explicit heads to a fixed decision hidden
state. The backbone reads the task and compact state summary; the heads emit
worker logits, workflow logits, verifier probability, abstention probability,
and eventually turn-level action and memory-update logits.

Fast routing should remain a classification decision over normalized worker
ids. The model can use language-model representations internally, but it should
not rely on free-form text generation to decide which worker to call.

## Worker Pool Requirements

Workers should be accessed through adapters. An adapter can wrap a hosted model,
an open-weight model, a local specialist, or a tool-using agent. The coordinator
should see normalized capabilities and constraints rather than provider-specific
details.

At minimum, each worker entry should expose:

- stable worker id
- modality and task strengths
- context limits
- expected latency
- expected cost
- data-handling constraints
- tool access
- availability and rate-limit state
