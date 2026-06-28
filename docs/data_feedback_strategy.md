# Data And Feedback Strategy

See also `docs/measured_routing_data.md` for the distinction between public
benchmark scores, measured routing records, and held-out agentic trajectories.

## What We Need To Learn

The orchestrator needs examples of tasks, available workers, decisions, and
outcomes. A useful training row answers:

```text
Given this task and this worker pool, which workflow maximizes solved quality
under cost, latency, and policy constraints?
```

## Canonical Training Record

```json
{
  "task_id": "string",
  "task_family": "code_data",
  "prompt_features": {
    "length_chars": 1234,
    "requires_code": true,
    "requires_math": false,
    "requires_tools": false
  },
  "constraints": {
    "max_cost_usd": 0.05,
    "max_latency_ms": 20000,
    "require_verification": true
  },
  "worker_pool": [
    {
      "worker_id": "local-small",
      "adapter": "ollama",
      "observed_cost_usd": 0.0,
      "observed_latency_ms": 1200
    }
  ],
  "attempts": [
    {
      "workflow_kind": "route",
      "worker_ids": ["local-small"],
      "verifier_id": null,
      "passed": true,
      "reward": 0.91,
      "failure_mode": null
    }
  ]
}
```

## Feedback Sources

- executable benchmark tests
- unit tests after code generation
- exact-answer tasks
- LLM judge only when objective checking is unavailable
- user acceptance or rejection
- repair success after verifier feedback
- trace-level signals such as number of turns or tool failures

## Memory Distillation

Useful traces should not stay only in raw logs. Promote them through a controlled
pipeline:

1. raw ledger event
2. sanitized trace
3. distilled task-outcome record
4. training example
5. evaluated policy update

This lets repeated experience become learned behavior while preserving audit and
rollback.

## Using Local Agent Traces

Local traces can be valuable because they contain realistic task distributions.
They should first become metadata, not raw training text.

Recommended extraction:

- classify task domain
- summarize task intent
- count turns/tool calls
- detect whether tests or commands passed
- detect whether final answer claimed completion
- remove secrets and private content

Use these traces first for task-distribution modeling and synthetic benchmark
selection. Only later use sanitized text for supervised routing.
