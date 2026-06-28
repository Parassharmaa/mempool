# Routing Record Schema

Routing records are measured local outcomes, not copied leaderboard scores. See
`docs/measured_routing_data.md` for why the orchestrator needs per-task,
per-worker, per-harness evidence.

Routing records group per-worker outcomes by task and provide a soft target
distribution for training a router.

Required fields:

- `task_id`
- `benchmark_id`
- `task_family`
- `prompt`
- `prompt_features`
- `workers`
- `target_worker_id`
- `target_distribution`

Each `workers` entry includes:

- `worker_id`
- `model`
- `passed`
- `score`
- `latency_ms`
- `cost_usd`
- `failure_mode`
- `reward`
- `target_probability`

The first reward formula is:

```text
reward = solved - latency_weight * normalized_latency - cost_usd
target = softmax(reward / temperature)
```
