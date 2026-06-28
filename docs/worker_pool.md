# Worker Pool Design

The worker pool should make strong models and specialist agents available
without binding the project to any single vendor, framework, or model name.

## Adapter Shape

Each adapter should implement the same conceptual operations:

- describe capabilities and constraints
- accept a normalized task payload
- return a normalized result payload
- expose cost, latency, and usage metadata
- surface recoverable and non-recoverable failures

## Worker Classes

- Hosted top-tier model: high capability, higher cost, variable data policy.
- Open-weight model: controllable deployment, lower policy risk, local hardware
  constraints.
- Specialist model: narrow domain strength such as code, math, extraction, or
  verification.
- Tool-using agent: can inspect files, run commands, call APIs, or operate an
  environment.
- Verifier: optimized for critique, tests, rubric checks, or consistency.

## Routing Metadata

The coordinator should not need provider-specific code. It should see:

- task strengths
- context size
- modalities
- cost estimate
- latency estimate
- availability
- data policy
- tool permissions
- historical win rate by task family
- calibration and failure history

## Upgrade Principle

Adding a better worker should require a new adapter configuration and eval run,
not coordinator rewrites. If the coordinator cannot benefit from a better worker
through the same interface, the interface is too weak.

## Current Adapter Target

The first real adapter target is OpenAI-compatible chat completions. This covers
local Ollama, cloud-backed Ollama models, and future hosted endpoints with the
same request shape.

## Ollama Cloud Profile

Local Ollama uses:

```text
http://localhost:11434/v1
```

Ollama Cloud can be configured as an OpenAI-compatible endpoint with:

```text
https://ollama.com/v1
```

Use `OLLAMA_API_KEY` for the cloud API key. Do not commit the key. Start by
listing the model catalog:

```bash
OLLAMA_API_KEY=... PYTHONPATH=src python3 tools/list_worker_pool.py \
  --base-url https://ollama.com/v1 \
  --api-key-env OLLAMA_API_KEY
```

Then copy `research/evals/ollama_cloud_worker_pool.example.json` to a local
config and replace the example model ids with the exact cloud model ids returned
by the account.

Worker-pool configs may include `chat_options`, which are merged into every
`/chat/completions` request. The client always sets `temperature: 0`; add
provider-supported options such as `seed` or `top_p` only after confirming the
endpoint accepts them.

## Refreshing The Current Cloud Pool

Save the live model catalog:

```bash
OLLAMA_API_KEY=... PYTHONPATH=src python3 tools/list_worker_pool.py \
  --base-url https://ollama.com/v1 \
  --api-key-env OLLAMA_API_KEY \
  > research/evals/ollama_cloud_model_catalog_raw.json
```

Then build the current candidate pool:

```bash
PYTHONPATH=src python3 tools/build_worker_pool_from_catalog.py \
  --catalog research/evals/ollama_cloud_model_catalog_raw.json \
  --output research/evals/ollama_cloud_worker_pool_current.json \
  --report research/evals/ollama_cloud_worker_pool_current_report.json \
  --limit 7
```

Use the model ids returned by `/v1/models` as runnable ids. Public Ollama pages
may show CLI aliases such as `:cloud`, but worker-pool configs should follow the
OpenAI-compatible endpoint catalog.

Keep measured winners in the active evaluation pool until a new candidate has a
repeatable benchmark result. New catalog candidates should first run through a
small smoke/repeatability slice before they affect the active router dataset.
