# Ollama Cloud Worker Profile

## Change

Added an Ollama Cloud worker-pool template:

- `research/evals/ollama_cloud_worker_pool.example.json`

The real-worker runner now reads `api_key_env` and `timeout_seconds` from the
worker-pool config, so the same OpenAI-compatible adapter can call local Ollama
or Ollama Cloud.

## Learning

Cloud models should enter the system as worker models. The trainable component
remains the orchestrator/router that chooses between local, cloud, and future
provider workers based on measured outcomes.

Use `OLLAMA_API_KEY` for cloud authentication, and list the available cloud
model ids before creating a real worker-pool config. The example config should
not be treated as the account's exact model catalog.

## Next Step

Once an `OLLAMA_API_KEY` is available, list cloud models with
`tools/list_worker_pool.py`, create a real cloud worker config, and run a small
eligible BigCodeBench slice against the cloud workers to collect stronger
positive routing labels.
