# Cloud Model Catalog Refresh

The Ollama Cloud `/v1/models` endpoint is the runnable source of truth for
worker-pool ids. Public pages may display CLI-oriented aliases with `:cloud`,
but the OpenAI-compatible catalog currently returns runnable ids such as
`glm-5.2`, `deepseek-v4-pro`, `kimi-k2.7-code`, `qwen3-coder:480b`,
`qwen3-coder-next`, `qwen3.5:397b`, and `gpt-oss:120b`.

The current pool should not immediately replace the measured top-four pool.
Qwen Coder, Kimi, GLM, and DeepSeek already have benchmark-backed outcomes.
The newer catalog candidates should enter through a smoke/repeatability run,
then be merged into routing datasets only if they produce stable positive or
latency-improving evidence.
