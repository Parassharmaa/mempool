# Training Framework Choice

## Local Mac Path

Use MLX/MLX-LM first for local Mac fine-tuning. It is designed for Apple Silicon
and supports LoRA-style fine-tuning without needing NVIDIA CUDA.

Use this path for:

- small supervised routing model
- LoRA on a 1-2B local instruction model
- quick iteration on local traces

## NVIDIA GPU Path

Use Unsloth or TRL when an NVIDIA GPU is available.

Use this path for:

- faster LoRA/QLoRA
- larger model experiments
- reward-optimized training such as DPO/GRPO-style experiments

## Start Smaller Than LLM Fine-Tuning

Before fine-tuning a language model, train a simple router:

- embedding model or bag-of-features task encoder
- small MLP/ranker
- target labels from measured worker rewards

This lets us debug the dataset and reward design before spending GPU cycles.

## Decision

1. Build data collection through Ollama/OpenAI-compatible adapters.
2. Train an offline lightweight ranker.
3. If the ranker shows signal, move to MLX-LM LoRA on a small local model.
4. If GPU access is available, compare MLX results against Unsloth/TRL.
