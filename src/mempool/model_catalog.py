from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelCandidate:
    model: str
    worker_id: str
    role: str
    priority: int
    strengths: tuple[str, ...]
    rationale: str


DEFAULT_MODEL_CANDIDATES: tuple[ModelCandidate, ...] = (
    ModelCandidate(
        model="qwen3-coder:480b",
        worker_id="ollama-cloud-qwen3-coder-480b",
        role="code_frontier",
        priority=10,
        strengths=("code_easy", "code_text", "code_data", "bigcodebench_hard", "agentic_coding"),
        rationale="Current measured strongest/fastest coding worker in mempool BigCodeBench runs.",
    ),
    ModelCandidate(
        model="qwen3-coder-next",
        worker_id="ollama-cloud-qwen3-coder-next",
        role="code_efficient_agentic",
        priority=20,
        strengths=("code_easy", "code_text", "code_data", "bigcodebench_hard", "agentic_coding", "low_active_params"),
        rationale="Efficient coding-specialist candidate for lower-cost agentic coding comparisons.",
    ),
    ModelCandidate(
        model="kimi-k2.7-code",
        worker_id="ollama-cloud-kimi-k2.7-code",
        role="code_agentic_specialist",
        priority=30,
        strengths=("code_easy", "code_text", "code_data", "bigcodebench_hard", "agentic_coding"),
        rationale="Specialist code model with prior Kimi wins in mempool routing data.",
    ),
    ModelCandidate(
        model="glm-5.2",
        worker_id="ollama-cloud-glm-5.2",
        role="long_context_agentic",
        priority=40,
        strengths=("general", "code_easy", "code_text", "code_data", "bigcodebench_hard", "agentic_coding", "long_context"),
        rationale="Long-context agentic candidate and prior GLM specialist winner in mempool routing data.",
    ),
    ModelCandidate(
        model="deepseek-v4-pro",
        worker_id="ollama-cloud-deepseek-v4-pro",
        role="reasoning_specialist",
        priority=50,
        strengths=("general", "reasoning", "code_easy", "code_text", "code_data", "bigcodebench_hard"),
        rationale="Reasoning-heavy specialist with prior DeepSeek wins in mempool routing data.",
    ),
    ModelCandidate(
        model="qwen3.5:397b",
        worker_id="ollama-cloud-qwen3p5-397b",
        role="general_multimodal_frontier",
        priority=60,
        strengths=("general", "reasoning", "code_easy", "bigcodebench_hard", "multimodal"),
        rationale="Large current general candidate for checking whether coding-specialist routes miss generalist wins.",
    ),
    ModelCandidate(
        model="gpt-oss:120b",
        worker_id="ollama-cloud-gpt-oss-120b",
        role="open_weight_reasoning",
        priority=70,
        strengths=("general", "reasoning", "code_easy", "agentic_coding", "open_weight"),
        rationale="Open-weight reasoning model useful as a local/open baseline when available.",
    ),
)


def select_model_candidates(
    available_models: list[str] | tuple[str, ...],
    candidates: tuple[ModelCandidate, ...] = DEFAULT_MODEL_CANDIDATES,
    limit: int | None = None,
) -> dict[str, Any]:
    available = set(available_models)
    selected = []
    missing = []
    for candidate in sorted(candidates, key=lambda item: item.priority):
        row = {
            "model": candidate.model,
            "worker_id": candidate.worker_id,
            "role": candidate.role,
            "priority": candidate.priority,
            "strengths": list(candidate.strengths),
            "rationale": candidate.rationale,
        }
        if candidate.model in available:
            if limit is None or len(selected) < limit:
                selected.append(row)
        else:
            missing.append(row)

    return {
        "selected": selected,
        "missing": missing,
        "available_model_count": len(available),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
    }


def build_worker_pool(
    catalog_payload: dict[str, Any],
    base_url: str,
    api_key_env: str | None,
    timeout_seconds: int,
    limit: int | None = None,
) -> dict[str, Any]:
    models = catalog_payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError("catalog payload must contain a models list")
    selection = select_model_candidates([str(model) for model in models], limit=limit)
    return {
        "base_url": base_url,
        "api_key_env": api_key_env,
        "timeout_seconds": timeout_seconds,
        "source": "live_openai_compatible_model_catalog",
        "workers": [
            {
                "id": row["worker_id"],
                "model": row["model"],
                "role": row["role"],
                "strengths": row["strengths"],
                "cost_usd": 0.0,
            }
            for row in selection["selected"]
        ],
        "selection_report": selection,
    }
