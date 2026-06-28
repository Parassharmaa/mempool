from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .multi_head_orchestrator import MultiHeadOrchestrator
from .task_features import extract_task_features


def load_multi_head_orchestrator_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("model_type") != "linear-multi-head-orchestrator":
        raise ValueError(f"unsupported model_type: {payload.get('model_type')}")
    if not isinstance(payload.get("orchestrator"), dict):
        raise ValueError("model payload missing orchestrator")
    return payload


def load_multi_head_orchestrator(path: str | Path) -> tuple[MultiHeadOrchestrator, dict[str, Any]]:
    payload = load_multi_head_orchestrator_payload(path)
    return MultiHeadOrchestrator.from_dict(payload["orchestrator"]), payload


def build_prompt_record(
    *,
    prompt: str,
    task_id: str = "ad-hoc",
    benchmark_id: str = "ad-hoc",
    task_family: str = "ad_hoc",
    categories: list[str] | None = None,
    libraries: list[str] | None = None,
    missing_libraries: list[str] | None = None,
) -> dict[str, Any]:
    record = {
        "task_id": task_id,
        "benchmark_id": benchmark_id,
        "task_family": task_family,
        "prompt": prompt,
        "prompt_features": {
            "categories": categories or [],
            "libraries": libraries or [],
            "missing_libraries": missing_libraries or [],
        },
    }
    record["dense_features"] = extract_task_features(record)
    return record


def normalize_prediction_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    if not isinstance(normalized.get("dense_features"), dict):
        if "prompt" not in normalized:
            raise ValueError("prediction record needs dense_features or prompt")
        normalized["dense_features"] = extract_task_features(normalized)
    normalized.setdefault("task_id", "ad-hoc")
    normalized.setdefault("benchmark_id", "ad-hoc")
    normalized.setdefault("task_family", "ad_hoc")
    return normalized


def predict_orchestration(
    *,
    model_path: str | Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    model, payload = load_multi_head_orchestrator(model_path)
    normalized = normalize_prediction_record(record)
    prediction = model.predict(normalized)
    return {
        "model_path": str(model_path),
        "model_type": payload["model_type"],
        "substrate": payload.get("substrate"),
        "task_id": normalized["task_id"],
        "benchmark_id": normalized["benchmark_id"],
        "task_family": normalized["task_family"],
        "available_workers": model.worker_ids,
        "selected_worker_id": prediction["target_worker_id"],
        "selected_workflow": prediction["workflow_kind"],
        "verifier_probability": prediction["verifier_probability"],
        "abstain_probability": prediction["abstain_probability"],
        "worker_distribution": prediction["worker_distribution"],
        "workflow_distribution": prediction["workflow_distribution"],
    }
