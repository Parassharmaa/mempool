from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .logits_router import reward_target_vector
from .orchestrator_contract import WORKFLOW_KINDS, read_json, validate_multi_head_contract
from .routing_dataset import read_routing_records, validate_routing_records
from .task_features import CATEGORIES, extract_task_features


SCHEMA_VERSION = "mempool.small_orchestrator_substrate.v1"


def _round_distribution(distribution: dict[str, float]) -> dict[str, float]:
    return {
        key: round(float(value), 6)
        for key, value in sorted(distribution.items())
    }


def _prompt_feature_summary(record: dict[str, Any]) -> dict[str, Any]:
    prompt_features = record.get("prompt_features") or {}
    if not isinstance(prompt_features, dict):
        prompt_features = {}
    categories = [
        category
        for category in CATEGORIES
        if category in {str(value).lower() for value in prompt_features.get("categories", [])}
    ]
    libraries = sorted({str(value) for value in prompt_features.get("libraries", [])})
    missing_libraries = sorted({str(value) for value in prompt_features.get("missing_libraries", [])})
    return {
        "categories": categories,
        "libraries": libraries,
        "missing_libraries": missing_libraries,
        "environment_risk": float(prompt_features.get("environment_risk", 0.0) or 0.0),
        "plausibility_score": float(prompt_features.get("plausibility_score", 0.0) or 0.0),
    }


def _worker_summary(record: dict[str, Any]) -> list[dict[str, Any]]:
    workers = []
    for worker in sorted(record["workers"], key=lambda item: str(item["worker_id"])):
        workers.append(
            {
                "worker_id": worker["worker_id"],
                "model": worker.get("model"),
                "pass_rate": float(worker.get("pass_rate", 1.0 if worker.get("passed") else 0.0)),
                "passed": bool(worker.get("passed")),
                "mean_latency_ms": float(worker.get("mean_latency_ms", worker.get("latency_ms", 0.0))),
                "reward": float(worker.get("reward", 0.0)),
                "failure_mode": worker.get("failure_mode"),
            }
        )
    return workers


def _workflow_label(record: dict[str, Any]) -> str:
    if any(bool(worker.get("passed")) for worker in record["workers"]):
        return "direct"
    return "verify_then_fallback"


def _verifier_probability(record: dict[str, Any], worker_distribution: dict[str, float]) -> float:
    if not any(bool(worker.get("passed")) for worker in record["workers"]):
        return 1.0
    target_probability = float(worker_distribution.get(record["target_worker_id"], 0.0))
    return round(max(0.0, min(1.0, 1.0 - target_probability)), 6)


def _abstain_probability(record: dict[str, Any]) -> float:
    return 1.0 if not any(bool(worker.get("passed")) for worker in record["workers"]) else 0.0


def _instruction(record: dict[str, Any], worker_ids: list[str]) -> str:
    feature_summary = _prompt_feature_summary(record)
    return "\n".join(
        [
            "Choose an orchestration action for this task.",
            "Return JSON with worker_distribution, workflow_kind, verifier_probability, and abstain_probability.",
            f"task_id: {record['task_id']}",
            f"task_family: {record['task_family']}",
            f"categories: {', '.join(feature_summary['categories']) or 'none'}",
            f"libraries: {', '.join(feature_summary['libraries']) or 'none'}",
            f"available_workers: {', '.join(worker_ids)}",
            "task_prompt:",
            str(record["prompt"]),
        ]
    )


def build_orchestrator_example(
    record: dict[str, Any],
    *,
    worker_ids: list[str],
    reward_temperature: float,
) -> dict[str, Any]:
    target_vector = reward_target_vector(
        record,
        worker_ids,
        temperature=reward_temperature,
    )
    worker_distribution = _round_distribution(
        {
            worker_id: probability
            for worker_id, probability in zip(worker_ids, target_vector, strict=True)
        }
    )
    workflow_kind = _workflow_label(record)
    target = {
        "worker_distribution": worker_distribution,
        "target_worker_id": record["target_worker_id"],
        "workflow_kind": workflow_kind,
        "workflow_distribution": {
            label: 1.0 if label == workflow_kind else 0.0
            for label in WORKFLOW_KINDS
        },
        "verifier_probability": _verifier_probability(record, worker_distribution),
        "abstain_probability": _abstain_probability(record),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": record["task_id"],
        "benchmark_id": record["benchmark_id"],
        "task_family": record["task_family"],
        "prompt_features": _prompt_feature_summary(record),
        "dense_features": extract_task_features(record),
        "workers": _worker_summary(record),
        "target": target,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a compact orchestration model. "
                    "Predict routing and control heads as strict JSON."
                ),
            },
            {
                "role": "user",
                "content": _instruction(record, worker_ids),
            },
            {
                "role": "assistant",
                "content": json.dumps(target, sort_keys=True),
            },
        ],
    }


def build_orchestrator_substrate(
    *,
    routing_dataset_path: Path,
    contract_path: Path,
    output_path: Path,
    manifest_path: Path,
    reward_temperature: float = 0.05,
) -> dict[str, Any]:
    records = read_routing_records(routing_dataset_path)
    errors = validate_routing_records(records)
    if errors:
        raise ValueError(f"invalid routing dataset: {errors}")

    contract = read_json(contract_path)
    contract_validation = validate_multi_head_contract(contract)
    if not contract_validation["valid"]:
        raise ValueError(f"invalid orchestrator contract: {contract_validation['reasons']}")

    worker_ids = sorted({worker["worker_id"] for record in records for worker in record["workers"]})
    examples = [
        build_orchestrator_example(
            record,
            worker_ids=worker_ids,
            reward_temperature=reward_temperature,
        )
        for record in records
    ]
    target_counts: dict[str, int] = {}
    workflow_counts: dict[str, int] = {}
    abstain_positive = 0
    verifier_sum = 0.0
    for example in examples:
        target = example["target"]
        target_worker = str(target["target_worker_id"])
        workflow = str(target["workflow_kind"])
        target_counts[target_worker] = target_counts.get(target_worker, 0) + 1
        workflow_counts[workflow] = workflow_counts.get(workflow, 0) + 1
        abstain_positive += int(float(target["abstain_probability"]) >= 0.5)
        verifier_sum += float(target["verifier_probability"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(example, sort_keys=True) + "\n" for example in examples),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "routing_dataset": str(routing_dataset_path),
        "orchestrator_contract": str(contract_path),
        "output": str(output_path),
        "reward_temperature": reward_temperature,
        "record_count": len(examples),
        "worker_ids": worker_ids,
        "target_counts": target_counts,
        "workflow_counts": workflow_counts,
        "abstain_positive": abstain_positive,
        "mean_verifier_probability": verifier_sum / len(examples) if examples else 0.0,
        "contract_validation": contract_validation,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
