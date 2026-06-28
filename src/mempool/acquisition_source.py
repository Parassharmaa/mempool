from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AcquisitionSource:
    id: str
    kind: str
    report: dict[str, Any]
    cost_hint: str = "unknown"
    evidence: tuple[dict[str, Any], ...] = ()


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _decision(report: dict[str, Any]) -> str:
    return str(report.get("decision") or report.get("status") or "").lower()


def _candidate_count(report: dict[str, Any]) -> int:
    for key in ("candidate_count", "eligible_count", "scored_candidate_count"):
        if key in report:
            return _as_int(report.get(key))
    return len(report.get("ranked_candidates", []) or report.get("selected_task_ids", []))


def _is_exhausted(report: dict[str, Any]) -> bool:
    return (
        _as_int(report.get("scanned"), default=-1) == 0
        and _candidate_count(report) == 0
        and "next_offset" in report
    )


def _outcome_evidence_score(
    evidence_items: tuple[dict[str, Any], ...],
    reasons: list[str],
) -> float:
    score = 0.0
    for evidence in evidence_items:
        by_task = evidence.get("by_task", [])
        if not isinstance(by_task, list):
            continue
        task_count = len(by_task)
        universal_failures = sum(1 for item in by_task if item.get("universal_failure"))
        candidates = [item for item in by_task if item.get("candidate_for_conversion")]
        nonqwen_candidates = [
            item
            for item in candidates
            if "qwen" not in str(item.get("best_worker_id", "")).lower()
        ]
        if task_count:
            failure_rate = universal_failures / task_count
            penalty = failure_rate * 12.0
            score -= penalty
            if universal_failures:
                reasons.append(f"{universal_failures}/{task_count} evidence tasks were universal failures")
        if candidates:
            score += min(len(candidates), 5) * 1.5
            reasons.append(f"{len(candidates)} evidence tasks were conversion candidates")
        if nonqwen_candidates:
            score += min(len(nonqwen_candidates), 5) * 2.0
            reasons.append(f"{len(nonqwen_candidates)} evidence candidates were non-Qwen wins")
        elif candidates:
            score -= 10.0
            reasons.append("evidence candidates did not produce stable non-Qwen wins")
    return score


def score_acquisition_source(source: AcquisitionSource) -> dict[str, Any]:
    report = source.report
    candidate_count = _candidate_count(report)
    selected_count = len(report.get("selected_task_ids", []) or [])
    decision = _decision(report)
    reasons: list[str] = []
    score = 0.0

    if _is_exhausted(report):
        reasons.append("source is exhausted")
        score -= 100.0
    else:
        score += min(candidate_count, 50) * 0.4
        if candidate_count:
            reasons.append(f"{candidate_count} candidates available")
        if selected_count:
            score += min(selected_count, 10) * 0.6
            reasons.append(f"{selected_count} preselected tasks")

    if source.kind in {"nonqwen_specialist_pressure", "specialist_pressure"}:
        score += 8.0
        reasons.append("targets non-Qwen specialist evidence")
    elif source.kind == "dependency_profile":
        score += 5.0
        reasons.append("widens benchmark dependency coverage")
    elif source.kind == "normal_offset_scan":
        score -= 2.0
        reasons.append("normal offset scan has lower novelty")

    if decision == "quarantine":
        score += 2.0
        reasons.append("prior gate produced useful negative evidence")
    elif decision in {"pass", "keep"}:
        score += 1.0
        reasons.append("prior gate was healthy")

    if source.cost_hint == "no_model_calls":
        score += 2.0
        reasons.append("can be inspected without live worker spend")
    elif source.cost_hint == "model_calls":
        score -= 1.0
        reasons.append("requires live worker calls")

    score += _outcome_evidence_score(source.evidence, reasons)

    return {
        "id": source.id,
        "kind": source.kind,
        "score": round(score, 4),
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "decision": decision or None,
        "exhausted": _is_exhausted(report),
        "evidence_count": len(source.evidence),
        "reasons": reasons,
    }


def rank_acquisition_sources(sources: list[AcquisitionSource]) -> dict[str, Any]:
    scored = [score_acquisition_source(source) for source in sources]
    ranked = sorted(
        scored,
        key=lambda item: (
            bool(item["exhausted"]),
            -float(item["score"]),
            str(item["id"]),
        ),
    )
    recommendation = ranked[0] if ranked else None
    return {
        "recommended_source_id": recommendation["id"] if recommendation else None,
        "recommendation": recommendation,
        "ranked_sources": ranked,
    }


def summarize_router_miss_neighborhoods(
    *,
    router_report: dict[str, Any],
    routing_records: list[dict[str, Any]],
    max_neighborhoods: int = 8,
) -> dict[str, Any]:
    records_by_id = {str(record["task_id"]): record for record in routing_records}
    examples = router_report.get("leave_one_out", {}).get("examples", [])
    neighborhoods: dict[tuple[str, str, str], dict[str, Any]] = {}
    misses = []
    for example in examples:
        target = str(example.get("target_worker_id", ""))
        predicted = str(example.get("predicted_worker_id", ""))
        task_id = str(example.get("task_id", ""))
        if not target or not predicted or target == predicted:
            continue
        record = records_by_id.get(task_id, {})
        prompt_features = record.get("prompt_features") or {}
        categories = tuple(sorted(str(value) for value in prompt_features.get("categories", [])))
        libraries = tuple(sorted(str(value) for value in prompt_features.get("libraries", [])))
        category_key = "+".join(categories) or "uncategorized"
        library_key = "+".join(libraries[:4]) or "no_libraries"
        key = (target, predicted, category_key)
        neighborhood = neighborhoods.setdefault(
            key,
            {
                "target_worker_id": target,
                "predicted_worker_id": predicted,
                "category_key": category_key,
                "library_keys": {},
                "task_ids": [],
                "example_prompts": [],
            },
        )
        neighborhood["task_ids"].append(task_id)
        neighborhood["library_keys"][library_key] = neighborhood["library_keys"].get(library_key, 0) + 1
        prompt = " ".join(str(record.get("prompt", "")).split())
        if prompt and len(neighborhood["example_prompts"]) < 2:
            neighborhood["example_prompts"].append(prompt[:240])
        misses.append(
            {
                "task_id": task_id,
                "target_worker_id": target,
                "predicted_worker_id": predicted,
                "categories": list(categories),
                "libraries": list(libraries),
            }
        )

    ranked = []
    for neighborhood in neighborhoods.values():
        library_counts = sorted(
            neighborhood["library_keys"].items(),
            key=lambda item: (-item[1], item[0]),
        )
        ranked.append(
            {
                "target_worker_id": neighborhood["target_worker_id"],
                "predicted_worker_id": neighborhood["predicted_worker_id"],
                "category_key": neighborhood["category_key"],
                "miss_count": len(neighborhood["task_ids"]),
                "task_ids": sorted(neighborhood["task_ids"]),
                "top_library_keys": [
                    {"libraries": libraries, "count": count}
                    for libraries, count in library_counts[:3]
                ],
                "example_prompts": neighborhood["example_prompts"],
                "recommended_action": (
                    "mine repeated examples where the target worker is plausible "
                    "and the predicted worker is currently over-selected"
                ),
            }
        )
    ranked.sort(
        key=lambda item: (
            -int(item["miss_count"]),
            str(item["target_worker_id"]),
            str(item["predicted_worker_id"]),
            str(item["category_key"]),
        )
    )
    return {
        "miss_count": len(misses),
        "task_count": len(examples),
        "neighborhood_count": len(ranked),
        "neighborhoods": ranked[:max_neighborhoods],
        "misses": misses,
    }


def build_execution_manifest(
    *,
    source_plan: dict[str, Any],
    source_report: dict[str, Any],
    selected_tasks: list[dict[str, Any]],
    worker_pool: dict[str, Any],
    worker_pool_path: Path,
    tasks_output: Path,
    run_id: str,
    output_path: Path,
    outcomes_path: Path,
    repeat_count: int,
    eval_timeout_seconds: int,
    required_packages: list[str] | None = None,
    python_executable: str = "python3",
) -> dict[str, Any]:
    recommended_source_id = source_plan.get("recommended_source_id")
    if recommended_source_id is None:
        raise ValueError("source plan has no recommended source")
    selected_task_ids = [str(task["id"]) for task in selected_tasks]
    report_task_ids = [str(task_id) for task_id in source_report.get("selected_task_ids", [])]
    if report_task_ids and selected_task_ids != report_task_ids:
        raise ValueError("selected tasks do not match source report selected_task_ids")
    workers = worker_pool.get("workers", [])
    if not isinstance(workers, list) or not workers:
        raise ValueError("worker pool must contain at least one worker")
    worker_ids = [str(worker["id"]) for worker in workers]
    package_args = " ".join(f"--required-package {package}" for package in required_packages or [])
    package_args = f" {package_args}" if package_args else ""
    command_prefix = f"PYTHONPATH=src {python_executable}"
    run_command = (
        f"{command_prefix} tools/run_real_smoke_benchmark.py "
        f"--config {worker_pool_path} "
        f"--tasks {tasks_output} "
        f"--repeat-count {repeat_count} "
        f"--eval-timeout-seconds {eval_timeout_seconds}"
        f"{package_args} "
        f"--run-id {run_id} "
        f"--output {output_path} "
        f"--outcomes {outcomes_path} "
        "--resume --progress --quiet"
    )
    summary_path = outcomes_path.with_name(outcomes_path.stem + "-summary.json")
    return {
        "benchmark_id": "bigcodebench-hard-recommended-acquisition",
        "recommended_source_id": recommended_source_id,
        "source_recommendation": source_plan.get("recommendation"),
        "source_candidate_count": source_report.get("candidate_count"),
        "run_id": run_id,
        "repeat_count": repeat_count,
        "worker_ids_to_run": worker_ids,
        "selected_task_ids": selected_task_ids,
        "selected_task_metadata": {
            str(item["task_id"]): item
            for item in source_report.get("selected", [])
            if str(item.get("task_id")) in selected_task_ids
        },
        "task_count": len(selected_tasks),
        "call_count": len(selected_tasks) * len(worker_ids) * repeat_count,
        "tasks_output": str(tasks_output),
        "worker_pool": str(worker_pool_path),
        "required_packages": required_packages or [],
        "python_executable": python_executable,
        "output": str(output_path),
        "outcomes": str(outcomes_path),
        "run_command": run_command,
        "summary_command": (
            f"PYTHONPATH=src python3 tools/summarize_repeated_outcomes.py "
            f"--input {outcomes_path} "
            f"--output {summary_path}"
        ),
    }
