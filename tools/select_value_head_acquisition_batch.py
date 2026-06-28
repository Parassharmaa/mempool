from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task
from mempool.multi_head_orchestrator import read_substrate
from mempool.second_attempt_value import ranked_worker_ids

try:
    from tools.evaluate_active_policy import load_active_router
    from tools.select_fallback_opportunity_batch import read_routing_task_ids, routing_summary
    from tools.select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids
    from tools.select_similar_tasks import task_similarity
except ModuleNotFoundError:
    from evaluate_active_policy import load_active_router
    from select_fallback_opportunity_batch import read_routing_task_ids, routing_summary
    from select_fresh_bigcodebench_batch import load_task_union, read_outcome_task_ids
    from select_similar_tasks import task_similarity


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def outcome_files_from_dirs(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        if not path.exists():
            continue
        files.extend(sorted(candidate for candidate in path.glob("*.jsonl") if candidate.is_file()))
    return files


def records_by_task(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["task_id"]): record for record in records}


def predictions_by_task(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    predictions = (report.get("leave_one_out") or {}).get("predictions") or []
    return {str(prediction["task_id"]): prediction for prediction in predictions}


def record_prompt(record: dict[str, Any]) -> str:
    prompt = record.get("prompt")
    if isinstance(prompt, str):
        return prompt
    messages = record.get("messages") or []
    user_messages = [
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if not user_messages:
        return ""
    content = user_messages[-1]
    marker = "task_prompt:\n"
    if marker not in content:
        return content
    after_marker = content.split(marker, 1)[1]
    end_marker = "\nYou may use these Python libraries"
    if end_marker in after_marker:
        return after_marker.split(end_marker, 1)[0].strip()
    return after_marker.strip()


def value_error_task_ids(value_report: dict[str, Any]) -> dict[str, set[str]]:
    examples = (value_report.get("leave_one_out") or {}).get("examples") or []
    false_negatives = {
        str(example["task_id"])
        for example in examples
        if bool(example.get("value_label")) and len(example.get("attempts") or []) == 1
    }
    false_positives = {
        str(example["task_id"])
        for example in examples
        if (not bool(example.get("value_label"))) and len(example.get("attempts") or []) > 1
    }
    return {
        "false_negative": false_negatives,
        "false_positive": false_positives,
    }


def seed_analysis(
    record: dict[str, Any],
    prediction: dict[str, Any] | None,
    group: str,
) -> dict[str, Any]:
    analysis = classify_task(
        {
            "id": record["task_id"],
            "prompt": record_prompt(record),
            "tests": [],
        }
    )
    prompt_features = record.get("prompt_features") or {}
    if isinstance(prompt_features, dict):
        libraries = prompt_features.get("libraries") or []
        categories = prompt_features.get("categories") or []
        primary_category = prompt_features.get("primary_category")
        if libraries:
            analysis["libraries"] = sorted({str(value) for value in libraries})
        if categories:
            analysis["categories"] = sorted({str(value) for value in categories})
        if primary_category:
            analysis["primary_category"] = str(primary_category)
    analysis["seed_task_id"] = record["task_id"]
    analysis["target_group"] = group
    if prediction:
        ranked = ranked_worker_ids(prediction)
        analysis["top_worker"] = ranked[0] if ranked else None
        analysis["second_worker"] = ranked[1] if len(ranked) > 1 else None
    else:
        analysis["top_worker"] = None
        analysis["second_worker"] = None
    return analysis


def build_error_seeds(
    *,
    substrate_records: list[dict[str, Any]],
    source_report: dict[str, Any],
    value_report: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    by_task = records_by_task(substrate_records)
    predictions = predictions_by_task(source_report)
    error_ids = value_error_task_ids(value_report)
    seeds: dict[str, list[dict[str, Any]]] = {"false_negative": [], "false_positive": []}
    for group, ids in error_ids.items():
        for task_id in sorted(ids):
            record = by_task.get(task_id)
            if record is None:
                continue
            seeds[group].append(seed_analysis(record, predictions.get(task_id), group))
    return seeds


def max_similarity(seeds: list[dict[str, Any]], analysis: dict[str, Any]) -> float:
    return max((task_similarity(seed, analysis) for seed in seeds), default=0.0)


def worker_match_bonus(routing: dict[str, Any], seeds: list[dict[str, Any]]) -> float:
    seed_second_workers = {
        str(seed["second_worker"])
        for seed in seeds
        if seed.get("second_worker")
    }
    second_worker = str(routing.get("second_worker", ""))
    top_worker = str(routing.get("top_worker", ""))
    bonus = 0.0
    if second_worker in seed_second_workers:
        bonus += 2.0
    if top_worker in seed_second_workers:
        bonus += 0.5
    return bonus


def build_contrast_priors(paths: list[Path]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            task_id = str(row.get("task_id", ""))
            if not task_id:
                continue
            by_task.setdefault(task_id, []).append(row)

    priors = []
    for task_id, rows in sorted(by_task.items()):
        prompt = next((str(row.get("prompt", "")) for row in rows if row.get("prompt")), "")
        if not prompt:
            continue
        analysis = classify_task({"id": task_id, "prompt": prompt, "tests": []})
        worker_count = len({str(row.get("worker_id", "")) for row in rows if row.get("worker_id")})
        passed = sum(1 for row in rows if bool(row.get("passed")))
        total = len(rows)
        pass_rate = passed / total if total else 0.0
        disagreement = 1.0 - abs(pass_rate - 0.5) * 2.0
        priors.append(
            {
                **analysis,
                "seed_task_id": task_id,
                "worker_count": worker_count,
                "outcome_count": total,
                "pass_rate": pass_rate,
                "disagreement_score": max(0.0, disagreement),
                "uniform_score": abs(pass_rate - 0.5) * 2.0,
            }
        )
    return priors


def contrast_prior_details(
    analysis: dict[str, Any],
    contrast_priors: list[dict[str, Any]],
) -> dict[str, float]:
    mixed = []
    uniform = []
    for prior in contrast_priors:
        similarity = task_similarity(prior, analysis)
        mixed.append(similarity * float(prior["disagreement_score"]))
        uniform.append(similarity * float(prior["uniform_score"]))
    return {
        "contrast_similarity": max(mixed, default=0.0),
        "uniform_similarity": max(uniform, default=0.0),
    }


def score_candidate(
    *,
    analysis: dict[str, Any],
    routing: dict[str, Any],
    false_negative_seeds: list[dict[str, Any]],
    false_positive_seeds: list[dict[str, Any]],
    contrast_priors: list[dict[str, Any]] | None = None,
    target_group: str,
) -> tuple[float, dict[str, Any]]:
    false_negative_similarity = max_similarity(false_negative_seeds, analysis)
    false_positive_similarity = max_similarity(false_positive_seeds, analysis)
    contrast_details = contrast_prior_details(analysis, contrast_priors or [])
    margin = float(routing["first_second_margin"])
    uncertainty = max(0.0, 1.0 - margin) * 2.0
    low_margin_bonus = 1.0 if margin <= 0.15 else 0.0
    risk_penalty = float(analysis["environment_risk"]) * 0.5
    plausibility_penalty = float(analysis["plausibility_score"]) * 0.05
    contrast_bonus = float(contrast_details["contrast_similarity"]) * 1.5
    uniform_penalty = float(contrast_details["uniform_similarity"]) * 1.0

    if target_group == "missed-positive-value":
        score = (
            false_negative_similarity * 3.0
            - false_positive_similarity
            + worker_match_bonus(routing, false_negative_seeds)
            + uncertainty
            + low_margin_bonus
            + contrast_bonus
            - risk_penalty
            - plausibility_penalty
            - uniform_penalty
        )
    elif target_group == "false-spend-value":
        score = (
            false_positive_similarity * 3.0
            - false_negative_similarity
            + worker_match_bonus(routing, false_positive_seeds)
            + uncertainty
            + low_margin_bonus
            + contrast_bonus
            - risk_penalty
            - plausibility_penalty
            - uniform_penalty
        )
    else:
        raise ValueError(f"unknown target group: {target_group}")
    return score, {
        "false_negative_similarity": round(false_negative_similarity, 4),
        "false_positive_similarity": round(false_positive_similarity, 4),
        "contrast_similarity": round(float(contrast_details["contrast_similarity"]), 4),
        "uniform_similarity": round(float(contrast_details["uniform_similarity"]), 4),
    }


def report_item(item: dict[str, Any]) -> dict[str, Any]:
    routing = item["routing"]
    analysis = item["analysis"]
    return {
        "task_id": item["task"]["id"],
        "target_group": item["target_group"],
        "score": item["score"],
        "false_negative_similarity": item["false_negative_similarity"],
        "false_positive_similarity": item["false_positive_similarity"],
        "contrast_similarity": item["contrast_similarity"],
        "uniform_similarity": item["uniform_similarity"],
        "top_worker": routing["top_worker"],
        "top_probability": routing["top_probability"],
        "second_worker": routing["second_worker"],
        "second_probability": routing["second_probability"],
        "first_second_margin": routing["first_second_margin"],
        "categories": analysis["categories"],
        "libraries": analysis["libraries"],
        "environment_risk": analysis["environment_risk"],
        "plausibility_score": analysis["plausibility_score"],
    }


def select_value_head_acquisition_batch(
    *,
    tasks: list[dict[str, Any]],
    router: Any,
    false_negative_seeds: list[dict[str, Any]],
    false_positive_seeds: list[dict[str, Any]],
    contrast_priors: list[dict[str, Any]] | None = None,
    max_uniform_similarity: float | None = None,
    exclude_ids: set[str],
    limit: int,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    candidates = []
    for task in tasks:
        if str(task["id"]) in exclude_ids:
            continue
        analysis = classify_task(task)
        routing = routing_summary(task, analysis, router)
        for target_group in ("missed-positive-value", "false-spend-value"):
            score, details = score_candidate(
                analysis=analysis,
                routing=routing,
                false_negative_seeds=false_negative_seeds,
                false_positive_seeds=false_positive_seeds,
                contrast_priors=contrast_priors,
                target_group=target_group,
            )
            if (
                max_uniform_similarity is not None
                and float(details["uniform_similarity"]) > max_uniform_similarity
            ):
                continue
            candidates.append(
                {
                    "task": task,
                    "analysis": analysis,
                    "routing": routing,
                    "target_group": target_group,
                    "score": round(score, 4),
                    **details,
                }
            )

    per_group_limit = max(1, limit // 2)
    selected = []
    selected_ids = set()
    for target_group in ("missed-positive-value", "false-spend-value"):
        if len(selected) >= limit:
            break
        ranked_group = sorted(
            [
                item
                for item in candidates
                if item["target_group"] == target_group and float(item["score"]) > 0.0
            ],
            key=lambda item: (
                -float(item["score"]),
                float(item["routing"]["first_second_margin"]),
                float(item["analysis"]["environment_risk"]),
                str(item["task"]["id"]),
            ),
        )
        for item in ranked_group:
            if len(selected) >= limit:
                break
            if sum(1 for chosen in selected if chosen["target_group"] == target_group) >= per_group_limit:
                break
            if item["task"]["id"] in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item["task"]["id"])

    ranked_all = sorted(
        candidates,
        key=lambda item: (
            -float(item["score"]),
            float(item["routing"]["first_second_margin"]),
            float(item["analysis"]["environment_risk"]),
            str(item["task"]["id"]),
        ),
    )
    for item in ranked_all:
        if len(selected) >= limit:
            break
        if item["task"]["id"] in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item["task"]["id"])

    ranked_candidates = sorted(
        candidates,
        key=lambda item: (
            item["target_group"],
            -float(item["score"]),
            float(item["routing"]["first_second_margin"]),
            float(item["analysis"]["environment_risk"]),
            str(item["task"]["id"]),
        ),
    )
    return {
        "candidate_count": len({item["task"]["id"] for item in candidates}),
        "scored_candidate_count": len(candidates),
        "excluded_count": len(exclude_ids),
        "false_negative_seed_ids": [seed["seed_task_id"] for seed in false_negative_seeds],
        "false_positive_seed_ids": [seed["seed_task_id"] for seed in false_positive_seeds],
        "contrast_prior_count": len(contrast_priors or []),
        "max_uniform_similarity": max_uniform_similarity,
        "selected_task_ids": [item["task"]["id"] for item in selected],
        "selected_tasks": [item["task"] for item in selected],
        "selected": [report_item(item) for item in selected],
        "ranked_candidates": [report_item(item) for item in ranked_candidates],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select fresh tasks to improve second-attempt value-head labels."
    )
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--value-head-report", type=Path, required=True)
    parser.add_argument("--exclude-routing-dataset", type=Path, action="append", default=[])
    parser.add_argument("--exclude-outcomes", type=Path, action="append", default=[])
    parser.add_argument("--exclude-outcome-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--contrast-outcomes",
        type=Path,
        action="append",
        default=[],
        help="Outcome JSONL files used as a pass/fail disagreement prior for acquisition scoring.",
    )
    parser.add_argument(
        "--max-uniform-similarity",
        type=float,
        help="Drop candidates whose closest uniform all-pass/all-fail prior similarity exceeds this value.",
    )
    parser.add_argument("--exclude-task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    router, active = load_active_router(args.registry)
    tasks = load_task_union(args.tasks)
    seeds = build_error_seeds(
        substrate_records=read_substrate(args.substrate),
        source_report=read_json(args.source_report),
        value_report=read_json(args.value_head_report),
    )
    exclude_ids = set(args.exclude_task_id)
    exclude_ids.update(read_routing_task_ids(args.exclude_routing_dataset))
    exclude_outcome_paths = args.exclude_outcomes + outcome_files_from_dirs(args.exclude_outcome_dir)
    exclude_ids.update(read_outcome_task_ids(exclude_outcome_paths))
    exclude_ids.update(seed["seed_task_id"] for seed in seeds["false_negative"])
    exclude_ids.update(seed["seed_task_id"] for seed in seeds["false_positive"])
    contrast_priors = build_contrast_priors(args.contrast_outcomes)

    selection = select_value_head_acquisition_batch(
        tasks=tasks,
        router=router,
        false_negative_seeds=seeds["false_negative"],
        false_positive_seeds=seeds["false_positive"],
        contrast_priors=contrast_priors,
        max_uniform_similarity=args.max_uniform_similarity,
        exclude_ids=exclude_ids,
        limit=args.limit,
    )
    report = {
        "registry": str(args.registry),
        "active_model": active["model"],
        "active_dataset": active["dataset"],
        "task_sources": [str(path) for path in args.tasks],
        "substrate": str(args.substrate),
        "source_report": str(args.source_report),
        "value_head_report": str(args.value_head_report),
        "excluded_routing_datasets": [str(path) for path in args.exclude_routing_dataset],
        "excluded_outcomes": [str(path) for path in exclude_outcome_paths],
        "excluded_outcome_dirs": [str(path) for path in args.exclude_outcome_dir],
        "contrast_outcomes": [str(path) for path in args.contrast_outcomes],
        **{key: value for key, value in selection.items() if key != "selected_tasks"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection["selected_tasks"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
