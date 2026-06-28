from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.bigcodebench import classify_task
from mempool.orchestrator_runtime import build_prompt_record, predict_orchestration


def select_evaluable_prompt_set(
    *,
    task_file: Path,
    model_path: Path,
    output_path: Path,
    limit: int = 3,
    max_prompt_chars: int = 1200,
) -> dict[str, Any]:
    tasks = json.loads(task_file.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("task_file must contain a task list")

    candidates = []
    for task in tasks:
        features = classify_task(task)
        record = build_prompt_record(
            prompt=str(task["prompt"]),
            task_id=str(task["id"]),
            benchmark_id="bigcodebench_hard",
            task_family=str(task.get("family") or "bigcodebench_hard"),
            categories=list(features["categories"]),
            libraries=list(features["libraries"]),
            missing_libraries=list(features["missing_libraries"]),
        )
        prediction = predict_orchestration(model_path=model_path, record=record)
        candidates.append(
            {
                "task": task,
                "features": features,
                "prediction": prediction,
                "confidence": max(float(value) for value in prediction["worker_distribution"].values()),
            }
        )

    selected_candidates = []
    seen_workers: set[str] = set()
    ordered_candidates = sorted(candidates, key=lambda item: -item["confidence"])
    for candidate in ordered_candidates:
        if len(selected_candidates) >= limit:
            break
        worker_id = str(candidate["prediction"]["selected_worker_id"])
        if worker_id in seen_workers:
            continue
        selected_candidates.append(candidate)
        seen_workers.add(worker_id)
    for candidate in ordered_candidates:
        if len(selected_candidates) >= limit:
            break
        if candidate in selected_candidates:
            continue
        selected_candidates.append(candidate)

    selected = []
    for candidate in selected_candidates:
        worker_id = str(candidate["prediction"]["selected_worker_id"])
        if len(selected) >= limit:
            break
        task = candidate["task"]
        features = candidate["features"]
        selected.append(
            {
                "task_id": task["id"],
                "benchmark_id": "bigcodebench_hard",
                "task_family": task.get("family", "bigcodebench_hard"),
                "categories": features["categories"],
                "libraries": features["libraries"],
                "missing_libraries": features["missing_libraries"],
                "predicted_worker_id": worker_id,
                "worker_distribution": candidate["prediction"]["worker_distribution"],
                "prompt": str(task["prompt"])[:max_prompt_chars],
            }
        )

    payload = {
        "schema_version": "mempool.evaluable_prompt_set.v1",
        "task_file": str(task_file),
        "model": str(model_path),
        "limit": limit,
        "prompt_count": len(selected),
        "prompts": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a small prompt set from locally evaluable tasks using the trained router for diversity."
    )
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-prompt-chars", type=int, default=1200)
    args = parser.parse_args()
    payload = select_evaluable_prompt_set(
        task_file=args.task_file,
        model_path=args.model,
        output_path=args.output,
        limit=args.limit,
        max_prompt_chars=args.max_prompt_chars,
    )
    print(json.dumps({key: payload[key] for key in ["prompt_count", "limit"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
