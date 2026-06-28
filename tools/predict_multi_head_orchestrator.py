from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.orchestrator_runtime import (
    build_prompt_record,
    predict_orchestration,
)


def _csv_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a trained multi-head orchestrator checkpoint on a task record or prompt."
    )
    parser.add_argument("--model", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--record-json", type=str)
    source.add_argument("--record-file", type=Path)
    source.add_argument("--prompt", type=str)
    parser.add_argument("--task-id", default="ad-hoc")
    parser.add_argument("--benchmark-id", default="ad-hoc")
    parser.add_argument("--task-family", default="ad_hoc")
    parser.add_argument("--categories", default="")
    parser.add_argument("--libraries", default="")
    parser.add_argument("--missing-libraries", default="")
    args = parser.parse_args()

    if args.record_json:
        record = json.loads(args.record_json)
    elif args.record_file:
        record = json.loads(args.record_file.read_text(encoding="utf-8"))
    else:
        record = build_prompt_record(
            prompt=args.prompt,
            task_id=args.task_id,
            benchmark_id=args.benchmark_id,
            task_family=args.task_family,
            categories=_csv_values(args.categories),
            libraries=_csv_values(args.libraries),
            missing_libraries=_csv_values(args.missing_libraries),
        )

    result = predict_orchestration(model_path=args.model, record=record)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
