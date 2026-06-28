from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.mined_fallback_head import (
    evaluate_head,
    leave_one_out_evaluation,
    select_threshold,
    train_mined_fallback_head,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dedupe_by_task_id(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = str(record["task_id"])
        previous = deduped.get(task_id)
        if previous is None or (
            bool(record.get("useful_any_fallback"))
            and not bool(previous.get("useful_any_fallback"))
        ):
            deduped[task_id] = record
    return list(deduped.values())


def train_and_report(
    dataset: Path,
    thresholds: list[float],
    epochs: int,
    learning_rate: float,
    l2: float,
    label_field: str,
    dedupe_task_id: bool = True,
) -> dict[str, Any]:
    raw_records = read_jsonl(dataset)
    records = dedupe_by_task_id(raw_records) if dedupe_task_id else raw_records
    trained, history = train_mined_fallback_head(
        records,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        label_field=label_field,
    )
    probabilities = [trained.probability(record) for record in records]
    selected_threshold, threshold_metrics = select_threshold(
        records,
        probabilities,
        thresholds,
        label_field=label_field,
    )
    trained.threshold = selected_threshold
    training_metrics = evaluate_head(records, trained)
    loo = leave_one_out_evaluation(
        records,
        thresholds=thresholds,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        label_field=label_field,
    )
    never_fallback = {
        key: value
        for key, value in select_threshold(
            records,
            [0.0 for _ in records],
            [1.0],
            label_field=label_field,
        )[1].items()
        if key != "examples"
    }
    always_fallback = {
        key: value
        for key, value in select_threshold(
            records,
            [1.0 for _ in records],
            [0.0],
            label_field=label_field,
        )[1].items()
        if key != "examples"
    }
    return {
        "dataset": str(dataset),
        "label_field": label_field,
        "dedupe_task_id": dedupe_task_id,
        "raw_record_count": len(raw_records),
        "record_count": len(records),
        "unique_task_count": len({record["task_id"] for record in records}),
        "positive_count": sum(1 for record in records if bool(record.get(label_field))),
        "thresholds": thresholds,
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "history": history,
        },
        "selected_threshold": selected_threshold,
        "threshold_selection_metrics": {
            key: value
            for key, value in threshold_metrics.items()
            if key != "examples"
        },
        "training_metrics": training_metrics,
        "leave_one_out": loo,
        "baselines": {
            "never_fallback": never_fallback,
            "always_fallback": always_fallback,
        },
        "model": {
            "policy": "mined-fallback-logit-head",
            "head": trained.to_dict(),
            "dataset": str(dataset),
            "selected_threshold": selected_threshold,
            "training_metrics": {
                key: value
                for key, value in training_metrics.items()
                if key != "examples"
            },
            "leave_one_out_metrics": {
                key: value
                for key, value in loo.get("metrics", {}).items()
                if key != "examples"
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a fallback-action logit head from mined fallback cases."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.4, 0.5, 0.6, 0.8])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--label-field", default="useful_any_fallback")
    parser.add_argument(
        "--keep-duplicate-task-ids",
        action="store_true",
        help="Train on every row instead of keeping one mined record per task id.",
    )
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()

    report = train_and_report(
        dataset=args.dataset,
        thresholds=args.thresholds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        label_field=args.label_field,
        dedupe_task_id=not args.keep_duplicate_task_ids,
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.model_output.write_text(
        json.dumps(report["model"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
