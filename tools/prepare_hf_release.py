from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_dataset_card(path: Path, *, rows_name: str, row_count: int) -> None:
    path.write_text(
        "\n".join(
            [
                "---",
                "license: apache-2.0",
                "task_categories:",
                "- text-classification",
                "language:",
                "- en",
                "tags:",
                "- orchestration",
                "- routing",
                "- qwen",
                "- logits-head",
                "pretty_name: mempool Qwen Logits Orchestrator Rows",
                "---",
                "",
                "# mempool Qwen Logits Orchestrator Rows",
                "",
                "This dataset contains measured task-level orchestration training rows for the",
                "`mempool` Qwen-small logits-head orchestrator smoke run.",
                "",
                f"- Rows: `{row_count}`",
                f"- Data file: `{rows_name}`",
                "- Targets: worker distribution, workflow distribution, verifier probability, abstain probability",
                "- Source: measured BigCodeBench task-worker outcome substrate",
                "",
                "The rows are intended for training an explicit routing-head orchestrator.",
                "They are not raw user traces and do not contain API keys.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_model_card(path: Path, *, checkpoint_name: str, report: dict) -> None:
    worker_ids = "\n".join(f"- `{worker}`" for worker in report.get("worker_ids", []))
    path.write_text(
        "\n".join(
            [
                "---",
                "license: apache-2.0",
                "base_model: Qwen/Qwen2.5-0.5B-Instruct",
                "library_name: transformers",
                "tags:",
                "- orchestration",
                "- routing",
                "- qwen",
                "- logits-head",
                "pretty_name: mempool Qwen Logits Orchestrator Smoke",
                "---",
                "",
                "# mempool Qwen Logits Orchestrator Smoke",
                "",
                "This repository contains the first smoke checkpoint for the `mempool`",
                "Qwen-small logits-head orchestrator path.",
                "",
                "The checkpoint stores only the trained routing heads, not the Qwen base",
                "model weights. Load the base model separately and attach the heads.",
                "",
                f"- Checkpoint: `{checkpoint_name}`",
                f"- Training rows: `{report.get('record_count')}`",
                f"- Final smoke loss: `{(report.get('history') or [{}])[-1].get('loss')}`",
                "",
                "Worker labels:",
                "",
                worker_ids,
                "",
                "This is a smoke artifact, not a promoted production policy.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def prepare_hf_release(
    *,
    rows_path: Path,
    plan_path: Path,
    readiness_path: Path,
    model_dir: Path,
    output_root: Path,
) -> dict:
    dataset_dir = output_root / "dataset"
    model_export_dir = output_root / "model"
    train_report_path = model_dir / "train_report.json"
    checkpoint_path = model_dir / "qwen_logits_heads.pt"
    report = json.loads(train_report_path.read_text(encoding="utf-8"))
    rows = [line for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    copy_file(rows_path, dataset_dir / rows_path.name)
    copy_file(plan_path, dataset_dir / plan_path.name)
    copy_file(readiness_path, dataset_dir / readiness_path.name)
    write_dataset_card(dataset_dir / "README.md", rows_name=rows_path.name, row_count=len(rows))

    copy_file(checkpoint_path, model_export_dir / checkpoint_path.name)
    copy_file(train_report_path, model_export_dir / train_report_path.name)
    copy_file(plan_path, model_export_dir / plan_path.name)
    copy_file(readiness_path, model_export_dir / readiness_path.name)
    write_model_card(model_export_dir / "README.md", checkpoint_name=checkpoint_path.name, report=report)

    manifest = {
        "schema_version": "mempool.hf_release_manifest.v1",
        "dataset_dir": str(dataset_dir),
        "model_dir": str(model_export_dir),
        "row_count": len(rows),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "train_report": report,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Hugging Face dataset/model export folders for mempool.")
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "research/datasets/20260628-qwen-small-logits-orchestrator-smoke-rows.jsonl",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "research/models/20260628-qwen-small-logits-orchestrator-smoke-plan.json",
    )
    parser.add_argument(
        "--readiness",
        type=Path,
        default=ROOT / "research/models/20260628-qwen-training-readiness-py311.json",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "research/models/20260628-qwen-small-logits-orchestrator-smoke",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "research/hf_export/qwen-logits-smoke")
    args = parser.parse_args()
    manifest = prepare_hf_release(
        rows_path=args.rows,
        plan_path=args.plan,
        readiness_path=args.readiness,
        model_dir=args.model_dir,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
