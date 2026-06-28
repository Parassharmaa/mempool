from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_dataset_card(
    path: Path,
    *,
    train_rows_name: str,
    train_row_count: int,
    heldout_rows_name: str | None = None,
    heldout_row_count: int | None = None,
    split_manifest_name: str | None = None,
) -> None:
    split_lines = [
        f"- Train rows: `{train_row_count}`",
        f"- Train data file: `{train_rows_name}`",
    ]
    if heldout_rows_name is not None and heldout_row_count is not None:
        split_lines.extend(
            [
                f"- Held-out rows: `{heldout_row_count}`",
                f"- Held-out data file: `{heldout_rows_name}`",
            ]
        )
    if split_manifest_name is not None:
        split_lines.append(f"- Split manifest: `{split_manifest_name}`")
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
                "`mempool` Qwen-small logits-head orchestrator path.",
                "",
                *split_lines,
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


def write_model_card(
    path: Path,
    *,
    checkpoint_name: str,
    report: dict,
    train_eval_report: dict | None = None,
    heldout_eval_report: dict | None = None,
) -> None:
    worker_ids = "\n".join(f"- `{worker}`" for worker in report.get("worker_ids", []))
    eval_lines = []
    if train_eval_report:
        eval_lines = [
            "",
            "Train-row evaluation:",
            "",
            f"- Worker accuracy: `{train_eval_report.get('worker_accuracy')}`",
            f"- Workflow accuracy: `{train_eval_report.get('workflow_accuracy')}`",
            f"- Mean worker loss: `{train_eval_report.get('mean_worker_loss')}`",
            f"- Mean workflow loss: `{train_eval_report.get('mean_workflow_loss')}`",
        ]
    if heldout_eval_report:
        eval_lines.extend(
            [
                "",
                "Held-out evaluation:",
                "",
                f"- Worker accuracy: `{heldout_eval_report.get('worker_accuracy')}`",
                f"- Workflow accuracy: `{heldout_eval_report.get('workflow_accuracy')}`",
                f"- Mean worker loss: `{heldout_eval_report.get('mean_worker_loss')}`",
                f"- Mean workflow loss: `{heldout_eval_report.get('mean_workflow_loss')}`",
            ]
        )
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
                "# mempool Qwen Logits Orchestrator Split Smoke",
                "",
                "This repository contains a split-smoke checkpoint for the `mempool`",
                "Qwen-small logits-head orchestrator path with a deterministic held-out gate.",
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
                *eval_lines,
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
    heldout_rows_path: Path | None = None,
    split_manifest_path: Path | None = None,
) -> dict:
    dataset_dir = output_root / "dataset"
    model_export_dir = output_root / "model"
    train_report_path = model_dir / "train_report.json"
    eval_report_path = model_dir / "eval_report.json"
    train_eval_report_path = model_dir / "train_eval_report.json"
    heldout_eval_report_path = model_dir / "heldout_eval_report.json"
    checkpoint_path = model_dir / "qwen_logits_heads.pt"
    report = json.loads(train_report_path.read_text(encoding="utf-8"))
    legacy_eval_report = json.loads(eval_report_path.read_text(encoding="utf-8")) if eval_report_path.exists() else None
    train_eval_report = (
        json.loads(train_eval_report_path.read_text(encoding="utf-8"))
        if train_eval_report_path.exists()
        else legacy_eval_report
    )
    heldout_eval_report = (
        json.loads(heldout_eval_report_path.read_text(encoding="utf-8"))
        if heldout_eval_report_path.exists()
        else None
    )
    rows = [line for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    heldout_rows = (
        [line for line in heldout_rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if heldout_rows_path is not None
        else []
    )

    copy_file(rows_path, dataset_dir / rows_path.name)
    if heldout_rows_path is not None:
        copy_file(heldout_rows_path, dataset_dir / heldout_rows_path.name)
    if split_manifest_path is not None:
        copy_file(split_manifest_path, dataset_dir / split_manifest_path.name)
    copy_file(plan_path, dataset_dir / plan_path.name)
    copy_file(readiness_path, dataset_dir / readiness_path.name)
    write_dataset_card(
        dataset_dir / "README.md",
        train_rows_name=rows_path.name,
        train_row_count=len(rows),
        heldout_rows_name=heldout_rows_path.name if heldout_rows_path is not None else None,
        heldout_row_count=len(heldout_rows) if heldout_rows_path is not None else None,
        split_manifest_name=split_manifest_path.name if split_manifest_path is not None else None,
    )

    copy_file(checkpoint_path, model_export_dir / checkpoint_path.name)
    copy_file(train_report_path, model_export_dir / train_report_path.name)
    if eval_report_path.exists():
        copy_file(eval_report_path, model_export_dir / eval_report_path.name)
    if train_eval_report_path.exists():
        copy_file(train_eval_report_path, model_export_dir / train_eval_report_path.name)
    if heldout_eval_report_path.exists():
        copy_file(heldout_eval_report_path, model_export_dir / heldout_eval_report_path.name)
    copy_file(plan_path, model_export_dir / plan_path.name)
    copy_file(readiness_path, model_export_dir / readiness_path.name)
    write_model_card(
        model_export_dir / "README.md",
        checkpoint_name=checkpoint_path.name,
        report=report,
        train_eval_report=train_eval_report,
        heldout_eval_report=heldout_eval_report,
    )

    manifest = {
        "schema_version": "mempool.hf_release_manifest.v1",
        "dataset_dir": str(dataset_dir),
        "model_dir": str(model_export_dir),
        "row_count": len(rows),
        "train_row_count": len(rows),
        "heldout_row_count": len(heldout_rows) if heldout_rows_path is not None else 0,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "train_report": report,
        "eval_report": train_eval_report,
        "train_eval_report": train_eval_report,
        "heldout_eval_report": heldout_eval_report,
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
        default=ROOT / "research/datasets/20260628-qwen-small-logits-orchestrator-split-train.jsonl",
    )
    parser.add_argument(
        "--heldout-rows",
        type=Path,
        default=ROOT / "research/datasets/20260628-qwen-small-logits-orchestrator-split-heldout.jsonl",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "research/datasets/20260628-qwen-small-logits-orchestrator-split-manifest.json",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "research/models/20260628-qwen-small-logits-orchestrator-split-smoke-plan.json",
    )
    parser.add_argument(
        "--readiness",
        type=Path,
        default=ROOT / "research/models/20260628-qwen-training-readiness-py311.json",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "research/models/20260628-qwen-small-logits-orchestrator-split-smoke",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "research/hf_export/qwen-logits-smoke")
    args = parser.parse_args()
    manifest = prepare_hf_release(
        rows_path=args.rows,
        plan_path=args.plan,
        readiness_path=args.readiness,
        model_dir=args.model_dir,
        output_root=args.output_root,
        heldout_rows_path=args.heldout_rows,
        split_manifest_path=args.split_manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
