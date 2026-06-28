from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> dict:
    env = dict(os.environ)
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    try:
        parsed_stdout = json.loads(completed.stdout)
    except json.JSONDecodeError:
        parsed_stdout = {"stdout": completed.stdout}
    return {
        "command": command,
        "stdout": parsed_stdout,
        "stderr": completed.stderr,
    }


def run_full_orchestrator(
    *,
    train_rows: Path,
    heldout_rows: Path,
    plan_output: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    device: str,
    base_model: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_result = run(
        [
            sys.executable,
            "tools/train_qwen_logits_orchestrator.py",
            "--training-rows",
            str(train_rows),
            "--plan-output",
            str(plan_output),
            "--rows-output",
            str(output_dir / "unused_rows.jsonl"),
            "--output-dir",
            str(output_dir),
            "--base-model",
            base_model,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
            "--max-length",
            str(max_length),
            "--device",
            device,
            "--train",
        ]
    )
    checkpoint = output_dir / "qwen_logits_heads.pt"
    train_eval = run(
        [
            sys.executable,
            "tools/evaluate_qwen_logits_orchestrator.py",
            "--rows",
            str(train_rows),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output_dir / "train_eval_report.json"),
        ]
    )
    heldout_eval = run(
        [
            sys.executable,
            "tools/evaluate_qwen_logits_orchestrator.py",
            "--rows",
            str(heldout_rows),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output_dir / "heldout_eval_report.json"),
        ]
    )
    manifest = {
        "schema_version": "mempool.qwen_full_orchestrator_run.v1",
        "base_model": base_model,
        "train_rows": str(train_rows),
        "heldout_rows": str(heldout_rows),
        "plan_output": str(plan_output),
        "output_dir": str(output_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "device": device,
        "train_result": train_result["stdout"],
        "train_eval_report": train_eval["stdout"],
        "heldout_eval_report": heldout_eval["stdout"],
    }
    manifest_path = output_dir / "full_run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the first usable Qwen logits orchestrator run.")
    parser.add_argument(
        "--train-rows",
        type=Path,
        default=ROOT / "research/datasets/20260628-qwen-small-logits-orchestrator-split-train.jsonl",
    )
    parser.add_argument(
        "--heldout-rows",
        type=Path,
        default=ROOT / "research/datasets/20260628-qwen-small-logits-orchestrator-split-heldout.jsonl",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=ROOT / "research/models/20260628-qwen-small-logits-orchestrator-full-plan.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "research/models/20260628-qwen-small-logits-orchestrator-full",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()
    manifest = run_full_orchestrator(
        train_rows=args.train_rows,
        heldout_rows=args.heldout_rows,
        plan_output=args.plan_output,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        device=args.device,
        base_model=args.base_model,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
