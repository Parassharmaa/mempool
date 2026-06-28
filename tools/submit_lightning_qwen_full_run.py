from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_job_command(
    *,
    repo_url: str,
    git_ref: str,
    output_name: str,
    plan_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
) -> str:
    return "\n".join(
        [
            "set -euxo pipefail",
            "python --version",
            "if command -v apt-get >/dev/null 2>&1; then apt-get update && apt-get install -y git; fi",
            f"git clone --depth 1 --branch {git_ref} {repo_url} mempool",
            "cd mempool",
            "python -m pip install -U pip",
            "python -m pip install -e '.[qwen-train]'",
            "python - <<'PY'",
            "import torch",
            "print('torch', torch.__version__)",
            "print('cuda_available', torch.cuda.is_available())",
            "print('cuda_device_count', torch.cuda.device_count())",
            "PY",
            "PYTHONPATH=src python tools/run_qwen_full_orchestrator.py \\",
            f"  --epochs {epochs} \\",
            f"  --batch-size {batch_size} \\",
            f"  --learning-rate {learning_rate} \\",
            f"  --max-length {max_length} \\",
            "  --device cuda \\",
            f"  --output-dir research/models/{output_name} \\",
            f"  --plan-output research/models/{plan_name}",
            "mkdir -p /artifacts",
            f"cp -R research/models/{output_name} /artifacts/{output_name}",
            f"cp research/models/{plan_name} /artifacts/{plan_name}",
        ]
    )


def submit_lightning_job(
    *,
    name: str,
    machine: str,
    image: str,
    command: str,
    artifacts_local: Path,
    max_runtime: int,
    user: str | None,
    teamspace: str | None,
    org: str | None,
) -> dict:
    if not os.environ.get("LIGHTNING_USER_ID"):
        raise RuntimeError("LIGHTNING_USER_ID must be set in the environment")
    if not os.environ.get("LIGHTNING_API_KEY"):
        raise RuntimeError("LIGHTNING_API_KEY must be set in the environment")

    from lightning_sdk import Job

    artifacts_local.mkdir(parents=True, exist_ok=True)
    job = Job.run(
        name=name,
        machine=machine,
        image=image,
        command=command,
        artifacts_local=str(artifacts_local),
        artifacts_remote="/artifacts",
        max_runtime=max_runtime,
        user=user,
        teamspace=teamspace,
        org=org,
    )
    result = {
        "schema_version": "mempool.lightning_qwen_full_job.v1",
        "name": name,
        "machine": machine,
        "image": image,
        "artifacts_local": str(artifacts_local),
        "status": str(job.status),
        "link": getattr(job, "link", None),
        "id": getattr(job, "id", None),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    (artifacts_local / "submitted_job.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit the Qwen orchestrator full training run to Lightning AI.")
    parser.add_argument("--repo-url", default="https://github.com/Parassharmaa/mempool.git")
    parser.add_argument("--git-ref", default="main")
    parser.add_argument("--name", default="mempool-qwen-orchestrator-full")
    parser.add_argument("--machine", default="L4")
    parser.add_argument("--image", default="pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-runtime", type=int, default=3 * 60 * 60)
    parser.add_argument("--user")
    parser.add_argument("--teamspace")
    parser.add_argument("--org")
    parser.add_argument(
        "--artifacts-local",
        type=Path,
        default=ROOT / "research/lightning_artifacts/qwen-full",
    )
    args = parser.parse_args()
    output_name = "20260628-qwen-small-logits-orchestrator-full-gpu"
    plan_name = "20260628-qwen-small-logits-orchestrator-full-gpu-plan.json"
    command = build_job_command(
        repo_url=args.repo_url,
        git_ref=args.git_ref,
        output_name=output_name,
        plan_name=plan_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
    result = submit_lightning_job(
        name=args.name,
        machine=args.machine,
        image=args.image,
        command=command,
        artifacts_local=args.artifacts_local,
        max_runtime=args.max_runtime,
        user=args.user,
        teamspace=args.teamspace,
        org=args.org,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
