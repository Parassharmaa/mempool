from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.orchestrated_executor import (
    execute_orchestrated_prompt,
    load_env_file,
)


ROOT = Path(__file__).resolve().parents[1]


def _csv_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Route a prompt with the trained orchestrator and execute it on the selected worker."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "research/models/20260628-m5-current-task-66task-multihead.json",
    )
    parser.add_argument(
        "--worker-pool",
        type=Path,
        default=ROOT / "research/evals/ollama_cloud_worker_pool_top4.json",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--task-id", default="ad-hoc")
    parser.add_argument("--benchmark-id", default="ad-hoc")
    parser.add_argument("--task-family", default="ad_hoc")
    parser.add_argument("--categories", default="")
    parser.add_argument("--libraries", default="")
    parser.add_argument("--missing-libraries", default="")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful worker model. Return the best answer you can.",
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    result = execute_orchestrated_prompt(
        model_path=args.model,
        worker_pool_path=args.worker_pool,
        prompt=args.prompt,
        task_id=args.task_id,
        benchmark_id=args.benchmark_id,
        task_family=args.task_family,
        categories=_csv_values(args.categories),
        libraries=_csv_values(args.libraries),
        missing_libraries=_csv_values(args.missing_libraries),
        system_prompt=args.system_prompt,
        dry_run=args.dry_run,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
