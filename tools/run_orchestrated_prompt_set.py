from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.orchestrated_executor import (
    execute_fixed_worker_prompt,
    execute_orchestrated_prompt,
    flatten_orchestrated_execution,
    load_env_file,
)


ROOT = Path(__file__).resolve().parents[1]


def _csv_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        prompts = payload.get("prompts", [])
    else:
        prompts = payload
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt set needs a non-empty prompts list")
    return [dict(item) for item in prompts]


def _prompt_categories(item: dict[str, Any]) -> list[str]:
    values = item.get("categories", [])
    return values if isinstance(values, list) else _csv_values(str(values))


def _prompt_libraries(item: dict[str, Any]) -> list[str]:
    values = item.get("libraries", [])
    return values if isinstance(values, list) else _csv_values(str(values))


def run_prompt_set(
    *,
    prompt_set: Path,
    model: Path,
    worker_pool: Path,
    fixed_worker_id: str,
    output: Path,
    outcomes_output: Path,
    env_file: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    load_env_file(env_file)
    prompts = _load_prompts(prompt_set)
    executions = []
    rows = []
    for item in prompts:
        common = {
            "model_path": model,
            "worker_pool_path": worker_pool,
            "prompt": str(item["prompt"]),
            "task_id": str(item.get("task_id") or item.get("id") or "ad-hoc"),
            "benchmark_id": str(item.get("benchmark_id") or "local-prompt-set"),
            "task_family": str(item.get("task_family") or "ad_hoc"),
            "categories": _prompt_categories(item),
            "libraries": _prompt_libraries(item),
            "missing_libraries": [],
            "system_prompt": str(item.get("system_prompt") or "You are a helpful worker model. Return the best answer you can."),
            "dry_run": dry_run,
        }
        orchestrated = execute_orchestrated_prompt(**common)
        fixed = execute_fixed_worker_prompt(
            **common,
            fixed_worker_id=fixed_worker_id,
        )
        executions.extend([orchestrated, fixed])
        rows.extend([flatten_orchestrated_execution(orchestrated), flatten_orchestrated_execution(fixed)])

    summary = {
        "schema_version": "mempool.orchestrated_prompt_set_comparison.v1",
        "prompt_set": str(prompt_set),
        "model": str(model),
        "worker_pool": str(worker_pool),
        "fixed_worker_id": fixed_worker_id,
        "dry_run": dry_run,
        "prompt_count": len(prompts),
        "execution_count": len(executions),
        "executions": executions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    outcomes_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outcomes_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare orchestrator-selected execution against a fixed worker on a small prompt set."
    )
    parser.add_argument("--prompt-set", type=Path, required=True)
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
    parser.add_argument("--fixed-worker-id", default="ollama-cloud-qwen3-coder-480b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outcomes-output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = run_prompt_set(
        prompt_set=args.prompt_set,
        model=args.model,
        worker_pool=args.worker_pool,
        fixed_worker_id=args.fixed_worker_id,
        output=args.output,
        outcomes_output=args.outcomes_output,
        env_file=args.env_file,
        dry_run=args.dry_run,
    )
    print(json.dumps({key: summary[key] for key in ["prompt_count", "execution_count", "dry_run"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
