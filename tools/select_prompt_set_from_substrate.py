from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extract_task_prompt(record: dict[str, Any]) -> str:
    messages = record.get("messages") or []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        marker = "task_prompt:\n"
        if marker in content:
            return content.split(marker, 1)[1].strip()
    raise ValueError(f"record {record.get('task_id')} does not include task_prompt")


def select_prompt_set(
    *,
    substrate_path: Path,
    output_path: Path,
    target_workers: list[str],
    limit_per_worker: int = 1,
    max_prompt_chars: int = 1200,
) -> dict[str, Any]:
    records = read_jsonl(substrate_path)
    selected = []
    seen_tasks = set()
    for worker_id in target_workers:
        count = 0
        for record in records:
            target = record.get("target") or {}
            task_id = str(record.get("task_id"))
            if target.get("target_worker_id") != worker_id or task_id in seen_tasks:
                continue
            prompt_features = record.get("prompt_features") or {}
            selected.append(
                {
                    "task_id": task_id,
                    "benchmark_id": record.get("benchmark_id"),
                    "task_family": record.get("task_family"),
                    "target_worker_id": worker_id,
                    "categories": list(prompt_features.get("categories") or []),
                    "libraries": list(prompt_features.get("libraries") or []),
                    "missing_libraries": list(prompt_features.get("missing_libraries") or []),
                    "prompt": extract_task_prompt(record)[:max_prompt_chars],
                }
            )
            seen_tasks.add(task_id)
            count += 1
            if count >= limit_per_worker:
                break
    payload = {
        "schema_version": "mempool.prompt_set_from_substrate.v1",
        "substrate": str(substrate_path),
        "target_workers": target_workers,
        "limit_per_worker": limit_per_worker,
        "prompt_count": len(selected),
        "prompts": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select prompt-set examples from a small-orchestrator substrate by target worker."
    )
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-worker", action="append", required=True)
    parser.add_argument("--limit-per-worker", type=int, default=1)
    parser.add_argument("--max-prompt-chars", type=int, default=1200)
    args = parser.parse_args()
    payload = select_prompt_set(
        substrate_path=args.substrate,
        output_path=args.output,
        target_workers=args.target_worker,
        limit_per_worker=args.limit_per_worker,
        max_prompt_chars=args.max_prompt_chars,
    )
    print(json.dumps({key: payload[key] for key in ["prompt_count", "target_workers"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
