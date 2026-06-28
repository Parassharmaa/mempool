from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def filter_worker_pool(pool: dict[str, Any], worker_ids: list[str]) -> dict[str, Any]:
    requested = list(dict.fromkeys(worker_ids))
    workers_by_id = {str(worker["id"]): worker for worker in pool.get("workers", [])}
    missing = [worker_id for worker_id in requested if worker_id not in workers_by_id]
    if missing:
        raise ValueError(f"worker id(s) not found: {', '.join(missing)}")
    filtered = {key: value for key, value in pool.items() if key != "workers"}
    filtered["workers"] = [workers_by_id[worker_id] for worker_id in requested]
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter a worker-pool JSON file to selected workers.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--worker-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pool = json.loads(args.input.read_text(encoding="utf-8"))
    filtered = filter_worker_pool(pool, args.worker_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(filtered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "worker_ids": [worker["id"] for worker in filtered["workers"]],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
