from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.terminal_bench import validate_terminal_bench_trajectories


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate metadata-safe Terminal-Bench trajectory JSONL."
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    records = read_jsonl(args.path)
    errors = validate_terminal_bench_trajectories(records)
    payload = {"records": len(records), "valid": not errors, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
