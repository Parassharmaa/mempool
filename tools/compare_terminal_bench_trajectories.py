from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.terminal_bench import (
    compare_terminal_bench_trajectories,
    read_terminal_bench_trajectory_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare metadata-safe Terminal-Bench trajectory JSONL files."
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for path in args.input:
        records.extend(read_terminal_bench_trajectory_jsonl(path))
    report = compare_terminal_bench_trajectories(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
