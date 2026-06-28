from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.bigcodebench import BigCodeBenchSource, fetch_rows, load_rows_from_path, materialize_tasks


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize BigCodeBench-Hard tasks into mempool smoke format.")
    parser.add_argument("--input", type=Path, help="Optional local BigCodeBench JSON/JSONL export.")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "evals" / "bigcodebench_hard_smoke_tasks.json")
    parser.add_argument("--dataset", default="bigcode/bigcodebench-hard")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="v0.1.4")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--mode", choices=("instruct", "complete"), default="instruct")
    args = parser.parse_args()

    rows = load_rows_from_path(args.input) if args.input else fetch_rows(
        BigCodeBenchSource(
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            offset=args.offset,
            limit=args.limit,
        )
    )
    tasks = materialize_tasks(rows[: args.limit], mode=args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "task_count": len(tasks),
                "dataset": args.dataset,
                "split": args.split,
                "mode": args.mode,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
