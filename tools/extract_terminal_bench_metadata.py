from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.terminal_bench import extract_terminal_bench_metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract metadata-only Terminal-Bench task rows from JSON/JSONL exports or task directories."
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows = extract_terminal_bench_metadata(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "metadata_rows": len(rows),
        "output": str(args.output),
        "selected_fields": sorted({key for row in rows for key in row}),
        "task_ids": [row["id"] for row in rows],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
