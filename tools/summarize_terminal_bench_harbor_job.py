from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.terminal_bench import summarize_harbor_job


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a metadata-safe summary for a Harbor Terminal-Bench job."
    )
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = summarize_harbor_job(args.job_dir)
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
