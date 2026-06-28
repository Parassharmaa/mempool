from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.terminal_bench import evaluate_terminal_bench_readiness


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether Terminal-Bench preflight summaries are ready for worker runs."
    )
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = evaluate_terminal_bench_readiness(args.summary)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
