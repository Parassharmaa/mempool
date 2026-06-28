from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.terminal_bench import refresh_terminal_bench_preflight_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh a Terminal-Bench preflight summary from safe Harbor metadata."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = refresh_terminal_bench_preflight_summary(args.summary)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
