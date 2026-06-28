from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.model_catalog import build_worker_pool


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a worker-pool config from a saved OpenAI-compatible model catalog."
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--base-url", default="https://ollama.com/v1")
    parser.add_argument("--api-key-env", default="OLLAMA_API_KEY")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    worker_pool = build_worker_pool(
        catalog,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout_seconds,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({k: v for k, v in worker_pool.items() if k != "selection_report"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(worker_pool["selection_report"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(worker_pool["selection_report"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
