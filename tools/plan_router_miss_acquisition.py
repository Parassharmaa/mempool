from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.acquisition_source import summarize_router_miss_neighborhoods
from mempool.routing_dataset import read_routing_records, validate_routing_records


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan data acquisition from held-out router miss neighborhoods."
    )
    parser.add_argument("--router-report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-neighborhoods", type=int, default=8)
    args = parser.parse_args()

    records = read_routing_records(args.dataset)
    errors = validate_routing_records(records)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1

    summary = summarize_router_miss_neighborhoods(
        router_report=read_json(args.router_report),
        routing_records=records,
        max_neighborhoods=args.max_neighborhoods,
    )
    payload = {
        "router_report": str(args.router_report),
        "dataset": str(args.dataset),
        "schema_version": "mempool.router_miss_acquisition.v1",
        **summary,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
