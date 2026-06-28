from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.routing_dataset import read_routing_records, validate_routing_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate routing dataset JSONL.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    records = read_routing_records(args.path)
    errors = validate_routing_records(records)
    payload = {"records": len(records), "valid": not errors, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
