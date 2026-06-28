from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mempool.terminal_bench import DEFAULT_CATEGORIES, select_terminal_bench_pilot


def read_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected a list of metadata rows in {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a metadata-only Terminal-Bench pilot subset."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--exclude-id", action="append", default=[])
    parser.add_argument("--preferred-category", action="append", default=[])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = select_terminal_bench_pilot(
        read_rows(args.metadata),
        limit=args.limit,
        excluded_ids=set(args.exclude_id),
        preferred_categories=tuple(args.preferred_category or DEFAULT_CATEGORIES),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
