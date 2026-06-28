from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_score(row: dict[str, Any], seed: int) -> str:
    key = f"{seed}:{row.get('task_id')}:{row.get('text', '')[:80]}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def split_rows(
    *,
    rows_path: Path,
    train_output: Path,
    heldout_output: Path,
    manifest_output: Path,
    heldout_fraction: float = 0.2,
    seed: int = 7,
) -> dict[str, Any]:
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be between 0 and 1")
    rows = read_jsonl(rows_path)
    ordered = sorted(rows, key=lambda row: stable_score(row, seed))
    heldout_count = max(1, round(len(ordered) * heldout_fraction)) if ordered else 0
    heldout = ordered[:heldout_count]
    train = ordered[heldout_count:]
    train_output.parent.mkdir(parents=True, exist_ok=True)
    heldout_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    train_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in train),
        encoding="utf-8",
    )
    heldout_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in heldout),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "mempool.qwen_logits_split.v1",
        "source": str(rows_path),
        "train_output": str(train_output),
        "heldout_output": str(heldout_output),
        "seed": seed,
        "heldout_fraction": heldout_fraction,
        "record_count": len(rows),
        "train_count": len(train),
        "heldout_count": len(heldout),
        "heldout_task_ids": [row.get("task_id") for row in heldout],
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Split Qwen logits training rows into deterministic train/heldout files.")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--heldout-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    manifest = split_rows(
        rows_path=args.rows,
        train_output=args.train_output,
        heldout_output=args.heldout_output,
        manifest_output=args.manifest_output,
        heldout_fraction=args.heldout_fraction,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
