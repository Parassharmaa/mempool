from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.outcome_mining import rank_outcome_sources, summarize_outcome_source

try:
    from tools.build_repeated_routing_dataset import build_records
    from tools.summarize_repeated_outcomes import read_jsonl
except ModuleNotFoundError:
    from build_repeated_routing_dataset import build_records
    from summarize_repeated_outcomes import read_jsonl


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank repeated outcome files by merge-ready stable routing signal."
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--latency-weight", type=float, default=0.05)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print a compact source summary instead of the full JSON report.",
    )
    args = parser.parse_args()

    summaries = []
    skipped = []
    for path in args.input:
        try:
            rows = read_jsonl(path)
            records = build_records(
                rows,
                temperature=args.temperature,
                latency_weight=args.latency_weight,
            )
        except Exception as exc:  # pragma: no cover - defensive report path
            skipped.append({"source": str(path), "error": str(exc)})
            continue
        summaries.append(
            summarize_outcome_source(
                source_path=path,
                records=records,
                min_pass_rate=args.min_pass_rate,
            )
        )

    ranked = rank_outcome_sources(summaries)
    report = {
        "schema_version": "mempool.stable_outcome_sources.v1",
        "input_count": len(args.input),
        "skipped": skipped,
        "min_pass_rate": args.min_pass_rate,
        "temperature": args.temperature,
        "latency_weight": args.latency_weight,
        "ranked_sources": ranked[: args.top],
        "source_count": len(summaries),
    }
    write_json(args.output, report)
    if args.quiet:
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "input_count": report["input_count"],
                    "source_count": report["source_count"],
                    "skipped_count": len(skipped),
                    "top_sources": [
                        {
                            "source": item["source"],
                            "score": item["score"],
                            "merge_ready_records": item["merge_ready_records"],
                            "exclusive_stable_nonqwen_targets": item[
                                "exclusive_stable_nonqwen_targets"
                            ],
                            "stable_nonqwen_targets": item["stable_nonqwen_targets"],
                            "broad_pass_latency_rows": item["broad_pass_latency_rows"],
                        }
                        for item in ranked[: min(args.top, 10)]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
