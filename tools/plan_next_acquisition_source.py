from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.acquisition_source import AcquisitionSource, rank_acquisition_sources


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_source(value: str) -> AcquisitionSource:
    parts = value.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "source must be id:kind:cost_hint:report_path"
        )
    source_id, kind, cost_hint, report_path = parts
    return AcquisitionSource(
        id=source_id,
        kind=kind,
        cost_hint=cost_hint,
        report=read_json(Path(report_path)),
    )


def parse_evidence(value: str) -> tuple[str, dict]:
    source_id, separator, path = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("evidence must be source_id:path")
    return source_id, read_json(Path(path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank candidate acquisition sources from existing evidence."
    )
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        required=True,
        help="id:kind:cost_hint:report_path",
    )
    parser.add_argument(
        "--evidence",
        action="append",
        type=parse_evidence,
        default=[],
        help="source_id:summary_path",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence_by_source: dict[str, list[dict]] = {}
    for source_id, evidence in args.evidence:
        evidence_by_source.setdefault(source_id, []).append(evidence)
    sources = [
        AcquisitionSource(
            id=source.id,
            kind=source.kind,
            cost_hint=source.cost_hint,
            report=source.report,
            evidence=tuple(evidence_by_source.get(source.id, [])),
        )
        for source in args.source
    ]
    plan = rank_acquisition_sources(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
