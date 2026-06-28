from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.adaptive_refresh import build_privacy_manifest, build_refresh_cycle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an auditable adaptive-memory refresh cycle artifact."
    )
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--distilled-dataset", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--policy-gate", type=Path, required=True)
    parser.add_argument("--active-registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--privacy-manifest", type=Path)
    parser.add_argument("--write-privacy-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    privacy_manifest = args.privacy_manifest
    if args.write_privacy_manifest is not None:
        build_privacy_manifest(
            distilled_dataset_path=args.distilled_dataset,
            output_path=args.write_privacy_manifest,
        )
        privacy_manifest = args.write_privacy_manifest

    cycle = build_refresh_cycle(
        cycle_id=args.cycle_id,
        distilled_dataset_path=args.distilled_dataset,
        candidate_model_path=args.candidate_model,
        candidate_report_path=args.candidate_report,
        gate_path=args.policy_gate,
        active_registry_path=args.active_registry,
        output_path=args.output,
        ledger_path=args.ledger,
        privacy_manifest_path=privacy_manifest,
    )
    print(json.dumps(cycle, indent=2, sort_keys=True))
    return 0 if cycle["decision"] == "promote" else 2


if __name__ == "__main__":
    raise SystemExit(main())
