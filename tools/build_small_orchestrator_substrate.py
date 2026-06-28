from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.small_orchestrator_substrate import build_orchestrator_substrate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export routing records as supervised multi-head small-orchestrator examples."
    )
    parser.add_argument("--routing-dataset", type=Path, required=True)
    parser.add_argument(
        "--orchestrator-contract",
        type=Path,
        default=Path("research/models/20260627-active-multi-head-orchestrator-contract.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--reward-temperature", type=float, default=0.05)
    args = parser.parse_args()

    manifest = build_orchestrator_substrate(
        routing_dataset_path=args.routing_dataset,
        contract_path=args.orchestrator_contract,
        output_path=args.output,
        manifest_path=args.manifest_output,
        reward_temperature=args.reward_temperature,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
