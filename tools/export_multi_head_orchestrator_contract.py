from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.orchestrator_contract import build_multi_head_contract


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a bridge multi-head action-logit contract for the small orchestrator."
    )
    parser.add_argument("--registry", type=Path, default=Path("research/policies/active_policy.json"))
    parser.add_argument(
        "--fallback-report",
        type=Path,
        default=Path("research/evals/results/20260627-selected-gated-fallback-active-23task.json"),
    )
    parser.add_argument(
        "--regression-report",
        type=Path,
        default=Path("research/evals/results/20260627-selected-gated-fallback-regression-slices.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/models/20260627-active-multi-head-orchestrator-contract.json"),
    )
    parser.add_argument("--direct-workflow-probability", type=float, default=0.85)
    parser.add_argument("--fallback-workflow-probability", type=float, default=0.15)
    parser.add_argument("--abstain-probability", type=float, default=0.05)
    args = parser.parse_args()

    contract = build_multi_head_contract(
        active_policy_registry=args.registry,
        fallback_report_path=args.fallback_report,
        regression_report_path=args.regression_report,
        output_path=args.output,
        direct_workflow_probability=args.direct_workflow_probability,
        fallback_workflow_probability=args.fallback_workflow_probability,
        abstain_probability=args.abstain_probability,
    )
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0 if contract["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
