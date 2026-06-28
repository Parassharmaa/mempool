from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.small_orchestrator_readiness import audit_small_orchestrator_readiness


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit readiness to move from the active logits router to M5 small orchestrator training."
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
        "--orchestrator-contract",
        type=Path,
        default=Path("research/models/20260627-active-multi-head-orchestrator-contract.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-tasks", type=int, default=50)
    parser.add_argument("--min-target-workers", type=int, default=4)
    parser.add_argument("--min-workers-per-task", type=int, default=4)
    parser.add_argument("--min-loo-target-accuracy", type=float, default=0.75)
    parser.add_argument("--min-loo-solvable-pass-at-1", type=float, default=0.80)
    parser.add_argument("--max-loo-latency-regret-ms", type=float, default=750.0)
    parser.add_argument("--min-fallback-solvable-pass-at-1", type=float, default=0.90)
    parser.add_argument(
        "--allow-missing-workflow-head",
        action="store_true",
        help="Do not require an explicit workflow-kind logits head.",
    )
    parser.add_argument(
        "--allow-missing-abstain-head",
        action="store_true",
        help="Do not require an explicit abstain probability head.",
    )
    args = parser.parse_args()

    report = audit_small_orchestrator_readiness(
        registry_path=args.registry,
        fallback_report_path=args.fallback_report,
        regression_report_path=args.regression_report,
        orchestrator_contract_path=args.orchestrator_contract,
        min_tasks=args.min_tasks,
        min_target_workers=args.min_target_workers,
        min_workers_per_task=args.min_workers_per_task,
        min_loo_target_accuracy=args.min_loo_target_accuracy,
        min_loo_solvable_pass_at_1=args.min_loo_solvable_pass_at_1,
        max_loo_latency_regret_ms=args.max_loo_latency_regret_ms,
        min_fallback_solvable_pass_at_1=args.min_fallback_solvable_pass_at_1,
        require_workflow_head=not args.allow_missing_workflow_head,
        require_abstain_head=not args.allow_missing_abstain_head,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready_for_m5_small_orchestrator"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
