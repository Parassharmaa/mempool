from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.terminal_bench import (
    harbor_job_to_terminal_bench_trajectories,
    validate_terminal_bench_trajectories,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert safe Harbor trial results into Terminal-Bench trajectory JSONL."
    )
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--selected-workflow", default="terminal-agent")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = harbor_job_to_terminal_bench_trajectories(
        args.job_dir,
        run_id=args.run_id,
        agent_id=args.agent_id,
        worker_id=args.worker_id,
        policy_id=args.policy_id,
        selected_workflow=args.selected_workflow,
    )
    errors = validate_terminal_bench_trajectories(records)
    if errors:
        for error in errors:
            print(error)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({"records": len(records), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
