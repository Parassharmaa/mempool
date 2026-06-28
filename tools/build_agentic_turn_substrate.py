from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.agentic_turn_substrate import build_agentic_turn_substrate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export sanitized agentic trajectories as turn-level orchestration examples."
    )
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_agentic_turn_substrate(
        trajectory_path=args.trajectories,
        output_path=args.output,
        manifest_path=args.manifest_output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
