from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.milestone_audit import apply_milestone_audit, audit_milestones


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit mempool milestone status from artifacts.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--milestones", type=Path, default=Path("research/programs/milestones.json"))
    parser.add_argument("--update-milestones", action="store_true")
    args = parser.parse_args()

    audit = audit_milestones(root=args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.update_milestones:
        updated = apply_milestone_audit(args.milestones, audit)
        args.milestones.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
