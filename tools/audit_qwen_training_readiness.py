from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.qwen_logits_orchestrator import audit_qwen_training_readiness


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether the current environment can train the Qwen-small logits-head orchestrator."
    )
    parser.add_argument("--backend", choices=["transformers", "mlx"], default="transformers")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_qwen_training_readiness(
        backend=args.backend,
        require_gpu=args.require_gpu,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["ready_for_local_head_training"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
