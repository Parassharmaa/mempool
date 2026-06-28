from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.qwen_logits_orchestrator import run_transformers_evaluation


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained Qwen-small logits-head orchestrator checkpoint.")
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "research/datasets/20260628-qwen-small-logits-orchestrator-smoke-rows.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "research/models/20260628-qwen-small-logits-orchestrator-smoke/qwen_logits_heads.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_transformers_evaluation(
        training_rows_path=args.rows,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
