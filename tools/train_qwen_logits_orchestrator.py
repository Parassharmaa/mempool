from __future__ import annotations

import argparse
import json
from pathlib import Path

from mempool.qwen_logits_orchestrator import (
    DEFAULT_BASE_MODEL,
    QwenLogitsTrainingConfig,
    build_qwen_logits_training_plan,
    build_qwen_logits_training_plan_from_rows,
    run_transformers_training,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or train a Qwen-small logits-head orchestrator from mempool substrate rows."
    )
    parser.add_argument(
        "--substrate",
        type=Path,
        default=ROOT / "research/datasets/20260628-m5-current-task-66task-substrate.jsonl",
    )
    parser.add_argument(
        "--training-rows",
        type=Path,
        help="Use an existing qwen_logits_orchestrator_row JSONL file instead of rebuilding rows from substrate.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--backend", choices=["transformers", "mlx"], default="transformers")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    config = QwenLogitsTrainingConfig(
        base_model=args.base_model,
        backend=args.backend,
        train_backbone=args.train_backbone,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    if args.training_rows:
        plan = build_qwen_logits_training_plan_from_rows(
            rows_path=args.training_rows,
            output_path=args.plan_output,
            config=config,
        )
        training_rows = args.training_rows
    else:
        plan = build_qwen_logits_training_plan(
            substrate_path=args.substrate,
            output_path=args.plan_output,
            rows_output_path=args.rows_output,
            config=config,
        )
        training_rows = args.rows_output
    if not args.train:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.backend != "transformers":
        raise SystemExit("only transformers training is implemented; MLX path is planned but not implemented yet")
    if not args.output_dir:
        raise SystemExit("--output-dir is required with --train")
    report = run_transformers_training(
        training_rows_path=training_rows,
        output_dir=args.output_dir,
        config=config,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
