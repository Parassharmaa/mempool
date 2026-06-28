from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mempool.qwen_logits_orchestrator import run_transformers_prediction


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Route text with a trained Qwen logits-head orchestrator checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "research/models/20260628-qwen-small-logits-orchestrator-full-gpu-l40s/qwen_logits_heads.pt",
    )
    parser.add_argument("--text", action="append", help="Text/task to route. Can be passed multiple times.")
    parser.add_argument("--input-jsonl", type=Path, help="Optional JSONL file with `text` fields.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    texts = list(args.text or [])
    if args.input_jsonl:
        for line in args.input_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            texts.append(str(row["text"]))
    if not texts:
        texts = [sys.stdin.read()]
    texts = [text for text in texts if text.strip()]
    if not texts:
        raise SystemExit("provide --text, --input-jsonl, or stdin text")

    prediction = run_transformers_prediction(
        checkpoint_path=args.checkpoint,
        texts=texts,
    )
    rendered = json.dumps(prediction, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
