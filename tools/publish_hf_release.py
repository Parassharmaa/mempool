from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def hf_whoami() -> str:
    completed = run(["hf", "auth", "whoami"])
    return parse_hf_whoami(completed.stdout)


def parse_hf_whoami(output: str) -> str:
    line = output.strip().splitlines()[-1].strip()
    for part in line.split():
        if part.startswith("user="):
            return part.split("=", 1)[1].strip()
    return line


def create_and_upload(*, repo_id: str, repo_type: str, folder: Path) -> None:
    run(["hf", "repo", "create", repo_id, "--type", repo_type, "--exist-ok"])
    run(["hf", "upload", repo_id, str(folder), ".", "--repo-type", repo_type])


def publish_hf_release(
    *,
    export_root: Path,
    namespace: str | None = None,
    dataset_name: str = "mempool-qwen-logits-orchestrator-rows",
    model_name: str = "mempool-qwen-logits-orchestrator-smoke",
) -> dict[str, str]:
    owner = namespace or hf_whoami()
    dataset_repo = f"{owner}/{dataset_name}"
    model_repo = f"{owner}/{model_name}"
    create_and_upload(repo_id=dataset_repo, repo_type="dataset", folder=export_root / "dataset")
    create_and_upload(repo_id=model_repo, repo_type="model", folder=export_root / "model")
    result = {
        "dataset_repo": dataset_repo,
        "model_repo": model_repo,
        "dataset_url": f"https://huggingface.co/datasets/{dataset_repo}",
        "model_url": f"https://huggingface.co/{model_repo}",
    }
    (export_root / "published.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and upload mempool Hugging Face dataset/model repos.")
    parser.add_argument("--export-root", type=Path, default=ROOT / "research/hf_export/qwen-logits-smoke")
    parser.add_argument("--namespace")
    parser.add_argument("--dataset-name", default="mempool-qwen-logits-orchestrator-rows")
    parser.add_argument("--model-name", default="mempool-qwen-logits-orchestrator-smoke")
    args = parser.parse_args()
    result = publish_hf_release(
        export_root=args.export_root,
        namespace=args.namespace,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
