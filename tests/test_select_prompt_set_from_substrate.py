import json
import tempfile
import unittest
from pathlib import Path

from tools.select_prompt_set_from_substrate import (
    extract_task_prompt,
    select_prompt_set,
)


def row(task_id: str, worker_id: str, prompt: str) -> dict:
    return {
        "task_id": task_id,
        "benchmark_id": "bench",
        "task_family": "bigcodebench_hard",
        "prompt_features": {
            "categories": ["filesystem"],
            "libraries": ["pathlib"],
            "missing_libraries": [],
        },
        "target": {"target_worker_id": worker_id},
        "messages": [
            {"role": "system", "content": "x"},
            {"role": "user", "content": f"metadata\ntask_prompt:\n{prompt}"},
        ],
    }


class SelectPromptSetFromSubstrateTest(unittest.TestCase):
    def test_extracts_task_prompt_from_user_message(self) -> None:
        self.assertEqual(extract_task_prompt(row("t1", "glm", "do work")), "do work")

    def test_selects_one_prompt_per_target_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            substrate = root / "substrate.jsonl"
            substrate.write_text(
                "\n".join(
                    json.dumps(item)
                    for item in [
                        row("t1", "glm", "do glm work"),
                        row("t2", "qwen", "do qwen work"),
                        row("t3", "kimi", "do kimi work"),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "prompts.json"

            payload = select_prompt_set(
                substrate_path=substrate,
                output_path=output,
                target_workers=["glm", "kimi"],
                limit_per_worker=1,
            )
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["prompt_count"], 2)
        self.assertEqual(saved["prompts"][0]["target_worker_id"], "glm")
        self.assertEqual(saved["prompts"][1]["prompt"], "do kimi work")
        self.assertEqual(saved["prompts"][0]["libraries"], ["pathlib"])


if __name__ == "__main__":
    unittest.main()
