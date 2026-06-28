import tempfile
import unittest
from pathlib import Path

from tools.select_latency_safe_matched_controls import select_latency_safe_matched_controls


def task(task_id: str, prompt: str) -> dict:
    return {
        "id": task_id,
        "family": "bigcodebench_hard",
        "prompt": prompt,
        "function_name": "task_func",
        "tests": [],
    }


class SelectLatencySafeMatchedControlsTest(unittest.TestCase):
    def test_selects_balanced_safe_and_unsafe_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "routing.jsonl"
            dataset.write_text(
                "\n".join(
                    [
                        '{"task_id":"safe-seed","prompt":"Zip files with pathlib os.","benchmark_id":"b","task_family":"unit","prompt_features":{},"target_worker_id":"a","target_distribution":{"a":0.5,"b":0.5},"workers":[{"worker_id":"a","model":"a","passed":true,"pass_rate":1.0,"score":1.0,"latency_ms":1,"cost_usd":0,"failure_mode":null,"reward":1.0,"target_probability":0.5},{"worker_id":"b","model":"b","passed":true,"pass_rate":1.0,"score":1.0,"latency_ms":2,"cost_usd":0,"failure_mode":null,"reward":1.0,"target_probability":0.5}]}',
                        '{"task_id":"unsafe-seed","prompt":"Fetch http URL with requests.","benchmark_id":"b","task_family":"unit","prompt_features":{},"target_worker_id":"a","target_distribution":{"a":0.5,"b":0.5},"workers":[{"worker_id":"a","model":"a","passed":true,"pass_rate":1.0,"score":1.0,"latency_ms":1,"cost_usd":0,"failure_mode":null,"reward":1.0,"target_probability":0.5},{"worker_id":"b","model":"b","passed":false,"pass_rate":0.0,"score":0.0,"latency_ms":2,"cost_usd":0,"failure_mode":"failed","reward":0.0,"target_probability":0.5}]}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            tasks = [
                task("safe-close", "Archive zip files with pathlib os."),
                task("unsafe-close", "Download http url with requests."),
                task("far", "Compute random statistics mean."),
            ]

            selection = select_latency_safe_matched_controls(
                tasks=tasks,
                seed_dataset=dataset,
                exclude_ids=set(),
                limit=2,
                per_label_limit=1,
            )

        self.assertEqual(len(selection["selected_task_ids"]), 2)
        self.assertEqual(
            {item["label"] for item in selection["selected"]},
            {"latency_safe_candidate", "unsafe_control_candidate"},
        )
        self.assertEqual(selection["safe_seed_count"], 1)
        self.assertEqual(selection["unsafe_seed_count"], 1)


if __name__ == "__main__":
    unittest.main()
