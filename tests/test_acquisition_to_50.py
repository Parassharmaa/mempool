import json
import tempfile
import unittest
from pathlib import Path

from mempool.acquisition_to_50 import build_acquisition_to_50_plan


def routing_row(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "benchmark_id": "bench",
        "task_family": "code",
        "prompt": "solve",
        "prompt_features": {},
        "target_worker_id": "w1",
        "target_distribution": {"w1": 1.0},
        "workers": [
            {
                "worker_id": "w1",
                "model": "m1",
                "passed": True,
                "score": 1.0,
                "latency_ms": 10,
                "cost_usd": 0.0,
                "failure_mode": None,
                "reward": 1.0,
                "target_probability": 1.0,
            }
        ],
    }


class AcquisitionTo50Test(unittest.TestCase):
    def test_plans_unique_overselected_tasks_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            active = root / "active.jsonl"
            active.write_text(
                "".join(json.dumps(routing_row(f"t{index}")) + "\n" for index in range(3)),
                encoding="utf-8",
            )
            readiness = root / "readiness.json"
            readiness.write_text(json.dumps({"decision": "continue-m3-data-and-heads"}), encoding="utf-8")
            task_file = root / "tasks.json"
            task_file.write_text(
                json.dumps([{"id": "t1"}, {"id": "t3"}, {"id": "t4"}, {"id": "t5"}]),
                encoding="utf-8",
            )
            output_tasks = root / "wave.json"

            plan = build_acquisition_to_50_plan(
                active_dataset=active,
                readiness_report=readiness,
                candidate_task_files=[task_file],
                output_tasks=output_tasks,
                worker_pool=root / "pool.json",
                run_id="wave",
                min_tasks=5,
                repeat_count=2,
                overselect_multiplier=1.5,
                required_packages=["numpy"],
            )
            output_exists = output_tasks.exists()

        self.assertEqual(plan["active_rows"], 3)
        self.assertEqual(plan["rows_needed"], 2)
        self.assertEqual(plan["selected_task_ids"], ["t3", "t4", "t5"])
        self.assertTrue(plan["covers_rows_needed_if_all_merge_ready"])
        self.assertEqual(plan["estimated_model_calls"], 24)
        self.assertIn("--required-package numpy", plan["stages"][0]["commands"]["run_repeated_eval"])
        self.assertTrue(output_exists)


if __name__ == "__main__":
    unittest.main()
