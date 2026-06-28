import unittest
from pathlib import Path

from mempool.benchmark import BenchmarkPlan


class BenchmarkPlanTest(unittest.TestCase):
    def test_loads_bigcodebench_plan(self) -> None:
        plan = BenchmarkPlan.from_json(
            Path("research/evals/bigcodebench_hard_plan.json")
        )

        self.assertEqual(plan.benchmark_id, "bigcodebench-hard-instruct")
        self.assertEqual(plan.stage("smoke")["target_tasks"], 10)
        self.assertIn("pass_at_1", plan.primary_metrics)

    def test_loads_terminal_bench_pilot_plan(self) -> None:
        plan = BenchmarkPlan.from_json(
            Path("research/evals/terminal_bench_2p1_pilot_plan.json")
        )

        self.assertEqual(plan.benchmark_id, "terminal-bench-2.1")
        self.assertEqual(plan.stage("metadata_selection")["target_tasks"], 5)
        self.assertIn("task_success_rate", plan.primary_metrics)
        self.assertIn(
            "active_bigcodebench_logits_router_as_initial_worker_selector",
            plan.baselines,
        )


if __name__ == "__main__":
    unittest.main()
