import unittest
from pathlib import Path

from mempool.smoke_benchmark import SmokeCodeBenchmarkAdapter, SmokeCodeTask, fixture_output


class SmokeBenchmarkTest(unittest.TestCase):
    def test_evaluates_fixture_outputs(self) -> None:
        adapter = SmokeCodeBenchmarkAdapter(Path("research/evals/smoke_code_tasks.json"))
        tasks = adapter.load_tasks()

        cheap_results = [
            adapter.evaluate_output(task, fixture_output("cheap-baseline", task.id))
            for task in tasks
        ]
        strong_results = [
            adapter.evaluate_output(task, fixture_output("strong-fixture", task.id))
            for task in tasks
        ]

        self.assertEqual(sum(result.passed for result in cheap_results), 5)
        self.assertEqual(sum(result.passed for result in strong_results), 10)

    def test_can_limit_tasks(self) -> None:
        adapter = SmokeCodeBenchmarkAdapter(Path("research/evals/smoke_code_tasks.json"))

        self.assertEqual(len(adapter.load_tasks(limit=2)), 2)

    def test_evaluation_timeout_is_structured_failure(self) -> None:
        adapter = SmokeCodeBenchmarkAdapter(
            Path("research/evals/smoke_code_tasks.json"),
            timeout_seconds=1,
        )
        task = SmokeCodeTask(
            id="timeout",
            prompt="Loop forever.",
            family="code_easy",
            function_name="task_func",
            tests=("assert task_func() == 1",),
        )

        result = adapter.evaluate_output(
            task,
            "def task_func():\n    while True:\n        pass\n",
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.failure_mode, "eval_timeout")
        self.assertEqual(result.metadata["timeout_seconds"], 1)


if __name__ == "__main__":
    unittest.main()
