import json
import tempfile
import unittest
from pathlib import Path

import mempool.bigcodebench as bigcodebench
from mempool.benchmark import BenchmarkResult
from mempool.bigcodebench import (
    BigCodeBenchSource,
    classify_task,
    canonical_output,
    ensure_unittest_runner,
    load_rows_from_path,
    materialize_tasks,
    merge_scan_reports,
    merge_task_lists,
    normalize_bigcodebench_row,
    parse_prompt_libraries,
    probe_canonical_solution,
    scan_canonical_pass_rows,
    select_minipilot_tasks,
)


SAMPLE_ROW = {
    "task_id": "BigCodeBench/999",
    "instruct_prompt": "Return the square of a number.",
    "complete_prompt": "def task_func(x):",
    "test": (
        "import unittest\n\n"
        "class TestCases(unittest.TestCase):\n"
        "    def test_square(self):\n"
        "        self.assertEqual(task_func(3), 9)\n"
    ),
    "entry_point": "task_func",
    "libs": "['math']",
}


class BigCodeBenchMaterializerTest(unittest.TestCase):
    def test_builds_canonical_output(self) -> None:
        row = {"code_prompt": "def task_func(x):\n", "canonical_solution": "    return x\n"}

        self.assertEqual(canonical_output(row), "def task_func(x):\n    return x\n")

    def test_normalizes_row_to_smoke_task(self) -> None:
        task = normalize_bigcodebench_row(SAMPLE_ROW)

        self.assertEqual(task.id, "bigcodebench-hard-BigCodeBench-999")
        self.assertEqual(task.family, "bigcodebench_hard")
        self.assertEqual(task.function_name, "task_func")
        self.assertIn("Return the square", task.prompt)
        self.assertIn("Define `task_func` exactly", task.prompt)
        self.assertIn("unittest.main()", task.tests[0])

    def test_preserves_existing_unittest_runner(self) -> None:
        source = "import unittest\nunittest.main()\n"

        self.assertEqual(ensure_unittest_runner(source), source)

    def test_loads_dataset_server_row_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.json"
            path.write_text(
                json.dumps({"rows": [{"row": SAMPLE_ROW}]}),
                encoding="utf-8",
            )

            rows = load_rows_from_path(path)

        self.assertEqual(rows, [SAMPLE_ROW])

    def test_materializes_json_records(self) -> None:
        records = materialize_tasks([SAMPLE_ROW])

        self.assertEqual(records[0]["id"], "bigcodebench-hard-BigCodeBench-999")
        self.assertEqual(records[0]["tests"][0].count("unittest.main()"), 1)

    def test_parses_prompt_libraries(self) -> None:
        prompt = "Do work.\n\nYou may use these Python libraries if helpful: ['os', 'csv']."

        self.assertEqual(parse_prompt_libraries(prompt), ["os", "csv"])

    def test_classifies_task_traits(self) -> None:
        task = materialize_tasks([SAMPLE_ROW])[0]
        analysis = classify_task(task)

        self.assertEqual(analysis["task_id"], "bigcodebench-hard-BigCodeBench-999")
        self.assertIn("math", analysis["libraries"])
        self.assertIn("general", analysis["categories"])
        self.assertEqual(analysis["missing_libraries"], [])

    def test_selects_diverse_minipilot_tasks(self) -> None:
        tasks = [
            {
                "id": "subprocess-task",
                "prompt": "Run a process.\n\nYou may use these Python libraries if helpful: ['subprocess'].",
                "tests": ["import unittest\nclass TestCases(unittest.TestCase):\n    pass\n"],
            },
            {
                "id": "filesystem-task",
                "prompt": "Zip files in a directory.\n\nYou may use these Python libraries if helpful: ['zipfile', 'os'].",
                "tests": ["import unittest\nclass TestCases(unittest.TestCase):\n    pass\n"],
            },
            {
                "id": "datasci-task",
                "prompt": "Use pandas on a dataframe.\n\nYou may use these Python libraries if helpful: ['pandas'].",
                "tests": ["import unittest\nclass TestCases(unittest.TestCase):\n    pass\n"],
            },
        ]

        selection = select_minipilot_tasks(tasks, count=3)

        self.assertEqual(
            selection["selected_task_ids"],
            ["subprocess-task", "filesystem-task", "datasci-task"],
        )

    def test_selects_only_eligible_minipilot_tasks(self) -> None:
        tasks = [
            {
                "id": "subprocess-task",
                "prompt": "Run a process.\n\nYou may use these Python libraries if helpful: ['subprocess'].",
                "tests": [""],
            },
            {
                "id": "filesystem-task",
                "prompt": "Zip files in a directory.\n\nYou may use these Python libraries if helpful: ['zipfile', 'os'].",
                "tests": [""],
            },
            {
                "id": "datasci-task",
                "prompt": "Use pandas on a dataframe.\n\nYou may use these Python libraries if helpful: ['pandas'].",
                "tests": [""],
            },
        ]

        selection = select_minipilot_tasks(
            tasks,
            count=2,
            eligibility={
                "subprocess-task": True,
                "filesystem-task": False,
                "datasci-task": True,
            },
        )

        self.assertEqual(selection["selected_task_ids"], ["subprocess-task", "datasci-task"])

    def test_probes_canonical_solution_with_adapter(self) -> None:
        class FakeAdapter:
            def evaluate_output(self, task, output):
                return BenchmarkResult(
                    task_id=task.id,
                    passed="return" in output,
                    score=1.0,
                    failure_mode=None,
                    metadata={"stderr_tail": "ok"},
                )

        probe = probe_canonical_solution(
            {**SAMPLE_ROW, "code_prompt": "def task_func(x):\n", "canonical_solution": "    return x * x\n"},
            adapter=FakeAdapter(),
        )

        self.assertTrue(probe.passed)
        self.assertEqual(probe.stderr_tail, "ok")

    def test_scans_until_target_passes(self) -> None:
        rows = [
            {**SAMPLE_ROW, "task_id": "BigCodeBench/1", "canonical_solution": "pass\n"},
            {**SAMPLE_ROW, "task_id": "BigCodeBench/2", "canonical_solution": "return 2\n"},
            {**SAMPLE_ROW, "task_id": "BigCodeBench/3", "canonical_solution": "return 3\n"},
        ]

        class FakeAdapter:
            def evaluate_output(self, task, output):
                return BenchmarkResult(
                    task_id=task.id,
                    passed="return" in output,
                    score=1.0 if "return" in output else 0.0,
                    failure_mode=None if "return" in output else "test_failure",
                    metadata={"stderr_tail": ""},
                )

        original_fetch_rows = bigcodebench.fetch_rows
        try:
            bigcodebench.fetch_rows = lambda source: rows[source.offset : source.offset + source.limit]
            scan = scan_canonical_pass_rows(
                BigCodeBenchSource(offset=0, limit=2),
                adapter=FakeAdapter(),
                target_passes=2,
                page_size=2,
                max_rows=3,
            )
        finally:
            bigcodebench.fetch_rows = original_fetch_rows

        self.assertEqual(scan["scanned"], 3)
        self.assertEqual(scan["next_offset"], 3)
        self.assertEqual(len(scan["passed_rows"]), 2)

    def test_merges_task_lists_by_id(self) -> None:
        merged = merge_task_lists(
            [
                [{"id": "b", "prompt": "first"}, {"id": "a", "prompt": "second"}],
                [{"id": "b", "prompt": "ignored"}, {"id": "c", "prompt": "third"}],
            ]
        )

        self.assertEqual([task["id"] for task in merged], ["a", "b", "c"])
        self.assertEqual([task["prompt"] for task in merged], ["second", "first", "third"])

    def test_merges_scan_reports(self) -> None:
        merged_tasks = [{"id": "a"}, {"id": "b"}]
        report = merge_scan_reports(
            [
                {
                    "offset": 0,
                    "next_offset": 10,
                    "scanned": 10,
                    "analysis": [{"task_id": "a", "kind": "one"}],
                    "canonical_probe": [{"task_id": "a", "passed": True}],
                },
                {
                    "offset": 10,
                    "next_offset": 15,
                    "scanned": 5,
                    "analysis": [{"task_id": "b", "kind": "two"}],
                    "canonical_probe": [{"task_id": "b", "passed": True}],
                },
            ],
            merged_tasks,
        )

        self.assertEqual(report["eligible_count"], 2)
        self.assertEqual(report["scanned"], 15)
        self.assertEqual(report["next_offset"], 15)
        self.assertEqual(report["selected_task_ids"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
