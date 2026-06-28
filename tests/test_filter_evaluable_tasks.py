import json
import tempfile
import unittest
from pathlib import Path

from tools.filter_evaluable_tasks import filter_evaluable_tasks, imported_roots


class FilterEvaluableTasksTest(unittest.TestCase):
    def test_extracts_import_roots_from_source_lines(self) -> None:
        source = "\n".join(
            [
                "import os, json as js",
                "from pathlib import Path",
                "import pandas as pd",
                "not an import",
            ]
        )

        self.assertEqual(imported_roots(source), ["json", "os", "pandas", "pathlib"])

    def test_filters_tasks_with_missing_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks = root / "tasks.json"
            tasks.write_text(
                json.dumps(
                    [
                        {
                            "id": "task-ok",
                            "prompt": "```python\nimport os\ndef task_func():\n```",
                            "family": "unit",
                            "function_name": "task_func",
                            "tests": ["import unittest\nassert task_func() == 1"],
                        },
                        {
                            "id": "task-missing",
                            "prompt": "```python\nimport definitely_missing_mempool_dep\ndef task_func():\n```",
                            "family": "unit",
                            "function_name": "task_func",
                            "tests": ["assert task_func() == 1"],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "evaluable.json"
            report = root / "report.json"

            payload = filter_evaluable_tasks(task_paths=[tasks], output_path=output, report_path=report)
            kept = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["input_task_count"], 2)
        self.assertEqual(payload["evaluable_task_count"], 1)
        self.assertEqual(kept[0]["id"], "task-ok")
        self.assertEqual(payload["missing_import_roots"], ["definitely_missing_mempool_dep"])


if __name__ == "__main__":
    unittest.main()
