import tempfile
import unittest
from pathlib import Path

from tools.analyze_bigcodebench_dependency_gaps import (
    analyze_reports,
    missing_module,
    package_for_module,
)


class AnalyzeBigCodeBenchDependencyGapsTest(unittest.TestCase):
    def test_extracts_missing_module(self) -> None:
        self.assertEqual(
            missing_module("ModuleNotFoundError: No module named 'pandas'"),
            "pandas",
        )

    def test_maps_import_module_to_package_name(self) -> None:
        self.assertEqual(package_for_module("PIL"), "pillow")
        self.assertEqual(package_for_module("sklearn"), "scikit-learn")
        self.assertEqual(package_for_module("cgi"), "legacy-cgi")
        self.assertEqual(package_for_module("flask_login"), "flask-login")
        self.assertEqual(package_for_module("flask_mail"), "flask-mail")
        self.assertEqual(package_for_module("Levenshtein"), "python-Levenshtein")
        self.assertEqual(package_for_module("docx"), "python-docx")
        self.assertEqual(package_for_module("pandas"), "pandas")

    def test_analyzes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            path.write_text(
                """
{
  "scanned": 3,
  "eligible_count": 1,
  "canonical_probe": [
    {"task_id": "t1", "passed": true, "stderr_tail": ""},
    {"task_id": "t2", "passed": false, "failure_mode": "test_failure", "stderr_tail": "ModuleNotFoundError: No module named 'pandas'"},
    {"task_id": "t3", "passed": false, "failure_mode": "test_failure", "stderr_tail": "ModuleNotFoundError: No module named 'PIL'"}
  ]
}
""",
                encoding="utf-8",
            )

            result = analyze_reports([path])

        self.assertEqual(result["scanned"], 3)
        self.assertEqual(result["eligible"], 1)
        self.assertEqual(result["unique_dependency_blocked_tasks"], 2)
        self.assertEqual(result["ranked_packages"][0]["blocked_task_count"], 1)
        self.assertEqual(
            {item["package"] for item in result["ranked_packages"]},
            {"pandas", "pillow"},
        )


if __name__ == "__main__":
    unittest.main()
