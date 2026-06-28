import unittest

from tools.plan_dependency_profile_expansion import plan_expansion


class PlanDependencyProfileExpansionTest(unittest.TestCase):
    def test_skips_current_profile_and_counts_incremental_unlocks(self) -> None:
        gap_report = {
            "scanned": 10,
            "eligible": 3,
            "unique_dependency_blocked_tasks": 5,
            "ranked_packages": [
                {
                    "package": "pandas",
                    "blocked_task_count": 3,
                    "modules": ["pandas"],
                    "task_ids": ["t1", "t2", "t3"],
                },
                {
                    "package": "scipy",
                    "blocked_task_count": 2,
                    "modules": ["scipy"],
                    "task_ids": ["t3", "t4"],
                },
                {
                    "package": "bs4",
                    "blocked_task_count": 1,
                    "modules": ["bs4"],
                    "task_ids": ["t5"],
                },
            ],
        }

        plan = plan_expansion(
            gap_report,
            current_packages=["pandas"],
            package_limit=2,
        )

        self.assertEqual(plan["expanded_profile"], ["pandas", "scipy", "bs4"])
        self.assertEqual(plan["projected_unique_unlocks"], 3)
        self.assertEqual(
            [item["package"] for item in plan["selected_packages"]],
            ["scipy", "bs4"],
        )
        self.assertEqual(plan["selected_packages"][0]["incremental_task_ids"], ["t3", "t4"])
        self.assertEqual(plan["source_scanned_tasks"], 10)

    def test_normalizes_current_profile_aliases(self) -> None:
        gap_report = {
            "ranked_packages": [
                {
                    "package": "scikit-learn",
                    "blocked_task_count": 2,
                    "modules": ["sklearn"],
                    "task_ids": ["t1", "t2"],
                },
                {
                    "package": "nltk",
                    "blocked_task_count": 1,
                    "modules": ["nltk"],
                    "task_ids": ["t3"],
                },
            ],
        }

        plan = plan_expansion(
            gap_report,
            current_packages=["sklearn"],
            package_limit=1,
        )

        self.assertEqual(plan["current_profile"], ["scikit-learn"])
        self.assertEqual(plan["expanded_profile"], ["scikit-learn", "nltk"])
        self.assertEqual(plan["selected_packages"][0]["package"], "nltk")

    def test_normalizes_benchmark_module_aliases(self) -> None:
        gap_report = {
            "ranked_packages": [
                {
                    "package": "legacy-cgi",
                    "blocked_task_count": 1,
                    "modules": ["cgi"],
                    "task_ids": ["t1"],
                },
                {
                    "package": "flask-login",
                    "blocked_task_count": 1,
                    "modules": ["flask_login"],
                    "task_ids": ["t2"],
                },
                {
                    "package": "python-docx",
                    "blocked_task_count": 1,
                    "modules": ["docx"],
                    "task_ids": ["t3"],
                },
                {
                    "package": "pillow",
                    "blocked_task_count": 1,
                    "modules": ["PIL"],
                    "task_ids": ["t4"],
                },
                {
                    "package": "opencv-python",
                    "blocked_task_count": 1,
                    "modules": ["cv2"],
                    "task_ids": ["t5"],
                },
                {
                    "package": "python-Levenshtein",
                    "blocked_task_count": 1,
                    "modules": ["Levenshtein"],
                    "task_ids": ["t6"],
                },
                {
                    "package": "nltk",
                    "blocked_task_count": 1,
                    "modules": ["nltk"],
                    "task_ids": ["t7"],
                },
            ],
        }

        plan = plan_expansion(
            gap_report,
            current_packages=[
                "cgi",
                "flask_login",
                "docx",
                "PIL",
                "cv2",
                "Levenshtein",
            ],
            package_limit=1,
        )

        self.assertEqual(
            plan["current_profile"],
            [
                "legacy-cgi",
                "flask-login",
                "python-docx",
                "pillow",
                "opencv-python",
                "python-Levenshtein",
            ],
        )
        self.assertEqual(plan["selected_packages"][0]["package"], "nltk")

    def test_rejects_empty_package_limit(self) -> None:
        with self.assertRaises(ValueError):
            plan_expansion({"ranked_packages": []}, package_limit=0)


if __name__ == "__main__":
    unittest.main()
