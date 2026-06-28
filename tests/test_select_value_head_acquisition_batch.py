import unittest
import tempfile
from pathlib import Path

from tools.select_value_head_acquisition_batch import (
    build_error_seeds,
    build_contrast_priors,
    outcome_files_from_dirs,
    record_prompt,
    select_value_head_acquisition_batch,
    value_error_task_ids,
)


class StaticRouter:
    def distribution(self, record: dict) -> dict[str, float]:
        prompt = record["prompt"].lower()
        if "deepseek" in prompt:
            return {"qwen": 0.45, "deepseek": 0.40, "kimi": 0.15}
        if "network" in prompt:
            return {"qwen": 0.42, "kimi": 0.39, "deepseek": 0.19}
        return {"qwen": 0.60, "glm": 0.30, "kimi": 0.10}


def substrate_record(task_id: str, prompt: str) -> dict:
    return {
        "task_id": task_id,
        "task_family": "bigcodebench_hard",
        "prompt": prompt,
        "prompt_features": {
            "categories": ["filesystem"] if "file" in prompt else ["network"],
            "libraries": ["pathlib"] if "file" in prompt else ["requests"],
            "primary_category": "filesystem" if "file" in prompt else "network",
        },
        "target": {"target_worker_id": "qwen"},
        "workers": [],
    }


def task(task_id: str, prompt: str) -> dict:
    return {
        "id": task_id,
        "prompt": prompt,
        "family": "bigcodebench_hard",
        "function_name": "task_func",
        "tests": [],
    }


class SelectValueHeadAcquisitionBatchTest(unittest.TestCase):
    def test_expands_outcome_directories_to_jsonl_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "a.jsonl"
            second = root / "b.jsonl"
            ignored = root / "c.json"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")
            ignored.write_text("{}", encoding="utf-8")

            self.assertEqual(outcome_files_from_dirs([root]), [first, second])

    def test_extracts_prompt_from_substrate_messages(self) -> None:
        record = {
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": (
                        "Choose an orchestration action.\n"
                        "task_prompt:\n"
                        "Use pathlib to inspect files.\n"
                        "You may use these Python libraries if helpful: ['pathlib']."
                    ),
                },
            ]
        }

        self.assertEqual(record_prompt(record), "Use pathlib to inspect files.")

    def test_extracts_value_head_error_task_ids(self) -> None:
        report = {
            "leave_one_out": {
                "examples": [
                    {"task_id": "fn", "value_label": 1.0, "attempts": [{"worker_id": "qwen"}]},
                    {
                        "task_id": "fp",
                        "value_label": 0.0,
                        "attempts": [{"worker_id": "qwen"}, {"worker_id": "kimi"}],
                    },
                    {"task_id": "ok", "value_label": 0.0, "attempts": [{"worker_id": "qwen"}]},
                ]
            }
        }

        self.assertEqual(
            value_error_task_ids(report),
            {"false_negative": {"fn"}, "false_positive": {"fp"}},
        )

    def test_builds_error_seeds_with_ranked_workers(self) -> None:
        substrate = [
            substrate_record("fn", "Use pathlib files for deepseek rescue."),
            substrate_record("fp", "Use requests for network fetch."),
        ]
        source_report = {
            "leave_one_out": {
                "predictions": [
                    {"task_id": "fn", "worker_distribution": {"qwen": 0.6, "deepseek": 0.4}},
                    {"task_id": "fp", "worker_distribution": {"qwen": 0.7, "kimi": 0.3}},
                ]
            }
        }
        value_report = {
            "leave_one_out": {
                "examples": [
                    {"task_id": "fn", "value_label": 1.0, "attempts": [{"worker_id": "qwen"}]},
                    {
                        "task_id": "fp",
                        "value_label": 0.0,
                        "attempts": [{"worker_id": "qwen"}, {"worker_id": "kimi"}],
                    },
                ]
            }
        }

        seeds = build_error_seeds(
            substrate_records=substrate,
            source_report=source_report,
            value_report=value_report,
        )

        self.assertEqual(seeds["false_negative"][0]["second_worker"], "deepseek")
        self.assertEqual(seeds["false_negative"][0]["libraries"], ["pathlib"])
        self.assertEqual(seeds["false_positive"][0]["second_worker"], "kimi")
        self.assertEqual(seeds["false_positive"][0]["categories"], ["network"])

    def test_selects_balanced_value_error_neighborhoods(self) -> None:
        tasks = [
            task("candidate-deepseek", "Use pathlib files for deepseek rescue."),
            task("candidate-network", "Use requests for network fetch."),
            task("candidate-http", "Use requests for http parsing."),
            task("candidate-other", "Sort general strings."),
        ]
        false_negative_seeds = [
            {
                "seed_task_id": "fn",
                "libraries": ["pathlib"],
                "categories": ["filesystem"],
                "primary_category": "filesystem",
                "environment_risk": 0,
                "plausibility_score": 1.0,
                "second_worker": "deepseek",
            }
        ]
        false_positive_seeds = [
            {
                "seed_task_id": "fp",
                "libraries": ["requests"],
                "categories": ["network"],
                "primary_category": "network",
                "environment_risk": 3,
                "plausibility_score": 3.0,
                "second_worker": "kimi",
            }
        ]

        selection = select_value_head_acquisition_batch(
            tasks=tasks,
            router=StaticRouter(),
            false_negative_seeds=false_negative_seeds,
            false_positive_seeds=false_positive_seeds,
            exclude_ids=set(),
            limit=4,
        )

        groups = {item["target_group"] for item in selection["selected"]}
        self.assertEqual(selection["candidate_count"], 4)
        self.assertIn("missed-positive-value", groups)
        self.assertIn("false-spend-value", groups)
        self.assertEqual(len(selection["selected_task_ids"]), 4)

    def test_builds_contrast_priors_from_outcome_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "outcomes.jsonl"
            rows = [
                {
                    "task_id": "mixed",
                    "prompt": "Use pathlib files for deepseek rescue.",
                    "worker_id": "qwen",
                    "passed": True,
                },
                {
                    "task_id": "mixed",
                    "prompt": "Use pathlib files for deepseek rescue.",
                    "worker_id": "kimi",
                    "passed": False,
                },
                {
                    "task_id": "uniform",
                    "prompt": "Use requests for network fetch.",
                    "worker_id": "qwen",
                    "passed": True,
                },
                {
                    "task_id": "uniform",
                    "prompt": "Use requests for network fetch.",
                    "worker_id": "kimi",
                    "passed": True,
                },
            ]
            path.write_text(
                "".join(__import__("json").dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            priors = {prior["seed_task_id"]: prior for prior in build_contrast_priors([path])}

        self.assertEqual(priors["mixed"]["worker_count"], 2)
        self.assertEqual(priors["mixed"]["pass_rate"], 0.5)
        self.assertEqual(priors["mixed"]["disagreement_score"], 1.0)
        self.assertEqual(priors["uniform"]["pass_rate"], 1.0)
        self.assertEqual(priors["uniform"]["uniform_score"], 1.0)

    def test_contrast_prior_promotes_disagreement_candidates(self) -> None:
        tasks = [
            task("candidate-mixed", "Use pathlib files for deepseek rescue."),
            task("candidate-uniform", "Use requests for network fetch."),
        ]
        false_negative_seeds = [
            {
                "seed_task_id": "fn",
                "libraries": ["pathlib"],
                "categories": ["filesystem"],
                "primary_category": "filesystem",
                "environment_risk": 0,
                "plausibility_score": 1.0,
                "second_worker": "deepseek",
            }
        ]
        false_positive_seeds = [
            {
                "seed_task_id": "fp",
                "libraries": ["requests"],
                "categories": ["network"],
                "primary_category": "network",
                "environment_risk": 0,
                "plausibility_score": 1.0,
                "second_worker": "kimi",
            }
        ]
        contrast_priors = [
            {
                "seed_task_id": "mixed",
                "libraries": ["pathlib"],
                "categories": ["filesystem"],
                "primary_category": "filesystem",
                "environment_risk": 0,
                "plausibility_score": 1.0,
                "disagreement_score": 1.0,
                "uniform_score": 0.0,
            },
            {
                "seed_task_id": "uniform",
                "libraries": ["requests"],
                "categories": ["network"],
                "primary_category": "network",
                "environment_risk": 0,
                "plausibility_score": 1.0,
                "disagreement_score": 0.0,
                "uniform_score": 1.0,
            },
        ]

        selection = select_value_head_acquisition_batch(
            tasks=tasks,
            router=StaticRouter(),
            false_negative_seeds=false_negative_seeds,
            false_positive_seeds=false_positive_seeds,
            contrast_priors=contrast_priors,
            exclude_ids=set(),
            limit=1,
        )

        self.assertEqual(selection["contrast_prior_count"], 2)
        self.assertEqual(selection["selected_task_ids"], ["candidate-mixed"])
        self.assertGreater(selection["selected"][0]["contrast_similarity"], 0.0)

    def test_max_uniform_similarity_filters_uniform_neighbors(self) -> None:
        tasks = [
            task("candidate-mixed", "Sort general strings."),
            task("candidate-uniform", "Use files in a directory."),
        ]
        seed = {
            "seed_task_id": "seed",
            "libraries": ["pathlib"],
            "categories": ["filesystem"],
            "primary_category": "filesystem",
            "environment_risk": 0,
            "plausibility_score": 1.0,
            "second_worker": "deepseek",
        }
        contrast_priors = [
            {
                "seed_task_id": "mixed",
                "libraries": [],
                "categories": ["general"],
                "primary_category": "general",
                "environment_risk": 0,
                "plausibility_score": 1.0,
                "disagreement_score": 1.0,
                "uniform_score": 0.0,
            },
            {
                "seed_task_id": "uniform",
                "libraries": [],
                "categories": ["filesystem"],
                "primary_category": "filesystem",
                "environment_risk": 0,
                "plausibility_score": 1.0,
                "disagreement_score": 0.0,
                "uniform_score": 1.0,
            },
        ]

        selection = select_value_head_acquisition_batch(
            tasks=tasks,
            router=StaticRouter(),
            false_negative_seeds=[seed],
            false_positive_seeds=[seed],
            contrast_priors=contrast_priors,
            max_uniform_similarity=1.0,
            exclude_ids=set(),
            limit=2,
        )

        self.assertEqual(selection["max_uniform_similarity"], 1.0)
        self.assertEqual(selection["selected_task_ids"], ["candidate-mixed"])


if __name__ == "__main__":
    unittest.main()
