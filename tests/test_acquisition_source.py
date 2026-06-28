import unittest
from pathlib import Path

from mempool.acquisition_source import (
    AcquisitionSource,
    build_execution_manifest,
    rank_acquisition_sources,
    summarize_router_miss_neighborhoods,
)


class AcquisitionSourceTest(unittest.TestCase):
    def test_ranks_nonqwen_pressure_above_exhausted_offset(self) -> None:
        plan = rank_acquisition_sources(
            [
                AcquisitionSource(
                    id="normal-offset676",
                    kind="normal_offset_scan",
                    cost_hint="no_model_calls",
                    report={
                        "eligible_count": 0,
                        "scanned": 0,
                        "next_offset": 676,
                        "selected_task_ids": [],
                    },
                ),
                AcquisitionSource(
                    id="nonqwen-pressure",
                    kind="nonqwen_specialist_pressure",
                    cost_hint="no_model_calls",
                    report={
                        "candidate_count": 33,
                        "selected_task_ids": ["a", "b", "c"],
                    },
                ),
            ]
        )

        self.assertEqual(plan["recommended_source_id"], "nonqwen-pressure")
        self.assertTrue(plan["ranked_sources"][1]["exhausted"])

    def test_quarantined_dependency_profile_can_still_rank_as_evidence_source(self) -> None:
        plan = rank_acquisition_sources(
            [
                AcquisitionSource(
                    id="small-pool",
                    kind="specialist_pressure",
                    cost_hint="model_calls",
                    report={"candidate_count": 2, "selected_task_ids": ["a"]},
                ),
                AcquisitionSource(
                    id="expanded-profile",
                    kind="dependency_profile",
                    cost_hint="no_model_calls",
                    report={
                        "candidate_count": 20,
                        "selected_task_ids": ["x", "y"],
                        "decision": "quarantine",
                    },
                ),
            ]
        )

        self.assertEqual(plan["recommended_source_id"], "expanded-profile")
        self.assertIn("prior gate", " ".join(plan["recommendation"]["reasons"]))

    def test_build_execution_manifest_includes_worker_calls_and_requirements(self) -> None:
        manifest = build_execution_manifest(
            source_plan={
                "recommended_source_id": "nonqwen-pressure",
                "recommendation": {"score": 1.0},
            },
            source_report={
                "candidate_count": 3,
                "selected_task_ids": ["t1"],
                "selected": [{"task_id": "t1", "environment_risk": 0}],
            },
            selected_tasks=[{"id": "t1", "prompt": "p"}],
            worker_pool={"workers": [{"id": "glm"}, {"id": "kimi"}]},
            worker_pool_path=Path("pool.json"),
            tasks_output=Path("tasks.json"),
            run_id="run",
            output_path=Path("out.json"),
            outcomes_path=Path("out.jsonl"),
            repeat_count=2,
            eval_timeout_seconds=20,
            required_packages=["numpy"],
            python_executable=".venv/bin/python",
        )

        self.assertEqual(manifest["call_count"], 4)
        self.assertEqual(manifest["worker_ids_to_run"], ["glm", "kimi"])
        self.assertEqual(manifest["selected_task_metadata"]["t1"]["environment_risk"], 0)
        self.assertIn("--required-package numpy", manifest["run_command"])
        self.assertIn(".venv/bin/python", manifest["run_command"])

    def test_outcome_evidence_penalizes_failed_nonqwen_source(self) -> None:
        plan = rank_acquisition_sources(
            [
                AcquisitionSource(
                    id="nonqwen-pressure",
                    kind="nonqwen_specialist_pressure",
                    cost_hint="no_model_calls",
                    report={"candidate_count": 33, "selected_task_ids": ["a", "b"]},
                    evidence=(
                        {
                            "by_task": [
                                {"task_id": "a", "universal_failure": True},
                                {
                                    "task_id": "b",
                                    "universal_failure": False,
                                    "candidate_for_conversion": True,
                                    "best_worker_id": "ollama-cloud-qwen3-coder-480b",
                                },
                            ]
                        },
                    ),
                ),
                AcquisitionSource(
                    id="expanded-profile",
                    kind="dependency_profile",
                    cost_hint="no_model_calls",
                    report={"candidate_count": 20, "selected_task_ids": ["x", "y"]},
                ),
            ]
        )

        self.assertEqual(plan["recommended_source_id"], "expanded-profile")
        self.assertIn("stable non-Qwen", " ".join(plan["ranked_sources"][1]["reasons"]))

    def test_summarizes_router_miss_neighborhoods(self) -> None:
        report = {
            "leave_one_out": {
                "examples": [
                    {
                        "task_id": "t1",
                        "target_worker_id": "glm",
                        "predicted_worker_id": "qwen",
                    },
                    {
                        "task_id": "t2",
                        "target_worker_id": "glm",
                        "predicted_worker_id": "qwen",
                    },
                    {
                        "task_id": "t3",
                        "target_worker_id": "qwen",
                        "predicted_worker_id": "qwen",
                    },
                ]
            }
        }
        records = [
            {
                "task_id": "t1",
                "prompt": "Zip filesystem files",
                "prompt_features": {
                    "categories": ["filesystem"],
                    "libraries": ["zipfile", "os"],
                },
            },
            {
                "task_id": "t2",
                "prompt": "Create archive from pathlib matches",
                "prompt_features": {
                    "categories": ["filesystem"],
                    "libraries": ["pathlib", "zipfile"],
                },
            },
            {
                "task_id": "t3",
                "prompt": "Already correct",
                "prompt_features": {"categories": ["general"], "libraries": []},
            },
        ]

        summary = summarize_router_miss_neighborhoods(
            router_report=report,
            routing_records=records,
        )

        self.assertEqual(summary["miss_count"], 2)
        self.assertEqual(summary["neighborhood_count"], 1)
        neighborhood = summary["neighborhoods"][0]
        self.assertEqual(neighborhood["target_worker_id"], "glm")
        self.assertEqual(neighborhood["predicted_worker_id"], "qwen")
        self.assertEqual(neighborhood["category_key"], "filesystem")
        self.assertEqual(neighborhood["task_ids"], ["t1", "t2"])


if __name__ == "__main__":
    unittest.main()
