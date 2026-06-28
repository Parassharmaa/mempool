import unittest

from tools.build_repeated_routing_dataset import (
    build_records,
    filter_rows_by_evaluator_packages,
)


def row(worker_id: str, sample_index: int, passed: bool, latency_ms: int) -> dict:
    return {
        "benchmark_id": "bench",
        "task_id": "task-1",
        "task_family": "code",
        "prompt": "Write a function using os and zipfile.",
        "worker_id": worker_id,
        "model": worker_id,
        "sample_index": sample_index,
        "passed": passed,
        "failure_mode": None if passed else "test_failure",
        "latency_ms": latency_ms,
    }


def row_with_packages(
    worker_id: str,
    sample_index: int,
    passed: bool,
    latency_ms: int,
    packages: dict[str, bool],
) -> dict:
    item = row(worker_id, sample_index, passed, latency_ms)
    item["evaluator_required_packages"] = packages
    return item


def library_row(
    worker_id: str,
    sample_index: int,
    passed: bool,
    latency_ms: int,
    packages: dict[str, bool],
) -> dict:
    item = row_with_packages(worker_id, sample_index, passed, latency_ms, packages)
    item["prompt"] = (
        "Prepare text features.\n"
        "You may use these Python libraries if helpful: ['pandas', 're', 'sklearn'].\n"
    )
    return item


class BuildRepeatedRoutingDatasetTest(unittest.TestCase):
    def test_builds_targets_from_pass_rate_and_latency(self) -> None:
        records = build_records(
            [
                row("fast", 0, True, 10),
                row("fast", 1, True, 10),
                row("slow", 0, True, 30),
                row("slow", 1, False, 30),
            ]
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["target_worker_id"], "fast")
        workers = {worker["worker_id"]: worker for worker in records[0]["workers"]}
        self.assertEqual(workers["fast"]["pass_rate"], 1.0)
        self.assertEqual(workers["slow"]["pass_rate"], 0.5)
        self.assertGreater(
            records[0]["target_distribution"]["fast"],
            records[0]["target_distribution"]["slow"],
        )

    def test_filters_rows_by_required_evaluator_packages(self) -> None:
        rows = [
            row_with_packages("valid", 0, True, 10, {"numpy": True, "pandas": True}),
            row_with_packages("invalid", 0, True, 10, {"numpy": True, "pandas": False}),
            row("missing-provenance", 0, True, 10),
        ]

        kept, skipped = filter_rows_by_evaluator_packages(rows, ["numpy", "pandas"])

        self.assertEqual([item["worker_id"] for item in kept], ["valid"])
        self.assertEqual(
            [item["worker_id"] for item in skipped],
            ["invalid", "missing-provenance"],
        )

    def test_build_records_can_require_evaluator_packages(self) -> None:
        records = build_records(
            [
                row_with_packages("valid", 0, True, 10, {"numpy": True}),
                row_with_packages("invalid", 0, True, 10, {"numpy": False}),
            ],
            required_evaluator_packages=["numpy"],
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(
            [worker["worker_id"] for worker in records[0]["workers"]],
            ["valid"],
        )

    def test_prompt_features_use_evaluator_package_provenance(self) -> None:
        records = build_records(
            [
                library_row("a", 0, True, 10, {"pandas": True, "sklearn": True}),
                library_row("a", 1, True, 10, {"pandas": True, "sklearn": True}),
                library_row("b", 0, False, 20, {"pandas": True, "sklearn": True}),
                library_row("b", 1, False, 20, {"pandas": True, "sklearn": True}),
            ],
            required_evaluator_packages=["pandas", "sklearn"],
        )

        features = records[0]["prompt_features"]

        self.assertEqual(features["libraries"], ["pandas", "re", "sklearn"])
        self.assertEqual(features["missing_libraries"], [])
        self.assertEqual(features["missing_library_count"] if "missing_library_count" in features else 0, 0)


if __name__ == "__main__":
    unittest.main()
