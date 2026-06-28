import json
import tempfile
import unittest
from pathlib import Path

from mempool.orchestrator_contract import (
    REQUIRED_HEADS,
    build_multi_head_contract,
    validate_multi_head_contract,
)


class OrchestratorContractTest(unittest.TestCase):
    def write_json(self, path: Path, payload: dict) -> Path:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_builds_valid_multi_head_contract_from_active_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = self.write_json(
                root / "model.json",
                {
                    "router": {
                        "worker_ids": ["w1", "w2"],
                        "feature_names": ["bias"],
                        "weights": [[0.1], [-0.1]],
                    }
                },
            )
            registry = self.write_json(
                root / "registry.json",
                {"active": {"model": str(model), "dataset": "dataset.jsonl"}},
            )
            fallback = self.write_json(
                root / "fallback.json",
                {
                    "evaluation": {
                        "policy": "gated-fallback",
                        "fallback_rate": 0.25,
                        "solvable_pass_at_1": 1.0,
                    }
                },
            )
            regression = self.write_json(root / "regression.json", {"passed": True})
            output = root / "contract.json"

            contract = build_multi_head_contract(
                active_policy_registry=registry,
                fallback_report_path=fallback,
                regression_report_path=regression,
                output_path=output,
            )
            output_exists = output.exists()

        self.assertTrue(contract["validation"]["valid"])
        self.assertEqual(set(contract["heads"]), REQUIRED_HEADS)
        self.assertEqual(contract["heads"]["worker_distribution"]["worker_ids"], ["w1", "w2"])
        self.assertEqual(contract["heads"]["workflow_kind"]["labels"], ["direct", "verify_then_fallback"])
        self.assertTrue(output_exists)

    def test_validation_rejects_missing_heads(self) -> None:
        report = validate_multi_head_contract({"heads": {"worker_distribution": {}}})

        self.assertFalse(report["valid"])
        self.assertTrue(any("missing required heads" in reason for reason in report["reasons"]))


if __name__ == "__main__":
    unittest.main()
