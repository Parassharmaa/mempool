import json
import tempfile
import unittest
from pathlib import Path

from mempool.adaptive_refresh import build_privacy_manifest, build_refresh_cycle


class AdaptiveRefreshTest(unittest.TestCase):
    def write_json(self, path: Path, payload: dict) -> Path:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_quarantines_when_gate_quarantines_even_with_guardrails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset.jsonl"
            dataset.write_text("{}\n", encoding="utf-8")
            model = self.write_json(root / "model.json", {"model_type": "x"})
            report = self.write_json(root / "report.json", {"leave_one_out": {"target_accuracy": 0.5}})
            gate = self.write_json(root / "gate.json", {"decision": "quarantine", "reasons": ["low accuracy"]})
            registry = self.write_json(root / "registry.json", {"active": {"model": "old"}})
            ledger = root / "ledger.jsonl"
            ledger.write_text("{}\n", encoding="utf-8")
            privacy = root / "privacy.json"
            build_privacy_manifest(distilled_dataset_path=dataset, output_path=privacy)
            output = root / "cycle.json"

            cycle = build_refresh_cycle(
                cycle_id="cycle",
                distilled_dataset_path=dataset,
                candidate_model_path=model,
                candidate_report_path=report,
                gate_path=gate,
                active_registry_path=registry,
                output_path=output,
                ledger_path=ledger,
                privacy_manifest_path=privacy,
            )

        self.assertEqual(cycle["decision"], "quarantine")
        self.assertIn("low accuracy", cycle["reasons"])
        self.assertTrue(all(check["passed"] for check in cycle["guardrails"] if check["name"] != "candidate_artifact"))
        self.assertFalse(cycle["promotion"]["allowed"])

    def test_blocks_promotion_when_privacy_guardrail_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset.jsonl"
            dataset.write_text("{}\n", encoding="utf-8")
            model = self.write_json(root / "model.json", {"model_type": "x"})
            report = self.write_json(root / "report.json", {"leave_one_out": {"target_accuracy": 1.0}})
            gate = self.write_json(root / "gate.json", {"decision": "promote", "reasons": []})
            registry = self.write_json(root / "registry.json", {"active": {"model": "old"}})
            ledger = root / "ledger.jsonl"
            ledger.write_text("{}\n", encoding="utf-8")
            privacy = self.write_json(
                root / "privacy.json",
                {
                    "memory_scope": "general",
                    "training_scope": "general",
                    "contains_raw_private_text": True,
                    "explicit_private_training_approval": False,
                },
            )

            cycle = build_refresh_cycle(
                cycle_id="cycle",
                distilled_dataset_path=dataset,
                candidate_model_path=model,
                candidate_report_path=report,
                gate_path=gate,
                active_registry_path=registry,
                output_path=root / "cycle.json",
                ledger_path=ledger,
                privacy_manifest_path=privacy,
            )

        self.assertEqual(cycle["decision"], "quarantine")
        self.assertTrue(any("privacy_filter" in reason for reason in cycle["reasons"]))

    def test_carries_promotion_profile_from_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset.jsonl"
            dataset.write_text("{}\n", encoding="utf-8")
            model = self.write_json(root / "model.json", {"model_type": "x"})
            report = self.write_json(root / "report.json", {"leave_one_out": {"target_accuracy": 1.0}})
            gate = self.write_json(
                root / "gate.json",
                {
                    "decision": "quarantine",
                    "promotion_profile": "preserve_accuracy",
                    "reasons": ["accuracy drop"],
                },
            )
            registry = self.write_json(root / "registry.json", {"active": {"model": "old"}})
            ledger = root / "ledger.jsonl"
            ledger.write_text("{}\n", encoding="utf-8")
            privacy = root / "privacy.json"
            build_privacy_manifest(distilled_dataset_path=dataset, output_path=privacy)

            cycle = build_refresh_cycle(
                cycle_id="cycle",
                distilled_dataset_path=dataset,
                candidate_model_path=model,
                candidate_report_path=report,
                gate_path=gate,
                active_registry_path=registry,
                output_path=root / "cycle.json",
                ledger_path=ledger,
                privacy_manifest_path=privacy,
            )

        self.assertEqual(cycle["promotion_profile"], "preserve_accuracy")
        self.assertEqual(cycle["promotion"]["profile"], "preserve_accuracy")


if __name__ == "__main__":
    unittest.main()
