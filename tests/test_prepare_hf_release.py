import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_hf_release import prepare_hf_release


class PrepareHfReleaseTest(unittest.TestCase):
    def test_prepares_dataset_and_model_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = root / "rows.jsonl"
            rows.write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
            plan = root / "plan.json"
            plan.write_text(json.dumps({"plan": True}), encoding="utf-8")
            readiness = root / "readiness.json"
            readiness.write_text(json.dumps({"ready": True}), encoding="utf-8")
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "train_report.json").write_text(
                json.dumps({"record_count": 1, "worker_ids": ["w"], "history": [{"loss": 1.0}]}),
                encoding="utf-8",
            )
            (model_dir / "eval_report.json").write_text(
                json.dumps({"worker_accuracy": 1.0, "workflow_accuracy": 1.0}),
                encoding="utf-8",
            )
            (model_dir / "qwen_logits_heads.pt").write_bytes(b"heads")

            manifest = prepare_hf_release(
                rows_path=rows,
                plan_path=plan,
                readiness_path=readiness,
                model_dir=model_dir,
                output_root=root / "export",
            )

            self.assertEqual(manifest["row_count"], 1)
            self.assertEqual(manifest["checkpoint_bytes"], 5)
            self.assertTrue((Path(manifest["dataset_dir"]) / "README.md").exists())
            self.assertTrue((Path(manifest["model_dir"]) / "qwen_logits_heads.pt").exists())
            self.assertTrue((Path(manifest["model_dir"]) / "eval_report.json").exists())


if __name__ == "__main__":
    unittest.main()
