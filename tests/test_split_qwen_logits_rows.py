import json
import tempfile
import unittest
from pathlib import Path

from tools.split_qwen_logits_rows import split_rows


class SplitQwenLogitsRowsTest(unittest.TestCase):
    def test_splits_rows_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = root / "rows.jsonl"
            rows.write_text(
                "".join(json.dumps({"task_id": f"task-{index}", "text": str(index)}) + "\n" for index in range(10)),
                encoding="utf-8",
            )
            manifest = split_rows(
                rows_path=rows,
                train_output=root / "train.jsonl",
                heldout_output=root / "heldout.jsonl",
                manifest_output=root / "manifest.json",
                heldout_fraction=0.2,
                seed=1,
            )
            train_lines = (root / "train.jsonl").read_text(encoding="utf-8").splitlines()
            heldout_lines = (root / "heldout.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(manifest["record_count"], 10)
        self.assertEqual(manifest["train_count"], 8)
        self.assertEqual(manifest["heldout_count"], 2)
        self.assertEqual(len(train_lines), 8)
        self.assertEqual(len(heldout_lines), 2)


if __name__ == "__main__":
    unittest.main()
