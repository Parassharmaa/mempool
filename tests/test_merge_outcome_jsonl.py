import tempfile
import unittest
from pathlib import Path

from tools.merge_outcome_jsonl import merge_rows


class MergeOutcomeJsonlTest(unittest.TestCase):
    def test_merges_unique_rows_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left = Path(tmpdir) / "left.jsonl"
            right = Path(tmpdir) / "right.jsonl"
            left.write_text(
                '{"run_id":"r1","worker_id":"w2","task_id":"t1","sample_index":0}\n'
                '{"run_id":"r1","worker_id":"w1","task_id":"t1","sample_index":0}\n',
                encoding="utf-8",
            )
            right.write_text(
                '{"run_id":"r1","worker_id":"w1","task_id":"t1","sample_index":0}\n'
                '{"run_id":"r2","worker_id":"w1","task_id":"t0","sample_index":0}\n',
                encoding="utf-8",
            )

            rows = merge_rows([left, right])

        self.assertEqual(
            [(row["task_id"], row["worker_id"], row["run_id"]) for row in rows],
            [("t0", "w1", "r2"), ("t1", "w1", "r1"), ("t1", "w2", "r1")],
        )


if __name__ == "__main__":
    unittest.main()
