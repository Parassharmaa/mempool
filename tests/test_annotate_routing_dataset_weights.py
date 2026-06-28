import unittest

from tools.annotate_routing_dataset_weights import annotate_weights


class AnnotateRoutingDatasetWeightsTest(unittest.TestCase):
    def test_applies_default_and_task_specific_weights(self) -> None:
        records = [
            {"task_id": "old", "prompt": "old task"},
            {"task_id": "new", "prompt": "new task"},
        ]

        annotated = annotate_weights(
            records,
            default_weight=1.0,
            task_weights={"new": 0.2},
        )

        self.assertEqual(annotated[0]["training_weight"], 1.0)
        self.assertEqual(annotated[1]["training_weight"], 0.2)
        self.assertNotIn("training_weight", records[0])


if __name__ == "__main__":
    unittest.main()
