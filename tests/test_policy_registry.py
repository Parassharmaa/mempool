import unittest

from tools.policy_registry import apply_refresh, rollback


def refresh(decision: str = "promote") -> dict:
    return {
        "decision": decision,
        "timestamp": "2026-06-27T00:00:00+00:00",
        "warnings": ["careful"],
        "candidate": {
            "report": "model.json",
            "dataset": "dataset.jsonl",
            "loo": {"target_accuracy": 0.75},
            "target_mix": {"target_worker_count": 3},
        },
    }


class PolicyRegistryTest(unittest.TestCase):
    def test_apply_refresh_sets_active_and_history(self) -> None:
        registry = {"active": None, "previous": None, "history": []}

        updated = apply_refresh(registry, refresh())

        self.assertEqual(updated["active"]["model"], "model.json")
        self.assertIsNone(updated["previous"])
        self.assertEqual(updated["history"][0]["action"], "promote")
        self.assertEqual(updated["history"][0]["warnings"], ["careful"])

    def test_apply_refresh_rejects_quarantine(self) -> None:
        registry = {"active": None, "previous": None, "history": []}

        with self.assertRaises(ValueError):
            apply_refresh(registry, refresh(decision="quarantine"))

    def test_rollback_swaps_active_and_previous(self) -> None:
        registry = {
            "active": {"model": "new.json"},
            "previous": {"model": "old.json"},
            "history": [],
        }

        updated = rollback(registry)

        self.assertEqual(updated["active"]["model"], "old.json")
        self.assertEqual(updated["previous"]["model"], "new.json")
        self.assertEqual(updated["history"][0]["action"], "rollback")


if __name__ == "__main__":
    unittest.main()
