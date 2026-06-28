import unittest
import tempfile
from pathlib import Path

from tools.select_logits_router_temperature import (
    run_selection,
    select_best_candidate,
    temperature_slug,
)


def candidate(
    decision: str,
    accuracy: float,
    pass_at_1: float,
    latency_regret_ms: float,
    mean_kl: float,
) -> dict:
    return {
        "refresh": {
            "decision": decision,
            "candidate": {
                "loo": {
                    "target_accuracy": accuracy,
                    "pass_at_1": pass_at_1,
                    "mean_latency_regret_ms": latency_regret_ms,
                    "mean_kl": mean_kl,
                }
            },
        }
    }


class SelectLogitsRouterTemperatureTest(unittest.TestCase):
    def test_temperature_slug_is_path_safe(self) -> None:
        self.assertEqual(temperature_slug(0.1), "0p1")
        self.assertEqual(temperature_slug(1.0), "1")

    def test_selects_best_promotable_candidate(self) -> None:
        weaker = candidate("promote", 0.7, 0.8, 900.0, 0.2)
        stronger = candidate("promote", 0.8, 0.9, 500.0, 0.4)
        quarantined = candidate("quarantine", 1.0, 1.0, 0.0, 0.0)

        selected = select_best_candidate([weaker, stronger, quarantined])

        self.assertIs(selected, stronger)

    def test_returns_none_when_all_candidates_are_quarantined(self) -> None:
        selected = select_best_candidate([candidate("quarantine", 0.9, 0.9, 0.0, 0.1)])

        self.assertIsNone(selected)

    def test_breaks_ties_on_lower_latency_regret(self) -> None:
        slow = candidate("promote", 0.8, 0.9, 900.0, 0.1)
        fast = candidate("promote", 0.8, 0.9, 100.0, 0.2)

        selected = select_best_candidate([slow, fast])

        self.assertIs(selected, fast)

    def test_selection_records_promotion_profile(self) -> None:
        record = {
            "task_id": "t1",
            "prompt": "Return a value",
            "task_family": "unit",
            "target_worker_id": "w1",
            "target_distribution": {"w1": 1.0},
            "workers": [
                {
                    "worker_id": "w1",
                    "passed": True,
                    "pass_rate": 1.0,
                    "latency_ms": 1.0,
                    "cost_usd": 0.0,
                    "reward": 1.0,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "candidate.jsonl"
            dataset.write_text(
                '{"task_id":"t1","target_worker_id":"w1"}\n'
                '{"task_id":"t2","target_worker_id":"w1"}\n',
                encoding="utf-8",
            )
            selection = run_selection(
                records=[record, {**record, "task_id": "t2"}],
                dataset=dataset,
                temperatures=[0.05],
                model_dir=root / "models",
                report_dir=root / "reports",
                prefix="unit-profile",
                epochs=1,
                learning_rate=0.0001,
                l2=0.0,
                baseline_report=None,
                baseline_dataset=None,
                operational_reference=None,
                min_loo_accuracy=0.7,
                max_loo_accuracy_drop=0.1,
                min_loo_solvable_pass_at_1=None,
                max_loo_latency_regret_ms=None,
                max_loo_latency_regret_increase_ms=None,
                promotion_profile="preserve_accuracy",
            )

        self.assertEqual(selection["promotion_profile"], "preserve_accuracy")
        self.assertEqual(selection["candidates"][0]["refresh"]["promotion_profile"], "preserve_accuracy")


if __name__ == "__main__":
    unittest.main()
