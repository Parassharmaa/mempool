import unittest

from tools.filter_worker_pool import filter_worker_pool


class FilterWorkerPoolTest(unittest.TestCase):
    def test_filters_workers_in_requested_order(self) -> None:
        pool = {
            "base_url": "https://example.test/v1",
            "workers": [
                {"id": "a", "model": "ma"},
                {"id": "b", "model": "mb"},
            ],
        }

        filtered = filter_worker_pool(pool, ["b", "a"])

        self.assertEqual([worker["id"] for worker in filtered["workers"]], ["b", "a"])
        self.assertEqual(filtered["base_url"], "https://example.test/v1")

    def test_raises_for_missing_worker(self) -> None:
        with self.assertRaisesRegex(ValueError, "not found"):
            filter_worker_pool({"workers": [{"id": "a"}]}, ["missing"])


if __name__ == "__main__":
    unittest.main()
