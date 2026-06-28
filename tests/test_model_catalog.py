import unittest

from mempool.model_catalog import build_worker_pool, select_model_candidates


class ModelCatalogTest(unittest.TestCase):
    def test_selects_available_candidates_by_priority(self) -> None:
        result = select_model_candidates(
            ["kimi-k2.7-code", "qwen3-coder:480b", "glm-5.2"],
            limit=2,
        )

        self.assertEqual(result["selected_count"], 2)
        self.assertEqual(
            [row["model"] for row in result["selected"]],
            ["qwen3-coder:480b", "kimi-k2.7-code"],
        )
        self.assertTrue(any(row["model"] == "deepseek-v4-pro" for row in result["missing"]))

    def test_builds_worker_pool_without_api_key_value(self) -> None:
        payload = {"models": ["qwen3-coder-next", "deepseek-v4-pro"]}

        pool = build_worker_pool(
            payload,
            base_url="https://example.test/v1",
            api_key_env="SECRET_ENV",
            timeout_seconds=30,
        )

        self.assertEqual(pool["base_url"], "https://example.test/v1")
        self.assertEqual(pool["api_key_env"], "SECRET_ENV")
        self.assertEqual(
            [worker["model"] for worker in pool["workers"]],
            ["qwen3-coder-next", "deepseek-v4-pro"],
        )
        self.assertNotIn("SECRET_ENV_VALUE", str(pool))


if __name__ == "__main__":
    unittest.main()
