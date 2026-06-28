import unittest

from mempool.task_features import extract_task_features, feature_distance


class TaskFeaturesTest(unittest.TestCase):
    def test_extracts_keyword_features(self) -> None:
        record = {
            "prompt": "Write a function that lowercases dict keys",
            "task_family": "code_data",
        }

        features = extract_task_features(record)

        self.assertEqual(features["family_code_data"], 1.0)
        self.assertEqual(features["kw_dict"], 1.0)
        self.assertEqual(features["kw_lowercase"], 0.0)

    def test_distance_is_zero_for_same_features(self) -> None:
        features = {"bias": 1.0, "kw_dict": 1.0}

        self.assertEqual(feature_distance(features, features), 0.0)

    def test_uses_prompt_feature_categories(self) -> None:
        record = {
            "prompt": "Download a URL and base64 encode a hex value",
            "task_family": "bigcodebench_hard",
            "prompt_features": {
                "categories": ["filesystem", "network"],
                "libraries": ["urllib", "base64", "weird-lib.name"],
                "missing_libraries": [],
                "environment_risk": 0,
                "plausibility_score": 2.5,
            },
        }

        features = extract_task_features(record)

        self.assertEqual(features["family_bigcodebench_hard"], 1.0)
        self.assertEqual(features["category_filesystem"], 1.0)
        self.assertEqual(features["category_network"], 1.0)
        self.assertEqual(features["library_count"], 3.0)
        self.assertEqual(features["lib_urllib"], 1.0)
        self.assertEqual(features["lib_base64"], 1.0)
        self.assertEqual(features["lib_weird_lib_name"], 1.0)
        self.assertEqual(features["kw_download"], 1.0)
        self.assertEqual(features["kw_url"], 1.0)
        self.assertEqual(features["kw_base64"], 1.0)
        self.assertEqual(features["kw_hex"], 1.0)
        self.assertEqual(features["environment_risk"], 0.0)

    def test_adds_network_archive_interaction_features(self) -> None:
        record = {
            "prompt": "Fetch an archive over HTTP, unpack it, and plot counts from files",
            "task_family": "bigcodebench_hard",
            "prompt_features": {
                "categories": ["network", "filesystem", "plotting"],
                "libraries": ["requests", "zipfile", "matplotlib", "os"],
                "missing_libraries": [],
                "environment_risk": 3,
                "plausibility_score": 8.0,
            },
        }

        features = extract_task_features(record)

        self.assertEqual(features["kw_archive"], 1.0)
        self.assertEqual(features["kw_http"], 1.0)
        self.assertEqual(features["kw_request"], 0.0)
        self.assertEqual(features["kw_requests"], 0.0)
        self.assertEqual(features["signal_network"], 1.0)
        self.assertEqual(features["signal_archive"], 1.0)
        self.assertEqual(features["signal_plotting"], 1.0)
        self.assertEqual(features["signal_filesystem"], 1.0)
        self.assertEqual(features["combo_network_archive"], 1.0)
        self.assertEqual(features["combo_network_plotting"], 1.0)
        self.assertEqual(features["combo_network_filesystem"], 1.0)


if __name__ == "__main__":
    unittest.main()
