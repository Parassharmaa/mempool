import unittest

from mempool.smoke_benchmark import extract_python_code


class SmokeCodeExtractionTest(unittest.TestCase):
    def test_extracts_fenced_python(self) -> None:
        text = "Here is code:\n```python\ndef f():\n    return 1\n```"

        self.assertEqual(extract_python_code(text), "def f():\n    return 1\n")

    def test_keeps_plain_code(self) -> None:
        self.assertEqual(extract_python_code("def f():\n    return 1"), "def f():\n    return 1\n")


if __name__ == "__main__":
    unittest.main()
