import unittest

from tools.publish_hf_release import parse_hf_whoami


class PublishHfReleaseTest(unittest.TestCase):
    def test_parses_current_hf_whoami_output(self) -> None:
        output = "user=blazeofchi orgs=autify,kidaura\n"

        self.assertEqual(parse_hf_whoami(output), "blazeofchi")

    def test_keeps_legacy_plain_username_output(self) -> None:
        self.assertEqual(parse_hf_whoami("blazeofchi\n"), "blazeofchi")


if __name__ == "__main__":
    unittest.main()
