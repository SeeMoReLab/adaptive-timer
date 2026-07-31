import unittest

from reward_utils import reward_from_report_row


class RewardExtractionTest(unittest.TestCase):
    def test_derives_quadratic_reward_from_reward_report_throughput(self) -> None:
        row = {
            "reward_throughput_tps": "500",
            "reward": "999999",
        }

        self.assertEqual(reward_from_report_row(row, row_number=2), 250.0)

    def test_keeps_throughput_linear_at_and_above_threshold(self) -> None:
        self.assertEqual(
            reward_from_report_row(
                {"reward_throughput_tps": "1500"}, row_number=2
            ),
            1500.0,
        )

    def test_requires_reward_report_throughput(self) -> None:
        with self.assertRaisesRegex(ValueError, "reward_throughput_tps"):
            reward_from_report_row({"reward": "500"}, row_number=2)


if __name__ == "__main__":
    unittest.main()
