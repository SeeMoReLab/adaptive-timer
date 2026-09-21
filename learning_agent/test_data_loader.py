import unittest

import agent_pb2
from reward_utils import (
    calculate_pbft_reward,
    calculate_reward_from_report_row,
    calculate_sbft_reward,
    calculate_tendermint_reward,
    reward_from_report_row,
)


class RewardExtractionTest(unittest.TestCase):
    def test_gives_full_credit_to_throughput_below_the_knee(self) -> None:
        row = {
            "reward_throughput_tps": "500",
            "reward": "999999",
        }

        self.assertEqual(reward_from_report_row(row, row_number=2), 0.5)

    def test_discounts_throughput_above_the_knee(self) -> None:
        # 500 tps past the 1000 tps knee is worth 0.25 * 500 / 1000 = 0.125,
        # a quarter of what the same 500 tps is worth below the knee.
        self.assertEqual(
            reward_from_report_row(
                {"reward_throughput_tps": "1500"}, row_number=2
            ),
            1.125,
        )

    def test_requires_reward_report_throughput(self) -> None:
        with self.assertRaisesRegex(ValueError, "reward_throughput_tps"):
            reward_from_report_row({"reward": "500"}, row_number=2)

    def test_complete_protocol_reports_keep_current_reward_formula(self) -> None:
        common = {
            "total_transactions": 100,
            "total_consensus_instances": 10,
            "avg_consensus_latency_ms": 20.0,
            "throughput_tps": 500.0,
            "avg_batch_size": 10.0,
            "leader_change_count": 1,
            "regency_change_count": 1,
        }
        pbft_reward = calculate_pbft_reward(
            **common,
            p50_consensus_latency_ms=18.0,
            p95_consensus_latency_ms=30.0,
            p95_batch_size=15.0,
            timeout_ms=300,
            avg_inter_commit_gap_ms=25.0,
            p50_inter_commit_gap_ms=22.0,
            p95_inter_commit_gap_ms=40.0,
            view_change_count=1,
            no_progress_view_change_count=1,
        )
        sbft_reward = calculate_sbft_reward(
            **common,
            p50_consensus_latency_ms=18.0,
            p95_consensus_latency_ms=30.0,
            p95_batch_size=15.0,
            timeout_ms=300,
            avg_inter_commit_gap_ms=25.0,
            p50_inter_commit_gap_ms=22.0,
            p95_inter_commit_gap_ms=40.0,
            view_change_count=1,
            no_progress_view_change_count=1,
            pre_prepare_latency_ms=5.0,
            prepare_latency_ms=6.0,
            commit_latency_ms=7.0,
            slow_path_timeout_ms=150,
            fast_path_count=90,
            slow_path_count=10,
        )
        tendermint_reward = calculate_tendermint_reward(
            **common,
            p50_consensus_latency_ms=18.0,
            p90_consensus_latency_ms=28.0,
            timeout_violation_rate=0.1,
            p50_batch_size=9.0,
            p90_batch_size=14.0,
            propose_latency_ms=10.0,
            prevote_latency_ms=5.0,
            precommit_latency_ms=5.0,
        )

        self.assertEqual(pbft_reward, 0.5)
        # SBFT discounts the throughput reward by p95 consensus latency: the
        # timers move the tail, not the median.
        self.assertAlmostEqual(sbft_reward, 0.5 * (300.0 / 330.0))
        self.assertAlmostEqual(tendermint_reward, 0.4944)

    def test_tendermint_csv_reward_uses_complete_reward_report(self) -> None:
        row = {
            "reward_total_transactions": "100",
            "reward_total_consensus_instances": "10",
            "reward_avg_consensus_latency_ms": "20",
            "reward_p50_consensus_latency_ms": "18",
            "reward_p90_consensus_latency_ms": "500",
            "reward_throughput_tps": "4000",
            "reward_timeout_violation_rate": "0.1",
            "reward_avg_batch_size": "10",
            "reward_p50_batch_size": "9",
            "reward_p90_batch_size": "14",
            "reward_leader_change_count": "1",
            "reward_regency_change_count": "1",
            "reward_propose_latency_ms": "10",
            "reward_prevote_latency_ms": "5",
            "reward_precommit_latency_ms": "5",
        }

        reward = calculate_reward_from_report_row(
            agent_pb2.PROTOCOL_TENDERMINT, row, row_number=2
        )

        self.assertEqual(reward, 3.9)


if __name__ == "__main__":
    unittest.main()
