import unittest

import agent_pb2
from reward_utils import (
    DEFAULT_REWARD_DESIGN,
    calculate_pbft_reward,
    calculate_reward_from_report_row,
    calculate_sbft_reward,
    calculate_tendermint_reward,
    get_reward_design,
    reward_from_report_row,
    set_reward_design,
)


class RewardExtractionTest(unittest.TestCase):
    def setUp(self) -> None:
        # The design is process-wide, so a test that changes it has to put it
        # back or it leaks into whatever runs next.
        previous = get_reward_design()
        self.addCleanup(set_reward_design, previous)
        set_reward_design(DEFAULT_REWARD_DESIGN)

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
            p95_consensus_latency_ms=30.0,
            p99_consensus_latency_ms=35.0,
            timeout_violation_rate=0.1,
            p95_batch_size=15.0,
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

        # Under the default design every protocol scores throughput alone, so
        # the three agree; Tendermint's old (tps - p90)/4000 is retired.
        self.assertEqual(pbft_reward, 0.5)
        self.assertEqual(sbft_reward, 0.5)
        self.assertEqual(tendermint_reward, 0.5)

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

        # 4000 tps: 1.0 for the first 1000, then 0.25 * 3 for the excess.
        self.assertEqual(reward, 1.75)

    def test_reward_mixed_discounts_throughput_by_the_fault_cost(self) -> None:
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
            "propose_timeout_milliseconds_used": "450",
            "prevote_timeout_milliseconds_used": "150",
            "precommit_timeout_milliseconds_used": "150",
        }
        set_reward_design("reward_mixed")

        reward = calculate_reward_from_report_row(
            agent_pb2.PROTOCOL_TENDERMINT, row, row_number=2
        )

        # window = 1000 * 100 / 4000 = 25ms; overhead = (450+150+150) + 20.
        self.assertAlmostEqual(reward, 1.75 * 25.0 / (25.0 + 770.0))

    def test_reward_mixed_rejects_a_tendermint_row_without_its_timeouts(
        self,
    ) -> None:
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
        set_reward_design("reward_mixed")

        with self.assertRaisesRegex(ValueError, "applied timeout for tendermint"):
            calculate_reward_from_report_row(
                agent_pb2.PROTOCOL_TENDERMINT, row, row_number=2
            )

    def test_unknown_reward_design_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown reward design"):
            set_reward_design("latency_capped")


if __name__ == "__main__":
    unittest.main()
