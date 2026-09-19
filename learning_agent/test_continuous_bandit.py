from pathlib import Path
import tempfile
import unittest

import numpy as np

import agent_pb2
from continuous_bandit import (
    ContinuousThompsonSamplingModel,
    ContinuousTimeoutActionCodec,
)
from models import (
    PBFT_ACCEPTABLE_TIMEOUTS_MS_BY_PHASE,
    ProtocolTimeoutActionCodec,
    ScheduledTimeoutActionCodec,
    ScheduledTimeoutModel,
    get_action_space_size,
    get_timeout_values_ms,
)


class DiscreteTimeoutActionCodecTest(unittest.TestCase):
    def test_pbft_uses_updated_discrete_arms(self) -> None:
        self.assertEqual(
            get_timeout_values_ms(agent_pb2.PROTOCOL_PBFT),
            (300, 800, 1800),
        )

    def test_pbft_phase_accuracy_policy(self) -> None:
        self.assertEqual(
            PBFT_ACCEPTABLE_TIMEOUTS_MS_BY_PHASE,
            {
                0: frozenset({300, 800, 1800}),
                1: frozenset({300}),
                2: frozenset({300, 800}),
                3: frozenset({800, 1800}),
                4: frozenset({800}),
                5: frozenset({1800}),
                6: frozenset({300, 800, 1800}),
            },
        )

    def test_existing_pbft_arm_encoding_is_unchanged(self) -> None:
        codec = ProtocolTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)
        timeout = agent_pb2.PbftTimeout(election_timeout_milliseconds=800)

        action = codec.encode(timeout)
        decoded = codec.decode(action)

        self.assertEqual(float(action.item()), 1.0)
        self.assertEqual(decoded.pbft.election_timeout_milliseconds, 800)

    def test_tendermint_arms_couple_phase_timeouts(self) -> None:
        codec = ProtocolTimeoutActionCodec(agent_pb2.PROTOCOL_TENDERMINT)

        decoded = codec.decode(np.asarray(1.0)).tendermint
        action = codec.encode(
            agent_pb2.TendermintTimeout(
                propose_timeout_milliseconds=1100,
                prevote_timeout_milliseconds=367,
                precommit_timeout_milliseconds=367,
            )
        )

        self.assertEqual(decoded.propose_timeout_milliseconds, 700)
        self.assertEqual(decoded.prevote_timeout_milliseconds, 233)
        self.assertEqual(decoded.precommit_timeout_milliseconds, 233)
        self.assertEqual(float(action.item()), 2.0)

    def test_tendermint_encode_rejects_uncoupled_timeouts(self) -> None:
        codec = ProtocolTimeoutActionCodec(agent_pb2.PROTOCOL_TENDERMINT)

        with self.assertRaises(ValueError):
            codec.encode(
                agent_pb2.TendermintTimeout(
                    propose_timeout_milliseconds=450,
                    prevote_timeout_milliseconds=50,
                    precommit_timeout_milliseconds=50,
                )
            )


class ScheduledTimeoutActionCodecTest(unittest.TestCase):
    def test_pbft_accepts_non_discrete_timeout(self) -> None:
        codec = ScheduledTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)

        decoded = codec.decode(np.asarray(437.0))
        encoded = codec.encode(decoded)

        self.assertEqual(decoded.pbft.election_timeout_milliseconds, 437)
        self.assertEqual(encoded.tolist(), [437.0])

    def test_tendermint_derives_phase_timeouts(self) -> None:
        codec = ScheduledTimeoutActionCodec(agent_pb2.PROTOCOL_TENDERMINT)

        decoded = codec.decode(np.asarray(437.0))
        encoded = codec.encode(decoded)

        self.assertEqual(decoded.tendermint.propose_timeout_milliseconds, 437)
        self.assertEqual(decoded.tendermint.prevote_timeout_milliseconds, 146)
        self.assertEqual(decoded.tendermint.precommit_timeout_milliseconds, 146)
        self.assertEqual(encoded.tolist(), [437.0])

    def test_sbft_accepts_three_non_discrete_timeouts(self) -> None:
        codec = ScheduledTimeoutActionCodec(agent_pb2.PROTOCOL_SBFT)

        decoded = codec.decode(np.asarray([437.0, 521.0, 89.0]))
        encoded = codec.encode(decoded)

        self.assertEqual(decoded.sbft.election_timeout_milliseconds, 437)
        self.assertEqual(decoded.sbft.slow_path_timeout_milliseconds, 521)
        self.assertEqual(decoded.sbft.batch_timeout_milliseconds, 89)
        self.assertEqual(encoded.tolist(), [437.0, 521.0, 89.0])

    def test_rejects_values_outside_uint32_range(self) -> None:
        codec = ScheduledTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)

        for invalid_value in (0, -1, 4294967296):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(ValueError):
                    codec.decode(np.asarray(float(invalid_value)))

    def test_rejects_fractional_milliseconds(self) -> None:
        codec = ScheduledTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)

        with self.assertRaises(ValueError):
            codec.decode(np.asarray(437.5))


class TendermintTimeoutOptionsTest(unittest.TestCase):
    def test_tendermint_has_three_proposal_driven_arms(self) -> None:
        self.assertEqual(
            get_timeout_values_ms(agent_pb2.PROTOCOL_TENDERMINT),
            (300, 700, 1100),
        )
        self.assertEqual(
            get_action_space_size(agent_pb2.PROTOCOL_TENDERMINT),
            3,
        )

    def test_schedule_reads_only_propose_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schedule.csv"
            path.write_text(
                "# time_s,propose_timeout_ms\n"
                "0,437\n"
                "10,821\n",
                encoding="utf-8",
            )

            model = ScheduledTimeoutModel.from_csv(
                agent_pb2.PROTOCOL_TENDERMINT,
                path,
            )

        self.assertEqual(model._schedule, [(0.0, (437,)), (10.0, (821,))])
        timeout = ScheduledTimeoutActionCodec(
            agent_pb2.PROTOCOL_TENDERMINT
        ).decode(model.predict_action(np.asarray([])))
        self.assertEqual(timeout.tendermint.propose_timeout_milliseconds, 437)

    def test_schedule_reports_invalid_timeout_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schedule.csv"
            path.write_text("0,0\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1"):
                ScheduledTimeoutModel.from_csv(
                    agent_pb2.PROTOCOL_TENDERMINT,
                    path,
                )


class ContinuousTimeoutActionCodecTest(unittest.TestCase):
    def test_pbft_round_trip_accepts_non_discrete_timeout(self) -> None:
        codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)
        timeout = agent_pb2.PbftTimeout(election_timeout_milliseconds=437)

        decoded = codec.decode(codec.encode(timeout))

        self.assertEqual(decoded.pbft.election_timeout_milliseconds, 437)

    def test_tendermint_round_trip_accepts_non_discrete_timeouts(self) -> None:
        codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_TENDERMINT)
        timeout = agent_pb2.TendermintTimeout(
            propose_timeout_milliseconds=437,
            prevote_timeout_milliseconds=146,
            precommit_timeout_milliseconds=146,
        )

        decoded = codec.decode(codec.encode(timeout))

        self.assertEqual(decoded.tendermint.propose_timeout_milliseconds, 437)
        self.assertEqual(decoded.tendermint.prevote_timeout_milliseconds, 146)
        self.assertEqual(decoded.tendermint.precommit_timeout_milliseconds, 146)

    def test_decode_clamps_model_output_to_protocol_bounds(self) -> None:
        pbft_codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)
        tendermint_codec = ContinuousTimeoutActionCodec(
            agent_pb2.PROTOCOL_TENDERMINT
        )

        pbft = pbft_codec.decode(np.asarray([-1.0]))
        tendermint = tendermint_codec.decode(np.asarray([2.0]))

        self.assertEqual(pbft.pbft.election_timeout_milliseconds, 300)
        self.assertEqual(tendermint.tendermint.propose_timeout_milliseconds, 600)
        self.assertEqual(tendermint.tendermint.prevote_timeout_milliseconds, 200)
        self.assertEqual(tendermint.tendermint.precommit_timeout_milliseconds, 200)


class ContinuousThompsonSamplingModelTest(unittest.TestCase):
    def _new_model(
        self, codec: ContinuousTimeoutActionCodec
    ) -> ContinuousThompsonSamplingModel:
        return ContinuousThompsonSamplingModel(
            action_dim=codec.action_dim,
            n_features=2,
            anchor_actions=codec.anchor_actions(),
            initial_design_size=3,
            memory_size=32,
            n_heads=4,
            candidate_count=32,
            seed=7,
        )

    def test_reset_replays_deterministic_initial_design(self) -> None:
        codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)
        model = self._new_model(codec)
        features = np.asarray([1.0, 2.0])

        first = model.predict_action(features)
        second = model.predict_action(features)
        model.reset_model()

        self.assertTrue(np.array_equal(first, model.predict_action(features)))
        self.assertTrue(np.array_equal(second, model.predict_action(features)))

    def test_tendermint_update_and_prediction_stay_in_bounds(self) -> None:
        codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_TENDERMINT)
        model = self._new_model(codec)
        features = np.asarray([10.0, 20.0])
        action = codec.encode(
            agent_pb2.TendermintTimeout(
                propose_timeout_milliseconds=437,
                prevote_timeout_milliseconds=146,
                precommit_timeout_milliseconds=146,
            )
        )

        model.update_model(features, action, 1234.0)
        for _ in range(5):
            predicted = model.predict_action(features)
            self.assertEqual(tuple(predicted.shape), (1,))
            self.assertTrue(bool(np.all(predicted >= 0.0)))
            self.assertTrue(bool(np.all(predicted <= 1.0)))
            timeout = codec.decode(predicted).tendermint
            self.assertGreaterEqual(timeout.propose_timeout_milliseconds, 150)
            self.assertLessEqual(timeout.propose_timeout_milliseconds, 600)
            self.assertEqual(
                timeout.prevote_timeout_milliseconds,
                (timeout.propose_timeout_milliseconds + 1) // 3,
            )
            self.assertEqual(
                timeout.precommit_timeout_milliseconds,
                (timeout.propose_timeout_milliseconds + 1) // 3,
            )


if __name__ == "__main__":
    unittest.main()
