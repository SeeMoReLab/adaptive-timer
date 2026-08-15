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
    ProtocolTimeoutActionCodec,
    ScheduledTimeoutModel,
    get_action_space_size,
    get_timeout_values_ms,
)


class DiscreteTimeoutActionCodecTest(unittest.TestCase):
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
                propose_timeout_milliseconds=600,
                prevote_timeout_milliseconds=200,
                precommit_timeout_milliseconds=200,
            )
        )

        self.assertEqual(decoded.propose_timeout_milliseconds, 300)
        self.assertEqual(decoded.prevote_timeout_milliseconds, 100)
        self.assertEqual(decoded.precommit_timeout_milliseconds, 100)
        self.assertEqual(float(action.item()), 2.0)

    def test_tendermint_encode_rejects_uncoupled_timeouts(self) -> None:
        codec = ProtocolTimeoutActionCodec(agent_pb2.PROTOCOL_TENDERMINT)

        with self.assertRaises(ValueError):
            codec.encode(
                agent_pb2.TendermintTimeout(
                    propose_timeout_milliseconds=300,
                    prevote_timeout_milliseconds=50,
                    precommit_timeout_milliseconds=50,
                )
            )


class TendermintTimeoutOptionsTest(unittest.TestCase):
    def test_tendermint_has_three_proposal_driven_arms(self) -> None:
        self.assertEqual(
            get_timeout_values_ms(agent_pb2.PROTOCOL_TENDERMINT),
            (150, 300, 600),
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
                "0,150\n"
                "10,600\n",
                encoding="utf-8",
            )

            model = ScheduledTimeoutModel.from_csv(
                agent_pb2.PROTOCOL_TENDERMINT,
                path,
            )

        self.assertEqual(model._schedule, [(0.0, 0), (10.0, 2)])


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
