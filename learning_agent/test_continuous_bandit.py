import unittest

import torch

import agent_pb2
from continuous_bandit import (
    ContinuousThompsonSamplingModel,
    ContinuousTimeoutActionCodec,
)
from models import ProtocolTimeoutActionCodec


class DiscreteTimeoutActionCodecTest(unittest.TestCase):
    def test_existing_pbft_arm_encoding_is_unchanged(self) -> None:
        codec = ProtocolTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)
        timeout = agent_pb2.PbftTimeout(election_timeout_milliseconds=800)

        action = codec.encode(timeout)
        decoded = codec.decode(action)

        self.assertEqual(float(action.item()), 1.0)
        self.assertEqual(decoded.pbft.election_timeout_milliseconds, 800)


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
            prevote_timeout_milliseconds=286,
            precommit_timeout_milliseconds=611,
        )

        decoded = codec.decode(codec.encode(timeout))

        self.assertEqual(decoded.tendermint.propose_timeout_milliseconds, 437)
        self.assertEqual(decoded.tendermint.prevote_timeout_milliseconds, 286)
        self.assertEqual(decoded.tendermint.precommit_timeout_milliseconds, 611)

    def test_decode_clamps_model_output_to_protocol_bounds(self) -> None:
        pbft_codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)
        tendermint_codec = ContinuousTimeoutActionCodec(
            agent_pb2.PROTOCOL_TENDERMINT
        )

        pbft = pbft_codec.decode(torch.tensor([-1.0]))
        tendermint = tendermint_codec.decode(torch.tensor([2.0, -1.0, 1.0]))

        self.assertEqual(pbft.pbft.election_timeout_milliseconds, 300)
        self.assertEqual(tendermint.tendermint.propose_timeout_milliseconds, 900)
        self.assertEqual(tendermint.tendermint.prevote_timeout_milliseconds, 150)
        self.assertEqual(tendermint.tendermint.precommit_timeout_milliseconds, 900)


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
        features = torch.tensor([1.0, 2.0])

        first = model.predict_action(features)
        second = model.predict_action(features)
        model.reset_model()

        self.assertTrue(torch.equal(first, model.predict_action(features)))
        self.assertTrue(torch.equal(second, model.predict_action(features)))

    def test_tendermint_update_and_prediction_stay_in_bounds(self) -> None:
        codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_TENDERMINT)
        model = self._new_model(codec)
        features = torch.tensor([10.0, 20.0])
        action = codec.encode(
            agent_pb2.TendermintTimeout(
                propose_timeout_milliseconds=437,
                prevote_timeout_milliseconds=286,
                precommit_timeout_milliseconds=611,
            )
        )

        model.update_model(features, action, 1234.0)
        for _ in range(5):
            predicted = model.predict_action(features)
            self.assertEqual(tuple(predicted.shape), (3,))
            self.assertTrue(bool(torch.all(predicted >= 0.0)))
            self.assertTrue(bool(torch.all(predicted <= 1.0)))
            timeout = codec.decode(predicted).tendermint
            self.assertGreaterEqual(timeout.propose_timeout_milliseconds, 150)
            self.assertLessEqual(timeout.propose_timeout_milliseconds, 900)


if __name__ == "__main__":
    unittest.main()
