from pathlib import Path
import math
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import agent_pb2
from continuous_bandit import (
    CONTEXT_SPECS,
    CONTINUOUS_TENDERMINT_BOUNDS_MS,
    DEFAULT_TENDERMINT_PRIOR_PATH,
    ContinuousThompsonSamplingModel,
    NumpyMLP,
    ContinuousTimeoutActionCodec,
    SimPriorGPModel,
    SimReturnGPModel,
    context_spec,
    report_context,
    return_prior_path,
    tendermint_context,
)
from models import (
    PBFT_ACCEPTABLE_TIMEOUTS_MS_BY_PHASE,
    SBFT_REPORT_FEATURE_FIELD_NAMES,
    ProtocolTimeoutActionCodec,
    ScheduledTimeoutActionCodec,
    ScheduledTimeoutModel,
    get_action_space_size,
    get_report_feature_field_names,
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

    def test_sbft_accepts_two_non_discrete_timeouts(self) -> None:
        codec = ScheduledTimeoutActionCodec(agent_pb2.PROTOCOL_SBFT)

        decoded = codec.decode(np.asarray([437.0, 521.0]))
        encoded = codec.encode(decoded)

        self.assertEqual(decoded.sbft.election_timeout_milliseconds, 437)
        self.assertEqual(decoded.sbft.slow_path_timeout_milliseconds, 521)
        self.assertEqual(encoded.tolist(), [437.0, 521.0])

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

    def _sbft_codec(self) -> ContinuousTimeoutActionCodec:
        return ContinuousTimeoutActionCodec(
            agent_pb2.PROTOCOL_SBFT, bounds_ms=(100, 3000), wait_bounds_ms=(10, 500)
        )

    def test_sbft_round_trip_is_election_then_wait(self) -> None:
        codec = self._sbft_codec()
        self.assertEqual(codec.action_dim, 2)
        self.assertEqual(codec.bounds_ms, ((100, 3000), (10, 500)))
        for election_ms, wait_ms in ((100, 10), (250, 50), (525, 150), (1100, 37), (3000, 500)):
            with self.subTest(election_ms=election_ms, wait_ms=wait_ms):
                inner = agent_pb2.SbftTimeout(
                    election_timeout_milliseconds=election_ms,
                    slow_path_timeout_milliseconds=wait_ms,
                )
                action = codec.encode(inner)
                self.assertEqual(action.shape, (2,))
                self.assertEqual(float(action[0]), codec.normalize_ms(election_ms, 0))
                self.assertEqual(float(action[1]), codec.normalize_ms(wait_ms, 1))
                self.assertTrue(
                    np.array_equal(action, codec.encode(agent_pb2.Timeout(sbft=inner)))
                )
                decoded = codec.decode(action)
                self.assertEqual(decoded.WhichOneof("value"), "sbft")
                self.assertEqual(decoded.sbft.election_timeout_milliseconds, election_ms)
                self.assertEqual(decoded.sbft.slow_path_timeout_milliseconds, wait_ms)
                self.assertTrue(np.array_equal(codec.encode(decoded), action))

        low = codec.decode(np.asarray([-1.0, -1.0])).sbft
        high = codec.decode(np.asarray([2.0, 2.0])).sbft
        self.assertEqual(
            (low.election_timeout_milliseconds, low.slow_path_timeout_milliseconds), (100, 10)
        )
        self.assertEqual(
            (high.election_timeout_milliseconds, high.slow_path_timeout_milliseconds),
            (3000, 500),
        )
        with self.assertRaises(ValueError):
            codec.decode(np.asarray([0.5]))

    def test_sbft_out_of_range_timeouts_raise(self) -> None:
        codec = self._sbft_codec()
        # 8000 is the CppBedrock default election timeout, 0 the wait before
        # the agent has set one.
        for election_ms, wait_ms in ((8000, 150), (99, 150), (525, 0), (525, 9), (525, 501)):
            with self.subTest(election_ms=election_ms, wait_ms=wait_ms):
                with self.assertRaises(ValueError):
                    codec.encode(
                        agent_pb2.SbftTimeout(
                            election_timeout_milliseconds=election_ms,
                            slow_path_timeout_milliseconds=wait_ms,
                        )
                    )

    def test_wait_bounds_are_sbft_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "wait_bounds_ms"):
            ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_SBFT, bounds_ms=(100, 3000))
        for protocol in (
            agent_pb2.PROTOCOL_TENDERMINT,
            agent_pb2.PROTOCOL_PBFT,
            agent_pb2.PROTOCOL_SMARTBFT,
        ):
            with self.subTest(protocol=protocol):
                with self.assertRaisesRegex(ValueError, "wait_bounds_ms"):
                    ContinuousTimeoutActionCodec(
                        protocol, bounds_ms=(100, 3000), wait_bounds_ms=(10, 500)
                    )
        with self.assertRaises(ValueError):
            ContinuousTimeoutActionCodec(
                agent_pb2.PROTOCOL_SBFT, bounds_ms=(100, 3000), wait_bounds_ms=(50, 50)
            )


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


class SimPriorGPModelTest(unittest.TestCase):
    def _model(self) -> tuple:
        protocol = agent_pb2.PROTOCOL_TENDERMINT
        codec = ContinuousTimeoutActionCodec(
            protocol, bounds_ms=CONTINUOUS_TENDERMINT_BOUNDS_MS
        )
        names = get_report_feature_field_names(protocol)
        model = SimPriorGPModel(
            action_codec=codec,
            field_names=names,
            prior_path=DEFAULT_TENDERMINT_PRIOR_PATH,
            reward_design=NumpyMLP(DEFAULT_TENDERMINT_PRIOR_PATH).reward_design,
        )
        return codec, names, model

    def test_actions_stay_in_range_and_learn_online(self) -> None:
        codec, names, model = self._model()
        rng = np.random.default_rng(0)
        for _ in range(20):
            features = rng.uniform(1.0, 500.0, size=len(names))
            features[names.index("timeout_violation_rate")] = rng.random()
            action = model.predict_action(features)
            propose = codec.decode(action).tendermint.propose_timeout_milliseconds
            self.assertGreaterEqual(propose, 100)
            self.assertLessEqual(propose, 3000)
            model.update_model(features, action, float(rng.normal(1.5, 0.1)))

    def test_update_with_wrong_feature_count_is_an_error(self) -> None:
        _, names, model = self._model()
        with self.assertRaises(ValueError):
            model.update_model(np.ones(len(names) - 1), np.asarray([0.5]), 1.0)


class SimReturnGPModelTest(unittest.TestCase):
    def _write_return_prior(self, directory: str, y_std: float, floor: float = -2.0) -> str:
        """H18's prior re-labelled as a floored fit with a chosen y_std."""
        data = dict(np.load(DEFAULT_TENDERMINT_PRIOR_PATH))
        data["target_floor"] = np.asarray(floor)
        data["y_std"] = np.asarray(y_std)
        path = str(Path(directory) / "return_prior.npz")
        np.savez(path, **data)
        return path

    def _model(self, prior_path: str) -> tuple:
        protocol = agent_pb2.PROTOCOL_TENDERMINT
        codec = ContinuousTimeoutActionCodec(
            protocol, bounds_ms=CONTINUOUS_TENDERMINT_BOUNDS_MS
        )
        names = get_report_feature_field_names(protocol)
        model = SimReturnGPModel(
            action_codec=codec,
            field_names=names,
            prior_path=prior_path,
            reward_design=NumpyMLP(prior_path).reward_design,
        )
        return codec, names, model

    def test_return_prior_path_is_next_to_the_protocol_prior(self) -> None:
        self.assertEqual(
            return_prior_path(agent_pb2.PROTOCOL_TENDERMINT),
            DEFAULT_TENDERMINT_PRIOR_PATH.replace("_prior.npz", "_return_prior.npz"),
        )

    def test_refuses_a_prior_fit_to_the_raw_reward(self) -> None:
        with self.assertRaises(ValueError):
            self._model(DEFAULT_TENDERMINT_PRIOR_PATH)

    def test_residuals_are_floored_and_in_units_of_the_prior_y_std(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_return_prior(directory, y_std=2.0, floor=-2.0)
            codec, names, model = self._model(path)
            self.assertEqual(model.reward_floor, -2.0)
            prior = NumpyMLP(path)
            rng = np.random.default_rng(0)
            features = rng.uniform(1.0, 500.0, size=len(names))
            features[names.index("timeout_violation_rate")] = rng.random()
            action = model.predict_action(features)
            propose = codec.decode(action).tendermint.propose_timeout_milliseconds
            self.assertGreaterEqual(propose, 100)
            self.assertLessEqual(propose, 3000)
            inputs = np.append(
                report_context(context_spec(agent_pb2.PROTOCOL_TENDERMINT), names, features),
                float(action[0]),
            )
            predicted = float(prior.predict(inputs.reshape(1, -1))[0])
            self.assertAlmostEqual(
                model.update_model(features, action, 1.5), (1.5 - predicted) / 2.0, places=9
            )
            # A collapse below the floor counts as the floor.
            self.assertAlmostEqual(
                model.update_model(features, action, -7.0), (-2.0 - predicted) / 2.0, places=9
            )

            for _ in range(10):
                features = rng.uniform(1.0, 500.0, size=len(names))
                features[names.index("timeout_violation_rate")] = rng.random()
                action = model.predict_action(features)
                propose = codec.decode(action).tendermint.propose_timeout_milliseconds
                self.assertGreaterEqual(propose, 100)
                self.assertLessEqual(propose, 3000)
                model.update_model(features, action, float(rng.normal(0.0, 0.5)))

    # Grid indices of the first 50 decisions of the ORIGINAL one-dimensional code
    # (before SBFT's second dimension) on _pinned_sequence with the shipped
    # Tendermint return prior; captured by scripts/cloudlab/tmp/cal/sbc/capture_d1.py.
    # Retraining that prior changes them.
    _PINNED_TENDERMINT_INDICES = (
        31, 6, 29, 30, 3, 29, 30, 5, 29, 30, 5, 29, 30, 5, 29, 30, 8, 29, 31, 6,
        29, 30, 6, 29, 30, 7, 29, 30, 8, 29, 30, 7, 29, 30, 10, 29, 30, 7, 29, 30,
        7, 29, 30, 6, 29, 30, 3, 29, 30, 6,
    )

    def test_tendermint_decisions_are_unchanged(self) -> None:
        protocol = agent_pb2.PROTOCOL_TENDERMINT
        path = return_prior_path(protocol)
        codec, names, model = self._model(path)
        grid = np.linspace(0.0, 1.0, 64)
        rng = np.random.default_rng(11)
        bases = rng.uniform(5.0, 400.0, size=(3, len(names)))
        violation = names.index("timeout_violation_rate")
        bases[:, violation] = (0.0, 0.2, 0.6)
        targets = (0.3, 0.6, 0.8)
        for step, expected in enumerate(self._PINNED_TENDERMINT_INDICES):
            c = step % 3
            features = bases[c] * np.exp(rng.normal(0.0, 0.05, size=len(names)))
            features[violation] = bases[c, violation]
            action = model.predict_action(features)
            self.assertEqual(action.shape, (1,))
            self.assertEqual(float(action[0]), float(grid[expected]), f"step {step}")
            reward = -2.0 * abs(float(action[0]) - targets[c]) + float(rng.normal(0.0, 0.05))
            model.update_model(features, action, reward)

    def _write_sbft_prior(self, directory: str, input_dim: int = 16, hold: int = 1) -> str:
        """A small random ReLU MLP in NumpyMLP's format: context 14 + (T, W)."""
        rng = np.random.default_rng(5)
        path = str(Path(directory) / "sbft_return_prior.npz")
        np.savez(
            path,
            n_layers=np.asarray(2),
            W0=rng.normal(0.0, 0.05, size=(input_dim, 16)),
            b0=np.zeros(16),
            W1=rng.normal(0.0, 0.05, size=(16, 1)),
            b1=np.zeros(1),
            x_mean=np.zeros(input_dim),
            x_std=np.ones(input_dim),
            y_mean=np.asarray(0.0),
            y_std=np.asarray(1.0),
            reward_design=np.asarray("reward_tps_latency"),
            target_floor=np.asarray(-2.0),
            hold_episodes=np.asarray(hold),
        )
        return path

    def _sbft_model(self, prior_path: str) -> tuple:
        protocol = agent_pb2.PROTOCOL_SBFT
        spec = context_spec(protocol)
        codec = ContinuousTimeoutActionCodec(
            protocol, bounds_ms=spec.bounds_ms, wait_bounds_ms=spec.wait_bounds_ms
        )
        names = get_report_feature_field_names(protocol)
        model = SimReturnGPModel(
            action_codec=codec,
            field_names=names,
            prior_path=prior_path,
            reward_design="reward_tps_latency",
        )
        return codec, names, model

    def test_sbft_two_dimensional_actions_stay_in_bounds_and_react(self) -> None:
        timeouts = np.linspace(0.0, 1.0, 64)
        waits = np.linspace(0.0, 1.0, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_sbft_prior(directory)
            codec, names, model = self._sbft_model(path)
            spec = context_spec(agent_pb2.PROTOCOL_SBFT)
            rng = np.random.default_rng(0)
            for _ in range(20):
                features = rng.uniform(1.0, 500.0, size=len(names))
                action = model.predict_action(features)
                self.assertEqual(action.shape, (2,))
                self.assertIn(float(action[0]), timeouts)
                self.assertIn(float(action[1]), waits)
                timeout = codec.decode(action).sbft
                self.assertGreaterEqual(timeout.election_timeout_milliseconds, 100)
                self.assertLessEqual(timeout.election_timeout_milliseconds, 3000)
                self.assertGreaterEqual(timeout.slow_path_timeout_milliseconds, 10)
                self.assertLessEqual(timeout.slow_path_timeout_milliseconds, 500)
                model.update_model(features, action, float(rng.normal(0.0, 0.5)))

            # With no evidence in a context the model plays the prior's argmax
            # over the product grid (T major).
            model.reset_model()
            features = rng.uniform(1.0, 500.0, size=len(names))
            context = report_context(spec, names, features)
            grid = np.column_stack((np.repeat(timeouts, 16), np.tile(waits, 64)))
            inputs = np.hstack((np.repeat(context.reshape(1, -1), len(grid), axis=0), grid))
            best = grid[int(np.argmax(NumpyMLP(path).predict(inputs)))]
            first = model.predict_action(features)
            self.assertTrue(np.array_equal(first, best))

            # Evidence that another (T, W) pays in this context moves the choice there.
            target = grid[37 * 16 + 5]
            self.assertFalse(np.array_equal(first, target))
            for _ in range(4):
                model.update_model(features, target, 1.0)
            self.assertTrue(np.array_equal(model.predict_action(features), target))

            with self.assertRaises(ValueError):
                model.update_model(features, np.asarray([0.5]), 0.0)

    def test_sbft_refuses_a_one_dimensional_prior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_sbft_prior(directory, input_dim=15)
            with self.assertRaisesRegex(ValueError, "expected 16"):
                self._sbft_model(path)

    def test_held_return_prior_updates_with_the_mean_of_the_next_rewards(self) -> None:
        """hold_episodes h > 1: decision k's residual is the mean of rewards k..k+h-1
        (floored) against the prior at decision k's inputs, recorded h - 1 updates late."""
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_sbft_prior(directory, hold=3)
            codec, names, model = self._sbft_model(path)
            spec = context_spec(agent_pb2.PROTOCOL_SBFT)
            prior = NumpyMLP(path)
            rng = np.random.default_rng(1)
            steps = [(rng.uniform(1.0, 500.0, size=len(names)), rng.random(2), r)
                     for r in (0.4, -3.0, 0.1, 0.7, -0.2)]
            done = []
            for k, (features, action, reward) in enumerate(steps):
                out = model.update_model(features, action, reward)
                if k < 2:
                    self.assertTrue(math.isnan(out))
                    self.assertEqual(len(model._residuals), 0)
                else:
                    first = k - 2
                    f0, a0, _ = steps[first]
                    target = max(-2.0, float(np.mean([r for _, _, r in steps[first:k + 1]])))
                    inputs = np.append(report_context(spec, names, f0), a0)
                    expected = target - float(prior.predict(inputs.reshape(1, -1))[0])
                    self.assertAlmostEqual(out, expected, places=12)
                    self.assertTrue(np.array_equal(model._inputs_seen[-1], inputs))
                    done.append(out)
            self.assertEqual(len(model._residuals), 3)


class ContextSpecTest(unittest.TestCase):
    def test_tendermint_context_is_unchanged(self) -> None:
        # Pins the H18 context the shipped prior was trained on: log1p of the
        # eight fields (negatives clamp to 0), then the raw violation rate.
        names = get_report_feature_field_names(agent_pb2.PROTOCOL_TENDERMINT)
        values = {
            "total_transactions": 12000,
            "total_consensus_instances": 40,
            "avg_consensus_latency_ms": 210.5,
            "p50_consensus_latency_ms": 180.0,
            "p90_consensus_latency_ms": 350.25,
            "throughput_tps": 1500.0,
            "timeout_violation_rate": 0.125,
            "avg_batch_size": -3.0,
            "p50_batch_size": 290,
            "p90_batch_size": 310,
            "leader_change_count": 1,
            "regency_change_count": 2,
            "propose_latency_ms": 60.0,
            "prevote_latency_ms": 45.5,
            "precommit_latency_ms": 30.0,
        }
        row = np.asarray([values[name] for name in names], dtype=float)
        expected = [
            5.354224998486333,
            5.198497031265826,
            5.861498220647955,
            4.110873864173311,
            3.8394523125933104,
            3.4339872044851463,
            7.313886831633462,
            0.0,
            0.125,
        ]

        context = tendermint_context(names, row)

        np.testing.assert_allclose(context, expected, rtol=0.0, atol=1e-12)
        spec = CONTEXT_SPECS[agent_pb2.PROTOCOL_TENDERMINT]
        self.assertTrue(np.array_equal(context, report_context(spec, names, row)))
        self.assertEqual(spec.context_dim, 9)
        self.assertEqual(spec.bounds_ms, CONTINUOUS_TENDERMINT_BOUNDS_MS)
        self.assertEqual(spec.prior_path, DEFAULT_TENDERMINT_PRIOR_PATH)

    def test_pbft_context_ignores_timeout_ms(self) -> None:
        names = get_report_feature_field_names(agent_pb2.PROTOCOL_PBFT)
        row = np.arange(1.0, len(names) + 1.0)
        changed = row.copy()
        changed[names.index("timeout_ms")] = 9999.0

        for protocol in (agent_pb2.PROTOCOL_PBFT, agent_pb2.PROTOCOL_SMARTBFT):
            with self.subTest(protocol=protocol):
                spec = context_spec(protocol)
                self.assertNotIn("timeout_ms", spec.log_fields + spec.raw_fields)
                context = report_context(spec, names, row)
                self.assertEqual(context.shape, (7,))
                self.assertTrue(
                    np.array_equal(context, report_context(spec, names, changed))
                )
                moved = row.copy()
                moved[names.index("p50_consensus_latency_ms")] += 1.0
                self.assertFalse(
                    np.array_equal(context, report_context(spec, names, moved))
                )

    def test_sbft_context_and_bounds(self) -> None:
        spec = context_spec(agent_pb2.PROTOCOL_SBFT)
        self.assertEqual(
            spec.log_fields,
            (
                "avg_consensus_latency_ms",
                "p50_consensus_latency_ms",
                "p95_consensus_latency_ms",
                "throughput_tps",
                "avg_batch_size",
                "p50_inter_commit_gap_ms",
                "view_change_count",
                "pre_prepare_latency_ms",
                "prepare_latency_ms",
                "fast_path_count",
                "slow_path_count",
                "nth_prepare_gap_ms",
                "nth_prepare_seen_count",
                "nth_prepare_unseen_count",
            ),
        )
        self.assertEqual(spec.raw_fields, ())
        self.assertEqual(spec.context_dim, 14)
        finite = {
            name: scale
            for name, scale in zip(spec.log_fields, spec.length_scales)
            if np.isfinite(scale)
        }
        self.assertEqual(
            finite,
            {
                "p50_consensus_latency_ms": 0.3,
                "throughput_tps": 0.2,
                "view_change_count": 0.7,
                "fast_path_count": 0.7,
                "slow_path_count": 0.7,
            },
        )
        self.assertEqual(spec.bounds_ms, (100, 3000))
        self.assertEqual(spec.wait_bounds_ms, (10, 500))
        self.assertEqual(Path(spec.prior_path).name, "continuous_sbft_prior.npz")
        self.assertEqual(
            Path(return_prior_path(agent_pb2.PROTOCOL_SBFT)).name,
            "continuous_sbft_return_prior.npz",
        )
        self.assertEqual(spec.simulator, "sbft_sim.py")
        for protocol in (
            agent_pb2.PROTOCOL_TENDERMINT,
            agent_pb2.PROTOCOL_PBFT,
            agent_pb2.PROTOCOL_SMARTBFT,
        ):
            self.assertIsNone(context_spec(protocol).wait_bounds_ms)

        # The applied (T, W) of the context's window is not part of it (it drove
        # cycles through the model's own last pick); the leader's n-th prepare
        # fields, the last three entries, are read by the prior only.
        names = get_report_feature_field_names(agent_pb2.PROTOCOL_SBFT)
        self.assertEqual(names, SBFT_REPORT_FEATURE_FIELD_NAMES)
        row = np.arange(1.0, len(names) + 1.0)
        changed = row.copy()
        changed[names.index("timeout_ms")] = 9999.0
        changed[names.index("slow_path_timeout_ms")] = 9999.0
        context = report_context(spec, names, row)
        self.assertEqual(context.shape, (14,))
        self.assertTrue(np.array_equal(context, report_context(spec, names, changed)))
        self.assertTrue(np.all(np.isinf(np.asarray(spec.length_scales[11:]))))

        with tempfile.TemporaryDirectory() as tmp:
            codec = ContinuousTimeoutActionCodec(
                agent_pb2.PROTOCOL_SBFT,
                bounds_ms=spec.bounds_ms,
                wait_bounds_ms=spec.wait_bounds_ms,
            )
            with self.assertRaisesRegex(FileNotFoundError, "sbft_sim.py"):
                SimReturnGPModel(
                    action_codec=codec,
                    field_names=names,
                    prior_path=str(Path(tmp) / "continuous_sbft_return_prior.npz"),
                    reward_design="reward_tps_latency",
                )

    def test_empty_sbft_window_takes_its_length_as_latency(self) -> None:
        # Kevin 2026-10-01: CppBedrock reports latency 0 for a window in which no
        # batch executed; the agent and the simulator give it the window's length.
        from main import _fill_empty_window_latency
        import sbft_sim

        latency_fields = (
            "avg_consensus_latency_ms",
            "p50_consensus_latency_ms",
            "p95_consensus_latency_ms",
        )
        empty = agent_pb2.SbftReport(total_consensus_instances=0, throughput_tps=0.0)
        self.assertTrue(_fill_empty_window_latency(empty, 5000))
        for name in latency_fields:
            self.assertEqual(getattr(empty, name), 5000.0)
        busy = agent_pb2.SbftReport(
            total_consensus_instances=3,
            avg_consensus_latency_ms=210.0,
            p50_consensus_latency_ms=200.0,
            p95_consensus_latency_ms=260.0,
        )
        self.assertFalse(_fill_empty_window_latency(busy, 5000))
        self.assertEqual(
            [getattr(busy, name) for name in latency_fields], [210.0, 200.0, 260.0]
        )

        # The simulator applies it to the 3-report median, as main.py applies it to the
        # report the model reads: a stall of the cluster, not a lone empty replica.
        sim = sbft_sim.SbftSim(sbft_sim.SimConstants(), sbft_sim.spec_schedule(), seed=1)
        windows = [sbft_sim._Window(15.0, 20.0) for _ in range(sbft_sim.N)]
        self.assertEqual(windows[0].report(250.0, 50.0)["p50_consensus_latency_ms"], 0.0)
        median, _ = sim.median_report(windows, 250.0, 50.0, 20000.0)
        for name in latency_fields:
            self.assertEqual(median[name], 5000.0)
        reporting = sim._reporting(20000.0)
        self.assertEqual(len(reporting), 3)
        for k in reporting[1:]:
            windows[k].batch.append(40.0)
            windows[k].lat.append(180.0)
        median, _ = sim.median_report(windows, 250.0, 50.0, 20000.0)
        self.assertEqual(median["total_consensus_instances"], 1.0)
        self.assertEqual(median["p50_consensus_latency_ms"], 180.0)

    def test_sbft_nth_prepare_fields_come_from_the_leading_report(self) -> None:
        # Only the replica that led a sequence measures its n-th prepare; the
        # aggregate takes those fields from the report that led the most sequences
        # (lowest node on a tie) and the median for everything else, and the
        # simulator's median report does the same.
        from aggregator import MedianAggregator
        import sbft_sim

        batch = agent_pb2.ReportBatch(episode=7, protocol=agent_pb2.PROTOCOL_SBFT)
        for node, (p50, gap, seen, unseen) in {
            0: (200.0, 0.0, 0, 0), 1: (210.0, 95.0, 8, 2), 3: (190.0, 40.0, 1, 0),
        }.items():
            entry = batch.reports.add()
            entry.node_id = node
            entry.protocol = agent_pb2.PROTOCOL_SBFT
            entry.sbft_state.CopyFrom(agent_pb2.SbftReport(
                total_consensus_instances=10, p50_consensus_latency_ms=p50,
                nth_prepare_gap_ms=gap, nth_prepare_seen_count=seen,
                nth_prepare_unseen_count=unseen))
        merged = MedianAggregator(agent_pb2.PROTOCOL_SBFT).aggregate(0, batch)
        self.assertEqual(merged.p50_consensus_latency_ms, 200.0)
        self.assertEqual(
            (merged.nth_prepare_gap_ms, merged.nth_prepare_seen_count,
             merged.nth_prepare_unseen_count), (95.0, 8, 2))
        batch.reports[2].sbft_state.nth_prepare_seen_count = 10  # node 3 ties node 1
        merged = MedianAggregator(agent_pb2.PROTOCOL_SBFT).aggregate(0, batch)
        self.assertEqual(merged.nth_prepare_gap_ms, 95.0)

        window = sbft_sim._Window(0.0, 10.0)
        window.nth.extend([100.0, 120.0, 110.0])
        window.nth_unseen = 2
        report = window.report(1100.0, 10.0)
        self.assertEqual(
            (report["nth_prepare_gap_ms"], report["nth_prepare_seen_count"],
             report["nth_prepare_unseen_count"]), (110.0, 3.0, 2.0))
        sim = sbft_sim.SbftSim(sbft_sim.SimConstants(), sbft_sim.spec_schedule(), seed=1)
        reporting = sim._reporting(20000.0)
        windows = [sbft_sim._Window(15.0, 20.0) for _ in range(sbft_sim.N)]
        for k in reporting:
            windows[k].batch.append(40.0)
            windows[k].lat.append(180.0)
        windows[reporting[-1]].nth.extend([100.0, 120.0, 110.0])
        median, _ = sim.median_report(windows, 1100.0, 10.0, 20000.0)
        self.assertEqual(median["nth_prepare_gap_ms"], 110.0)
        self.assertEqual(median["nth_prepare_seen_count"], 3.0)

    def test_three_action_dimensions_are_refused(self) -> None:
        three_dimensional = SimpleNamespace(
            protocol=agent_pb2.PROTOCOL_SBFT, action_dim=3
        )
        with self.assertRaisesRegex(ValueError, "one or two timeout dimensions"):
            SimPriorGPModel(
                action_codec=three_dimensional,
                field_names=get_report_feature_field_names(agent_pb2.PROTOCOL_SBFT),
                prior_path=DEFAULT_TENDERMINT_PRIOR_PATH,
                reward_design="reward_mixed",
            )

    def test_missing_prior_fails_at_construction_naming_the_simulator(self) -> None:
        for protocol, simulator in (
            (agent_pb2.PROTOCOL_PBFT, "pbft_sim.py"),
            (agent_pb2.PROTOCOL_SMARTBFT, "smartbft_sim.py"),
        ):
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as tmp:
                spec = context_spec(protocol)
                self.assertEqual(spec.simulator, simulator)
                codec = ContinuousTimeoutActionCodec(protocol, bounds_ms=spec.bounds_ms)
                with self.assertRaisesRegex(FileNotFoundError, simulator):
                    SimPriorGPModel(
                        action_codec=codec,
                        field_names=get_report_feature_field_names(protocol),
                        prior_path=str(Path(tmp) / Path(spec.prior_path).name),
                        reward_design="reward_mixed",
                    )


if __name__ == "__main__":
    unittest.main()
