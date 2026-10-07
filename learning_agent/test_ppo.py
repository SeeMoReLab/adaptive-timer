import unittest

import numpy as np

import agent_pb2
from continuous_bandit import ContinuousPPO, ContinuousTimeoutActionCodec
from models import (
    DiscretePPO,
    PPOBase,
    ProtocolTimeoutActionCodec,
    _Adam,
    _TinyMLP,
    get_action_space_size,
    get_report_feature_dim,
    get_timeout_values_ms,
    signed_log_context,
)
from reward_utils import reward_tps

PROTOCOL = agent_pb2.PROTOCOL_PBFT
N_FEATURES = get_report_feature_dim(PROTOCOL)
N_ARMS = get_action_space_size(PROTOCOL)
ARMS_MS = tuple(sorted(get_timeout_values_ms(PROTOCOL)))


def central_difference(function, point, index, step=1e-6):
    high = np.array(point, dtype=float)
    low = np.array(point, dtype=float)
    high.reshape(-1)[index] += step
    low.reshape(-1)[index] -= step
    return (function(high) - function(low)) / (2.0 * step)


class TinyMLPTest(unittest.TestCase):
    def test_backward_matches_numerical_gradient(self):
        rng = np.random.default_rng(0)
        network = _TinyMLP((4, 6, 3), rng, output_scale=0.5)
        inputs = rng.normal(size=(7, 4))
        weights = rng.normal(size=(7, 3))

        def loss(flat_parameters):
            offset = 0
            for parameter in network.parameters:
                size = parameter.size
                parameter.reshape(-1)[:] = flat_parameters[offset : offset + size]
                offset += size
            output, _ = network.forward(inputs)
            return float(np.sum(output * weights))

        flat = np.concatenate([p.reshape(-1).copy() for p in network.parameters])
        output, cache = network.forward(inputs)
        analytic = np.concatenate(
            [g.reshape(-1) for g in network.backward(cache, weights)]
        )
        for index in range(flat.size):
            numeric = central_difference(loss, flat, index)
            loss(flat)  # restore
            self.assertAlmostEqual(numeric, analytic[index], places=6)

    def test_no_hidden_layers_is_a_single_affine_map(self):
        rng = np.random.default_rng(1)
        network = _TinyMLP((5, 2), rng)
        self.assertEqual(len(network.weights), 1)
        inputs = rng.normal(size=(3, 5))
        output, _ = network.forward(inputs)
        expected = inputs @ network.weights[0] + network.biases[0]
        np.testing.assert_allclose(output, expected)

    def test_seeded_initialization_is_reproducible(self):
        first = _TinyMLP((4, 3), np.random.default_rng(7))
        second = _TinyMLP((4, 3), np.random.default_rng(7))
        for left, right in zip(first.parameters, second.parameters):
            np.testing.assert_array_equal(left, right)


class AdamTest(unittest.TestCase):
    def test_descends_a_quadratic(self):
        parameters = [np.array([5.0, -3.0])]
        optimizer = _Adam(parameters, lr=0.1)
        for _ in range(500):
            optimizer.step(parameters, [2.0 * parameters[0]])
        np.testing.assert_allclose(parameters[0], np.zeros(2), atol=1e-3)


class SignedLogContextTest(unittest.TestCase):
    def test_prepends_bias_and_bounds_the_row(self):
        rows = np.array([[1e6] + [0.0] * (N_FEATURES - 1)], dtype=float)
        context = signed_log_context(rows, N_FEATURES)
        self.assertEqual(context.shape, (1, N_FEATURES + 1))
        self.assertEqual(context[0, 0], 1.0)
        self.assertLessEqual(float(np.linalg.norm(context[0, 1:])), 1.0 + 1e-9)

    def test_rejects_wrong_width_and_non_finite(self):
        with self.assertRaises(ValueError):
            signed_log_context(np.zeros((1, N_FEATURES + 1)), N_FEATURES)
        with self.assertRaises(ValueError):
            signed_log_context(np.full((1, N_FEATURES), np.nan), N_FEATURES)


class DiscretePPOGradientTest(unittest.TestCase):
    def setUp(self):
        self.model = DiscretePPO(
            action_dim=N_ARMS, n_features=N_FEATURES, seed=11
        )

    def test_log_prob_and_entropy_gradients(self):
        rng = np.random.default_rng(2)
        head = rng.normal(size=(5, N_ARMS))
        actions = rng.integers(0, N_ARMS, size=5).astype(float)
        _, _, d_log, d_entropy = self.model._head_terms(head, actions)

        for row in range(5):
            for column in range(N_ARMS):
                numeric_log = central_difference(
                    lambda h: self.model._head_terms(h, actions)[0][row],
                    head,
                    row * N_ARMS + column,
                )
                numeric_entropy = central_difference(
                    lambda h: self.model._head_terms(h, actions)[1][row],
                    head,
                    row * N_ARMS + column,
                )
                self.assertAlmostEqual(numeric_log, d_log[row, column], places=6)
                self.assertAlmostEqual(
                    numeric_entropy, d_entropy[row, column], places=6
                )

    def test_probabilities_sum_to_one_and_respect_the_floor(self):
        rng = np.random.default_rng(3)
        features = rng.normal(size=(4, N_FEATURES)) * 100.0
        probabilities = self.model.action_probabilities(features)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
        # Mixing 0.05 of uniform over three arms floors every arm at 1/60.
        self.assertTrue(np.all(probabilities >= 0.05 / N_ARMS - 1e-12))

    def test_no_exploration_floor_is_a_plain_softmax(self):
        model = DiscretePPO(
            action_dim=N_ARMS,
            n_features=N_FEATURES,
            seed=5,
            exploration_floor=0.0,
        )
        head = np.array([[1.0, 2.0, 3.0]])
        soft, mixed = model._probabilities(head)
        np.testing.assert_allclose(soft, mixed)


class DiscretePPOBehaviourTest(unittest.TestCase):
    def _features(self, rng):
        return rng.normal(size=N_FEATURES) * 1000.0

    def test_predict_action_decodes_to_a_real_arm(self):
        model = DiscretePPO(action_dim=N_ARMS, n_features=N_FEATURES, seed=1)
        codec = ProtocolTimeoutActionCodec(PROTOCOL)
        rng = np.random.default_rng(4)
        for _ in range(25):
            scores = model.predict_action(self._features(rng))
            timeout = codec.decode(scores)
            self.assertIn(
                timeout.pbft.election_timeout_milliseconds, ARMS_MS
            )

    def test_sampled_arm_is_the_argmax_of_the_returned_row(self):
        # The codec argmaxes the row, so sampling is only honoured if the
        # sampled entry is strictly the maximum.
        model = DiscretePPO(action_dim=N_ARMS, n_features=N_FEATURES, seed=2)
        rng = np.random.default_rng(5)
        seen = set()
        for _ in range(200):
            row = model.predict_action(self._features(rng))
            self.assertEqual(row.shape, (1, N_ARMS))
            seen.add(int(np.argmax(row)))
        # Over 200 draws from a near-uniform initial policy every arm appears.
        self.assertEqual(seen, set(range(N_ARMS)))

    def test_updates_only_fire_on_the_configured_cadence(self):
        model = DiscretePPO(
            action_dim=N_ARMS, n_features=N_FEATURES, seed=3, update_every=4
        )
        rng = np.random.default_rng(6)
        for step in range(12):
            model.update_model(
                self._features(rng), np.asarray(0.0), np.asarray(5.0)
            )
            self.assertEqual(model.update_count, (step + 1) // 4)
        self.assertEqual(model.observation_count, 12)

    def test_learns_a_stationary_best_arm(self):
        model = DiscretePPO(action_dim=N_ARMS, n_features=N_FEATURES, seed=9)
        rng = np.random.default_rng(7)
        features = np.ones(N_FEATURES) * 1000.0
        payoffs = [2.0, 9.0, 3.0]
        hits = 0
        for step in range(200):
            action = int(np.argmax(model.predict_action(features)))
            reward = payoffs[action] + 0.3 * rng.normal()
            model.update_model(
                features, np.asarray(float(action)), np.asarray(reward)
            )
            if step >= 150:
                hits += int(action == 1)
        self.assertGreater(hits / 50.0, 0.7)

    def test_rejects_out_of_range_actions(self):
        model = DiscretePPO(action_dim=N_ARMS, n_features=N_FEATURES, seed=3)
        with self.assertRaises(ValueError):
            model.update_model(
                np.zeros(N_FEATURES), np.asarray(float(N_ARMS)), np.asarray(1.0)
            )

    def test_backward_false_is_a_no_op(self):
        model = DiscretePPO(action_dim=N_ARMS, n_features=N_FEATURES, seed=3)
        model.update_model(
            np.zeros(N_FEATURES), np.asarray(0.0), np.asarray(1.0), backward=False
        )
        self.assertEqual(model.observation_count, 0)

    def test_reset_restores_the_initial_policy(self):
        model = DiscretePPO(action_dim=N_ARMS, n_features=N_FEATURES, seed=3)
        features = np.ones(N_FEATURES) * 500.0
        before = model.action_probabilities(features).copy()
        rng = np.random.default_rng(8)
        for _ in range(40):
            model.update_model(
                self._features(rng), np.asarray(1.0), np.asarray(9.0)
            )
        model.reset_model()
        np.testing.assert_allclose(
            model.action_probabilities(features), before, atol=1e-12
        )
        self.assertEqual(model.observation_count, 0)

    def test_two_models_with_one_seed_agree(self):
        # Sharing mode aggregates rewards only if every replica draws the same
        # action from the same agreed features.
        rng = np.random.default_rng(10)
        rows = [self._features(rng) for _ in range(30)]
        first = DiscretePPO(action_dim=N_ARMS, n_features=N_FEATURES, seed=42)
        second = DiscretePPO(action_dim=N_ARMS, n_features=N_FEATURES, seed=42)
        for row in rows:
            left = first.predict_action(row)
            right = second.predict_action(row)
            np.testing.assert_array_equal(left, right)
            action = float(np.argmax(left))
            first.update_model(row, np.asarray(action), np.asarray(6.0))
            second.update_model(row, np.asarray(action), np.asarray(6.0))

    def test_rejects_bad_hyperparameters(self):
        for kwargs in (
            {"clip_ratio": 0.0},
            {"epochs": 0},
            {"update_every": 0},
            {"gamma": 1.5},
            {"entropy_coef": -1.0},
            {"exploration_floor": 1.0},
        ):
            with self.assertRaises(ValueError):
                DiscretePPO(action_dim=N_ARMS, n_features=N_FEATURES, **kwargs)
        with self.assertRaises(ValueError):
            DiscretePPO(action_dim=0, n_features=N_FEATURES)


class ContinuousPPOTest(unittest.TestCase):
    def setUp(self):
        self.codec = ContinuousTimeoutActionCodec(PROTOCOL)
        self.model = ContinuousPPO(
            action_codec=self.codec, n_features=N_FEATURES, seed=13
        )

    def test_log_prob_and_entropy_gradients(self):
        rng = np.random.default_rng(11)
        head = rng.normal(size=(6, 2))
        positions = rng.uniform(0.05, 0.95, size=6)
        _, _, d_log, d_entropy = self.model._head_terms(head, positions)
        for row in range(6):
            for column in range(2):
                numeric_log = central_difference(
                    lambda h: self.model._head_terms(h, positions)[0][row],
                    head,
                    row * 2 + column,
                )
                numeric_entropy = central_difference(
                    lambda h: self.model._head_terms(h, positions)[1][row],
                    head,
                    row * 2 + column,
                )
                self.assertAlmostEqual(numeric_log, d_log[row, column], places=5)
                self.assertAlmostEqual(
                    numeric_entropy, d_entropy[row, column], places=5
                )

    def test_density_integrates_to_one(self):
        from scipy.integrate import quad

        rng = np.random.default_rng(12)
        head = rng.normal(size=(1, 2))
        density = lambda x: float(
            np.exp(self.model._head_terms(head, np.asarray([x]))[0][0])
        )
        total, _ = quad(density, 0.0, 1.0, limit=200)
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_concentrations_start_near_uniform(self):
        features = np.ones(N_FEATURES) * 1000.0
        head = self.model._actor_head(self.model._contexts(features.reshape(1, -1)))
        alpha, beta = self.model._concentrations(head)
        # alpha = beta = 1 is exactly uniform; the bias puts them just above it.
        self.assertLess(abs(float(alpha[0]) - 1.0), 0.3)
        self.assertLess(abs(float(beta[0]) - 1.0), 0.3)

    def test_concentrations_never_fall_below_one(self):
        head = np.array([[-500.0, 500.0]])
        alpha, beta = self.model._concentrations(head)
        self.assertGreaterEqual(float(alpha[0]), 1.0)
        self.assertGreaterEqual(float(beta[0]), 1.0)

    def test_actions_stay_inside_the_arm_range(self):
        rng = np.random.default_rng(14)
        for _ in range(200):
            action = self.model.predict_action(rng.normal(size=N_FEATURES) * 800.0)
            timeout = self.codec.denormalize_ms(float(np.asarray(action).reshape(-1)[0]))
            self.assertGreaterEqual(timeout, min(ARMS_MS))
            self.assertLessEqual(timeout, max(ARMS_MS))
            self.assertEqual(timeout, self.model.timeout_ms)

    def test_round_trips_a_reported_action_to_a_position(self):
        for timeout_ms in ARMS_MS:
            normalized = self.codec.normalize_ms(timeout_ms)
            position = self.model._store_action(normalized)
            self.assertAlmostEqual(
                self.model._timeout_of(position), timeout_ms, delta=1
            )

    def test_rejects_actions_outside_the_unit_interval(self):
        with self.assertRaises(ValueError):
            self.model.update_model(
                np.zeros(N_FEATURES), np.asarray(1.5), np.asarray(1.0)
            )

    def test_learns_to_prefer_a_paying_region(self):
        # Reward pays only near the long end; the policy should move there.
        model = ContinuousPPO(
            action_codec=self.codec, n_features=N_FEATURES, seed=17
        )
        rng = np.random.default_rng(15)
        features = np.ones(N_FEATURES) * 1000.0
        early, late = [], []
        for step in range(400):
            action = model.predict_action(features)
            timeout = self.codec.denormalize_ms(float(np.asarray(action).reshape(-1)[0]))
            reward = 9.0 if timeout >= 900 else 2.0
            reward += 0.3 * rng.normal()
            model.update_model(
                features,
                np.asarray(float(np.asarray(action).reshape(-1)[0])),
                np.asarray(reward),
            )
            (early if step < 100 else late).append(timeout >= 900)
        self.assertGreater(np.mean(late), np.mean(early) + 0.15)

    def test_reset_restores_the_initial_policy(self):
        features = np.ones(N_FEATURES) * 700.0
        context = self.model._contexts(features.reshape(1, -1))
        before = self.model._actor_head(context).copy()
        rng = np.random.default_rng(16)
        for _ in range(40):
            action = self.model.predict_action(rng.normal(size=N_FEATURES) * 500.0)
            self.model.update_model(
                np.zeros(N_FEATURES),
                np.asarray(float(np.asarray(action).reshape(-1)[0])),
                np.asarray(8.0),
            )
        self.model.reset_model()
        np.testing.assert_allclose(
            self.model._actor_head(context), before, atol=1e-12
        )

    def test_rejects_a_multi_dimensional_codec(self):
        class TwoDimensional:
            action_dim = 2
            protocol = PROTOCOL
            bounds_ms = ((100, 10000), (100, 10000))

        with self.assertRaises(ValueError):
            ContinuousPPO(action_codec=TwoDimensional(), n_features=N_FEATURES)

    def test_rejects_a_clamp_outside_the_codec_bounds(self):
        with self.assertRaises(ValueError):
            ContinuousPPO(
                action_codec=self.codec,
                n_features=N_FEATURES,
                clamp_ms=(50, 20000),
            )


class PPOBaseTest(unittest.TestCase):
    def test_zero_gamma_advantage_is_reward_minus_value(self):
        model = DiscretePPO(action_dim=N_ARMS, n_features=N_FEATURES, seed=3)
        rewards = np.array([5.0, 2.0, 8.0])
        values = np.array([4.0, 4.0, 4.0])
        advantages, returns = model._advantages(rewards, values)
        np.testing.assert_allclose(advantages, rewards - values)
        np.testing.assert_allclose(returns, rewards)

    def test_positive_gamma_bootstraps_from_the_next_value(self):
        model = DiscretePPO(
            action_dim=N_ARMS,
            n_features=N_FEATURES,
            seed=3,
            gamma=0.5,
            gae_lambda=0.0,
        )
        rewards = np.array([1.0, 1.0])
        values = np.array([0.0, 0.0])
        advantages, _ = model._advantages(rewards, values)
        # lambda=0 is the one-step TD error; the last step has no successor.
        np.testing.assert_allclose(advantages, [1.0 + 0.5 * 0.0, 1.0])

    def test_is_a_tensor_model(self):
        self.assertTrue(issubclass(DiscretePPO, PPOBase))
        self.assertTrue(issubclass(ContinuousPPO, PPOBase))

    def test_parameter_count_matches_discrete_linucb(self):
        # A linear actor over (bias + features) for each arm: the point of the
        # default is that PPO is not given more capacity than the bandit it is
        # being compared against.
        model = DiscretePPO(action_dim=N_ARMS, n_features=N_FEATURES, seed=3)
        total = sum(p.size for p in model._actor.parameters)
        self.assertEqual(total, (N_FEATURES + 1) * N_ARMS + N_ARMS)


if __name__ == "__main__":
    unittest.main()
