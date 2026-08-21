import unittest

import numpy as np

from models import DiscountedLinUCB


class DiscountedLinUCBTest(unittest.TestCase):
    def new_model(self, **overrides) -> DiscountedLinUCB:
        options = {
            "action_dim": 3,
            "n_features": 2,
            "confidence_scale": 1.0,
            "discount_factor": 0.95,
            "exploration_floor": 0.0,
            "reward_scale": 1.0,
            "seed": 7,
        }
        options.update(overrides)
        return DiscountedLinUCB(**options)

    def test_initializes_every_arm_before_using_finite_ucb_scores(self) -> None:
        model = self.new_model()
        features = np.asarray([2.0, 0.5])

        for expected_arm in range(3):
            scores = model.predict_action(features)
            self.assertEqual(int(np.argmax(scores, axis=1)[0]), expected_arm)
            model.update_model(features, np.asarray(expected_arm), 1.0)

        self.assertTrue(bool(np.all(np.isfinite(model.predict_action(features)))))

    def test_exploits_arm_with_highest_learned_reward(self) -> None:
        model = self.new_model(confidence_scale=0.0, discount_factor=1.0)
        features = np.asarray([2.0, 0.5])

        for arm, reward in enumerate((0.2, 1.0, 0.4)):
            for _ in range(12):
                model.update_model(features, np.asarray(arm), reward)

        scores = model.predict_action(features)
        self.assertEqual(int(np.argmax(scores, axis=1)[0]), 1)

    def test_defaults_converge_with_normalized_simulator_rewards(self) -> None:
        model = DiscountedLinUCB(action_dim=3, n_features=2, seed=7)
        features = np.asarray([2.0, 0.5])
        rewards = (0.2, 2.0, 0.4)
        selected = []

        for _ in range(100):
            arm = int(np.argmax(model.predict_action(features), axis=1)[0])
            selected.append(arm)
            model.update_model(features, np.asarray(arm), rewards[arm])

        self.assertEqual(selected[-20:], [1] * 20)

    def test_uncertainty_bonus_can_select_less_observed_arm(self) -> None:
        model = self.new_model(
            action_dim=2,
            confidence_scale=2.0,
            discount_factor=1.0,
        )
        features = np.asarray([2.0, 0.5])

        for _ in range(30):
            model.update_model(features, np.asarray(0), 1.0)
        model.update_model(features, np.asarray(1), 0.0)

        scores = model.predict_action(features)
        self.assertEqual(int(np.argmax(scores, axis=1)[0]), 1)

    def test_discounting_restores_uncertainty_to_unobserved_arm(self) -> None:
        model = self.new_model(
            action_dim=2,
            confidence_scale=1.0,
            discount_factor=0.9,
        )
        features = np.asarray([2.0, 0.5])

        for _ in range(30):
            model.update_model(features, np.asarray(0), 1.0)
        model.update_model(features, np.asarray(1), 0.0)
        self.assertEqual(int(np.argmax(model.predict_action(features), axis=1)[0]), 0)

        for _ in range(25):
            model.update_model(features, np.asarray(0), 1.0)
        self.assertEqual(int(np.argmax(model.predict_action(features), axis=1)[0]), 1)

    def test_batch_update_matches_ordered_single_updates(self) -> None:
        batch = self.new_model(action_dim=2)
        sequential = self.new_model(action_dim=2)
        features = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        actions = np.asarray([0.0, 1.0, 0.0])
        rewards = np.asarray([0.2, 1.0, 0.4])

        batch.update_model(features, actions, rewards)
        for row, action, reward in zip(features, actions, rewards):
            sequential.update_model(row, action, reward)

        self.assertTrue(np.allclose(batch._precision, sequential._precision))
        self.assertTrue(np.allclose(batch._covariance, sequential._covariance))
        self.assertTrue(
            np.allclose(batch._reward_vectors, sequential._reward_vectors)
        )
        self.assertTrue(
            np.array_equal(batch._observation_counts, sequential._observation_counts)
        )

    def test_reset_clears_learning_and_replays_seeded_exploration(self) -> None:
        model = self.new_model(exploration_floor=1.0)
        features = np.asarray([2.0, 0.5])
        for arm in range(3):
            model.update_model(features, np.asarray(arm), float(arm))

        first = model.predict_action(features)
        model.reset_model()
        for arm in range(3):
            model.update_model(features, np.asarray(arm), float(arm))

        self.assertTrue(np.array_equal(first, model.predict_action(features)))


if __name__ == "__main__":
    unittest.main()
