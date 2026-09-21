import unittest

import numpy as np

import agent_pb2
from continuous_bandit import (
    ContinuousThompsonSamplingModel,
    ContinuousTimeoutActionCodec,
    NormalizedLatencyFloor,
)
from latency_floor import (
    LatencyFloor,
    LatencyFloorPolicy,
    feasible_timeout_mask,
    latency_feature_indices,
)
from models import (
    TensorModel,
    get_report_feature_dim,
    get_report_feature_field_names,
    get_timeout_values_ms,
)


PBFT_ARMS_MS = (300, 800, 1800)
PBFT_FEATURE_DIM = 16


def pbft_features(latency_ms: float, instances: float = 10.0) -> np.ndarray:
    """One report row carrying `latency_ms` in the field the floor reads.

    Every latency field is set, so the helper keeps working if the floor's
    default field changes again; latency_feature_indices decides which one is
    actually consulted.
    """
    names = get_report_feature_field_names(agent_pb2.PROTOCOL_PBFT)
    row = np.zeros(len(names), dtype=float)
    for field in (
        "avg_consensus_latency_ms",
        "p50_consensus_latency_ms",
        "p95_consensus_latency_ms",
    ):
        row[names.index(field)] = latency_ms
    row[names.index("total_consensus_instances")] = instances
    return row


class FixedArmModel(TensorModel):
    """One-hot model that always asks for the same arm."""

    def __init__(self, action_dim: int, arm: int) -> None:
        self._action_dim = action_dim
        self._arm = arm

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features, dtype=float)
        single = rows.ndim == 1
        if single:
            rows = rows.reshape(1, -1)
        out = np.zeros((len(rows), self._action_dim), dtype=float)
        out[:, self._arm] = 1.0
        return out[0] if single else out

    def update_model(self, features, action, reward) -> float:
        self.last_update = (features, action, reward)
        return 1.0

    def reset_model(self) -> None:
        self.was_reset = True


class ScoreModel(TensorModel):
    """Score-vector model, so the floor has something to re-rank."""

    def __init__(self, scores) -> None:
        self._scores = np.asarray(scores, dtype=float)

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features, dtype=float)
        if rows.ndim == 1:
            return self._scores.copy()
        return np.repeat(self._scores.reshape(1, -1), len(rows), axis=0)

    def update_model(self, features, actions, rewards, backward: bool = True) -> float:
        return 0.0

    def reset_model(self) -> None:
        pass


class LatencyFloorTest(unittest.TestCase):
    def test_cold_start_places_no_bound(self) -> None:
        self.assertEqual(LatencyFloor().floor_ms, 0.0)

    def test_floor_tracks_the_low_quantile_of_observed_latency(self) -> None:
        floor = LatencyFloor(multiplier=1.0, quantile=0.25)
        for latency_ms in (400.0, 420.0, 800.0, 1600.0):
            floor.observe(latency_ms)

        # 25th percentile of the window, not its mean: the two inflated windows
        # do not carry the bound.
        self.assertAlmostEqual(floor.floor_ms, 415.0)

    def test_multiplier_scales_the_bound(self) -> None:
        floor = LatencyFloor(multiplier=1.5, quantile=0.0)
        floor.observe(400.0)

        self.assertAlmostEqual(floor.floor_ms, 600.0)

    def test_stalled_and_unmeasured_windows_are_dropped(self) -> None:
        floor = LatencyFloor(multiplier=1.0, quantile=0.0)
        floor.observe(500.0)
        floor.observe(0.0, alive=True)  # committed nothing: no measurement
        floor.observe(10.0, alive=False)  # window did not commit
        floor.observe(float("nan"))

        self.assertEqual(floor.observation_count, 1)
        self.assertAlmostEqual(floor.floor_ms, 500.0)

    def test_window_forgets_old_observations(self) -> None:
        floor = LatencyFloor(multiplier=1.0, quantile=0.0, window=2)
        floor.observe(100.0)
        floor.observe(900.0)
        floor.observe(800.0)

        self.assertAlmostEqual(floor.floor_ms, 800.0)

    def test_max_floor_caps_the_bound(self) -> None:
        floor = LatencyFloor(multiplier=1.0, quantile=0.0, max_floor_ms=700.0)
        floor.observe(5000.0)

        self.assertAlmostEqual(floor.floor_ms, 700.0)

    def test_reset_clears_the_window(self) -> None:
        floor = LatencyFloor()
        floor.observe(500.0)
        floor.reset()

        self.assertEqual(floor.floor_ms, 0.0)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            LatencyFloor(multiplier=0.0)
        with self.assertRaises(ValueError):
            LatencyFloor(window=0)
        with self.assertRaises(ValueError):
            LatencyFloor(quantile=1.5)


class FeasibleTimeoutMaskTest(unittest.TestCase):
    def test_arms_below_the_floor_are_hidden(self) -> None:
        mask = feasible_timeout_mask(PBFT_ARMS_MS, 700.0)

        self.assertEqual(mask.tolist(), [False, True, True])

    def test_a_floor_above_every_arm_keeps_the_longest(self) -> None:
        mask = feasible_timeout_mask(PBFT_ARMS_MS, 9000.0)

        self.assertEqual(mask.tolist(), [False, False, True])

    def test_no_floor_keeps_every_arm(self) -> None:
        mask = feasible_timeout_mask(PBFT_ARMS_MS, 0.0)

        self.assertEqual(mask.tolist(), [True, True, True])


class LatencyFloorPolicyTest(unittest.TestCase):
    def build(self, model: TensorModel, **kwargs) -> LatencyFloorPolicy:
        names = get_report_feature_field_names(agent_pb2.PROTOCOL_PBFT)
        latency_index, instances_index = latency_feature_indices(names)
        return LatencyFloorPolicy(
            model=model,
            timeouts_ms=PBFT_ARMS_MS,
            latency_index=latency_index,
            n_features=len(names),
            instances_index=instances_index,
            **kwargs,
        )

    def test_short_arm_survives_while_consensus_is_fast(self) -> None:
        policy = self.build(
            FixedArmModel(3, arm=0),
            latency_floor=LatencyFloor(multiplier=1.0, quantile=0.0),
        )

        action = policy.predict_action(pbft_features(120.0))

        self.assertEqual(action.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(policy.override_count, 0)

    def test_arm_under_the_measured_latency_is_replaced(self) -> None:
        policy = self.build(
            FixedArmModel(3, arm=0),
            latency_floor=LatencyFloor(multiplier=1.0, quantile=0.0),
        )

        action = policy.predict_action(pbft_features(900.0))

        # 300ms cannot cover a 900ms round, so the shortest arm that can is
        # played instead of spending the decision on a certain view change.
        self.assertEqual(action.tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(policy.override_count, 1)

    def test_stalled_window_does_not_lower_the_floor(self) -> None:
        policy = self.build(
            FixedArmModel(3, arm=0),
            latency_floor=LatencyFloor(multiplier=1.0, quantile=0.0),
        )
        policy.predict_action(pbft_features(900.0))

        action = policy.predict_action(pbft_features(0.0, instances=0.0))

        self.assertEqual(action.tolist(), [0.0, 0.0, 1.0])

    def test_scores_are_reranked_over_the_feasible_arms(self) -> None:
        policy = self.build(
            ScoreModel([9.0, 1.0, 5.0]),
            latency_floor=LatencyFloor(multiplier=1.0, quantile=0.0),
        )

        action = policy.predict_action(pbft_features(900.0))

        # 300 scores best but is infeasible; 1800 is the best of what is left.
        self.assertEqual(action.tolist(), [0.0, 0.0, 1.0])

    def test_batches_are_constrained_row_by_row(self) -> None:
        policy = self.build(
            FixedArmModel(3, arm=0),
            latency_floor=LatencyFloor(multiplier=1.0, quantile=0.0),
        )

        actions = policy.predict_action(
            np.vstack([pbft_features(120.0), pbft_features(120.0)])
        )

        self.assertEqual(actions.shape, (2, 3))
        self.assertEqual(actions[0].tolist(), [1.0, 0.0, 0.0])

    def test_update_reaches_a_model_without_a_backward_parameter(self) -> None:
        inner = FixedArmModel(3, arm=0)
        policy = self.build(inner)

        loss = policy.update_model(
            pbft_features(500.0), np.asarray(2.0), 0.5, backward=True
        )

        self.assertEqual(loss, 1.0)
        self.assertEqual(inner.last_update[2], 0.5)

    def test_reset_clears_the_floor_and_the_wrapped_model(self) -> None:
        inner = FixedArmModel(3, arm=0)
        policy = self.build(
            inner, latency_floor=LatencyFloor(multiplier=1.0, quantile=0.0)
        )
        policy.predict_action(pbft_features(900.0))

        policy.reset_model()

        self.assertTrue(inner.was_reset)
        self.assertEqual(policy.latency_floor.floor_ms, 0.0)
        self.assertEqual(policy.override_count, 0)


class NormalizedLatencyFloorTest(unittest.TestCase):
    def build(self, **kwargs) -> NormalizedLatencyFloor:
        names = get_report_feature_field_names(agent_pb2.PROTOCOL_PBFT)
        latency_index, instances_index = latency_feature_indices(names)
        return NormalizedLatencyFloor(
            action_codec=ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT),
            latency_index=latency_index,
            n_features=len(names),
            instances_index=instances_index,
            **kwargs,
        )

    def test_unobserved_floor_leaves_the_range_open(self) -> None:
        self.assertEqual(self.build().lower_bound(), 0.0)

    def test_bound_maps_through_the_log_scale(self) -> None:
        codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)
        floor = self.build(latency_floor=LatencyFloor(multiplier=1.0, quantile=0.0))
        floor.observe_features(pbft_features(900.0))

        self.assertAlmostEqual(floor.lower_bound(), codec.normalize_ms(900))

    def test_floor_above_the_range_keeps_only_the_longest_timeout(self) -> None:
        floor = self.build(latency_floor=LatencyFloor(multiplier=1.0, quantile=0.0))
        floor.observe_features(pbft_features(50_000.0))

        self.assertEqual(floor.lower_bound(), 1.0)


class ContinuousModelFloorTest(unittest.TestCase):
    def build_model(self, multiplier: float = 1.0):
        codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)
        names = get_report_feature_field_names(agent_pb2.PROTOCOL_PBFT)
        latency_index, instances_index = latency_feature_indices(names)
        model = ContinuousThompsonSamplingModel(
            action_dim=codec.action_dim,
            n_features=len(names),
            anchor_actions=codec.anchor_actions(),
            initial_design_size=4,
            memory_size=64,
            n_heads=4,
            candidate_count=64,
            seed=7,
            latency_floor=NormalizedLatencyFloor(
                action_codec=codec,
                latency_index=latency_index,
                n_features=len(names),
                instances_index=instances_index,
                latency_floor=LatencyFloor(multiplier=multiplier, quantile=0.0),
            ),
        )
        return codec, model

    def timeouts_for(self, codec, model, latency_ms: float, draws: int):
        features = pbft_features(latency_ms)
        timeouts = []
        for _ in range(draws):
            action = model.predict_action(features)
            timeouts.append(
                codec.decode(action).pbft.election_timeout_milliseconds
            )
            model.update_model(features, action, 1.0)
        return timeouts

    def test_initial_design_stays_above_the_floor(self) -> None:
        codec, model = self.build_model()

        timeouts = self.timeouts_for(codec, model, 900.0, draws=4)

        self.assertTrue(all(value >= 900 for value in timeouts), timeouts)

    def test_fitted_search_stays_above_the_floor(self) -> None:
        codec, model = self.build_model()

        timeouts = self.timeouts_for(codec, model, 900.0, draws=20)

        self.assertTrue(all(value >= 900 for value in timeouts), timeouts)

    def test_design_is_not_piled_onto_the_bound(self) -> None:
        codec, model = self.build_model()

        timeouts = self.timeouts_for(codec, model, 900.0, draws=4)

        self.assertGreater(len(set(timeouts)), 1, timeouts)

    def test_no_floor_reaches_the_short_end_of_the_range(self) -> None:
        codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)
        model = ContinuousThompsonSamplingModel(
            action_dim=codec.action_dim,
            n_features=PBFT_FEATURE_DIM,
            anchor_actions=codec.anchor_actions(),
            initial_design_size=16,
            memory_size=64,
            n_heads=4,
            candidate_count=64,
            seed=7,
        )

        timeouts = self.timeouts_for(codec, model, 900.0, draws=16)

        self.assertTrue(any(value < 900 for value in timeouts), timeouts)


class ProtocolWiringTest(unittest.TestCase):
    def test_pbft_feature_indices(self) -> None:
        names = get_report_feature_field_names(agent_pb2.PROTOCOL_PBFT)

        # Default field is the window mean: the p95 is dominated by the rounds
        # the current timer already mishandled, so it reports the arm's own
        # thrashing back to the rule that chose it.
        self.assertEqual(names[2], "avg_consensus_latency_ms")
        self.assertEqual(latency_feature_indices(names), (2, 1))
        self.assertEqual(
            latency_feature_indices(names, "p95_consensus_latency_ms"), (4, 1)
        )
        self.assertEqual(len(names), get_report_feature_dim(agent_pb2.PROTOCOL_PBFT))

    def test_tendermint_arms_are_ordered_for_the_mask(self) -> None:
        arms = get_timeout_values_ms(agent_pb2.PROTOCOL_TENDERMINT)

        self.assertEqual(list(arms), sorted(arms))


if __name__ == "__main__":
    unittest.main()
