import unittest

import numpy as np

import agent_pb2
from continuous_bandit import (
    ContinuousLinUCB,
    SimReturnGPModel,
    context_spec,
    resolve_prior_path,
    return_prior_path,
    _refined_ladder_ms,
    ContinuousThompsonSamplingModel,
    ContinuousTimeoutActionCodec,
    LatencyFloorHeuristic,
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


class LatencyFloorHeuristicTest(unittest.TestCase):
    def build_model(self, **kwargs) -> tuple:
        codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)
        names = get_report_feature_field_names(agent_pb2.PROTOCOL_PBFT)
        latency_index, instances_index = latency_feature_indices(names)
        floor_kwargs = {"multiplier": kwargs.pop("floor_k", 1.0), "quantile": 0.0}
        model = LatencyFloorHeuristic(
            action_codec=codec,
            n_features=len(names),
            latency_floor=NormalizedLatencyFloor(
                action_codec=codec,
                latency_index=latency_index,
                n_features=len(names),
                instances_index=instances_index,
                latency_floor=LatencyFloor(**floor_kwargs),
            ),
            **kwargs,
        )
        return codec, model

    def timeout_for(self, codec, model, latency_ms: float, **kwargs) -> int:
        action = model.predict_action(pbft_features(latency_ms, **kwargs))
        return codec.decode(action).pbft.election_timeout_milliseconds

    def test_plays_the_floor_itself(self) -> None:
        codec, model = self.build_model()

        timeout_ms = self.timeout_for(codec, model, 640.0)

        self.assertEqual(model.floor_ms, 640.0)
        self.assertEqual(timeout_ms, 640)

    def test_multiplier_defaults_to_one(self) -> None:
        _, model = self.build_model()

        self.assertEqual(model.multiplier, 1.0)

    def test_floor_k_scales_the_timeout(self) -> None:
        codec, model = self.build_model(floor_k=2.0)

        self.assertEqual(self.timeout_for(codec, model, 640.0), 1280)

    def test_cold_start_holds_the_middle_arm(self) -> None:
        codec, model = self.build_model()

        timeout_ms = self.timeout_for(codec, model, 0.0, instances=0.0)

        self.assertEqual(model.floor_ms, 0.0)
        self.assertEqual(
            timeout_ms, sorted(get_timeout_values_ms(agent_pb2.PROTOCOL_PBFT))[1]
        )

    def test_a_stalled_window_holds_the_last_timeout(self) -> None:
        codec, model = self.build_model()

        measured = self.timeout_for(codec, model, 640.0)
        stalled = self.timeout_for(codec, model, 0.0, instances=0.0)

        self.assertEqual(stalled, measured)

    def test_the_decision_is_deterministic(self) -> None:
        codec, model = self.build_model()

        first = self.timeout_for(codec, model, 640.0)
        repeats = [self.timeout_for(codec, model, 640.0) for _ in range(5)]

        self.assertEqual(repeats, [first] * 5)

    def test_arm_family_clamps_the_floor(self) -> None:
        arms = sorted(get_timeout_values_ms(agent_pb2.PROTOCOL_PBFT))
        codec, model = self.build_model(clamp_ms=(arms[0], arms[-1]))

        # A healthy floor is far under the shortest arm, and a stalled one far
        # over the longest; both land on the arm range the discrete baselines
        # are measured on rather than on the codec's wider bounds.
        self.assertEqual(self.timeout_for(codec, model, 37.0), arms[0])
        model.reset_model()
        self.assertEqual(self.timeout_for(codec, model, 50_000.0), arms[-1])

    def test_clamp_outside_the_codec_bounds_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.build_model(clamp_ms=(10, 1800))

    def test_codec_bounds_clamp_the_floor_by_default(self) -> None:
        codec, model = self.build_model()

        self.assertEqual(self.timeout_for(codec, model, 20.0), 100)
        model.reset_model()
        self.assertEqual(self.timeout_for(codec, model, 50_000.0), 10_000)

    def test_updates_do_not_move_the_rule(self) -> None:
        codec, model = self.build_model()
        features = pbft_features(640.0)
        action = model.predict_action(features)

        loss = model.update_model(features, action, 9999.0)

        self.assertEqual(loss, 0.0)
        self.assertEqual(self.timeout_for(codec, model, 640.0), 640)

    def test_reset_clears_the_floor_and_the_held_timeout(self) -> None:
        codec, model = self.build_model()
        self.timeout_for(codec, model, 640.0)

        model.reset_model()

        self.assertEqual(model.floor_ms, 0.0)
        self.assertEqual(
            model.timeout_ms,
            sorted(get_timeout_values_ms(agent_pb2.PROTOCOL_PBFT))[1],
        )

    def test_rejects_a_missing_floor(self) -> None:
        codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)

        with self.assertRaises(ValueError):
            LatencyFloorHeuristic(
                action_codec=codec,
                n_features=PBFT_FEATURE_DIM,
                latency_floor=None,
            )


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


class ContinuousLinUCBTest(unittest.TestCase):
    """codex_v1's estimator over a continuous timeout, and the floor on top."""

    def build_model(self, floor_k: float = 0.0, **kwargs) -> tuple:
        codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)
        names = get_report_feature_field_names(agent_pb2.PROTOCOL_PBFT)
        latency_index, instances_index = latency_feature_indices(names)
        latency_floor = None
        if floor_k > 0.0:
            latency_floor = NormalizedLatencyFloor(
                action_codec=codec,
                latency_index=latency_index,
                n_features=len(names),
                instances_index=instances_index,
                latency_floor=LatencyFloor(multiplier=floor_k, quantile=0.0),
            )
        model = ContinuousLinUCB(
            action_codec=codec,
            n_features=len(names),
            latency_floor=latency_floor,
            seed=7,
            **kwargs,
        )
        return codec, model

    def timeout_for(self, codec, model, latency_ms: float, **kwargs) -> int:
        action = model.predict_action(pbft_features(latency_ms, **kwargs))
        return codec.decode(action).pbft.election_timeout_milliseconds

    def play(self, codec, model, latency_ms: float, reward_for) -> int:
        """One decision and the reward it earns, as the agent applies them."""
        features = pbft_features(latency_ms)
        action = model.predict_action(features)
        timeout_ms = codec.decode(action).pbft.election_timeout_milliseconds
        model.update_model(features, action, float(reward_for(timeout_ms)))
        return timeout_ms

    def test_the_action_space_is_the_arm_range(self) -> None:
        codec, model = self.build_model(exploration_floor=1.0)

        timeouts = [
            self.play(codec, model, 100.0, lambda _: 1.0) for _ in range(40)
        ]

        arms_ms = get_timeout_values_ms(agent_pb2.PROTOCOL_PBFT)
        self.assertTrue(
            all(min(arms_ms) <= value <= max(arms_ms) for value in timeouts),
            timeouts,
        )

    def test_every_basis_centre_is_played_before_anything_is_fitted(
        self,
    ) -> None:
        codec, model = self.build_model()

        centers_ms = list(
            _refined_ladder_ms(get_timeout_values_ms(agent_pb2.PROTOCOL_PBFT))
        )
        opening = [
            self.play(codec, model, 100.0, lambda _: 1.0)
            for _ in range(len(centers_ms))
        ]

        # codex_v1 scores an unpulled arm at +inf, so it opens by playing each
        # arm once. Here the same rule sweeps the centres, which are the arms
        # with the midpoint of each pair inserted.
        self.assertEqual(opening, centers_ms)
        self.assertTrue(
            set(get_timeout_values_ms(agent_pb2.PROTOCOL_PBFT)).issubset(
                set(centers_ms)
            ),
            centers_ms,
        )

    def test_a_reward_between_two_arms_is_reachable(self) -> None:
        # What the continuous action space is for: with centres only on the
        # ladder an average over them is maximised at an arm, and a reward
        # that only pays between two arms could never be collected.
        codec, model = self.build_model(exploration_floor=0.0)

        for _ in range(40):
            timeout_ms = self.play(
                codec,
                model,
                100.0,
                lambda value: 1.0 if 700 <= value <= 900 else 0.0,
            )

        self.assertTrue(700 <= timeout_ms <= 900, timeout_ms)

    def test_floor_trims_the_bottom_of_the_range(self) -> None:
        codec, model = self.build_model(floor_k=1.0, exploration_floor=1.0)

        # 5+175 in config/pbft/failure_spec.xml: a ~705ms healthy tail puts
        # both 250 and 550 out of reach, as feasible_timeout_mask would.
        timeouts = [
            self.play(codec, model, 705.0, lambda _: 1.0) for _ in range(30)
        ]

        self.assertEqual(model.floor_ms, 705.0)
        self.assertTrue(all(value >= 705 for value in timeouts), timeouts)

    def test_no_floor_leaves_the_shortest_arm_reachable(self) -> None:
        codec, model = self.build_model(exploration_floor=1.0)

        timeouts = [
            self.play(codec, model, 705.0, lambda _: 1.0) for _ in range(30)
        ]

        self.assertEqual(model.floor_ms, 0.0)
        self.assertTrue(any(value < 705 for value in timeouts), timeouts)

    def test_floor_above_the_range_keeps_the_longest_arm(self) -> None:
        codec, model = self.build_model(floor_k=1.0, exploration_floor=1.0)

        timeouts = [
            self.play(codec, model, 50_000.0, lambda _: 1.0) for _ in range(10)
        ]

        self.assertEqual(set(timeouts), {1200})

    def test_reward_moves_the_choice_toward_what_paid(self) -> None:
        codec, model = self.build_model(exploration_floor=0.0)

        def middle_pays(timeout_ms: int) -> float:
            return 1.0 if 450 <= timeout_ms <= 700 else 0.0

        for _ in range(20):
            timeout_ms = self.play(codec, model, 100.0, middle_pays)

        self.assertGreater(model.observation_count, 0)
        self.assertTrue(400 <= timeout_ms <= 800, timeout_ms)

    def test_a_long_reward_is_not_pulled_short_by_the_basis(self) -> None:
        # Unnormalized bumps peak inside the range when the reward is earned
        # at its end; the partition-of-unity basis has to reach 1200.
        codec, model = self.build_model(exploration_floor=0.0)

        for _ in range(20):
            timeout_ms = self.play(
                codec, model, 100.0, lambda value: 1.0 if value >= 1000 else 0.0
            )

        self.assertGreaterEqual(timeout_ms, 1000)

    def test_neighbouring_timeouts_share_evidence(self) -> None:
        # The property the ladder does not have: a reward earned at one
        # timeout has to move the estimate at the timeout next to it.
        codec, model = self.build_model(exploration_floor=0.0)
        features = pbft_features(100.0)
        row = model._context(features)
        near = model._position_of(600)
        far = model._position_of(1200)
        before = model._scores(row, np.asarray([near, far]))

        model.update_model(features, np.asarray([codec.normalize_ms(550)]), 4.0)

        after = model._scores(row, np.asarray([near, far]))
        self.assertGreater(after[0] - before[0], after[1] - before[1])

    def test_reward_follows_the_timeout_that_was_applied(self) -> None:
        # The replicas can run with a timeout this agent did not ask for; the
        # update lands on that timeout, not on the one this model chose.
        # Greedy on the fitted mean: with the bonus on, the regions this loop
        # never updates would rightly be explored instead.
        codec, model = self.build_model(
            exploration_floor=0.0, confidence_scale=0.0
        )
        features = pbft_features(100.0)
        applied = np.asarray([codec.normalize_ms(1150)])

        for _ in range(20):
            model.predict_action(features)
            model.update_model(features, applied, 1.0)

        self.assertGreaterEqual(self.timeout_for(codec, model, 100.0), 1000)

    def test_an_unlearned_region_keeps_its_confidence_bonus(self) -> None:
        # The discount is what lets an avoided region come back: with the
        # bonus switched off the model would sit on its first good reading.
        codec, model = self.build_model(exploration_floor=0.0)
        features = pbft_features(100.0)

        for _ in range(30):
            model.predict_action(features)
            model.update_model(
                features, np.asarray([codec.normalize_ms(250)]), 1.0
            )

        row = model._context(features)
        scores = model._scores(row, np.asarray([0.0, 1.0]))
        self.assertGreater(scores[1], 0.0)

    def _moving_optimum_regret(self, **kwargs) -> tuple:
        """Seven phases whose paying timeout moves under the agent.

        The reward is a bump around the phase's best timeout, on the 2..10
        scale reward_tps produces at this cluster's throughput - the scale is
        the point, because it is what the ridge prior is or is not centred on.
        """
        import math

        codec, model = self.build_model(**kwargs)
        best_by_phase = (250, 250, 250, 550, 550, 1200, 250)

        def pays(phase: int, timeout_ms: int) -> float:
            ratio = math.log(timeout_ms / best_by_phase[phase]) / 0.35
            return 9.0 * math.exp(-0.5 * ratio * ratio) + 1.0

        chosen, regret = [], 0.0
        for phase, best in enumerate(best_by_phase):
            for _ in range(15):
                features = pbft_features(100.0 + 80 * phase)
                action = model.predict_action(features)
                timeout_ms = codec.decode(action).pbft.election_timeout_milliseconds
                model.update_model(features, action, pays(phase, timeout_ms))
                regret += pays(phase, best) - pays(phase, timeout_ms)
                chosen.append((phase, timeout_ms))
        finals = sorted(t for p, t in chosen if p == len(best_by_phase) - 1)
        return regret, finals[len(finals) // 2]

    def test_centring_the_reward_is_what_lets_it_follow_a_phase(self) -> None:
        # Uncentred, an unplayed timeout's estimate decays toward the ridge
        # prior of 0, which on this reward scale is worse than anything ever
        # measured, so the confidence bonus can never bring it back.
        centred, _ = self._moving_optimum_regret(center_rewards=True)
        codex_like, _ = self._moving_optimum_regret(center_rewards=False)

        self.assertLess(centred, codex_like)

    def test_it_comes_back_down_when_the_phase_does(self) -> None:
        _, final_median = self._moving_optimum_regret()

        # The last phase pays at the shortest timeout again, after a phase
        # that paid at the longest: staying long is the failure this catches.
        self.assertLess(final_median, 550)

    def test_reset_replays_the_same_decisions(self) -> None:
        codec, model = self.build_model(floor_k=1.0)

        first = [self.play(codec, model, 405.0, lambda _: 1.0) for _ in range(8)]
        model.reset_model()
        second = [self.play(codec, model, 405.0, lambda _: 1.0) for _ in range(8)]

        self.assertEqual(first, second)

    def test_rejects_centres_outside_the_action_range(self) -> None:
        codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT)
        with self.assertRaises(ValueError):
            ContinuousLinUCB(
                action_codec=codec,
                n_features=get_report_feature_dim(agent_pb2.PROTOCOL_PBFT),
                basis_centers_ms=(250, 9000),
            )


class PretrainingAblationTest(unittest.TestCase):
    """The three --prior-path settings the Full RL blocks run."""

    def test_empty_selects_the_protocols_shipped_return_prior(self) -> None:
        for protocol in (
            agent_pb2.PROTOCOL_PBFT,
            agent_pb2.PROTOCOL_SMARTBFT,
            agent_pb2.PROTOCOL_TENDERMINT,
            agent_pb2.PROTOCOL_SBFT,
        ):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    resolve_prior_path(protocol, ""),
                    return_prior_path(protocol),
                )

    def test_none_is_untrained(self) -> None:
        self.assertIsNone(resolve_prior_path(agent_pb2.PROTOCOL_PBFT, "none"))

    def test_a_path_is_taken_as_the_ablation_prior(self) -> None:
        self.assertEqual(
            resolve_prior_path(agent_pb2.PROTOCOL_PBFT, "a/blind_prior.npz"),
            "a/blind_prior.npz",
        )

    def test_untrained_model_opens_mid_range_and_learns_online(self) -> None:
        protocol = agent_pb2.PROTOCOL_PBFT
        spec = context_spec(protocol)
        codec = ContinuousTimeoutActionCodec(
            protocol, bounds_ms=spec.bounds_ms, wait_bounds_ms=spec.wait_bounds_ms
        )
        model = SimReturnGPModel(
            action_codec=codec,
            field_names=get_report_feature_field_names(protocol),
            prior_path=None,
            reward_design="reward_tps_latency",
        )
        features = np.zeros(get_report_feature_dim(protocol))
        features[0], features[1], features[2], features[5] = 200000, 100, 80.0, 40000
        opening = codec.denormalize_ms(float(np.atleast_1d(model.predict_action(features))[0]))
        lower_ms, upper_ms = spec.bounds_ms
        # Nothing in a zero prior prefers an end of the range, so it must not
        # open at one: argmax over a flat grid would otherwise pick index 0.
        self.assertGreater(opening, lower_ms)
        self.assertLess(opening, upper_ms)
        # And the GP still takes an online update, fitting the reward itself.
        action = model.predict_action(features)
        model.update_model(features, action, 5.0)
        self.assertIsNotNone(model.predict_action(features))

    def test_untrained_sbft_opens_inside_both_dimensions(self) -> None:
        protocol = agent_pb2.PROTOCOL_SBFT
        spec = context_spec(protocol)
        codec = ContinuousTimeoutActionCodec(
            protocol, bounds_ms=spec.bounds_ms, wait_bounds_ms=spec.wait_bounds_ms
        )
        model = SimReturnGPModel(
            action_codec=codec,
            field_names=get_report_feature_field_names(protocol),
            prior_path=None,
            reward_design="reward_tps_latency",
        )
        features = np.zeros(get_report_feature_dim(protocol))
        features[0], features[1], features[2], features[5] = 200000, 100, 80.0, 40000
        action = np.atleast_1d(model.predict_action(features))
        self.assertEqual(action.size, 2)
        for dimension, (lower_ms, upper_ms) in enumerate(
            (spec.bounds_ms, spec.wait_bounds_ms)
        ):
            value = codec.denormalize_ms(float(action[dimension]), dimension)
            with self.subTest(dimension=dimension):
                self.assertGreater(value, lower_ms)
                self.assertLess(value, upper_ms)
