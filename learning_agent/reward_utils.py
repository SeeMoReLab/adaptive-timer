from typing import Optional, Tuple


# Throughput reward is concave: every transaction per second up to the knee is
# worth full credit, and everything above it is discounted. The agent should
# therefore treat losing 200 tps at 600 tps as a much bigger deal than losing
# 200 tps at 3000 tps, which is what we want when the timeout choice is really
# about staying out of the degraded regime rather than about peak speed.
REWARD_THROUGHPUT_KNEE_TPS = 1000.0

# Marginal value of throughput above the knee, relative to below it. 0.25 means
# 1000 extra tps past the knee is worth as much as 250 tps below it.
REWARD_THROUGHPUT_ABOVE_KNEE_WEIGHT = 0.25


def reward_tps(throughput_tps: float) -> float:
    """Apply the reward shaping used by the online learning agent."""
    value = float(max(0.0, throughput_tps))
    if value <= REWARD_THROUGHPUT_KNEE_TPS:
        return value / REWARD_THROUGHPUT_KNEE_TPS
    excess = (value - REWARD_THROUGHPUT_KNEE_TPS) / REWARD_THROUGHPUT_KNEE_TPS
    return 1.0 + REWARD_THROUGHPUT_ABOVE_KNEE_WEIGHT * excess

def reward_tps_latency(throughput_tps: float, p50_consensus_latency_ms: float) -> float:
    """Throughput and median consensus latency at a fixed exchange rate.

    10,000 tps is worth one unit and so is one second of p50 latency, so 100 ms
    trades for 1,000 tps. Within a fault phase this ranks the calibrated best
    SBFT pair first in all ten phase x rate cells of
    runs/2026-09-28-sbft-revalidation. Across phases the level moves with the
    network delay (about -0.8 from base to 5+150), which the contextual models
    must absorb through their features.
    """
    return float(throughput_tps) / 10000.0 - float(p50_consensus_latency_ms) / 1000.0


def reward_mixed(
    throughput_tps: float,
    timeout_ms: int,
    avg_consensus_latency_ms: float,
    total_transactions: int,
) -> float:
    """Throughput, discounted by the share of the window one fault would cost.

    A leader failure costs the timeout (to detect it) plus one consensus
    latency (to re-commit under the new leader). Expressed as a fraction of
    the measurement window, that is what a longer timer is actually buying
    you, so no tuning constant is needed to say how much it is worth.

    Latency enters only as an overhead relative to the window, never as a
    level - an absolute latency term ranks the delay environment instead of
    the arm, which is why it was taken out of this reward before.
    """
    if throughput_tps <= 0.0:
        return 0.0
    window_ms = 1000.0 * total_transactions / throughput_tps
    overhead_ms = float(timeout_ms) + max(0.0, avg_consensus_latency_ms)
    return reward_tps(throughput_tps) * window_ms / (window_ms + overhead_ms)

# --- Reward design selection -------------------------------------------------
#
# Which of the shapings above the calculate_*_reward functions apply. This
# is one process-wide setting rather than a per-call argument because those
# calculators are reached through signatures fixed by the protobuf report
# layout (main.reward_engineering, simulator's CSV path, plotting scripts), and
# threading a design parameter through all of them would touch every field
# list. One agent process runs one design for its whole run, so a module-level
# selection is the same thing with less plumbing. Set it once at startup, from
# --reward-design.
REWARD_DESIGNS: Tuple[str, ...] = ("reward_tps", "reward_mixed", "reward_tps_latency")
DEFAULT_REWARD_DESIGN = "reward_tps_latency"

_active_reward_design = DEFAULT_REWARD_DESIGN


def set_reward_design(name: str) -> str:
    """Select the reward shaping for this process, returning the name set."""
    global _active_reward_design
    if name not in REWARD_DESIGNS:
        raise ValueError(
            f"Unknown reward design {name!r}. Expected one of "
            f"{'|'.join(REWARD_DESIGNS)}."
        )
    _active_reward_design = name
    return _active_reward_design


def get_reward_design() -> str:
    return _active_reward_design


def _apply_reward_design(
    *,
    throughput_tps: float,
    timeout_ms: Optional[int],
    avg_consensus_latency_ms: float,
    total_transactions: int,
    p50_consensus_latency_ms: float,
    protocol_name: str = "",
) -> float:
    if _active_reward_design == "reward_tps":
        return reward_tps(throughput_tps)
    if _active_reward_design == "reward_tps_latency":
        return reward_tps_latency(throughput_tps, p50_consensus_latency_ms)
    if timeout_ms is None:
        target = f" for {protocol_name}" if protocol_name else ""
        raise ValueError(
            f"reward_design=reward_mixed needs the applied timeout{target}, "
            "but none was supplied with the report."
        )
    return reward_mixed(
        throughput_tps, timeout_ms, avg_consensus_latency_ms, total_transactions
    )


def calculate_pbft_reward(
    *,
    total_transactions: int,
    total_consensus_instances: int,
    avg_consensus_latency_ms: float,
    p50_consensus_latency_ms: float,
    p95_consensus_latency_ms: float,
    throughput_tps: float,
    avg_batch_size: float,
    p95_batch_size: float,
    leader_change_count: int,
    regency_change_count: int,
    timeout_ms: int,
    avg_inter_commit_gap_ms: float,
    p50_inter_commit_gap_ms: float,
    p95_inter_commit_gap_ms: float,
    view_change_count: int,
    no_progress_view_change_count: int,
) -> float:
    """Calculate an online PBFT reward from a complete reward report."""
    return _apply_reward_design(
        throughput_tps=throughput_tps,
        timeout_ms=timeout_ms,
        avg_consensus_latency_ms=avg_consensus_latency_ms,
        total_transactions=total_transactions,
        p50_consensus_latency_ms=p50_consensus_latency_ms,
    )


def calculate_smartbft_reward(
    *,
    total_transactions: int,
    total_consensus_instances: int,
    avg_consensus_latency_ms: float,
    p50_consensus_latency_ms: float,
    p95_consensus_latency_ms: float,
    throughput_tps: float,
    avg_batch_size: float,
    p95_batch_size: float,
    leader_change_count: int,
    regency_change_count: int,
    timeout_ms: int,
    avg_inter_commit_gap_ms: float,
    p50_inter_commit_gap_ms: float,
    p95_inter_commit_gap_ms: float,
    view_change_count: int,
    no_progress_view_change_count: int,
) -> float:
    """Calculate an online SmartBFT reward from a complete reward report.

    SmartbftReport has the same fields as PbftReport, so this takes the same
    arguments as calculate_pbft_reward but is tuned independently of the Rust
    PBFT baseline.
    """
    return _apply_reward_design(
        throughput_tps=throughput_tps,
        timeout_ms=timeout_ms,
        avg_consensus_latency_ms=avg_consensus_latency_ms,
        total_transactions=total_transactions,
        p50_consensus_latency_ms=p50_consensus_latency_ms,
    )



def calculate_sbft_reward(
    *,
    total_transactions: int,
    total_consensus_instances: int,
    avg_consensus_latency_ms: float,
    p50_consensus_latency_ms: float,
    p95_consensus_latency_ms: float,
    throughput_tps: float,
    avg_batch_size: float,
    p95_batch_size: float,
    leader_change_count: int,
    regency_change_count: int,
    timeout_ms: int,
    avg_inter_commit_gap_ms: float,
    p50_inter_commit_gap_ms: float,
    p95_inter_commit_gap_ms: float,
    view_change_count: int,
    no_progress_view_change_count: int,
    pre_prepare_latency_ms: float,
    prepare_latency_ms: float,
    commit_latency_ms: float,
    slow_path_timeout_ms: int,
    fast_path_count: int,
    slow_path_count: int,
) -> float:
    """Calculate an online SBFT reward from a complete reward report.

    SbftReport shares PbftReport's base fields and adds the SBFT phase and
    fast/slow-path context, all of which are available here for tuning.

    Unlike the report this dispatcher was first written against, SbftReport
    now carries the applied election timeout, so ``reward_mixed`` can build
    its fault-cost term here as it does for the other protocols.
    """
    return _apply_reward_design(
        throughput_tps=throughput_tps,
        timeout_ms=timeout_ms,
        avg_consensus_latency_ms=avg_consensus_latency_ms,
        total_transactions=total_transactions,
        p50_consensus_latency_ms=p50_consensus_latency_ms,
        protocol_name="sbft",
    )


def calculate_tendermint_reward(
    *,
    total_transactions: int,
    total_consensus_instances: int,
    avg_consensus_latency_ms: float,
    p50_consensus_latency_ms: float,
    p90_consensus_latency_ms: float,
    throughput_tps: float,
    timeout_violation_rate: float,
    avg_batch_size: float,
    p50_batch_size: float,
    p90_batch_size: float,
    leader_change_count: int,
    regency_change_count: int,
    propose_latency_ms: float,
    prevote_latency_ms: float,
    precommit_latency_ms: float,
    propose_timeout_ms: Optional[int] = None,
    prevote_timeout_ms: Optional[int] = None,
    precommit_timeout_ms: Optional[int] = None,
) -> float:
    """Calculate an online Tendermint reward from a complete reward report.

    TendermintReport carries no timeout field - the action is a triple, and
    only the report's *_latency_ms measurements come back - so the three
    timeouts that were applied are passed in alongside it by the caller
    (main.reward_engineering from the Timeout message, the simulator from the
    ``*_timeout_milliseconds_used`` CSV columns).

    ``reward_mixed`` prices one fault as the round budget the protocol has to
    burn before it can move on, which in Tendermint is all three phase timers,
    not the propose timer alone: a silent proposer leaves prevote and precommit
    to expire too. The three are locked together by
    ``models.tendermint_timeouts_from_propose`` (prevote = precommit =
    ceil(propose/3)), so the sum stays a fixed multiple of the arm and the
    ranking over arms is unchanged - only the scale of the discount differs.
    """
    timeout_ms = None
    if None not in (propose_timeout_ms, prevote_timeout_ms, precommit_timeout_ms):
        timeout_ms = (
            int(propose_timeout_ms)
            + int(prevote_timeout_ms)
            + int(precommit_timeout_ms)
        )
    return _apply_reward_design(
        throughput_tps=throughput_tps,
        timeout_ms=timeout_ms,
        avg_consensus_latency_ms=avg_consensus_latency_ms,
        total_transactions=total_transactions,
        p50_consensus_latency_ms=p50_consensus_latency_ms,
        protocol_name="tendermint",
    )


def calculate_hotstuff_reward(
    *,
    total_transactions: int,
    total_consensus_instances: int,
    avg_consensus_latency_ms: float,
    p50_consensus_latency_ms: float,
    p95_consensus_latency_ms: float,
    throughput_tps: float,
    avg_batch_size: float,
    p95_batch_size: float,
    leader_change_count: int,
    regency_change_count: int,
    timeout_ms: int,
    avg_inter_commit_gap_ms: float,
    p50_inter_commit_gap_ms: float,
    p95_inter_commit_gap_ms: float,
    view_change_count: int,
    no_progress_view_change_count: int,
) -> float:
    """Calculate an online HotStuff reward from a complete reward report."""
    return _apply_reward_design(
        throughput_tps=throughput_tps,
        timeout_ms=timeout_ms,
        avg_consensus_latency_ms=avg_consensus_latency_ms,
        total_transactions=total_transactions,
        p50_consensus_latency_ms=p50_consensus_latency_ms,
    )


def calculate_autobahn_reward(
    *,
    total_transactions: int,
    total_consensus_instances: int,
    avg_consensus_latency_ms: float,
    p50_consensus_latency_ms: float,
    p95_consensus_latency_ms: float,
    throughput_tps: float,
    avg_batch_size: float,
    p95_batch_size: float,
    leader_change_count: int,
    timeout_ms: int,
    avg_inter_commit_gap_ms: float,
    p50_inter_commit_gap_ms: float,
    p95_inter_commit_gap_ms: float,
    view_change_count: int,
    no_progress_view_change_count: int,
) -> float:
    """Calculate an online Autobahn reward from a complete reward report."""
    return _apply_reward_design(
        throughput_tps=throughput_tps,
        timeout_ms=timeout_ms,
        avg_consensus_latency_ms=avg_consensus_latency_ms,
        total_transactions=total_transactions,
        p50_consensus_latency_ms=p50_consensus_latency_ms,
    )
