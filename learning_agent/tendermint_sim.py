"""Analytic Tendermint (CometBFT) simulator for pretraining the continuous model.

Built from the protocol mechanics and the constants in config/tendermint, not
fit to run data. One height is simulated round by round, all four nodes in step:

- Proposer of (height h, round r) is (h + r) mod 4 (equal power, round robin).
  Node 0 is the maverick: it holds its proposal ``D`` ms after building it and
  drops it if its own propose step has already timed out
  (``delayedSendPreparedProposal``).
- A proposal reaches the other nodes one link hop after it is sent, but not
  before the proposer has learned their round state (``peer_hops`` extra hops
  of NewRoundStep gossip), plus gossip jitter.
- If it arrives before the propose timeout ``T + 150 r`` the round commits
  after the prevote step and the precommit step, one link hop each. Otherwise everyone prevotes nil, the nil prevote and
  precommit majorities form after two vote hops, and the precommit wait
  ``ceil(T / 3) + 50 r`` runs before round r + 1 (a nil precommit majority
  always waits, as in ``addVote``).
- Transactions arrive at ``load_tps`` into a mempool capped at 5000
  (config.toml ``size``); arrivals to a full mempool are lost. A block takes
  everything pending.

Episodes follow the wall-clock learning windows in config.toml: 7 s features,
2 s reply wait, 4 s warm-up, 7 s reward. Reports carry every TendermintReport
field (proto/agent.proto), with the same round-0 definitions as
``consensus/adaptive_timer.go``, as the agent reads them in sharing mode,
where every row runs: the median over the four nodes' reports, a window
statistic taken over the nodes that measured it, 0 only when none did
(aggregator.measured_only_fields). Until 2026-10-19 (restart D11) they were
node 0's own report, and node 0 is the maverick: in the heights it proposes,
its propose step ends when it sends its own proposal (after the hold, as
``delayedSendPreparedProposal`` enters prevote only then) and its prevote step
carries the block transfer that the other nodes' propose steps carry, and
under --peer-spread its nil prevotes follow its own timer. The nodes run in
step, so the consensus latency, batch and inter-commit gap statistics are the
same on every node; only the propose and prevote steps and, under
--peer-spread, the timeout differ between them.

Training runs draw from the shared pretraining recipe (sim_common; Kevin,
2026-10-19, restart D11):

- the offered load with draw_load_tps around N = 5,000 (reward_normalizer_tps),
  the fault onsets with fault_start_times after the base phase at 0, the extra
  link delay with draw_link_extra_ms, the base one-way delay with
  draw_base_delay_ms, no netem jitter (JITTER_RATIO), and the initial propose
  timeout with draw_initial_timeout_ms;
- this simulator's own fault mix (none 0.2, delayed proposer 0.35, link 0.35,
  both 0.1). A delayed proposer holds draw_hold_ms (60%) or is silent (40%): a
  hold 0.5-2x around the spec's 10 s, longer than any propose timeout in the
  action range, so its proposal is always dropped. 40% is the share of the old
  log-uniform 100-30,000 ms hold above the 3,000 ms action bound, so the silent
  proposer the spec runs stays in training;
- every constant 0.5-2x around its SimConstants value (around): the fitted
  ones (FREE in scripts/cloudlab/tmp/cal/blindall/tendermint/fit/sim_fit.py) as
  before, and the unfitted create_ms and exec_ms (5 ms, eyeballed from
  b1_codex) and peer_hops (1, now a float hop multiplier) in place of their old
  wide ranges;
- no initial backlog: a block takes the whole mempool, so a backlog drains in
  the first height.

spec_schedule() reads config/tendermint/failure_spec.xml and raises on anything
the simulator cannot represent.
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from prior_windows import PriorWindows, check_prior_windows
from sim_common import schedule_rows, scheduled_action
from sim_common import (
    JITTER_RATIO,
    around,
    draw_base_delay_ms,
    draw_hold_ms,
    draw_initial_timeout_ms,
    draw_link_extra_ms,
    draw_load_tps,
    fault_start_times,
)

N_NODES = 4
QUORUM = 3  # 2f+1 of 4: accepting nodes for a block, nil prevotes to end a round
MEMPOOL_CAP = 5000
PROPOSE_DELTA_MS = 150.0
VOTE_DELTA_MS = 50.0
FEATURE_S = 7.0
REPLY_S = 2.0
WARMUP_S = 4.0
REWARD_S = 7.0
EPISODE_S = FEATURE_S + REPLY_S + WARMUP_S + REWARD_S
SIM_WINDOWS = PriorWindows.from_seconds(FEATURE_S, REPLY_S, WARMUP_S, REWARD_S)


@dataclass
# 5,000 tx/s refit (2026-10-12, runs/2026-10-12-tm-recal5k): 12,000 fresh candidates (seeds 2-9) on the bl5k_ blind runs;
# real 5k check loss 0.068 (old 4k-fit constants: 0.125), regret 0.015. load_tps 5000, training draw 1250-10000.
# Fitted defaults: blind best-by-loss candidate i=1342 (loss 0.105) of s3fit_tendermint_blind_s9.jsonl, restart S3 blind refit (seeds 7,8,9,10,11 x 1500, best of the seeds; real-spec mean regret 0.034 PASS, 2026-10-20).
class SimConstants:
    """Defaults are an evaluation point eyeballed from b1_codex, used to
    pre-screen policies; training draws every field from sample_constants,
    the fitted ones around these defaults."""

    base_delay_ms: float = 5.0
    vote_proc_ms: float = 6.53284
    create_ms: float = 5.0
    create_per_tx_ms: float = 0.001656
    exec_ms: float = 5.0
    exec_per_tx_ms: float = 0.033648
    # NewRoundStep gossip hops before a proposal can go to a peer, a multiplier
    # of the hop time (a float drawn 0.5-2x since restart D11; 0, 1 or 2 until then).
    peer_hops: float = 1.0
    # netem jitter as a fraction of each hop's delay: 0, as every spec sets
    # (sim_common.JITTER_RATIO, Kevin, 2026-10-19; the applier's 0.2 default
    # until then, which the fitted constants were fit under).
    jitter_ratio: float = JITTER_RATIO
    # The reactor's gossip sleep and flush before a proposal goes out.
    gossip_proc_ms: float = 7.087048
    # Block parts take longer to arrive the bigger the block, and more so on a
    # slow link (the transfer is window-limited, so it scales with the RTT).
    propagate_per_tx_ms: float = 0.005694
    propagate_per_tx_per_delay_ms: float = 7e-06
    # Per-proposal lognormal spread of the transfer time: a timeout just above
    # the mean arrival still misses the slow tail.
    propagate_sigma: float = 0.211933
    load_tps: float = 5000.0  # the harness's offered rate (5 connections x 25 x 40; 4000 before 2026-10-12)
    # Offered load falls when the client's own links slow down.
    load_delay_sensitivity: float = 0.000302


@dataclass
class Phase:
    start_s: float
    extra_delay_ms: float = 0.0
    proposal_delay_ms: float = 0.0


@dataclass
class _Window:
    start_s: float
    end_s: float
    txs: int = 0
    heights: int = 0
    rounds_gt0: int = 0
    round_changes: int = 0
    lat: List[float] = field(default_factory=list)
    # The round-0 propose and prevote steps per node, node 0 first: the only
    # samples that differ between the nodes (module docstring).
    prop: List[List[float]] = field(default_factory=lambda: [[] for _ in range(N_NODES)])
    prev: List[List[float]] = field(default_factory=lambda: [[] for _ in range(N_NODES)])
    prec: List[float] = field(default_factory=list)
    batch: List[float] = field(default_factory=list)
    commit_ms: List[float] = field(default_factory=list)

    def report(self, node_timeouts_ms: Sequence[float]) -> Dict[str, float]:
        """Every TendermintReport field as the agent reads it in sharing mode
        (module docstring); ``node_timeouts_ms`` are the propose timeouts in
        force, node 0 first."""
        def avg(v):
            return float(np.mean(v)) if len(v) else 0.0

        def pct(v, q):
            # Nearest rank, sorted[ceil(q/100 n) - 1], as the engine's windowPercentile
            # (cometbft consensus/adaptive_timer.go).
            return float(np.percentile(v, q, method="inverted_cdf")) if len(v) else 0.0

        def family(v, avg_name: str, pct_name: str) -> Dict[str, float]:
            # The avg/p50/p90/p95 family of one sample list under the proto's names.
            out = {avg_name: avg(v)}
            out.update({pct_name.format(q): pct(v, q) for q in (50, 90, 95)})
            return out

        def shared(values: Sequence[float]) -> float:
            # aggregator.measured_only_fields: the median of the nodes that measured it.
            measured = [x for x in values if x > 0.0]
            return float(np.median(measured)) if measured else 0.0

        elapsed = self.end_s - self.start_s
        # main._fill_empty_window_latency: the agent gives a window in which
        # nothing committed its own length as its latency (the report gives 0).
        empty_ms = elapsed * 1000.0
        report = {
            "total_transactions": float(self.txs),
            "total_consensus_instances": float(self.heights),
            "throughput_tps": self.txs / elapsed if elapsed > 0 else 0.0,
            "timeout_violation_rate": (
                self.rounds_gt0 / self.heights if self.heights else 0.0
            ),
            "leader_change_count": float(self.rounds_gt0),
            "regency_change_count": float(self.round_changes),
            # Each node reports its own uint32 timeout; the plain median, rounded
            # as aggregator._array_to_report rounds an integer field.
            "timeout_ms": float(max(0, int(round(float(np.median(
                [int(round(t)) for t in node_timeouts_ms])))))),
        }
        latency = family(self.lat, "avg_consensus_latency_ms", "p{}_consensus_latency_ms")
        report.update({k: (v if self.lat else empty_ms) for k, v in latency.items()})
        report.update(family(self.batch, "avg_batch_size", "p{}_batch_size"))
        report.update(family(list(np.diff(self.commit_ms)), "avg_inter_commit_gap_ms",
                             "p{}_inter_commit_gap_ms"))
        report.update(family(self.prec, "precommit_latency_ms", "p{}_precommit_latency_ms"))
        per_node = [
            {**family(self.prop[i], "propose_latency_ms", "p{}_propose_latency_ms"),
             **family(self.prev[i], "prevote_latency_ms", "p{}_prevote_latency_ms")}
            for i in range(N_NODES)
        ]
        report.update({k: shared([node[k] for node in per_node]) for k in per_node[0]})
        return report


@functools.lru_cache(maxsize=None)
def check_report_fields() -> None:
    """Raise unless the simulated report has exactly TendermintReport's fields."""
    import agent_pb2
    from models import get_report_field_names

    have = set(_Window(0.0, 1.0).report([1.0] * N_NODES))
    want = set(get_report_field_names(agent_pb2.PROTOCOL_TENDERMINT))
    if have != want:
        raise ValueError(
            f"tendermint_sim report fields differ from TendermintReport: missing "
            f"{sorted(want - have)}, extra {sorted(have - want)}"
        )


class TendermintSim:
    """Continuous-time simulation driven one episode at a time."""

    def __init__(
        self,
        constants: SimConstants,
        phases: Sequence[Phase],
        seed: int,
        start_offset_s: float = 0.0,
        initial_timeout_ms: float = 300.0,
        peer_factors: Optional[Sequence[float]] = None,
    ):
        self.c = constants
        self.phases = sorted(phases, key=lambda p: p.start_s)
        self.rng = np.random.default_rng(seed)
        self.t_ms = 0.0
        self.height = 1
        self.pending = 0.0
        self.last_arrival_ms = 0.0
        self.timeout_ms = float(initial_timeout_ms)
        # The other three nodes' propose timeouts as multiples of node 0's. On
        # CloudLab each replica learns its own timeout, and a nil round ends
        # when the third-lowest timer fires, so a node with a low timer sees
        # its vote steps stretched by its peers' timers (2026-09-30: real p1
        # prevote step 24 ms under 130 ms, 102 ms at 800-1500, while the
        # lockstep simulation reads 10 ms at every timeout). Ones = lockstep.
        self.peer_factors = np.ones(N_NODES - 1) if peer_factors is None else np.asarray(peer_factors, dtype=float)
        if self.peer_factors.shape != (N_NODES - 1,) or np.any(self.peer_factors <= 0.0):
            raise ValueError(f"peer_factors must be {N_NODES - 1} positive multipliers, got {peer_factors}")
        check_report_fields()
        self.episode_start_s = float(start_offset_s)
        self._advance_to(start_offset_s * 1000.0, None)

    def node_timeouts_ms(self) -> np.ndarray:
        """The four nodes' propose timeouts, node 0 first."""
        return self.timeout_ms * np.concatenate(([1.0], self.peer_factors))

    def phase_at(self, t_s: float) -> Tuple[int, Phase]:
        idx = 0
        for i, p in enumerate(self.phases):
            if t_s >= p.start_s:
                idx = i
        return idx, self.phases[idx]

    def _arrive(self, now_ms: float) -> None:
        dt = max(0.0, now_ms - self.last_arrival_ms)
        _, ph = self.phase_at(self.last_arrival_ms / 1000.0)
        rate = self.c.load_tps / (1.0 + self.c.load_delay_sensitivity * ph.extra_delay_ms)
        self.pending = min(float(MEMPOOL_CAP), self.pending + rate * dt / 1000.0)
        self.last_arrival_ms = now_ms

    def _one_height(self) -> Tuple[float, Dict[str, Any]]:
        """Simulate the current height from self.t_ms; return commit time and sample."""
        c = self.c
        t0_first = self.t_ms
        t0 = t0_first
        # Per-node propose timeouts, node 0 first; the quorum timer is the
        # third-lowest (2f+1 of 4): a block needs three accepting nodes, and a
        # nil round advances when the third nil prevote is cast.
        Ts = self.node_timeouts_ms()
        T = float(Ts[0])
        Tq = float(np.sort(Ts)[QUORUM - 1])
        Tpc_q = float(max(1, (int(Tq) + 1) // 3))
        sample: Dict[str, Any] = {}
        r = 0
        while True:
            _, ph = self.phase_at(t0 / 1000.0)
            d = c.base_delay_ms + ph.extra_delay_ms
            proposer = (self.height + r) % N_NODES
            T_r = T + PROPOSE_DELTA_MS * r
            Ts_r = Ts + PROPOSE_DELTA_MS * r
            Tpc_r = Tpc_q + VOTE_DELTA_MS * r
            self._arrive(t0)
            ntx = int(self.pending)
            create = c.create_ms + c.create_per_tx_ms * ntx
            t_send = t0 + create
            if proposer == 0 and ph.proposal_delay_ms > 0.0:
                t_send += ph.proposal_delay_ms
                if t_send > t0 + T_r:
                    t_send = math.inf  # dropped: its own propose step timed out
            u = self.rng.uniform
            jr = c.jitter_ratio
            transfer = ntx * (c.propagate_per_tx_ms + c.propagate_per_tx_per_delay_ms * d)
            transfer *= float(self.rng.lognormal(0.0, c.propagate_sigma))
            # A block is many packets, and it is complete only when the last
            # one lands, so its hop sits in the upper half of the jitter band.
            data_hop = d * (1.0 + jr * float(u(0.0, 1.0)))
            state_hop = d * (1.0 + jr * float(u(-1.0, 1.0)))
            t_arr = (
                max(t_send, t0 + c.peer_hops * state_hop)
                + data_hop + transfer + float(u(0.0, c.gossip_proc_ms))
            )
            # A node accepts the proposal if it arrives before its own timer;
            # the block commits when a quorum accepts. Each node's prevote ends
            # its propose step: the proposer prevotes on its own proposal the
            # moment it sends it, the others on it when it arrives, and a node
            # whose timer fires first prevotes nil (node 0's rule before
            # restart D11, now kept for every node).
            accepted = t_arr <= t0 + Ts_r
            t_prevote = np.where(accepted, t_arr, t0 + Ts_r)
            if accepted[proposer]:
                t_prevote[proposer] = t_send
            if int(np.count_nonzero(accepted)) >= QUORUM:
                t_precommit = t_arr + d * (1.0 + jr * float(u(-1.0, 1.0))) + c.vote_proc_ms
                t_commit = t_precommit + d * (1.0 + jr * float(u(-1.0, 1.0))) + c.vote_proc_ms
                self._arrive(t_commit)
                ntx = min(ntx, int(self.pending))
                t_done = t_commit + c.exec_ms + c.exec_per_tx_ms * ntx
                self.pending -= ntx
                if r == 0:
                    sample["prop"] = t_prevote - t0_first
                    sample["prev"] = t_precommit - t_prevote
                    sample["prec"] = t_done - t_precommit
                else:
                    sample["prec"] = t_done - sample["_r0_precommit"]
                sample.pop("_r0_precommit", None)
                sample["lat"] = t_done - t0_first
                sample["ntx"] = float(ntx)
                sample["round"] = float(r)
                return t_done, sample
            # No quorum for the block: the third prevote (a nil one from a node
            # whose timer fired, or a block one from a node that accepted) makes
            # the +2/3 prevote set, the nil precommit majority follows one hop
            # later, and the quorum's precommit wait runs before round r + 1.
            prevotes = np.where(accepted, t_arr, t0 + Ts_r)
            t_prevote_q = float(np.sort(prevotes)[QUORUM - 1])
            t_precommit = t_prevote_q + d * (1.0 + jr * float(u(-1.0, 1.0))) + c.vote_proc_ms
            t_next = t_precommit + d * (1.0 + jr * float(u(-1.0, 1.0))) + c.vote_proc_ms + Tpc_r
            if r == 0:
                sample["prop"] = t_prevote - t0_first
                # A node whose timer fires after the quorum already has +2/3
                # prevotes when it casts its own: its prevote step is one
                # processing time, not a negative interval.
                sample["prev"] = np.maximum(t_precommit - t_prevote, c.vote_proc_ms)
                sample["_r0_precommit"] = t_precommit
            t0 = t_next
            r += 1
            if r > 60:
                raise RuntimeError("height did not commit in 60 rounds")

    def _advance_to(self, end_ms: float, windows: Optional[Sequence[_Window]]) -> None:
        while self.t_ms < end_ms:
            t_done, s = self._one_height()
            if windows is not None:
                t_s = t_done / 1000.0
                for w in windows:
                    if w.start_s <= t_s < w.end_s:
                        w.heights += 1
                        w.txs += int(s["ntx"])
                        w.rounds_gt0 += int(s["round"] > 0)
                        w.round_changes += int(s["round"])
                        w.lat.append(s["lat"])
                        if s["ntx"] > 0:
                            # accumulateHeight counts a batch only when txCount > 0
                            # (Kevin, 2026-10-19, restart D11; every height until then).
                            w.batch.append(s["ntx"])
                        w.commit_ms.append(t_done)
                        for i in range(N_NODES):
                            if "prop" in s:
                                w.prop[i].append(float(s["prop"][i]))
                            if "prev" in s:
                                w.prev[i].append(float(s["prev"][i]))
                        if "prec" in s:
                            w.prec.append(s["prec"])
            self.t_ms = t_done
            self.height += 1

    def run_episode(
        self, choose: Callable[[Dict[str, float]], float]
    ) -> Tuple[Dict[str, float], float, Dict[str, float], int]:
        """Run one episode; ``choose(features) -> timeout_ms`` is called at the
        feature deadline. Returns (features, timeout, reward report, phase index
        when the timeout applies)."""
        s = self.episode_start_s
        wa = _Window(s, s + FEATURE_S)
        self._advance_to((s + FEATURE_S) * 1000.0, [wa])
        features = wa.report(self.node_timeouts_ms())
        # The phase in force when the timeout applies, as main.py's dataset
        # phase: the onset episode carries the phase its reward window measures.
        phase_idx, _ = self.phase_at(s + FEATURE_S + REPLY_S)
        new_timeout = float(choose(features))
        # Applied at the end of the reply wait; heights already running keep
        # the old value, as ApplyPendingTimeout does.
        self._advance_to((s + FEATURE_S + REPLY_S) * 1000.0, None)
        self.timeout_ms = new_timeout
        wb = _Window(s + FEATURE_S + REPLY_S + WARMUP_S, s + EPISODE_S)
        self._advance_to((s + EPISODE_S) * 1000.0, [wb])
        self.episode_start_s = s + EPISODE_S
        return features, new_timeout, wb.report(self.node_timeouts_ms()), phase_idx



# --- Randomized environments -------------------------------------------------


def sample_constants(rng: np.random.Generator) -> SimConstants:
    # Every constant 0.5-2x around its SimConstants value: the blind fit's FREE
    # fields (blindall/tendermint/fit/sim_fit.py) as before, the unfitted
    # create_ms, exec_ms and peer_hops in place of their wide ranges (until
    # restart D11: 2-40, 2-40 and 0-2 hops); base delay, jitter and load from
    # the shared recipe (Kevin, 2026-10-19, restart D11; module docstring).
    from reward_utils import reward_normalizer_tps

    h = SimConstants()
    return SimConstants(
        base_delay_ms=draw_base_delay_ms(rng),
        vote_proc_ms=around(rng, h.vote_proc_ms),
        create_ms=around(rng, h.create_ms),
        create_per_tx_ms=around(rng, h.create_per_tx_ms),
        exec_ms=around(rng, h.exec_ms),
        exec_per_tx_ms=around(rng, h.exec_per_tx_ms),
        peer_hops=around(rng, h.peer_hops),
        jitter_ratio=JITTER_RATIO,
        gossip_proc_ms=around(rng, h.gossip_proc_ms),
        propagate_per_tx_ms=around(rng, h.propagate_per_tx_ms),
        propagate_per_tx_per_delay_ms=around(rng, h.propagate_per_tx_per_delay_ms),
        propagate_sigma=around(rng, h.propagate_sigma),
        load_tps=draw_load_tps(rng, reward_normalizer_tps("tendermint")),
        load_delay_sensitivity=around(rng, h.load_delay_sensitivity),
    )


NO_PROPOSAL_FAULTS = False  # train --no-proposal-faults: a prior that never saw a delayed or silent proposer

# The silent proposer (module docstring): the spec's hold, longer than any
# propose timeout in the action range, drawn 0.5-2x; the share of delayed
# proposers that are silent is the old 100-30,000 ms hold's mass above 3,000 ms,
# ln(10) / ln(300) (Kevin, 2026-10-19, restart D11).
SPEC_PROPOSAL_DELAY_MS = 10000.0
SILENT_PROPOSER_SHARE = 0.4


def _proposal_delay_ms(rng: np.random.Generator) -> float:
    if rng.random() < SILENT_PROPOSER_SHARE:
        return around(rng, SPEC_PROPOSAL_DELAY_MS)
    return draw_hold_ms(rng)


def sample_phase(rng: np.random.Generator, start_s: float) -> Phase:
    kind = rng.random()
    extra, prop = 0.0, 0.0
    if kind < 0.2:
        pass
    elif kind < 0.55:
        prop = _proposal_delay_ms(rng)
    elif kind < 0.9:
        extra = draw_link_extra_ms(rng)
    else:
        prop = _proposal_delay_ms(rng)
        extra = draw_link_extra_ms(rng)
    if NO_PROPOSAL_FAULTS:
        prop = 0.0
    return Phase(start_s=start_s, extra_delay_ms=extra, proposal_delay_ms=prop)


def sample_schedule(rng: np.random.Generator, total_s: float) -> List[Phase]:
    return [Phase(start_s=0.0)] + [sample_phase(rng, t) for t in fault_start_times(rng, total_s)]


SPEC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "tendermint", "failure_spec.xml"
)


def _xml_float(node: ET.Element, path: str, where: str) -> float:
    el = node.find(path)
    if el is None or el.text is None or not el.text.strip():
        raise ValueError(f"{where}: missing <{path}>")
    return float(el.text.strip())


def spec_schedule(path: str = SPEC_PATH) -> List[Phase]:
    """config/tendermint/failure_spec.xml as simulator phases, read from the
    file (Kevin, 2026-10-19, restart D11; literals until then): the atTime 0
    phase at 0, every later one at warm-up + atTime. Raises on anything the
    simulator cannot represent or that the spec runs' SimConstants defaults
    contradict: another warm-up, base delay or jitter, a burst (not
    continuous) link delay, per-source or per-edge delays, a protocol fault
    other than one global <proposalDelay>, an unknown phase element, or report
    injections. Used only to evaluate policies in simulation, never to train."""
    root = ET.parse(path).getroot()
    host = SimConstants()
    expected = (
        ("warmUpTime", WARMUP_PHASE_S),
        ("globalNetworkDelay/delayMs", host.base_delay_ms),
        ("globalNetworkDelay/jitterRatio", host.jitter_ratio),
    )
    for name, want in expected:
        got = _xml_float(root, name, path)
        if got != want:
            raise ValueError(f"{path}: <{name}> is {got:g}, the simulator's spec runs use {want:g}")
    phases: List[Phase] = []
    for k, el in enumerate(root.findall("./phases/phase")):
        where = f"{path} phase {k}"
        unknown = sorted({child.tag for child in el} - {"atTime", "network", "tendermint", "agent"})
        if unknown:
            raise ValueError(f"{where}: the simulator does not model {unknown}")
        at = _xml_float(el, "atTime", where)
        if (k == 0) != (at == 0.0):
            raise ValueError(f"{where}: only the first phase starts at atTime 0, got {at:g}")
        start_s = 0.0 if at == 0.0 else WARMUP_PHASE_S + at
        if phases and start_s <= phases[-1].start_s:
            raise ValueError(f"{where}: atTime {at:g} does not follow the previous phase")
        net = el.find("network")
        if net is None:
            raise ValueError(f"{where}: no <network>")
        if net.find("./sources/source") is not None or net.find("./edges/edge") is not None:
            raise ValueError(f"{where}: per-source and per-edge delays are not modeled")
        extra = _xml_float(net, "global/delayMs", where)
        duration = _xml_float(net, "global/burstDurationMs", where)
        interval = _xml_float(net, "global/burstIntervalMs", where)
        if extra > 0.0 and not (interval > 0.0 and duration >= interval):
            raise ValueError(f"{where}: a burst link delay ({duration:g}/{interval:g} ms) is not modeled, "
                             "only a continuous one")
        prop = 0.0
        tm = el.find("tendermint")
        if tm is not None:
            tags = [child.tag for child in tm]
            if tags not in ([], ["proposalDelay"]):
                raise ValueError(f"{where}: only one global <proposalDelay> is modeled, got {tags}")
            if tags:
                prop = _xml_float(tm, "proposalDelay/delayMs", where)
        if el.find("./agent//id") is not None:
            raise ValueError(f"{where}: report injections are not modeled")
        phases.append(Phase(start_s, extra_delay_ms=extra, proposal_delay_ms=prop))
    if not phases:
        raise ValueError(f"{path}: no phases")
    return phases



# --- Agent loop, data generation, prior training, evaluation -----------------


def feature_vector(report: Dict[str, float]) -> np.ndarray:
    from models import TENDERMINT_REPORT_FEATURE_FIELD_NAMES

    return np.asarray(
        [report[name] for name in TENDERMINT_REPORT_FEATURE_FIELD_NAMES], dtype=float
    )


def episode_reward(report: Dict[str, float], timeout_ms: float) -> float:
    """Reward of one reward window under the process-wide reward design, the
    same calculator main.reward_engineering uses."""
    from models import tendermint_timeouts_from_propose
    from reward_utils import calculate_tendermint_reward

    p, v, c = tendermint_timeouts_from_propose(int(round(timeout_ms)))
    fields = {k: report[k] for k in (
        "avg_consensus_latency_ms", "p50_consensus_latency_ms",
        "p90_consensus_latency_ms", "throughput_tps", "timeout_violation_rate",
        "avg_batch_size", "p50_batch_size", "p90_batch_size",
        "propose_latency_ms", "prevote_latency_ms", "precommit_latency_ms",
    )}
    return calculate_tendermint_reward(
        total_transactions=int(report["total_transactions"]),
        total_consensus_instances=int(report["total_consensus_instances"]),
        leader_change_count=int(report["leader_change_count"]),
        regency_change_count=int(report["regency_change_count"]),
        propose_timeout_ms=p,
        prevote_timeout_ms=v,
        precommit_timeout_ms=c,
        **fields,
    )


WARMUP_PHASE_S = 10.0  # failure_spec warmUpTime: decisions before it are phase -1
# Spec runs start so that episode e applies at warm-up + 20 (e - 1), on the phase onsets, as on every engine.
SPEC_START_OFFSET_S = WARMUP_PHASE_S - FEATURE_S - REPLY_S


def run_agent(
    sim: TendermintSim,
    episodes: int,
    predict: Callable[[np.ndarray], float],
    update: Optional[Callable[[np.ndarray, float, float], None]],
) -> List[Tuple[int, float, Dict[str, float]]]:
    """Drive ``sim`` like LearningAgentServicer: predict at the feature deadline,
    reward from the reward window, no update for warm-up episodes."""
    rows = []
    for _ in range(episodes):
        decision_s = sim.episode_start_s + FEATURE_S
        holder: Dict[str, np.ndarray] = {}

        def choose(report: Dict[str, float]) -> float:
            holder["x"] = feature_vector(report)
            return predict(holder["x"])

        _, timeout_ms, reward_report, phase_idx = sim.run_episode(choose)
        phase = phase_idx if decision_s >= WARMUP_PHASE_S else -1
        if update is not None and phase != -1:
            update(holder["x"], timeout_ms, episode_reward(reward_report, timeout_ms))
        rows.append((phase, timeout_ms, reward_report))
    return rows


def training_sim(
    rng: np.random.Generator, seed: int, peer_spread: float = 0.0
) -> Tuple[TendermintSim, int]:
    """One randomized training run: (simulator, episodes)."""
    import agent_pb2

    episodes = int(rng.integers(40, 120))
    # --peer-spread s > 0: the peers' timeouts are node 0's times fixed per-run
    # lognormal factors with sigma drawn from [0, s], so training covers the
    # lockstep case and the dispersed one the four independent learners make.
    sigma = float(rng.uniform(0.0, peer_spread)) if peer_spread > 0.0 else 0.0
    peer_factors = np.exp(rng.normal(0.0, sigma, N_NODES - 1)) if sigma > 0.0 else None
    sim = TendermintSim(
        sample_constants(rng),
        sample_schedule(rng, episodes * EPISODE_S),
        seed=seed,
        start_offset_s=float(rng.uniform(0.0, 20.0)),
        initial_timeout_ms=draw_initial_timeout_ms(rng, agent_pb2.PROTOCOL_TENDERMINT),
        peer_factors=peer_factors,
    )
    return sim, episodes


def _generate_one(args: Tuple[int, str, int, float]) -> Tuple[np.ndarray, np.ndarray]:
    from reward_utils import set_reward_design

    seed, design, hold, peer_spread = args
    set_reward_design(design)
    from continuous_bandit import (
        CONTINUOUS_TENDERMINT_BOUNDS_MS,
        ContinuousTimeoutActionCodec,
        tendermint_context,
    )
    from models import TENDERMINT_REPORT_FEATURE_FIELD_NAMES

    import agent_pb2

    rng = np.random.default_rng(seed)
    codec = ContinuousTimeoutActionCodec(
        agent_pb2.PROTOCOL_TENDERMINT, bounds_ms=CONTINUOUS_TENDERMINT_BOUNDS_MS
    )
    sim, episodes = training_sim(rng, seed, peer_spread)
    prev = codec.normalize_ms(int(round(sim.timeout_ms)))
    xs, ys = [], []
    # --hold-episodes h > 1: an action is drawn at the start of a hold and kept for
    # h episodes; the sample is (context at the hold's start, action, mean reward
    # over the hold), the held-action return that SimReturnGPModel's prior predicts.
    # h = 1 is SimPriorGPModel's data: a fresh draw every episode with its own reward.
    held: Dict[str, float] = {}
    rewards: List[float] = []
    for _ in range(episodes):

        def choose(report):
            x = feature_vector(report)
            ctx = tendermint_context(TENDERMINT_REPORT_FEATURE_FIELD_NAMES, x)
            if not held:
                if rng.random() < 0.6:
                    a = float(rng.random())
                else:
                    a = float(np.clip(prev + rng.normal(0.0, 0.08), 0.0, 1.0))
                ms = codec._denormalize_value(a, 0)
                held["ctx"] = ctx
                held["a"] = codec.normalize_ms(ms)
                held["ms"] = float(ms)
            return held["ms"]

        _, timeout_ms, reward_report, _ = sim.run_episode(choose)
        rewards.append(episode_reward(reward_report, timeout_ms))
        if len(rewards) == hold:
            xs.append(np.append(held["ctx"], held["a"]))
            ys.append(float(np.mean(rewards)))
            prev = held["a"]
            held.clear()
            rewards = []
    return np.asarray(xs), np.asarray(ys)


def generate(
    n_runs: int, seed: int, workers: int, design: str, hold: int = 1, peer_spread: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    from multiprocessing import Pool

    if int(hold) < 1:
        raise ValueError(f"hold must be >= 1 episode, got {hold}")
    if not math.isfinite(peer_spread) or peer_spread < 0.0:
        raise ValueError(f"peer_spread must be >= 0, got {peer_spread}")
    seeds = [(seed + i, design, int(hold), float(peer_spread)) for i in range(n_runs)]
    with Pool(workers) as pool:
        parts = pool.map(_generate_one, seeds, chunksize=8)
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


# The floor the pretraining recipe fits the GP prior with (train --target-floor; Kevin,
# 2026-10-19, restart D11; no floor by default until then).
TARGET_FLOOR = -2.0


from continuous_bandit import DEFAULT_TENDERMINT_PRIOR_PATH as DEFAULT_PRIOR_PATH  # noqa: E402
from continuous_bandit import opening_action_ms, train_gp_prior  # noqa: E402


def make_policy(kind: str, prior_path: str = DEFAULT_PRIOR_PATH, **kwargs):
    """(predict, update) closures around the real model classes, built as main.serve does."""
    import agent_pb2
    from continuous_bandit import (
        CONTINUOUS_TENDERMINT_BOUNDS_MS,
        ContinuousTimeoutActionCodec,
        SimPriorGPModel,
    )
    from latency_floor import LatencyFloor, LatencyFloorPolicy, latency_feature_indices
    from models import (
        DiscountedLinUCB,
        ProtocolTimeoutActionCodec,
        TENDERMINT_REPORT_FEATURE_FIELD_NAMES as names,
        get_timeout_values_ms,
    )

    protocol = agent_pb2.PROTOCOL_TENDERMINT
    if kind == "heuristic":
        from model_builder import build_model

        built = build_model(protocol, "continuous_heuristic", prior_windows=SIM_WINDOWS)
        model, codec = built.model, built.action_codec
    elif kind == "discrete_linucb":
        codec = ProtocolTimeoutActionCodec(protocol)
        latency_index, instances_index = latency_feature_indices(names)
        model = LatencyFloorPolicy(
            model=DiscountedLinUCB(action_dim=3, n_features=len(names), seed=42),
            timeouts_ms=get_timeout_values_ms(protocol),
            latency_index=latency_index,
            n_features=len(names),
            instances_index=instances_index,
            latency_floor=LatencyFloor(multiplier=1.0, quantile=0.5),
        )
    elif kind == "candidate":
        codec = ContinuousTimeoutActionCodec(protocol, bounds_ms=CONTINUOUS_TENDERMINT_BOUNDS_MS)
        from continuous_bandit import NumpyMLP, SimReturnGPModel
        from reward_utils import get_reward_design

        check_prior_windows(prior_path, SIM_WINDOWS)
        # A held-action return prior (train --hold-episodes > 1) belongs to
        # SimReturnGPModel; a one-episode prior to SimPriorGPModel (H18).
        cls = SimReturnGPModel if NumpyMLP(prior_path).hold_episodes > 1 else SimPriorGPModel
        model = cls(
            action_codec=codec,
            field_names=names,
            prior_path=prior_path,
            reward_design=get_reward_design(),
            **kwargs,
        )
    else:
        raise ValueError(kind)

    def predict(x: np.ndarray) -> float:
        timeout = codec.decode(model.predict_action(x))
        return float(timeout.tendermint.propose_timeout_milliseconds)

    def update(x: np.ndarray, timeout_ms: float, reward: float) -> None:
        from models import tendermint_timeouts_from_propose

        p, v, c = tendermint_timeouts_from_propose(int(round(timeout_ms)))
        used = agent_pb2.Timeout(
            tendermint=agent_pb2.TendermintTimeout(
                propose_timeout_milliseconds=p,
                prevote_timeout_milliseconds=v,
                precommit_timeout_milliseconds=c,
            )
        )
        model.update_model(x, codec.encode(used.tendermint), reward)

    return predict, update


def evaluate_spec(kind: str, constants: SimConstants, seed: int, **kwargs) -> np.ndarray:
    """Per-phase (tps, mean latency, median timeout) of one policy on the spec."""
    predict, update = make_policy(kind, **kwargs)
    phases = spec_schedule()
    sim = TendermintSim(constants, phases, seed=seed, start_offset_s=SPEC_START_OFFSET_S)
    rows = run_agent(sim, 90, predict, update)
    out = np.zeros((len(phases), 3))
    for ph in range(len(phases)):
        sel = [r for r in rows if r[0] == ph]
        out[ph] = (
            np.mean([r[2]["throughput_tps"] for r in sel]),
            np.mean([r[2]["avg_consensus_latency_ms"] for r in sel]),
            np.median([r[1] for r in sel]),
        )
    return out


def pretrain_adapter(peer_spread: float = 0.0):
    """This simulator as pretrain.py drives it: training runs as train draws
    them (peer factors only for ``peer_spread`` > 0, as train --peer-spread),
    spec runs as evaluate_spec builds them (host constants, lockstep peers, 90
    episodes); the spec phase is run_agent's, and the oracle plays
    learning_agent/config/schedule_tendermint.csv as ScheduledTimeoutModel does."""
    import agent_pb2
    from models import tendermint_timeouts_from_propose
    from pretrain import SimAdapter, timeout_message

    if not math.isfinite(peer_spread) or peer_spread < 0.0:
        raise ValueError(f"peer_spread must be >= 0, got {peer_spread}")
    spec_phases = spec_schedule()

    def to_timeout(timeout_ms: float) -> agent_pb2.Timeout:
        p, v, c = tendermint_timeouts_from_propose(int(round(timeout_ms)))
        return agent_pb2.Timeout(tendermint=agent_pb2.TendermintTimeout(
            propose_timeout_milliseconds=p,
            prevote_timeout_milliseconds=v,
            precommit_timeout_milliseconds=c,
        ))

    def from_timeout(timeout: agent_pb2.Timeout) -> float:
        return float(timeout_message(timeout, "tendermint").propose_timeout_milliseconds)

    rows = schedule_rows("tendermint")

    def oracle(sim) -> Callable[[np.ndarray], float]:
        # The apply time: the decision at the feature deadline reaches the
        # replicas a reply window later (Kevin, 2026-10-19, restart: the sim gate
        # compares against the oracle on all six).
        def predict(x: np.ndarray) -> float:
            return scheduled_action(rows, sim.episode_start_s + FEATURE_S + REPLY_S, WARMUP_PHASE_S)[0]

        return predict

    return SimAdapter(
        protocol=agent_pb2.PROTOCOL_TENDERMINT,
        simulator="tendermint_sim",
        training_run=lambda seed: training_sim(np.random.default_rng(seed), seed, peer_spread),
        spec_run=lambda seed: (
            # The engine opens at the middle rung (config.toml timeout_propose, D10b).
            TendermintSim(SimConstants(), spec_phases, seed=seed, start_offset_s=SPEC_START_OFFSET_S,
                          initial_timeout_ms=float(opening_action_ms(agent_pb2.PROTOCOL_TENDERMINT)[0])), 90),
        run_agent=run_agent,
        # A training schedule's phase 0 is a real phase, so run_agent already
        # updates every decision from 10 s.
        run_training=run_agent,
        to_timeout=to_timeout,
        from_timeout=from_timeout,
        row_phase=lambda row: row[0],
        row_reward=lambda row: episode_reward(row[2], row[1]),
        spec_phase_count=len(spec_phases),
        oracle=oracle,
    )


def main() -> None:
    import agent_pb2

    from reward_utils import DEFAULT_REWARD_DESIGN, reward_normalizer_tps, set_reward_design

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="generate randomized episodes and fit the prior")
    t.add_argument("--code-hash", required=True,
                   help="code_hash of the tree training the prior (scripts/cloudlab/code_state.py), recorded in the prior and compared with FREEZE by the harness")
    t.add_argument("--runs", type=int, default=4000)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    t.add_argument("--out", default=DEFAULT_PRIOR_PATH)
    t.add_argument("--reward-design", default=DEFAULT_REWARD_DESIGN)
    t.add_argument("--no-proposal-faults", action="store_true",
                   help="train on network-delay phases only; the prior never sees a delayed or silent proposer")
    # Tendermint rotates its proposer every height, so there is no leader pinned
    # for 60 s to hold an action across (PBFT, SBFT and SmartBFT hold 3).
    t.add_argument("--hold-episodes", type=int, default=1,
                   help="hold each drawn action this many episodes and fit the mean reward over the hold; "
                        "1 fits one episode's reward")
    t.add_argument("--target-floor", type=float, default=TARGET_FLOOR,
                   help="fit rewards below this value as this value (SimReturnGPModel applies the same "
                        f"floor online); default {TARGET_FLOOR:g}, as the pretraining recipe")
    t.add_argument("--peer-spread", type=float, default=0.0,
                   help="largest sigma of the per-run lognormal factors that set the peers' timeouts "
                        "relative to node 0's; 0 keeps all nodes on one timeout (every row runs in "
                        "sharing mode, and the pretraining recipe passes 0)")
    e = sub.add_parser("eval", help="compare policies on the spec schedule in simulation")
    e.add_argument("--seeds", type=int, default=10)
    e.add_argument("--prior", default=DEFAULT_PRIOR_PATH)
    e.add_argument("--randomize", action="store_true", help="random constants per seed")
    e.add_argument("--reward-design", default=DEFAULT_REWARD_DESIGN,
                   help="reward every policy learns from (reward_mixed for the baselines until restart D11)")
    args = parser.parse_args()

    if args.cmd == "train":
        global NO_PROPOSAL_FAULTS
        NO_PROPOSAL_FAULTS = bool(args.no_proposal_faults)  # inherited by the forked workers
        x, y = generate(args.runs, args.seed, args.workers, args.reward_design, args.hold_episodes,
                        args.peer_spread)
        rmse = train_gp_prior(
            agent_pb2.PROTOCOL_TENDERMINT, x, y, args.out, seed=args.seed, design=args.reward_design,
            hold=args.hold_episodes, target_floor=args.target_floor, windows=SIM_WINDOWS,
            sim_constants=asdict(SimConstants()), code_hash=args.code_hash, extra={"peer_spread": float(args.peer_spread)})
        print(f"samples={len(x)} hold_episodes={args.hold_episodes} target_floor={args.target_floor} "
              f"peer_spread={args.peer_spread} reward_std={y.std():.4f} "
              f"floored_std={np.maximum(y, args.target_floor).std():.4f} "
              f"holdout_rmse={rmse:.4f} -> {args.out}")
        return

    rng = np.random.default_rng(12345)
    spec_phases = spec_schedule()
    set_reward_design(args.reward_design)
    for s in range(args.seeds):
        constants = sample_constants(rng) if args.randomize else SimConstants()
        if args.randomize:
            constants.load_tps = reward_normalizer_tps("tendermint")
        res = {}
        for k in ("discrete_linucb", "heuristic", "candidate"):
            res[k] = evaluate_spec(k, constants, s, **({"prior_path": args.prior} if k == "candidate" else {}))
        wins = 0
        line = []
        for ph, phase in enumerate(spec_phases):
            # The six-phase loop's rule (runs/2026-09-29-tendermint-continuous-6ph):
            # strictly lower latency in a fault phase, within 5% in a fault-free one.
            faulty = phase.extra_delay_ms > 0.0 or phase.proposal_delay_ms > 0.0
            bt = max(res["discrete_linucb"][ph, 0], res["heuristic"][ph, 0])
            bl = min(res["discrete_linucb"][ph, 1], res["heuristic"][ph, 1])
            ct, cl = res["candidate"][ph, 0], res["candidate"][ph, 1]
            ok = ct >= 0.95 * bt and (cl < bl if faulty else cl <= 1.05 * bl)
            wins += ok
            line.append(
                f"p{ph} {ct:5.0f}/{cl:5.0f}@{res['candidate'][ph,2]:4.0f} "
                f"[{res['heuristic'][ph,0]:5.0f}/{res['heuristic'][ph,1]:5.0f}@{res['heuristic'][ph,2]:4.0f} "
                f"{res['discrete_linucb'][ph,0]:5.0f}/{res['discrete_linucb'][ph,1]:5.0f}@{res['discrete_linucb'][ph,2]:4.0f}]{'+' if ok else '-'}"
            )
        print(f"seed {s} phases_ok={wins}/{len(spec_phases)} hops={constants.peer_hops:.2f} | "
              + " | ".join(line))



if __name__ == "__main__":
    main()
