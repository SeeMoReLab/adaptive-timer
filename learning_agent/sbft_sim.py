"""Event-driven CppBedrock SBFT simulator for pretraining the two-dimensional continuous model.

Built from the engine's rules (CppBedrock EntityBatching.cpp, EntityConsensus.cpp,
EntityViewChange.cpp, EntityRecovery.cpp, PendingRequestTimer.cpp, AgentClient.cpp,
FailureSpec.cpp), with host constants taken from the 20k calibration runs on clnode384
(SimConstants). Four replicas apply the same election timeout T and fast-path wait W
(sharing mode); the model sees the field-wise median of the replicas' reports and is
rewarded on the median reward-window report.

Each replica keeps its own view, view-change state, request timer (backoff, floor),
proposal-tick phase, execution pointer and links:

- Requests arrive at ``load_tps`` into one pool capped at 32,768 unexecuted; arrivals
  beyond it are lost. A replica admits (starts timing) a request at its own next 100 ms
  proposal tick; the tick re-anchors when the replica starts a view change and when it
  installs a view (cancelDelayedProposals), so after a view change a follower ticks one
  NewView hop after its leader and its timer sees ``100 - (hop mod 100)`` ms of the
  oldest request's wait, not all of it.
- The leader of view v (v mod 4) proposes at its tick, while fewer than 64 batches are
  unexecuted at it, the oldest pending requests up to the 512 KiB byte budget
  (``batch_cap`` requests). A pinned leader holds the proposal D ms (the slot is taken);
  a view change drops held proposals.
- Each directed link is a FIFO that sends at most one congestion window per RTT: it
  starts at the TCP initial window, slow-starts while a send has to wait, up to
  ``max_window_bytes``, and stays warm (tcp_slow_start_after_idle=0). netem delays all
  of the straggler's egress, ACKs included, so links to and from it have RTT 2h + S. A
  replica's first leadership, and the leader when netem is first installed, starts
  cold, and a long or straggling link drains a backlog at most at window/RTT.
- Pre-prepare to follower j over that link, then a size-dependent parse and a processing
  time, then the prepare one hop back (a straggler's messages carry +S). The leader
  votes on its own proposal and reads W at the third vote: the fourth within W gives the
  fast path (certificate at the fourth), otherwise the slow path (certificate at third +
  W, commitAck from two followers, execute certificate; the W expiry and the execute
  certificate each wait an exponential lock time, ``slow_lock_ms``). Fewer than three
  live voters: no certificate. Execution is in sequence order on every replica and needs the batch
  bytes; the window slot frees at the leader's execution.
- Request timer: deadline = max(admission tick of the oldest unexecuted request, floor)
  + min(T << backoff, max(T, 30 s)). Expiry raises the backoff and starts a view change:
  the replica stops voting and executing consensus and sends ViewChange(v+1). Another
  replica joins at f+1 = 2 distinct senders above its view; the next leader builds the
  NewView at 2f+1 = 3 (``nv_build_ms``), followers install it one hop later plus
  ``nv_val_ms``; install floors the timer. Undecided batches with f+1 votes are
  re-proposed inside the NewView, headers only: each replica installs them one after
  another (a full pre-prepare validation each); one that never accepted the original
  pre-prepare asks every peer for the bytes as it handles the header, each holder
  answers with the full batch on its link, and it votes, acks and commits only once it
  holds them. The rest go back to pending. The view-change wait resends below 2f+1 senders and escalates at 2f+1
  (e522af3). Backoff resets when the watched request completes before an unfloored
  deadline, or nothing is left to watch; a deadline already past then fires at once.
  Lowering T floors every timer.
- A lone replica in a view change stays there until the others move up; meanwhile it
  executes only through recovery (once a second, the newest committed batch counted
  without a latency sample: p50 0, about a tenth of the throughput).
- Faults per phase: extra one-way link delay with jitter; a straggler replica with extra
  outgoing delay S. S >= 2,000 ms is a dead peer: its consensus channel is dead both
  ways (no pre-prepares reach it, its votes and acks never count, its NewView installs
  nowhere), but its ViewChanges arrive S late and it installs the others' NewViews. It
  watches the oldest request it admitted, which consensus never completes for it, so it
  times out T << backoff after each install and waits alone in its view change, and its
  standing claim turns any single live timeout into a view change (20260928_133956,
  160921, 162044). Outside a view change it recovers through checkpoints, so once
  T << backoff outlasts that lag it stops timing out (the 1500 s sharing runs). A leader
  proposal delay D is re-pinned to whoever leads at the first observation of each
  (phase, interval) key. Where the appliers first install netem (a qdisc replace) every
  link stalls for ``onset_stall_ms`` and the links in use lose their windows; later
  phases change the delay in place (tc qdisc change), which drops nothing. The harness
  now pre-applies the base netem, so no fault onset is a qdisc replace and
  ``onset_stall_ms`` defaults to 0 (no stall, no window loss).
- A straggler that times out alone while T is above the slow-path latency stays exiled
  until another replica times out, with no fast path for anyone: the 5+150 deep state
  (agent 2 at p50 0 and 0% fast at the others in the deep 15k draw 20260928_123338; the
  codex sharing run exiled it from 918 s to 1152 s).

Reports follow AgentClient::MetricsWindow on each replica (the 22 SbftReport fields):
one latency sample per executed batch, dated from its oldest request (none without a
firstSeen: lone recovery); nearest-rank percentiles; inter-commit gaps within the window
(n commits give n-1); phase latencies are means of the samples
above 0 (pre-prepare stamp to certificate, certificate to commit, commit to execution;
a fast instance's certificate-to-commit is microseconds, and counts; none for batches
executed after a later install); fast and slow counts on every replica; view changes
started (expiry, join, escalation; not resends); new views installed. The model sees
the field-wise median of the first 2f+1 reports, as the sharing agents seal an episode
at the third report: the straggler's or dead peer's report is never among them.

``train`` draws every constant per run from a range that includes the host's value and
every fault from wide ranges; it never sees config/sbft/failure_spec.xml, which only
``check`` and ``eval`` use (``spec_schedule()``).
"""

from __future__ import annotations

import argparse
import bisect
import heapq
import math
import os
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from prior_windows import PriorWindows, check_prior_windows

N = 4
F = 1
QUORUM = 2 * F + 1
INF = math.inf
PROPOSAL_INTERVAL_MS = 100.0
MAINTENANCE_MS = 250.0  # maintainConsensus cadence (Consensus.h)
RECOVERY_MIN_MS = 1000.0  # requestRecovery at most once per second
OBSERVE_MS = 200.0  # leader observation for the proposal-delay pin
WINDOW_BATCHES = 64  # run_sbft.sh MAX_INFLIGHT_BATCHES
MAX_PENDING_REQUESTS = 32768
BATCH_MAX_REQUESTS = 8192  # --batch-max-requests; the byte budget binds first (batch_cap)
MAX_ELECTION_BACKOFF_MS = 30000.0
BATCH_MAX_BYTES = 524288.0  # --batch-max-bytes
DEFAULT_ELECTION_TIMEOUT_MS = 8000.0  # config.sbft.yaml, before the first decision
DEFAULT_SLOW_PATH_TIMEOUT_MS = 20.0
FAST_PREP_MS = 0.0033  # prepare_latency_ms of an all-fast window (agent CSVs: 0.0032-0.0035)
CRASH_DELAY_MS = 2000.0  # a straggler this slow is a dead peer (gRPC keepalive 5 s)
FEATURE_S = 7.0
REPLY_S = 2.0
WARMUP_S = 4.0
REWARD_S = 7.0
EPISODE_S = FEATURE_S + REPLY_S + WARMUP_S + REWARD_S
SIM_WINDOWS = PriorWindows.from_seconds(FEATURE_S, REPLY_S, WARMUP_S, REWARD_S)
WARMUP_PHASE_S = 10.0  # failure_spec warmUpTime: decisions before it are phase -1

# Action bounds of --model-type=continuous_pretrained_gp on SBFT (design_model.md), per dimension.
T_BOUNDS_MS: Tuple[int, int] = (100, 3000)
W_BOUNDS_MS: Tuple[int, int] = (10, 500)

# Context the prior reads: log1p(max(0, v)) of these SbftReport fields, in this order.
SBFT_CONTEXT_FIELDS: Tuple[str, ...] = (
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
)

DEFAULT_PRIOR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pretrained_models", "continuous_sbft_return_prior.npz"
)


@dataclass
# Fitted defaults: blind best-by-loss candidate i=950 (loss 0.685) of fit7m_sbft_blind.jsonl, 7/2/4/7 calibration, replica-median targets (blindall, 2026-10-10).
class SimConstants:
    """Defaults are the calibration host (clnode384, 20k req/s), except load_tps.
    Training draws every field from sample_constants."""

    base_ms: float = 5.0  # one-way base link (h = 5)
    # Follower pre-prepare transfer and parse: about 12 ms at 2,000 requests (leader minus
    # follower pre_prepare_latency at every link level, 20k revalidation reports).
    per_tx_ms: float = 0.002679
    # Spread of that transfer across replicas and batches; follower stamps agree within
    # 1-2 ms at 2,000 requests.
    proc_spread: float = 0.123384
    # Prepare processing after the stamp (fast t4 - t0 = 2h + 14-15 ms).
    proc_base_ms: float = 1.989537
    proc_sigma: float = 0.3
    # Slow path: commitAck processing and the follower's handling of the execute
    # certificate (follower minus leader prepare_latency, measured 4-6 ms; 3 keeps the
    # base slow path near its measured p50 of 200 once the lock waits below are added:
    # with 5, p0 (250, 50) sat 30 ms under T and cascaded).
    # With the execute certificate's lock wait, the leader's prepare_latency is 2h + 6
    # (measured 2h + 4-7 at h = 5 / 55 / 155).
    ack_proc_ms: float = 1.175323
    exe_proc_ms: float = 2.331944
    # The slow path's two extra lock-bound leader events, the W expiry (a scheduled
    # task that needs the engine lock) and the execute certificate, each wait an
    # exponential time of this mean. The per-window p95 - p50 of the 20k slow cells is
    # 8.5-36 ms (median 19.5 over 13 cells; fast cells 3-5); 7 ms gives 18-21 here
    # (fast 2-5), and its means land in pre_prepare_latency (2h + W + 6-10, measured
    # 2h + W + 4-15). The wait does not grow with the batch: a mean proportional to the
    # batch (7 ms at 2,000) stretched the 5,566-request backlog batches of a p0 (250, 50)
    # election past T and cascaded it, where the measured cell has 0.3 elections per
    # window and recovers. It does grow with load: at 15k the measured slow spread is
    # 2.5-5 ms; training draws it per run, 0.5-2 x this, independently of load_tps.
    slow_lock_ms: float = 6.085705
    # Execution after commit: commit_latency 4-6 ms at 2,000 requests.
    exec_base_ms: float = 0.86319
    exec_per_tx_ms: float = 0.001778
    # NewView build at the new leader after its 2f+1-th ViewChange (30-95 ms in the
    # server logs of v20_p1_250_50_2 and v20_p3_525_150_1) and the followers' install
    # after receipt (follower install = leader install + hop - 15..+13 ms).
    nv_build_ms: float = 40.061676
    nv_val_ms: float = 4.309705
    vc_sigma: float = 0.4
    # A straggler's prepare lands S plus this much after the others (catch cost S + 0-15
    # at 20k, S + 20-30 in the 09-27 pairs log).
    strag_hold_ms: float = 4.266941
    # Link stall where the appliers first install netem (a qdisc replace drops the
    # queue): a retransmission timeout, the Linux RTO floor 200 ms plus the RTT, about
    # 300 ms. 0: the harness pre-applies the base netem, so fault onsets only change it.
    onset_stall_ms: float = 0.0
    # TCP initial window (Linux initcwnd 10 x 1448 B). A link that never carried a
    # pre-prepare, or lost its queue at the netem install, slow-starts from it; the harness sets
    # tcp_slow_start_after_idle=0 (lib_bedrock.sh:1081), so a warm link stays warm.
    init_window_bytes: float = 14480.0
    # Largest window a link reaches, so its throughput is at most this per RTT. The
    # onset and ramp drains do not bound it: the leader's Stats lines show those drains
    # limited by one commit per proposal tick, not by the link. What bounds it from below
    # is the straggler: at 5+150 with S = 100 its links have a 410 ms RTT and still keep
    # the fast path at 20k (3/3 calibration draws, the oracle), which 0.8 MB (19.5k req/s)
    # cannot; 1.2 MB reproduces the measured catch rates of p1 and p3.
    max_window_bytes: float = 1165717.324881
    load_tps: float = 15000.0  # run_sbft.sh --rate of the calibration and experiment runs
    # Requests per batch where the 512 KiB --batch-max-bytes budget binds: p95 batch size
    # tops out at 5,566-5,569 across the 20k revalidation reports.
    batch_cap: int = 7679
    initial_backlog: int = 0


@dataclass
class Phase:
    start_s: float
    extra_ms: float = 0.0
    jitter: float = 0.0  # jitterRatio: uniform +-jitter x (base + extra) per message
    straggler: int = -1
    straggler_ms: float = 0.0
    delay_ms: float = 0.0  # leader proposal delay D
    interval_s: float = 0.0  # re-pin interval; 0 = the whole phase

    @property
    def crashed(self) -> int:
        return self.straggler if self.straggler >= 0 and self.straggler_ms >= CRASH_DELAY_MS else -1

    def net_key(self) -> Tuple[float, float, int, float]:
        return (self.extra_ms, self.jitter, self.straggler, self.straggler_ms)


class _Window:
    __slots__ = ("start", "end", "txs", "lat", "batch", "gap", "last_commit", "pre", "prep", "com",
                 "vcs", "noprog", "nvs", "fast", "slow", "nth", "nth_unseen")

    def __init__(self, start_s: float, end_s: float):
        self.start = start_s * 1000.0
        self.end = end_s * 1000.0
        self.txs = 0.0
        self.lat: List[float] = []
        self.batch: List[float] = []
        self.gap: List[float] = []
        self.last_commit: Optional[float] = None  # gaps reset per window: n commits, n-1 gaps
        self.pre: List[float] = []
        self.prep: List[float] = []
        self.com: List[float] = []
        self.vcs = 0
        self.noprog = 0
        self.nvs = 0
        self.fast = 0
        self.slow = 0
        self.nth: List[float] = []  # as leader: n-th minus 2f+1-th prepare, n-th seen by execution
        self.nth_unseen = 0  # as leader: sequences whose n-th prepare had not come by execution

    def report(self, timeout_ms: float, wait_ms: float) -> Dict[str, float]:
        def avg(v):
            return float(np.mean(v)) if v else 0.0

        def pct(v, q):
            # Nearest rank, sorted[ceil(p n) - 1], as AgentClient.cpp percentile.
            if not v:
                return 0.0
            ordered = sorted(v)
            return float(ordered[max(0, math.ceil(q / 100.0 * len(ordered)) - 1)])

        elapsed_s = (self.end - self.start) / 1000.0
        txs = float(round(self.txs))
        return {
            "total_transactions": txs,
            "total_consensus_instances": float(len(self.batch)),
            "avg_consensus_latency_ms": avg(self.lat),
            "p50_consensus_latency_ms": pct(self.lat, 50),
            "p95_consensus_latency_ms": pct(self.lat, 95),
            "throughput_tps": txs / elapsed_s if elapsed_s > 0 else 0.0,
            "avg_batch_size": avg(self.batch),
            "p95_batch_size": pct(self.batch, 95),
            "leader_change_count": float(self.nvs),
            "regency_change_count": float(self.nvs),
            "timeout_ms": float(int(round(timeout_ms))),
            "avg_inter_commit_gap_ms": avg(self.gap),
            "p50_inter_commit_gap_ms": pct(self.gap, 50),
            "p95_inter_commit_gap_ms": pct(self.gap, 95),
            "view_change_count": float(self.vcs),
            "no_progress_view_change_count": float(self.noprog),
            "pre_prepare_latency_ms": avg(self.pre),
            "prepare_latency_ms": avg(self.prep),
            "commit_latency_ms": avg(self.com),
            "slow_path_timeout_ms": float(int(round(wait_ms))),
            "fast_path_count": float(self.fast),
            "slow_path_count": float(self.slow),
            "nth_prepare_gap_ms": pct(self.nth, 50),
            "nth_prepare_seen_count": float(len(self.nth)),
            "nth_prepare_unseen_count": float(self.nth_unseen),
        }


class _Batch:
    __slots__ = ("seq", "view", "leader", "t0", "t_send", "size", "oldest", "mean_arr", "chunks",
                 "pp", "stamp", "send", "arr", "u_pp", "u_ret", "u_cc", "u_ack", "u_ex", "xfer", "proc",
                 "ackp", "exep",
                 "cost", "t_dec", "fast", "commit", "cr", "acc_prev", "first_exec", "replies",
                 "reproposed", "body", "bodyprev", "wlag", "xlag", "t3", "t4")

    def __init__(self):
        self.t_dec = INF
        self.t3 = INF  # the leader's 2f+1-th and n-th valid prepare (INF: none)
        self.t4 = INF
        self.fast = False
        self.commit = [INF] * N
        self.cr = [INF] * N
        self.acc_prev = [False] * N
        self.first_exec = False
        self.replies: List[float] = []
        self.reproposed = False
        self.body: List[Optional[float]] = [None] * N  # when each replica holds the batch bytes
        self.bodyprev = [False] * N  # held the bytes before its latest re-proposal


class _Replica:
    __slots__ = ("i", "view", "in_vc", "backoff", "floor", "anchors", "next_tick", "wait_armed",
                 "claims", "last_nv_sent", "installed", "stopped", "resume_t", "last_installed",
                 "next_seq", "last_exec", "commits_since_vc", "rec_req", "rec_due",
                 "crashed", "dead_watch")

    def __init__(self, i: int, anchor: float):
        self.i = i
        self.view = 0
        self.in_vc = False
        self.backoff = 0
        self.floor = -INF
        self.anchors = [anchor]
        self.next_tick = anchor + PROPOSAL_INTERVAL_MS
        self.wait_armed: Optional[float] = None
        self.claims: Dict[int, set] = {}
        self.last_nv_sent = 0
        self.installed: Dict[int, float] = {0: -INF}
        self.stopped: Dict[int, float] = {}
        self.resume_t = -INF
        self.last_installed = 0
        self.next_seq = 0
        self.last_exec = -INF
        self.commits_since_vc = 0
        self.rec_req = -INF
        self.rec_due = INF
        self.crashed = False
        # A dead peer's oldest unexecuted arrival among batches the log has dropped.
        self.dead_watch = INF


# Event kinds of the heap and of the per-iteration scan.
_EV_VC, _EV_NV, _EV_BUILD = 0, 1, 2


class SbftSim:
    """Event-driven SBFT simulation driven one episode at a time."""

    def __init__(
        self,
        constants: SimConstants,
        phases: Sequence[Phase],
        seed: int,
        start_offset_s: float = 0.0,
        initial_timeout_ms: float = DEFAULT_ELECTION_TIMEOUT_MS,
        initial_wait_ms: float = DEFAULT_SLOW_PATH_TIMEOUT_MS,
    ):
        self.c = constants
        self.phases = sorted(phases, key=lambda p: p.start_s)
        if self.phases[0].start_s != 0.0:
            raise ValueError("the first phase must start at 0 s")
        self._starts = [p.start_s * 1000.0 for p in self.phases]
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.T = float(initial_timeout_ms)
        self.W = float(initial_wait_ms)
        self._w_hist: List[Tuple[float, float]] = [(-INF, self.W)]
        rng = self.rng
        self.reps = [_Replica(i, float(rng.uniform(-PROPOSAL_INTERVAL_MS, 0.0))) for i in range(N)]
        self.heap: List[Tuple[float, int, int, int, int, int]] = []
        self._hseq = 0
        self.log: Dict[int, _Batch] = {}
        self.log_seqs: List[int] = []
        self.next_alloc = 0
        self.pending: deque = deque()  # [oldest arrival, count, mean arrival, newest arrival]
        self.pending_n = 0.0
        self.inflight_n = 0.0
        self.last_arrival = 0.0
        self.lost = 0.0
        self.pin_key: Optional[Tuple[int, int]] = None
        self.pinned = -1
        self.next_obs = float(rng.uniform(0.0, OBSERVE_MS))
        self.late_reporter = int(np.random.default_rng([seed, 1]).integers(0, N))  # own stream
        # Links stall for onset_stall_ms where the appliers first install netem: that
        # is a qdisc replace, which drops the queue (apply_network_failures.py). Later
        # phases rewrite the delay in place (tc qdisc change), which drops nothing.
        self.stalls: List[Tuple[float, float]] = []
        for prev, ph in zip(self.phases, self.phases[1:]):
            if ph.net_key() != prev.net_key() and self.c.onset_stall_ms > 0.0:
                s = ph.start_s * 1000.0
                self.stalls.append((s, s + self.c.onset_stall_ms))
                break
        self._phase_i = 0
        self.next_phase_ms = self._starts[1] if len(self._starts) > 1 else INF
        self._apply_crash(self.phases[0])
        # Per directed link: congestion window (bytes), when its send queue drains, and
        # when it last carried a pre-prepare (the netem install's loss resets the active ones).
        self.cwnd = [[self.c.init_window_bytes] * N for _ in range(N)]
        self.tokens = [[self.c.init_window_bytes] * N for _ in range(N)]
        self.tok_t = [[0.0] * N for _ in range(N)]
        self.link_used = [[-INF] * N for _ in range(N)]
        self._stall_i = 0
        # maintainConsensus phase per replica: a missing batch body is requested on its ticks.
        self.maint_phase = [float(rng.uniform(0.0, MAINTENANCE_MS)) for _ in range(N)]
        self.windows: Optional[List[List[_Window]]] = None
        self.clat: List[Tuple[float, float, float]] = []  # (time of the f+1-th reply, latency, requests)
        self.trace: Optional[List[str]] = None  # debugging: view-change events when a list
        if constants.initial_backlog > 0:
            n = float(constants.initial_backlog)
            self.pending.append([0.0, n, 0.0, 0.0])
            self.pending_n = n
        self.episode_start_s = float(start_offset_s)
        self._advance_to(start_offset_s * 1000.0)

    # --- helpers ---------------------------------------------------------------

    def phase_index(self, t_s: float) -> int:
        return bisect.bisect_right(self._starts, t_s * 1000.0) - 1

    def phase_at(self, t_s: float) -> Tuple[int, Phase]:
        i = self.phase_index(t_s)
        return i, self.phases[i]

    def _ph(self, t_ms: float) -> Phase:
        return self.phases[bisect.bisect_right(self._starts, t_ms) - 1]

    def _push(self, t: float, kind: int, a: int, b: int = 0, c: int = 0) -> None:
        self._hseq += 1
        heapq.heappush(self.heap, (t, self._hseq, kind, a, b, c))

    def _teff(self, backoff: int) -> float:
        base = max(1.0, self.T)
        return min(base * float(1 << min(backoff, 30)), max(base, MAX_ELECTION_BACKOFF_MS))

    def _hop(self, i: int, j: int, t: float, u: float) -> float:
        """Arrival time at j of a message i sends at t (u: jitter draw in [-1, 1])."""
        ph = self.phases[bisect.bisect_right(self._starts, t) - 1]
        d = (self.c.base_ms + ph.extra_ms) * (1.0 + ph.jitter * u)
        if ph.straggler == i:
            d += ph.straggler_ms + self.c.strag_hold_ms
        a = t + d
        for s, e in self.stalls:
            if t < e and a > s:
                a = e + d if t >= s else e + (a - s)
        return a

    def _bulk(self, i: int, j: int, t: float, nbytes: float, u: float) -> float:
        """Arrival at j of nbytes that i sends at t, queued behind the link's earlier sends.
        A link sends at most one congestion window per RTT: a token bucket of depth cwnd
        refilled at cwnd/RTT.
        While a send has to wait the window doubles each RTT (slow start, no loss) up to
        max_window_bytes; the netem install's loss resets the windows of the links in use."""
        while self._stall_i < len(self.stalls) and self.stalls[self._stall_i][0] <= t:
            s0 = self.stalls[self._stall_i][0]
            for a in range(N):
                for b in range(N):
                    if self.link_used[a][b] >= s0 - 1000.0:
                        self.cwnd[a][b] = self.c.init_window_bytes
                        self.tokens[a][b] = min(self.tokens[a][b], self.c.init_window_bytes)
            self._stall_i += 1
        ph = self._ph(t)
        # netem delays everything the straggler sends, its ACKs and window updates
        # included, so a link to it has the same RTT as a link from it.
        rtt = 2.0 * (self.c.base_ms + ph.extra_ms) + (ph.straggler_ms if ph.straggler in (i, j) else 0.0)
        w = self.cwnd[i][j]
        # FIFO: a send queues behind the previous one on this link; tok_t is when that
        # one finished departing, and the bucket refills only from then.
        start = max(t, self.tok_t[i][j])
        tok = min(w, self.tokens[i][j] + w * (start - self.tok_t[i][j]) / rtt)
        depart = start
        if nbytes > tok:
            need = nbytes - tok
            # Each RTT of waiting sends one (doubled) window.
            while need > 0.0:
                depart += rtt
                w = min(self.c.max_window_bytes, 2.0 * w)
                need -= w
            tok = -need  # what is left of the last window
        else:
            tok -= nbytes
        self.cwnd[i][j] = w
        self.tokens[i][j] = tok
        self.tok_t[i][j] = depart
        self.link_used[i][j] = t
        return self._hop(i, j, depart, u)

    def _client_hop(self, j: int, t: float) -> float:
        ph = self._ph(t)
        if ph.straggler == j:
            return self.c.base_ms + ph.straggler_ms + self.c.strag_hold_ms
        return self.c.base_ms

    def _w_at(self, t: float) -> float:
        i = bisect.bisect_right(self._w_hist, (t, INF)) - 1
        return self._w_hist[max(0, i)][1]

    def _lognormal(self, mean: float, sigma: float) -> float:
        return mean * float(self.rng.lognormal(0.0, sigma)) if sigma > 0.0 else mean

    def _anchor(self, k: int) -> None:
        r = self.reps[k]
        r.anchors.append(self.t)
        if len(r.anchors) > 8:
            del r.anchors[0]
        r.next_tick = self.t + PROPOSAL_INTERVAL_MS

    @staticmethod
    def _accept_time(r: _Replica, a: float) -> float:
        """Replica r's proposal tick that admitted (started timing) a request arriving at a."""
        A = r.anchors
        i = bisect.bisect_right(A, a) - 1
        if i < 0:
            i = 0
            t = A[0] + PROPOSAL_INTERVAL_MS
        else:
            t = A[i] + PROPOSAL_INTERVAL_MS * (math.floor((a - A[i]) / PROPOSAL_INTERVAL_MS) + 1.0)
        while i + 1 < len(A) and t >= A[i + 1]:
            i += 1
            t = A[i] + PROPOSAL_INTERVAL_MS
        return t

    def _in_windows(self, k: int, t: float):
        if self.windows is None:
            return ()
        return [w for w in self.windows[k] if w.start <= t < w.end]

    # --- arrivals ---------------------------------------------------------------

    def _arrive(self, now: float) -> None:
        dt = now - self.last_arrival
        if dt <= 0.0:
            return
        offered = self.c.load_tps * dt / 1000.0
        room = max(0.0, MAX_PENDING_REQUESTS - self.pending_n - self.inflight_n)
        admitted = min(offered, room)
        self.lost += offered - admitted
        if admitted > 0.0:
            self.pending.append([self.last_arrival, admitted, 0.5 * (self.last_arrival + now), now])
            self.pending_n += admitted
        self.last_arrival = now

    # --- phases, crashes, pin ---------------------------------------------------

    def _apply_crash(self, ph: Phase) -> None:
        """A dead peer (S >= 2 s, past the gRPC keepalive) still times out, sends its
        ViewChange (S late) and installs NewViews, but its consensus channel is dead: it
        executes nothing through consensus, only through recovery (server logs of the 20k
        crash phase: replica 2 alone in view 1 from 12.6 s, its claim counted by the others,
        so one more timeout anywhere moves everyone). Leaving the phase it catches up."""
        dead = ph.crashed
        for r in self.reps:
            if r.i == dead:
                if not r.crashed:
                    r.dead_watch = INF
                    if not r.in_vc:
                        r.rec_due = self.t + RECOVERY_MIN_MS + 2.0 * (self.c.base_ms + ph.extra_ms) + ph.straggler_ms
                r.crashed = True
            elif r.crashed:
                r.crashed = False
                r.dead_watch = INF
                if not r.in_vc:
                    r.rec_due = INF
                if not r.in_vc:
                    r.next_seq = max(x.next_seq for x in self.reps)
                    r.last_exec = self.t
                    r.floor = self.t

    def _phase_change(self) -> None:
        self._phase_i += 1
        ph = self.phases[self._phase_i]
        self.next_phase_ms = self._starts[self._phase_i + 1] if self._phase_i + 1 < len(self._starts) else INF
        self._apply_crash(ph)

    def _pin_check(self, t: float, leader: int) -> None:
        idx = bisect.bisect_right(self._starts, t) - 1
        ph = self.phases[idx]
        if ph.delay_ms <= 0.0:
            self.pin_key = None
            self.pinned = -1
            return
        tick = int((t - ph.start_s * 1000.0) // (ph.interval_s * 1000.0)) if ph.interval_s > 0 else 0
        key = (idx, tick)
        if key != self.pin_key:
            self.pin_key = key
            self.pinned = leader

    def _observe(self) -> None:
        # Every replica outside a view change observes who leads (EntityRecovery.cpp:319);
        # the first observation of a key pins that leader, alive or not.
        obs = [r for r in self.reps if not r.in_vc]
        if obs:
            views = [r.view for r in obs]
            v = max(set(views), key=views.count)
            self._pin_check(self.t, v % N)

    # --- proposals and the consensus instance ------------------------------------

    def _inflight_at(self, k: int) -> int:
        return len(self.log_seqs) - bisect.bisect_left(self.log_seqs, self.reps[k].next_seq)

    def _tick(self, k: int) -> None:
        r = self.reps[k]
        r.next_tick += PROPOSAL_INTERVAL_MS
        self._pin_check(self.t, k)
        if not self.pending or self._inflight_at(k) >= WINDOW_BATCHES:
            return
        self._propose(k)

    def _new_batch(self, k: int) -> _Batch:
        c = self.c
        rng = self.rng
        chunks = []
        size = 0.0
        wsum = 0.0
        cap = float(min(self.c.batch_cap, BATCH_MAX_REQUESTS))
        while self.pending and size < cap - 1e-9:
            ch = self.pending[0]
            take = min(ch[1], cap - size)
            if take >= ch[1] - 1e-9:
                self.pending.popleft()
                chunks.append(ch)
                wsum += ch[1] * ch[2]
                take = ch[1]
            else:
                cut = ch[0] + (ch[3] - ch[0]) * take / ch[1]
                part = [ch[0], take, 0.5 * (ch[0] + cut), cut]
                chunks.append(part)
                wsum += take * part[2]
                ch[0], ch[1], ch[2] = cut, ch[1] - take, 0.5 * (cut + ch[3])
            size += take
        self.pending_n -= size
        self.inflight_n += size
        b = _Batch()
        b.seq = self.next_alloc
        self.next_alloc += 1
        b.size = size
        b.oldest = chunks[0][0]
        b.mean_arr = wsum / size if size > 0 else self.t
        b.chunks = chunks
        b.t0 = self.t
        u = rng.uniform(-1.0, 1.0, (5, N))
        b.u_ret, b.u_cc, b.u_ack, b.u_ex = u[1].tolist(), u[2].tolist(), u[3].tolist(), u[4].tolist()
        b.pp = [0.0] * N
        b.stamp = [0.0] * N
        b.send = [0.0] * N
        b.arr = [INF] * N
        if c.proc_sigma > 0:
            ln = rng.lognormal(0.0, c.proc_sigma, (3, N))
        else:
            ln = np.ones((3, N))
        b.proc = (c.proc_base_ms * ln[0]).tolist()
        b.ackp = (c.ack_proc_ms * ln[1]).tolist()
        b.exep = (c.exe_proc_ms * ln[2]).tolist()
        xf = c.per_tx_ms * size * (1.0 + np.abs(rng.normal(0.0, 1.0, N)) * c.proc_spread)
        b.xfer = xf.tolist()
        b.u_pp = u[0].tolist()
        b.cost = c.exec_base_ms + c.exec_per_tx_ms * size
        m = c.slow_lock_ms
        b.wlag, b.xlag = (float(x) for x in rng.exponential(m, 2)) if m > 0.0 else (0.0, 0.0)
        return b

    def _propose(self, k: int) -> None:
        b = self._batch_from_pending(k)
        self.log[b.seq] = b
        self.log_seqs.append(b.seq)
        self._resolve(b)

    def _batch_from_pending(self, k: int) -> _Batch:
        nb = self._new_batch(k)
        nb.view = self.reps[k].view
        nb.leader = k
        ph = self._ph(self.t)
        hold = ph.delay_ms if (ph.delay_ms > 0.0 and self.pinned == k) else 0.0
        self._timeline(nb, self.t + hold, None)
        return nb

    def _timeline(self, b: _Batch, t_send: float, installs: Optional[List[float]]) -> None:
        """Vote timeline of b sent by its leader at t_send. installs: for a re-proposal
        carried by a NewView, when each replica gets to it: finishNewView installs the
        re-proposals one after another under the engine lock, each a full pre-prepare
        validation, so the i-th is handled after the first i-1 (the caller accumulates)."""
        L = b.leader
        b.t_send = t_send
        # A dead peer's consensus channel is dead both ways (gRPC declares it dead every
        # keepalive cycle and drops what it queued): it gets no pre-prepare, its votes
        # and acks never count, and a view it leads makes no progress.
        dead = self._ph(t_send).crashed
        for j in range(N):
            if j != L and dead in (j, L):
                b.pp[j] = b.stamp[j] = b.send[j] = b.arr[j] = INF
                continue
            # A re-proposal is a header: advanceConsensus votes only once the replica
            # holds the bytes (EntityConsensus.cpp:390-396; _fetch_bodies set b.body).
            gate = b.body[j] if installs is not None and b.body[j] is not None else -INF
            if j == L:
                b.pp[j] = t_send
                b.stamp[j] = t_send if installs is None else installs[j]
                b.send[j] = max(b.stamp[j], gate) + b.proc[j]
                b.arr[j] = b.send[j]
                continue
            if installs is None:
                nbytes = b.size * BATCH_MAX_BYTES / max(1.0, float(self.c.batch_cap))
                b.pp[j] = self._bulk(L, j, t_send, nbytes, b.u_pp[j])
                b.stamp[j] = b.pp[j] + b.xfer[j]
            else:
                b.pp[j] = installs[j]
                b.stamp[j] = installs[j]
            b.send[j] = max(b.stamp[j], gate) + b.proc[j]
            b.arr[j] = self._hop(j, L, b.send[j], b.u_ret[j]) if b.send[j] < INF else INF

    def _valid(self, j: int, v: int, t_in: float, t_out: float) -> bool:
        """j is in view v when the message arrives (t_in) and still there when it answers (t_out)."""
        r = self.reps[j]
        return r.installed.get(v, INF) <= t_in and t_out < r.stopped.get(v, INF)

    @staticmethod
    def _gate(j: int, b: _Batch) -> float:
        """When j may act on a re-proposal's certificates: once it holds the bytes."""
        return b.body[j] if b.reproposed and b.body[j] is not None else -INF

    def _resolve(self, b: _Batch) -> None:
        """Decide b from its vote timeline and the replicas' current stop times."""
        v, L = b.view, b.leader
        rl = self.reps[L]
        stop_l = rl.stopped.get(v, INF)
        b.t_dec = INF
        if b.t_send >= stop_l:
            return  # a held proposal dropped by a view change
        votes = []
        for j in range(N):
            if b.arr[j] < INF and (j == L or self._valid(j, v, b.pp[j], b.send[j])):
                votes.append(b.arr[j])
        if len(votes) < QUORUM:
            return
        votes.sort()
        t3 = votes[QUORUM - 1]
        t4 = votes[QUORUM] if len(votes) > QUORUM else INF
        b.t3, b.t4 = t3, t4
        w = max(1.0, self._w_at(t3))
        if t4 <= t3 + w:
            fast = True
            t_cert = t4 + 0.5
        else:
            fast = False
            t_cert = t3 + w + b.wlag
        if t_cert >= stop_l:
            return
        cr = [INF] * N
        cr[L] = t_cert
        for j in range(N):
            if j != L:
                cr[j] = self._hop(L, j, t_cert, b.u_cc[j])
        if fast:
            t_dec = t_cert
        else:
            acks = []
            for j in range(N):
                if j == L or cr[j] == INF:
                    continue
                ts = max(cr[j], self._gate(j, b)) + b.ackp[j]
                if self._ph(cr[j]).crashed != j and self._valid(j, v, cr[j], ts):
                    acks.append(self._hop(j, L, ts, b.u_ack[j]))
            acks.sort()
            if len(acks) < QUORUM - 1 or acks[QUORUM - 2] == INF:
                return
            t_dec = acks[QUORUM - 2] + 0.5 + b.xlag
            if t_dec >= stop_l:
                return
        b.fast = fast
        b.t_dec = t_dec
        b.cr = cr
        for j in range(N):
            if self._ph(t_dec).crashed == j:
                b.commit[j] = INF
            elif j == L:
                b.commit[j] = t_dec
            elif fast:
                b.commit[j] = max(cr[j], self._gate(j, b)) + 0.5 if cr[j] < INF else INF
            else:
                b.commit[j] = max(self._hop(L, j, t_dec, b.u_ex[j]), self._gate(j, b)) + b.exep[j]

    def _stop(self, k: int, v: int) -> None:
        """Replica k leaves view v now: its later votes and acks do not count."""
        r = self.reps[k]
        if r.stopped.get(v, INF) <= self.t:
            return
        r.stopped[v] = self.t
        lo = bisect.bisect_left(self.log_seqs, min(x.next_seq for x in self.reps))
        for seq in self.log_seqs[lo:]:
            b = self.log[seq]
            if b.view == v and b.t_dec > self.t:
                self._resolve(b)

    # --- execution ---------------------------------------------------------------

    def _head(self, k: int) -> Optional[_Batch]:
        i = bisect.bisect_left(self.log_seqs, self.reps[k].next_seq)
        return self.log[self.log_seqs[i]] if i < len(self.log_seqs) else None

    def _exec_cand(self, k: int) -> Tuple[float, Optional[_Batch], int]:
        """Next execution at k: (time, batch, mode) with mode 1 = in its view, 2 = after
        a later install (no phase samples). A dead peer executes only through recovery."""
        r = self.reps[k]
        if r.crashed:
            return INF, None, 0
        b = self._head(k)
        if b is None or b.t_dec == INF:
            return INF, None, 0
        c = b.commit[k]
        stop = r.stopped.get(b.view, INF)
        if c < stop and r.installed.get(b.view, INF) <= c:
            return max(c, r.last_exec, self._body_time(k, b)) + b.cost, b, 1
        if not r.in_vc and r.resume_t >= min(stop, c) and r.resume_t < INF:
            return max(r.resume_t, r.last_exec, b.t_dec, self._body_time(k, b)) + b.cost, b, 2
        return INF, None, 0

    def _has_body(self, k: int, b: _Batch) -> bool:
        if b.reproposed:
            return b.bodyprev[k]
        return k == b.leader or self._valid(k, b.view, b.pp[k], b.pp[k])

    def _body_time(self, k: int, b: _Batch) -> float:
        """When k holds b's bytes. A re-proposal's are set when its NewView is handled
        (_fetch_bodies). A decided batch whose pre-prepare k never accepted (it was in
        its view change) is fetched at k's next maintenance tick (EntityRecovery.cpp:360),
        the first BatchResponse winning."""
        t = b.body[k]
        if t is not None:
            return t
        if self._has_body(k, b):
            t = b.t0 if k == b.leader else b.pp[k]
        else:
            # The response travels a link that carried bulk before (rotating leaders and
            # relays warm every link), so it costs a transfer, not a slow start.
            m = self.maint_phase[k]
            t_req = m + MAINTENANCE_MS * math.ceil((max(self.t, self.reps[k].resume_t) - m) / MAINTENANCE_MS)
            t = INF
            for p in range(N):
                if p != k and self._has_body(p, b):
                    ta = self._hop(k, p, t_req, float(self.rng.uniform(-1.0, 1.0)))
                    t = min(t, self._hop(p, k, ta, float(self.rng.uniform(-1.0, 1.0))) + self.c.per_tx_ms * b.size)
        b.body[k] = t
        return t

    def _accepted(self, k: int, b: _Batch) -> bool:
        return b.acc_prev[k] or (k == b.leader) or self._valid(k, b.view, b.pp[k], b.pp[k])

    def _watched(self, k: int) -> float:
        """Arrival of the oldest request replica k admitted and has not executed. A dead
        peer executes nothing through consensus, so that is the oldest batch the live
        replicas executed past it (dead_watch, once the log has dropped it): it times
        out T << backoff after each install and waits in its view change, as replica 2
        did in the 20k crash phase (20260928_133956: alone in view 1 from 12.6 s)."""
        b = self._head(k)
        a1 = b.oldest if b is not None else INF
        a2 = self.pending[0][0] if self.pending else INF
        a = a1 if a1 < a2 else a2
        r = self.reps[k]
        return min(a, r.dead_watch) if r.crashed else a

    def _deadline(self, k: int) -> Tuple[float, float, bool]:
        r = self.reps[k]
        a = self._watched(k)
        if a == INF:
            return INF, INF, False
        acc = self._accept_time(r, a)
        floored = r.floor > acc
        return max(acc, r.floor) + self._teff(r.backoff), acc, floored

    def _execute(self, k: int, b: _Batch, mode: int, t: float, sample: bool = True,
                 count: bool = True) -> None:
        r = self.reps[k]
        if mode != 0 and not r.in_vc and r.backoff:
            dl, _, floored = self._deadline(k)
            watched = self._watched(k)
        else:
            dl, floored, watched = INF, True, INF
        if count:
            lat_ok = sample and self._accepted(k, b) and t > b.oldest
            for w in self._in_windows(k, t):
                w.txs += b.size
                w.batch.append(b.size)
                if lat_ok:
                    w.lat.append(t - b.oldest)
                if w.last_commit is not None:
                    w.gap.append(t - w.last_commit)
                w.last_commit = t
                if mode == 1:
                    pre = b.cr[k] - b.stamp[k]
                    if pre > 0.0:
                        w.pre.append(pre)
                    # A fast certificate is prepared and committed in the same call,
                    # microseconds apart (EntityConsensus.cpp:436-438, 493), and the
                    # engine keeps that sample too (Entity.cpp:750-752).
                    prep = FAST_PREP_MS if b.fast else b.commit[k] - b.cr[k]
                    if prep > 0.0:
                        w.prep.append(prep)
                    w.com.append(t - b.commit[k])
                    # The leader's n-th prepare gap, censored at its execution
                    # (Entity.cpp markOperationProcessed, nthPrepareTs_).
                    if k == b.leader and b.t3 < INF:
                        if b.t4 <= t:
                            w.nth.append(b.t4 - b.t3)
                        else:
                            w.nth_unseen += 1
                if b.fast:
                    w.fast += 1
                else:
                    w.slow += 1
            r.commits_since_vc += 1
        r.next_seq = b.seq + 1
        r.last_exec = t
        if r.crashed:
            r.dead_watch = INF  # recovery jumps past everything the log dropped
        if not b.first_exec:
            b.first_exec = True
            self.inflight_n -= b.size
        if count:
            ch = self._client_hop(k, t)
            if ch < INF:
                b.replies.append(t + ch)
        # Backoff restores when the watched request completed in time on an unfloored
        # deadline, or nothing is left to watch (EntityRequests.cpp completeSequence).
        if r.backoff and mode != 0 and not r.in_vc and b.oldest <= watched:
            after = self._watched(k)
            if after == INF or (not floored and t < dl):
                r.backoff = 0
                if self.trace is not None:
                    dl2, acc2, fl2 = self._deadline(k)
                    hb = self._head(k)
                    self._log(f"r{k} backoff reset at exec seq {b.seq} (view {b.view}, oldest {b.oldest / 1000.0:.3f}); "
                              f"next deadline {dl2 / 1000.0:.3f} acc {acc2 / 1000.0:.3f} head seq "
                              f"{hb.seq if hb else None} oldest {hb.oldest / 1000.0 if hb else 0:.3f} "
                              f"dec {hb.t_dec / 1000.0 if hb else 0:.3f} pend0 "
                              f"{self.pending[0][0] / 1000.0 if self.pending else 0:.3f}")
        self._prune()

    def _prune(self) -> None:
        live = [r.next_seq for r in self.reps if not r.crashed]
        lo = min(live) if live else self.next_alloc
        while self.log_seqs and self.log_seqs[0] < lo:
            b = self.log.pop(self.log_seqs.pop(0))
            for r in self.reps:
                if r.crashed and r.next_seq <= b.seq and b.oldest < r.dead_watch:
                    r.dead_watch = b.oldest
            if len(b.replies) >= F + 1:
                rep = sorted(b.replies)[F]
                self.clat.append((rep, rep - b.mean_arr + self.c.base_ms, b.size))

    def _recovery(self, k: int) -> None:
        """A replica in a view change asks for recovery: committed batches it accepted are
        executed (with their latency); of those it never saw, all but the newest are
        skipped by the checkpoint and the newest is counted without a sample."""
        r = self.reps[k]
        r.rec_req = self.t - self._rtt(k)
        t = self.t
        if r.crashed and self.log_seqs and r.next_seq < self.log_seqs[0]:
            # A dead peer is not held in the log: what the live replicas all executed is
            # below the stable checkpoint, which the recovery installs without samples.
            r.next_seq = self.log_seqs[0]
            r.dead_watch = INF
        while True:
            b = self._head(k)
            if b is None or b.t_dec > self.t:
                break
            if self._accepted(k, b):
                t = max(t, r.last_exec) + b.cost
                self._execute(k, b, 0, t)
                continue
            # The run of committed batches it never saw: the checkpoint skips all but
            # the newest, which is learned from its committed proof.
            run = []
            i = bisect.bisect_left(self.log_seqs, r.next_seq)
            while i < len(self.log_seqs):
                bb = self.log[self.log_seqs[i]]
                if bb.t_dec > self.t or self._accepted(k, bb):
                    break
                run.append(bb)
                i += 1
            for bb in run[:-1]:
                self._execute(k, bb, 0, t, count=False)
            t = max(t, r.last_exec) + run[-1].cost
            self._execute(k, run[-1], 0, t, sample=False)
        if r.in_vc:
            nxt = max(self.t + MAINTENANCE_MS, r.rec_req + RECOVERY_MIN_MS)
        else:
            # A dead peer in a view asks after 1 s without progress (EntityRecovery.cpp:362).
            nxt = max(self.t, r.rec_req) + RECOVERY_MIN_MS
        r.rec_due = nxt + self._rtt(k)

    def _rtt(self, k: int) -> float:
        ph = self._ph(self.t)
        d = 2.0 * (self.c.base_ms + ph.extra_ms)
        if ph.straggler == k:
            d += ph.straggler_ms
        return d

    # --- view change -------------------------------------------------------------

    def _count_vc(self, k: int) -> None:
        r = self.reps[k]
        for w in self._in_windows(k, self.t):
            w.vcs += 1
            if r.commits_since_vc == 0:
                w.noprog += 1
        r.commits_since_vc = 0

    def _send_vc(self, s: int, v: int) -> None:
        for j in range(N):
            if j != s:
                ta = self._hop(s, j, self.t, float(self.rng.uniform(-1.0, 1.0)))
                if ta < INF:
                    self._push(ta, _EV_VC, j, s, v)

    def _log(self, msg: str) -> None:
        """Debugging trace (sim.trace = [] to enable); callers build msg only when enabled."""
        if self.trace is not None:
            self.trace.append(f"{self.t / 1000.0:9.3f} {msg}")

    def _start_vc(self, k: int, nv: int, why: str = "") -> None:
        r = self.reps[k]
        if nv <= r.view:
            return
        if self.trace is not None:
            self._log(f"r{k} vc->{nv} {why} backoff={r.backoff} pend={self.pending_n:.0f}")
        if not r.in_vc:
            self._stop(k, r.view)
        r.view = nv
        r.in_vc = True
        self._count_vc(k)
        self._anchor(k)
        for v in list(r.claims):
            r.claims[v].discard(k)
            if not r.claims[v]:
                del r.claims[v]
        r.claims.setdefault(nv, set()).add(k)
        r.wait_armed = self.t
        self._send_vc(k, nv)
        first = self.t + float(self.rng.uniform(0.0, MAINTENANCE_MS))
        r.rec_due = max(first, r.rec_req + RECOVERY_MIN_MS) + self._rtt(k)
        self._try_build(k, nv)

    def _on_vc(self, k: int, s: int, v: int) -> None:
        r = self.reps[k]
        if v < r.view or (v == r.view and not r.in_vc):
            if not r.in_vc and r.last_installed >= v:
                # A sender behind an installed view gets the last NewView back.
                ta = self._hop(k, s, self.t, float(self.rng.uniform(-1.0, 1.0)))
                if ta < INF:
                    self._push(ta + self._lognormal(self.c.nv_val_ms, self.c.vc_sigma), _EV_NV, s, r.last_installed)
            return
        for vv, ss in r.claims.items():
            if vv > v and s in ss:
                return
        for vv in list(r.claims):
            if vv != v:
                r.claims[vv].discard(s)
                if not r.claims[vv]:
                    del r.claims[vv]
        ss = r.claims.setdefault(v, set())
        inserted = s not in ss
        ss.add(s)
        if inserted and v == r.view and r.in_vc and len(ss) == QUORUM:
            r.wait_armed = self.t
        higher = set()
        smallest = None
        for vv, sv in r.claims.items():
            if vv > r.view:
                smallest = vv if smallest is None else min(smallest, vv)
                higher |= sv
        if len(higher) >= F + 1:
            self._start_vc(k, smallest, "join")
            return
        self._try_build(k, v)

    def _try_build(self, k: int, v: int) -> None:
        r = self.reps[k]
        if not r.in_vc or r.view != v or v % N != k or r.last_nv_sent >= v:
            return
        if len(r.claims.get(v, ())) < QUORUM:
            return
        r.last_nv_sent = v
        self._push(self.t + self._lognormal(self.c.nv_build_ms, self.c.vc_sigma), _EV_BUILD, k, v)

    def _wait_expired(self, k: int) -> None:
        r = self.reps[k]
        reached = sum(len(ss) for vv, ss in r.claims.items() if vv >= r.view)
        if reached < QUORUM and k in r.claims.get(r.view, ()):
            r.wait_armed = self.t
            self._send_vc(k, r.view)
            return
        r.backoff = min(r.backoff + 1, 31)
        self._start_vc(k, r.view + 1, "escalate")

    def _expire(self, k: int) -> None:
        r = self.reps[k]
        if self.trace is not None:
            dl, acc, floored = self._deadline(k)
            self._log(f"r{k} deadline {dl / 1000.0:.3f} acc {acc / 1000.0:.3f} floor {r.floor / 1000.0:.3f} "
                      f"watched {self._watched(k) / 1000.0:.3f} T={self.T:.0f}")
        r.backoff = min(r.backoff + 1, 31)
        self._start_vc(k, r.view + 1, "expire")

    def _on_build(self, k: int, v: int) -> None:
        r = self.reps[k]
        if not r.in_vc or r.view != v:
            return
        repro = self._classify_orphans(v)
        self._install(k, v)
        installs = [INF] * N
        installs[k] = self.t
        # A dead peer's NewView carries its re-proposals over its dead consensus channel
        # (and would land S late, after the others escalated): no one installs it.
        dead = r.crashed
        for j in range(N):
            if j == k or dead:
                continue
            ta = self._hop(k, j, self.t, float(self.rng.uniform(-1.0, 1.0)))
            if ta == INF:
                continue
            ti = ta + self._lognormal(self.c.nv_val_ms, self.c.vc_sigma)
            rj = self.reps[j]
            if v > rj.view or (v == rj.view and rj.in_vc):
                rj.installed[v] = ti  # scheduled; withdrawn if it refuses the NewView
                installs[j] = ti
            self._push(ti, _EV_NV, j, v)
        busy = list(installs)
        for b in repro:
            b.view = v
            b.leader = k
            b.body = [None] * N
            busy = [x + xf if x < INF else INF for x, xf in zip(busy, b.xfer)]
            self._fetch_bodies(b, busy)
            self._timeline(b, self.t, busy)
            self._resolve(b)

    def _fetch_bodies(self, b: _Batch, handled: List[float]) -> None:
        """When each replica handling re-proposal b holds its bytes. One that held them
        before attaches its copy; one that did not sends a BatchRequest to every peer
        as it handles the header (requestBatch, EntityConsensus.cpp:159-167) and takes
        the first answer; every holder answers with the full batch on its link.
        Not modelled: the repeat request on each maintenance tick while the bytes are
        missing (and its duplicate answers), and the stall-time retransmissions of
        EntityRecovery.cpp:320-353. Charging the duplicate answers to the links made
        the 5+150 storms twice as slow and 40% lower in throughput than measured, and
        no instrumented run says how much of either the real links carry."""
        nbytes = b.size * BATCH_MAX_BYTES / max(1.0, float(self.c.batch_cap))
        dead = self._ph(self.t).crashed
        holders = [p for p in range(N) if b.bodyprev[p] and p != dead]
        for j in range(N):
            t = handled[j]
            if t == INF:
                continue
            if j == dead:
                b.body[j] = INF
                continue
            if b.bodyprev[j]:
                b.body[j] = t
                continue
            got = INF
            for p in holders:
                if p != j:
                    ta = self._hop(j, p, t, float(self.rng.uniform(-1.0, 1.0)))
                    got = min(got, self._bulk(p, j, ta, nbytes, float(self.rng.uniform(-1.0, 1.0)))
                              + self.c.per_tx_ms * b.size)
            b.body[j] = got

    def _classify_orphans(self, v: int) -> List[_Batch]:
        """Undecided batches of older views: re-proposed in the NewView when f+1 replicas
        voted for them (or a later sequence is decided), back to pending otherwise."""
        repro: List[_Batch] = []
        dropped: List[_Batch] = []
        decided_max = -1
        for seq in self.log_seqs:
            if self.log[seq].t_dec < INF:
                decided_max = seq
        for seq in list(self.log_seqs):
            b = self.log[seq]
            if b.t_dec < INF or b.view >= v:
                continue
            votes = 0
            for j in range(N):
                if b.arr[j] < INF and (j == b.leader and b.t_send < self.reps[j].stopped.get(b.view, INF)
                                       or j != b.leader and self._valid(j, b.view, b.pp[j], b.send[j])):
                    votes += 1
            if votes >= F + 1 or seq < decided_max:
                for j in range(N):
                    b.bodyprev[j] = (b.bodyprev[j] or (b.body[j] is not None and b.body[j] <= self.t)
                                     or self._has_body(j, b))
                    b.acc_prev[j] = b.acc_prev[j] or self._valid(j, b.view, b.pp[j], b.pp[j]) or j == b.leader
                b.reproposed = True
                repro.append(b)
            else:
                dropped.append(b)
        if dropped:
            chunks = []
            for b in dropped:
                del self.log[b.seq]
                self.log_seqs.remove(b.seq)
                if not b.first_exec:
                    self.inflight_n -= b.size
                self.pending_n += b.size
                chunks.extend(b.chunks)
            for ch in reversed(chunks):
                self.pending.appendleft(ch)
        return repro

    def _on_nv(self, k: int, v: int) -> None:
        r = self.reps[k]
        if v < r.view or (v == r.view and not r.in_vc):
            if v < r.view and r.installed.get(v, INF) >= self.t:
                r.installed[v] = INF
                self._rerun(v)
            return
        self._install(k, v)

    def _rerun(self, v: int) -> None:
        for seq in self.log_seqs:
            b = self.log[seq]
            if b.view == v and b.t_dec > self.t:
                self._resolve(b)

    def _install(self, k: int, v: int) -> None:
        r = self.reps[k]
        if self.trace is not None:
            self._log(f"r{k} install {v} backoff={r.backoff}")
        if not r.in_vc and r.view != v:
            self._stop(k, r.view)
        prev_sched = r.installed.get(v, INF)
        r.view = v
        r.in_vc = False
        r.installed[v] = self.t
        r.stopped.pop(v, None)
        r.wait_armed = None
        r.resume_t = self.t
        r.last_installed = max(r.last_installed, v)
        r.rec_due = self.t + RECOVERY_MIN_MS + self._rtt(k) if r.crashed else INF
        self._anchor(k)
        r.floor = self.t
        for vv in list(r.claims):
            if vv <= v:
                del r.claims[vv]
        for w in self._in_windows(k, self.t):
            w.nvs += 1
        if abs(prev_sched - self.t) > 1e-6:
            self._rerun(v)

    # --- main loop -----------------------------------------------------------------

    def _advance_to(self, end_ms: float) -> None:
        reps = self.reps
        while True:
            now = self.t
            best = self.heap[0][0] if self.heap else INF
            kind, who, extra = 0, -1, None
            if self.next_obs < best:
                best, kind = self.next_obs, 1
            if self.next_phase_ms < best:
                best, kind = self.next_phase_ms, 2
            for k in range(N):
                r = reps[k]
                te, b, mode = self._exec_cand(k)
                if te < best:
                    best, kind, who, extra = max(now, te), 3, k, (b, mode)
                if r.in_vc:
                    if r.wait_armed is not None:
                        tw = max(now, r.wait_armed + self._teff(r.backoff))
                        if tw < best:
                            best, kind, who = tw, 4, k
                    if r.rec_due < best:
                        best, kind, who = max(now, r.rec_due), 5, k
                else:
                    td = max(now, self._deadline(k)[0])
                    if td < best:
                        best, kind, who = td, 6, k
                    if r.crashed and r.rec_due < best:
                        best, kind, who = max(now, r.rec_due), 5, k
                    if r.view % N == k and r.next_tick < best:
                        best, kind, who = max(now, r.next_tick), 7, k
            if best > end_ms:
                break
            self._arrive(best)
            self.t = best
            if kind == 0:
                _, _, ev, a, b2, c2 = heapq.heappop(self.heap)
                if ev == _EV_VC:
                    self._on_vc(a, b2, c2)
                elif ev == _EV_NV:
                    self._on_nv(a, b2)
                else:
                    self._on_build(a, b2)
            elif kind == 1:
                self.next_obs += OBSERVE_MS
                self._observe()
            elif kind == 2:
                self._phase_change()
            elif kind == 3:
                self._execute(who, extra[0], extra[1], best)
            elif kind == 4:
                self._wait_expired(who)
            elif kind == 5:
                self._recovery(who)
            elif kind == 6:
                self._expire(who)
            else:
                self._tick(who)
        self._arrive(end_ms)
        self.t = max(self.t, end_ms)

    # --- timeouts and episodes ------------------------------------------------------

    def apply_timeouts(self, timeout_ms: float, wait_ms: float) -> None:
        """applyAgentTimeouts: a lower T floors every request timer at the change."""
        if timeout_ms < self.T:
            for r in self.reps:
                r.floor = self.t
        self.T = float(timeout_ms)
        if float(wait_ms) != self.W:
            self.W = float(wait_ms)
            self._w_hist.append((self.t, self.W))
            if len(self._w_hist) > 64:
                del self._w_hist[1]

    def _reporting(self, t_ms: float) -> List[int]:
        """The 2f+1 reports the sharing agents aggregate. The agent seals an episode at
        its 2f+1-th report and rejects the rest (main.py _store_report_locked), so the
        batch is the first three to arrive (all 472 SBFT sharing batches held 3). The
        agent runs in its replica's netns, so a straggler's report carries S and is
        never among them (0 of 60 batches in p1-p4 of the four 1500 s sharing runs);
        nor is a dead peer's. Without one, the last report is the replica whose agent
        clock (started at its first consensus sample) runs latest: fixed per run."""
        ph = self._ph(t_ms)
        drop = ph.straggler if ph.straggler >= 0 and ph.straggler_ms > 0.0 else self.late_reporter
        return [k for k in range(N) if k != drop]

    def median_report(self, windows: List[_Window], timeout_ms: float, wait_ms: float,
                      t_ms: float) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        reports = [windows[k].report(timeout_ms, wait_ms) for k in range(N)]
        avail = self._reporting(t_ms)
        med = {name: float(np.median([reports[k][name] for k in avail])) for name in reports[0]}
        # The n-th prepare fields are the leader's: taken from the report that led the most
        # sequences, lowest replica on a tie (aggregator.py, as every agent does).
        lead = min(avail, key=lambda k: (-(reports[k]["nth_prepare_seen_count"]
                                           + reports[k]["nth_prepare_unseen_count"]), k))
        for name in NTH_PREPARE_FIELDS:
            med[name] = reports[lead][name]
        # An aggregate in which no batch executed takes the window's length as its latency,
        # as main.py sets it on the report the model reads (CppBedrock reports 0 there).
        if med["total_consensus_instances"] == 0.0:
            for name in ("avg_consensus_latency_ms", "p50_consensus_latency_ms", "p95_consensus_latency_ms"):
                med[name] = windows[0].end - windows[0].start
        return med, reports

    def run_episode(
        self, choose: Callable[[Dict[str, float]], Tuple[float, float]]
    ) -> Tuple[Dict[str, float], Tuple[float, float], Dict[str, float], int, List[Dict[str, float]]]:
        """One episode: ``choose(median feature report) -> (T, W)`` at the feature deadline,
        applied at the end of the reply wait. Returns (features, (T, W), median reward
        report, phase index at the apply instant, per-replica reward reports)."""
        s = self.episode_start_s
        fw = [_Window(s, s + FEATURE_S) for _ in range(N)]
        self.windows = [[w] for w in fw]
        self._advance_to((s + FEATURE_S) * 1000.0)
        features, _ = self.median_report(fw, self.T, self.W, self.t)
        # The phase in force when the pair applies, as main.py labels an episode, so the
        # onset episode carries the phase its reward window measures.
        phase_idx = self.phase_index(s + FEATURE_S + REPLY_S)
        self.windows = None
        new_t, new_w = choose(features)
        self._advance_to((s + FEATURE_S + REPLY_S) * 1000.0)
        self.apply_timeouts(new_t, new_w)
        rw = [_Window(s + FEATURE_S + REPLY_S + WARMUP_S, s + EPISODE_S) for _ in range(N)]
        self.windows = [[w] for w in rw]
        self._advance_to((s + EPISODE_S) * 1000.0)
        reward, per_rep = self.median_report(rw, new_t, new_w, self.t)
        self.windows = None
        self.episode_start_s = s + EPISODE_S
        return features, (float(new_t), float(new_w)), reward, phase_idx, per_rep


# --- Randomized environments ---------------------------------------------------


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _around(rng: np.random.Generator, host: float, lo: float = 0.5, hi: float = 2.0) -> float:
    """A host constant scaled by a log-uniform factor in [lo, hi]."""
    return float(host * math.exp(rng.uniform(math.log(lo), math.log(hi))))


# Offered load the training draw is centred on: the client rate the CloudLab
# calibration and experiment runs use (run_sbft.sh --rate 15000). Each run draws
# load_tps uniformly over LOAD_DRAW_FACTORS x this, the same 0.5-1.25 x bracket the
# draw had around the 20k calibration rate, so the deployment rate sits inside the
# support rather than past its top edge. It matters because the arm ranking moves with
# load: at 20k the 250 ms election wins phase 0 and at 30k it storms and loses to
# 525/1100. train --rate sets it, and the fitted prior records it as load_tps. eval and
# the pretrain spec runs use SPEC_RATE_TPS; check keeps MEASURED_RATE_TPS, the rate of
# the MEASURED_20K fixture it is scored against.
TRAIN_RATE_TPS = 15000.0
LOAD_DRAW_FACTORS: Tuple[float, float] = (0.5, 1.25)


def sample_constants(rng: np.random.Generator) -> SimConstants:
    """Per-run physics. The processing, transfer and view-change constants, and every
    constant the blind fit sets (sim_fit.py FREE, batch_cap and slow_lock_ms included),
    are drawn within 0.5-2 x their SimConstants defaults (each cited there): wider draws
    (up to 15 x) made a T just above the base floor storm in 15-17% of draws whose
    context the prior could not tell apart from the host's, and the prior then preferred
    a T that sleeps through a proposal delay. batch_cap stays an integer at most
    BATCH_MAX_REQUESTS. Load, the base link, the initial window and the backlog stay
    wide; onset_stall_ms stays at its default (the harness pre-applies the base netem);
    the fault values (link extra, straggler, crash, proposal delay) are drawn in
    sample_phase and never depend on the spec."""
    h = SimConstants()
    return SimConstants(
        base_ms=float(rng.uniform(3.0, 8.0)),
        per_tx_ms=_around(rng, h.per_tx_ms),
        proc_spread=_around(rng, h.proc_spread),
        proc_base_ms=_around(rng, h.proc_base_ms),
        proc_sigma=_around(rng, h.proc_sigma),
        ack_proc_ms=_around(rng, h.ack_proc_ms),
        slow_lock_ms=_around(rng, h.slow_lock_ms),
        exe_proc_ms=_around(rng, h.exe_proc_ms),
        exec_base_ms=_around(rng, h.exec_base_ms),
        exec_per_tx_ms=_around(rng, h.exec_per_tx_ms),
        nv_build_ms=_around(rng, h.nv_build_ms),
        nv_val_ms=_around(rng, h.nv_val_ms),
        vc_sigma=_around(rng, h.vc_sigma),
        strag_hold_ms=_around(rng, h.strag_hold_ms),
        init_window_bytes=1448.0 * float(rng.integers(4, 41)),
        max_window_bytes=_around(rng, h.max_window_bytes),
        load_tps=float(rng.uniform(LOAD_DRAW_FACTORS[0] * TRAIN_RATE_TPS,
                                   LOAD_DRAW_FACTORS[1] * TRAIN_RATE_TPS)),
        batch_cap=int(round(_around(rng, h.batch_cap, hi=min(2.0, BATCH_MAX_REQUESTS / h.batch_cap)))),
        initial_backlog=int(rng.integers(0, MAX_PENDING_REQUESTS + 1)) if rng.random() < 0.3 else 0,
    )


# train --no-proposal-faults: a prior that never saw a delayed or silent
# proposer, so its pretraining covers the network-delay regimes only. The
# "blind pretrained" arm of the ablation - see main.py --prior-path.
NO_PROPOSAL_FAULTS = False


def sample_phase(rng: np.random.Generator, start_s: float) -> Phase:
    extra = 0.0 if rng.random() < 0.35 else _log_uniform(rng, 10.0, 250.0)
    jitter = 0.0 if rng.random() < 0.6 else float(rng.uniform(0.02, 0.15))
    kind = rng.random()
    straggler, s_ms = -1, 0.0
    if kind >= 0.35:
        straggler = int(rng.integers(0, N))
        s_ms = _log_uniform(rng, 20.0, 800.0) if kind < 0.80 else _log_uniform(rng, CRASH_DELAY_MS, 10000.0)
    delay, interval = 0.0, 0.0
    if rng.random() >= 0.45:
        delay = _log_uniform(rng, 50.0, 1500.0)
        interval = float(rng.uniform(20.0, 120.0)) if rng.random() < 0.8 else 0.0
    if NO_PROPOSAL_FAULTS:
        straggler, s_ms, delay, interval = -1, 0.0, 0.0, 0.0
    return Phase(start_s=start_s, extra_ms=extra, jitter=jitter, straggler=straggler,
                 straggler_ms=s_ms, delay_ms=delay, interval_s=interval)


def sample_schedule(rng: np.random.Generator, total_s: float) -> List[Phase]:
    phases = [Phase(start_s=0.0)]
    t = float(rng.uniform(0.0, 20.0))
    while t < total_s:
        phases.append(sample_phase(rng, t))
        t += float(rng.uniform(40.0, 400.0))
    return phases


NTH_PREPARE_FIELDS = ("nth_prepare_gap_ms", "nth_prepare_seen_count", "nth_prepare_unseen_count")

SPEC_WARMUP_S = 10.0
# Spec replays start here, so episode e applies at SPEC_WARMUP_S + EPISODE_S (e - 1), on the phase onsets.
SPEC_START_OFFSET_S = SPEC_WARMUP_S - FEATURE_S - REPLY_S
SPEC_RUN_S = 1510.0
SPEC_RATE_TPS = 15000.0  # run_sbft.sh --rate of the calibration and experiment runs
MEASURED_RATE_TPS = 20000.0  # the rate of the MEASURED_20K fixture check compares against


def spec_phases() -> List[Phase]:
    """The five phases of config/sbft/failure_spec.xml at their atTime (s), replica 2 the
    faulty source. Used only by check and eval, never by train."""
    return [
        Phase(0.0, straggler=2, straggler_ms=5000.0, delay_ms=150.0, interval_s=60.0),
        Phase(300.0, extra_ms=50.0, straggler=2, straggler_ms=80.0, delay_ms=350.0, interval_s=60.0),
        Phase(600.0, extra_ms=50.0, straggler=2, straggler_ms=300.0, delay_ms=350.0, interval_s=60.0),
        Phase(900.0, extra_ms=150.0, straggler=2, straggler_ms=100.0),
        Phase(1200.0, extra_ms=150.0, straggler=2, straggler_ms=300.0),
    ]


def spec_schedule() -> List[Phase]:
    """Simulator schedule of the spec: a clean warm-up, then spec phase p at
    10 s + atTime (sim phase index p + 1)."""
    return [Phase(0.0)] + [replace(p, start_s=SPEC_WARMUP_S + p.start_s) for p in spec_phases()]


# --- Context, reward, model interface ----------------------------------------------


def report_vector(report: Dict[str, float]) -> np.ndarray:
    from models import SBFT_REPORT_FEATURE_FIELD_NAMES

    return np.asarray([report[name] for name in SBFT_REPORT_FEATURE_FIELD_NAMES], dtype=float)


def sbft_context(report: Dict[str, float]) -> np.ndarray:
    return np.asarray([math.log1p(max(0.0, float(report[name]))) for name in SBFT_CONTEXT_FIELDS])


def normalize_ms(ms: float, bounds: Tuple[int, int]) -> float:
    lo, hi = bounds
    if ms < lo or ms > hi:
        raise ValueError(f"{ms} ms outside {lo}..{hi}")
    return (math.log(ms) - math.log(lo)) / (math.log(hi) - math.log(lo))


def denormalize_ms(a: float, bounds: Tuple[int, int]) -> int:
    lo, hi = bounds
    a = min(1.0, max(0.0, float(a)))
    return min(hi, max(lo, int(round(math.exp(math.log(lo) + a * (math.log(hi) - math.log(lo)))))))


def check_model_spec():
    """The SBFT ContextSpec of continuous_bandit, if it exists yet, must read exactly the
    simulator's context and bounds. Returns the spec or None."""
    import agent_pb2
    import continuous_bandit as cb

    if int(agent_pb2.PROTOCOL_SBFT) not in cb.CONTEXT_SPECS:
        return None
    spec = cb.context_spec(agent_pb2.PROTOCOL_SBFT)
    if tuple(spec.log_fields) != SBFT_CONTEXT_FIELDS or tuple(spec.raw_fields) != ():
        raise ValueError(f"continuous_bandit SBFT context {spec.log_fields}/{spec.raw_fields} "
                         f"differs from sbft_sim.SBFT_CONTEXT_FIELDS {SBFT_CONTEXT_FIELDS}")
    wb = getattr(spec, "wait_bounds_ms", None)
    if tuple(spec.bounds_ms) != T_BOUNDS_MS or wb is None or tuple(wb) != W_BOUNDS_MS:
        raise ValueError(f"continuous_bandit SBFT bounds {spec.bounds_ms}/{wb} differ from "
                         f"sbft_sim {T_BOUNDS_MS}/{W_BOUNDS_MS}")
    return spec


_INT_FIELDS = (
    "total_transactions", "total_consensus_instances", "leader_change_count",
    "regency_change_count", "timeout_ms", "view_change_count", "no_progress_view_change_count",
    "slow_path_timeout_ms", "fast_path_count", "slow_path_count",
)


def episode_reward(report: Dict[str, float]) -> float:
    """Reward of one (median) reward report under the process-wide reward design, the
    calculator main.reward_engineering uses."""
    from reward_utils import calculate_sbft_reward

    return calculate_sbft_reward(**{
        name: int(round(value)) if name in _INT_FIELDS else float(value)
        for name, value in report.items() if name not in NTH_PREPARE_FIELDS
    })


def run_agent(
    sim: SbftSim,
    episodes: int,
    predict: Callable[[np.ndarray], Tuple[float, float]],
    update: Optional[Callable[[np.ndarray, Tuple[float, float], float], None]],
) -> List[Tuple[int, Tuple[float, float], Dict[str, float], float]]:
    """Drive sim like the sharing-mode agents: predict at the feature deadline from the
    median report, reward from the median reward report, no update before warm-up."""
    rows = []
    for _ in range(episodes):
        decision_s = sim.episode_start_s + FEATURE_S
        holder: Dict[str, np.ndarray] = {}

        def choose(report):
            holder["x"] = report_vector(report)
            return predict(holder["x"])

        _, pair, reward_report, phase_idx, _ = sim.run_episode(choose)
        phase = phase_idx if decision_s >= WARMUP_PHASE_S else -1
        reward = episode_reward(reward_report)
        if update is not None and phase != -1:
            update(holder["x"], pair, reward)
        rows.append((phase, pair, reward_report, reward))
    return rows


# --- Training data and prior ---------------------------------------------------------


def training_sim(rng: np.random.Generator, seed: int) -> Tuple[SbftSim, int]:
    """One randomized training run: (simulator, episodes)."""
    episodes = int(rng.integers(40, 120))
    if rng.random() < 0.25:
        t0, w0 = DEFAULT_ELECTION_TIMEOUT_MS, DEFAULT_SLOW_PATH_TIMEOUT_MS
    else:
        t0 = _log_uniform(rng, *T_BOUNDS_MS)
        w0 = _log_uniform(rng, *W_BOUNDS_MS)
    sim = SbftSim(
        sample_constants(rng),
        sample_schedule(rng, episodes * EPISODE_S + 20.0),
        seed=seed,
        start_offset_s=float(rng.uniform(0.0, 20.0)),
        initial_timeout_ms=t0,
        initial_wait_ms=w0,
    )
    return sim, episodes


def _generate_one(args: Tuple[int, str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """One randomized run. Every ``hold`` episodes an action is drawn and held for
    ``hold`` episodes; a sample is (context at the hold's first decision, action) ->
    the mean reward over the hold (hold 1 = per episode)."""
    from reward_utils import set_reward_design

    seed, design, hold = args
    set_reward_design(design)
    rng = np.random.default_rng(seed)
    sim, episodes = training_sim(rng, seed)
    prev = np.array([
        normalize_ms(min(max(sim.T, T_BOUNDS_MS[0]), T_BOUNDS_MS[1]), T_BOUNDS_MS),
        normalize_ms(min(max(sim.W, W_BOUNDS_MS[0]), W_BOUNDS_MS[1]), W_BOUNDS_MS),
    ])
    xs, ys = [], []
    for _ in range(episodes // hold):
        holder = {}

        def choose(report):
            if "a" in holder:
                return holder["ms"]
            holder["ctx"] = sbft_context(report)
            if rng.random() < 0.6:
                a = rng.random(2)
            else:
                a = np.clip(prev + rng.normal(0.0, 0.08, 2), 0.0, 1.0)
            t_ms = denormalize_ms(a[0], T_BOUNDS_MS)
            w_ms = denormalize_ms(a[1], W_BOUNDS_MS)
            holder["a"] = np.array([normalize_ms(t_ms, T_BOUNDS_MS), normalize_ms(w_ms, W_BOUNDS_MS)])
            holder["ms"] = (float(t_ms), float(w_ms))
            return holder["ms"]

        rewards = []
        for _ in range(hold):
            _, _, reward_report, _, _ = sim.run_episode(choose)
            rewards.append(episode_reward(reward_report))
        xs.append(np.concatenate([holder["ctx"], holder["a"]]))
        ys.append(float(np.mean(rewards)))
        prev = holder["a"]
    return np.asarray(xs), np.asarray(ys)


def _generate_policy_one(args: Tuple[int, str, int, str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """One randomized run driven by the deployed model (make_candidate on a prior), which
    decides and learns online every episode as on CloudLab. At a decision, with chance
    ``probe``, a probe starts: the policy's own pair with noise or a uniform pair (half
    each), held ``hold`` episodes; a sample is (context at the probe's decision, pair) ->
    the mean reward over the hold. The samples sit where the policy goes, which random
    actions rarely reach (its own cycles)."""
    from reward_utils import set_reward_design

    seed, design, hold, prior, probe = args
    set_reward_design(design)
    rng = np.random.default_rng(seed)
    episodes = int(rng.integers(40, 120))
    if rng.random() < 0.25:
        t0, w0 = DEFAULT_ELECTION_TIMEOUT_MS, DEFAULT_SLOW_PATH_TIMEOUT_MS
    else:
        t0 = _log_uniform(rng, *T_BOUNDS_MS)
        w0 = _log_uniform(rng, *W_BOUNDS_MS)
    sim = SbftSim(
        sample_constants(rng),
        sample_schedule(rng, episodes * EPISODE_S + 20.0),
        seed=seed,
        start_offset_s=float(rng.uniform(0.0, 20.0)),
        initial_timeout_ms=t0,
        initial_wait_ms=w0,
    )
    predict, update = make_candidate(prior)
    xs, ys = [], []
    active: Dict[str, object] = {}
    for _ in range(episodes):
        holder: Dict[str, np.ndarray] = {}

        def choose(report):
            holder["x"] = report_vector(report)
            if active:
                return active["ms"]
            if rng.random() >= probe:
                return predict(holder["x"])
            if rng.random() < 0.5:
                t_ms, w_ms = predict(holder["x"])
                a = np.array([normalize_ms(t_ms, T_BOUNDS_MS), normalize_ms(w_ms, W_BOUNDS_MS)])
                a = np.clip(a + rng.normal(0.0, 0.1, 2), 0.0, 1.0)
            else:
                a = rng.random(2)
            t_ms = denormalize_ms(a[0], T_BOUNDS_MS)
            w_ms = denormalize_ms(a[1], W_BOUNDS_MS)
            active.update(ctx=sbft_context(report), ms=(float(t_ms), float(w_ms)), rewards=[],
                          a=np.array([normalize_ms(t_ms, T_BOUNDS_MS), normalize_ms(w_ms, W_BOUNDS_MS)]))
            return active["ms"]

        decision_s = sim.episode_start_s + FEATURE_S
        _, pair, reward_report, _, _ = sim.run_episode(choose)
        reward = episode_reward(reward_report)
        if decision_s >= WARMUP_PHASE_S:  # the agents skip updates during warm-up (run_agent)
            update(holder["x"], pair, reward)
        if active:
            active["rewards"].append(reward)
            if len(active["rewards"]) == hold:
                xs.append(np.concatenate([active["ctx"], active["a"]]))
                ys.append(float(np.mean(active["rewards"])))
                active.clear()
    if not xs:
        return np.zeros((0, len(SBFT_CONTEXT_FIELDS) + 2)), np.zeros(0)
    return np.asarray(xs), np.asarray(ys)


def generate(n_runs: int, seed: int, workers: int, design: str, hold: int = 1,
             policy_prior: Optional[str] = None, probe: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
    from multiprocessing import Pool

    if policy_prior is None:
        fn, seeds = _generate_one, [(seed + i, design, hold) for i in range(n_runs)]
    else:
        fn, seeds = _generate_policy_one, [(seed + i, design, hold, policy_prior, probe) for i in range(n_runs)]
    if workers <= 1:
        parts = [fn(s) for s in seeds]
    else:
        with Pool(workers) as pool:
            parts = pool.map(fn, seeds, chunksize=4)
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def train_prior(x: np.ndarray, y: np.ndarray, out_path: str, seed: int, design: str,
                target_floor: float, hold: int = 1,
                hidden: Tuple[int, ...] = (128, 128, 64),
                rate_tps: Optional[float] = None) -> float:
    from sklearn.neural_network import MLPRegressor

    y = np.maximum(y, target_floor)
    x_mean, x_std = x.mean(axis=0), x.std(axis=0) + 1e-9
    y_mean, y_std = float(y.mean()), float(y.std() + 1e-9)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    cut = int(0.9 * len(x))
    tr, va = order[:cut], order[cut:]
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(hidden),
        activation="relu",
        alpha=1e-5,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=seed,
    )
    mlp.fit((x[tr] - x_mean) / x_std, (y[tr] - y_mean) / y_std)
    pred = mlp.predict((x[va] - x_mean) / x_std) * y_std + y_mean
    rmse = float(np.sqrt(np.mean((pred - y[va]) ** 2)))
    arrays = {"n_layers": np.asarray(len(mlp.coefs_))}
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        arrays[f"W{i}"] = w
        arrays[f"b{i}"] = b
    arrays.update(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=np.asarray(y_mean),
        y_std=np.asarray(y_std),
        reward_design=np.asarray(design),
        hold_episodes=np.asarray(int(hold)),
        target_floor=np.asarray(float(target_floor)),
        peer_spread=np.asarray(0.0),
        action_dim=np.asarray(2),
        # Centre of the load draw the episodes were generated at (train --rate), so a
        # prior deployed at a different client rate is recognisable as one.
        load_tps=np.asarray(float(TRAIN_RATE_TPS if rate_tps is None else rate_tps)),
    )
    arrays.update(SIM_WINDOWS.arrays())
    np.savez(out_path, **arrays)
    return rmse


# --- Check: fixed pairs on the spec ----------------------------------------------------

LADDER_PAIRS = [(250, 50), (250, 150), (525, 50), (525, 150), (1100, 50), (1100, 150)]
OFF_LADDER_PAIRS = [(400, 100), (800, 100), (1600, 300), (2400, 30)]

# runs/2026-09-28-sbft-revalidation, 20k: agent (median report, episodes 3-5) p50 and tps,
# client latency (cal/sb_v20_* client.log, t in [60, 125)); data.md section 3.
MEASURED_20K = {
    0: {(250, 50): (200, 19930, 191), (250, 150): (1763, 14238, 1022), (525, 50): (356, 20083, 272),
        (525, 150): (442, 19868, 395), (1100, 50): (351, 20002, 322), (1100, 150): (457, 20002, 431)},
    1: {(250, 50): (748, 19019, 858), (250, 150): (687, 17705, 825), (525, 50): (444, 19977, 425),
        (525, 150): (382, 20000, 351), (1100, 50): (814, 19903, 787), (1100, 150): (738, 20002, 695)},
    2: {(250, 50): (1286, 18636, 1487), (250, 150): (1211, 16994, 1305), (525, 50): (448, 19992, 445),
        (525, 150): (963, 19273, 1064), (1100, 50): (818, 19989, 773), (1100, 150): (921, 19823, 885)},
    3: {(250, 50): (2111, 15152, 2187), (250, 150): (1913, 13505, 2124), (525, 50): (1409, 19019, 1414),
        (525, 150): (1048, 17652, 1194), (1100, 50): (955, 19914, 918), (1100, 150): (685, 20002, 652)},
    4: {(250, 50): (2107, 15508, 3034), (250, 150): (2115, 14723, 3453), (525, 50): (2061, 16605, 2477),
        (525, 150): (2177, 18002, 2461), (1100, 50): (963, 19968, 925), (1100, 150): (1070, 20050, 1029)},
}
# The ranking the check must reproduce (design_sim.md), best first; "storm" pairs last.
MEASURED_RANKING = {
    0: [(250, 50), (525, 50), (1100, 50)],
    1: [(525, 150), (525, 50), (1100, 150), (1100, 50)],
    2: [(525, 50), (1100, 50), (1100, 150), (525, 150)],
    3: [(1100, 150), (1100, 50)],
    4: [(1100, 50), (1100, 150)],
}


def _phase_window_s(p: int, mode: str) -> Tuple[float, float]:
    """Interior of spec phase p in sim seconds: reward windows from 40 s after onset (60 s
    into the run for phase 0, the steady-state rule) to the phase end."""
    if mode == "cal":
        return 60.0, 125.0
    onset = SPEC_WARMUP_S + 300.0 * p
    lo = max(onset + 40.0, 60.0)
    return lo, onset + 300.0 if p < 4 else SPEC_RUN_S


def _fixed_pair_run(pair: Tuple[int, int], seed: int, mode: str, phase: int,
                    constants: SimConstants) -> Dict[int, Dict[str, float]]:
    """Run the spec (mode spec: all five phases; mode cal: one phase from 10 s, 130 s, as
    the calibration runs did) with one pair from the first decision; per-phase means."""
    if mode == "spec":
        sim = SbftSim(constants, spec_schedule(), seed=seed, start_offset_s=SPEC_START_OFFSET_S)
        episodes, phases = int(SPEC_RUN_S // EPISODE_S), range(5)
    else:
        ph = replace(spec_phases()[phase], start_s=SPEC_WARMUP_S)
        sim = SbftSim(constants, [Phase(0.0), ph], seed=seed, start_offset_s=SPEC_START_OFFSET_S)
        episodes, phases = 6, [phase]
    rows = []
    for _ in range(episodes):
        s0 = sim.episode_start_s
        _, _, rep, _, per = sim.run_episode(lambda f: (float(pair[0]), float(pair[1])))
        rows.append((s0 + FEATURE_S + REPLY_S + WARMUP_S, s0 + EPISODE_S, rep, per))
    out = {}
    for p in phases:
        lo, hi = _phase_window_s(p, mode)
        sel = [r for r in rows if r[0] >= lo and r[1] <= hi]
        reps = [r[2] for r in sel]
        if not reps:
            continue
        fast = sum(r["fast_path_count"] for r in reps)
        slow = sum(r["slow_path_count"] for r in reps)
        cl = [(lat, n) for t, lat, n in sim.clat if lo * 1000.0 <= t < hi * 1000.0]
        out[p] = {
            "p50": float(np.mean([r["p50_consensus_latency_ms"] for r in reps])),
            "tps": float(np.mean([r["throughput_tps"] for r in reps])),
            "fast": fast / (fast + slow) if fast + slow > 0 else 0.0,
            "vc": float(np.mean([r["view_change_count"] for r in reps])),
            "reward": float(np.mean([episode_reward(r) for r in reps])),
            "clat": float(sum(a * n for a, n in cl) / sum(n for _, n in cl)) if cl else float("nan"),
            "zero": float(np.mean([r["total_consensus_instances"] == 0.0 for r in reps])),
        }
    return out


def _check_job(args):
    pair, seed, mode, phase = args
    from reward_utils import set_reward_design

    set_reward_design("reward_tps_latency")
    t0 = time.time()
    c = SimConstants()
    c.load_tps = MEASURED_RATE_TPS
    res = _fixed_pair_run(pair, seed, mode, phase, c)
    return pair, seed, phase, res, time.time() - t0


def run_check(seeds: int, workers: int, mode: str, pairs: List[Tuple[int, int]], out: Optional[str]) -> str:
    from multiprocessing import Pool

    jobs = []
    for pair in pairs:
        for s in range(seeds):
            if mode == "spec":
                jobs.append((pair, 1000 + s, mode, -1))
            else:
                for p in range(5):
                    jobs.append((pair, 1000 + s, mode, p))
    if workers <= 1:
        results = [_check_job(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            results = pool.map(_check_job, jobs, chunksize=1)
    agg: Dict[Tuple[int, Tuple[int, int]], List[Dict[str, float]]] = {}
    secs = []
    for pair, seed, phase, res, dt in results:
        secs.append(dt)
        for p, v in res.items():
            agg.setdefault((p, pair), []).append(v)
    lines = [f"sbft_sim check mode={mode} seeds={seeds} rate={MEASURED_RATE_TPS:.0f} host constants "
             f"{SimConstants()}",
             f"wall s per simulated run: mean {np.mean(secs):.1f} (max {np.max(secs):.1f}) for "
             f"{'1510' if mode == 'spec' else '130'} s simulated",
             f"phase pair        sim: p50   clat    tps  fast%   vc/{REWARD_S:g}s  zero%  reward | measured 20k: agent p50   tps  client"]
    ranking_ok = []
    for p in range(5):
        cells = {}
        for pair in pairs:
            vals = agg.get((p, pair))
            if not vals:
                continue
            m = {k: float(np.nanmean([v[k] for v in vals])) for k in vals[0]}
            cells[pair] = m
            meas = MEASURED_20K[p].get(pair)
            ms = f"{meas[0]:6.0f} {meas[1]:6.0f} {meas[2]:6.0f}" if meas else "     -      -      -"
            lines.append(
                f"p{p}  ({pair[0]:4d},{pair[1]:3d})  {m['p50']:7.0f} {m['clat']:6.0f} {m['tps']:6.0f} "
                f"{100 * m['fast']:5.0f} {m['vc']:7.1f} {100 * m['zero']:5.0f} {m['reward']:7.3f} | "
                f"{ms}")
        want = [q for q in MEASURED_RANKING[p] if q in cells]
        by_reward = sorted(want, key=lambda q: -cells[q]["reward"])
        by_p50 = sorted(want, key=lambda q: cells[q]["p50"])
        # Adjacent pairs closer than TIE_REWARD (10 ms of p50 at equal tps) are ties, not an order.
        ties = [f"{a}~{b}" for a, b in zip(want, want[1:])
                if abs(cells[a]["reward"] - cells[b]["reward"]) < TIE_REWARD]
        # Pairs the sim orders against the measurement, and whether all are ties anyway.
        inverted = [(a, b) for i, a in enumerate(want) for b in want[i + 1:]
                    if cells[a]["reward"] < cells[b]["reward"]]
        tie_only = all(abs(cells[a]["reward"] - cells[b]["reward"]) < TIE_REWARD for a, b in inverted)
        ok_r = by_reward == want and not ties
        ok_l = by_p50 == want
        ranking_ok.append(ok_r)
        lines.append(f"p{p}  measured order {want}; sim by reward {by_reward} "
                     f"{'OK' if ok_r else ('TIE ' + ','.join(ties) if tie_only and ties else 'DIFF')}; "
                     f"sim by p50 {by_p50} {'OK' if ok_l else 'DIFF'}")
    lines.append(f"rankings by reward reproduced in {sum(ranking_ok)}/5 phases")
    text = "\n".join(lines)
    if out:
        with open(out, "w") as fh:
            fh.write(text + "\n")
    return text


# --- Eval: SimReturnGPModel against the oracle on the spec -----------------------------

TIE_REWARD = 0.01

ORACLE_ROWS = [(250, 50), (525, 150), (525, 50), (1100, 150), (1100, 50)]


def make_candidate(prior_path: str):
    """(predict, update) around continuous_bandit.SimReturnGPModel on SBFT, built as
    main.serve builds --model-type continuous_pretrained_gp (codec with wait_bounds_ms)."""
    import agent_pb2
    from continuous_bandit import ContinuousTimeoutActionCodec, SimReturnGPModel
    from models import SBFT_REPORT_FEATURE_FIELD_NAMES as names
    from reward_utils import get_reward_design

    spec = check_model_spec()
    if spec is None:
        raise RuntimeError("continuous_bandit has no SBFT ContextSpec yet; eval needs the 2-D model change")
    check_prior_windows(prior_path, SIM_WINDOWS)
    codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_SBFT, bounds_ms=spec.bounds_ms,
                                         wait_bounds_ms=spec.wait_bounds_ms)
    model = SimReturnGPModel(action_codec=codec, field_names=names, prior_path=prior_path,
                             reward_design=get_reward_design())

    def predict(x: np.ndarray) -> Tuple[float, float]:
        t = codec.decode(model.predict_action(x)).sbft
        return float(t.election_timeout_milliseconds), float(t.slow_path_timeout_milliseconds)

    def update(x: np.ndarray, pair: Tuple[float, float], reward: float) -> None:
        used = agent_pb2.SbftTimeout(election_timeout_milliseconds=int(round(pair[0])),
                                     slow_path_timeout_milliseconds=int(round(pair[1])))
        model.update_model(x, codec.encode(used), reward)

    return predict, update


def _eval_rows(kind: str, seed: int, prior: str, rate_tps: float = SPEC_RATE_TPS):
    c = SimConstants()
    c.load_tps = rate_tps
    sim = SbftSim(c, spec_schedule(), seed=seed, start_offset_s=SPEC_START_OFFSET_S)
    episodes = int(SPEC_RUN_S // EPISODE_S)
    if kind == "oracle":
        def predict(x):
            # The row of the phase the pair applies in (reply wait later): it switches at onset.
            p = max(0, sim.phase_index(sim.t / 1000.0 + REPLY_S) - 1)
            return tuple(float(v) for v in ORACLE_ROWS[p])
        update = None
    else:
        predict, update = make_candidate(prior)
    return run_agent(sim, episodes, predict, update)


def run_eval(seeds: int, prior: str, rate_tps: float = SPEC_RATE_TPS) -> str:
    from reward_utils import set_reward_design

    set_reward_design("reward_tps_latency")
    lines = [f"sbft_sim eval prior={prior} seeds={seeds} rate={rate_tps:.0f} "
             f"(phase: candidate p50/tps/reward @T,W | oracle)"]
    for s in range(seeds):
        res = {k: _eval_rows(k, 2000 + s, prior, rate_tps) for k in ("candidate", "oracle")}
        parts = []
        for p in range(5):
            out = []
            for k in ("candidate", "oracle"):
                sel = [r for r in res[k] if r[0] == p + 1]
                out.append((np.mean([r[2]["p50_consensus_latency_ms"] for r in sel]),
                            np.mean([r[2]["throughput_tps"] for r in sel]),
                            np.mean([r[3] for r in sel]),
                            np.median([r[1][0] for r in sel]), np.median([r[1][1] for r in sel])))
            cnd, orc = out
            parts.append(f"p{p} {cnd[0]:5.0f}/{cnd[1]:5.0f}/{cnd[2]:+.3f}@{cnd[3]:.0f},{cnd[4]:.0f} | "
                         f"{orc[0]:5.0f}/{orc[1]:5.0f}/{orc[2]:+.3f}")
        lines.append(f"seed {s}: " + " ; ".join(parts))
    return "\n".join(lines)


def pretrain_adapter():
    """This simulator as pretrain.py drives it: training runs as train draws
    them (at TRAIN_RATE_TPS), spec runs and the oracle as eval builds them (host
    constants at SPEC_RATE_TPS, SPEC_START_OFFSET_S, SPEC_RUN_S). Sim phase p + 1 is
    spec phase p."""
    import agent_pb2
    from pretrain import SimAdapter, timeout_message

    def to_timeout(pair: Tuple[float, float]) -> agent_pb2.Timeout:
        return agent_pb2.Timeout(sbft=agent_pb2.SbftTimeout(
            election_timeout_milliseconds=int(round(pair[0])),
            slow_path_timeout_milliseconds=int(round(pair[1])),
        ))

    def from_timeout(timeout: agent_pb2.Timeout) -> Tuple[float, float]:
        t = timeout_message(timeout, "sbft")
        return float(t.election_timeout_milliseconds), float(t.slow_path_timeout_milliseconds)

    def spec_run(seed: int) -> Tuple[SbftSim, int]:
        c = SimConstants()
        c.load_tps = SPEC_RATE_TPS
        sim = SbftSim(c, spec_schedule(), seed=seed, start_offset_s=SPEC_START_OFFSET_S)
        return sim, int(SPEC_RUN_S // EPISODE_S)

    def oracle(sim: SbftSim) -> Callable[[np.ndarray], Tuple[float, float]]:
        def predict(x: np.ndarray) -> Tuple[float, float]:
            # The row of the phase the pair applies in (reply wait later): it switches at onset.
            p = max(0, sim.phase_index(sim.t / 1000.0 + REPLY_S) - 1)
            return tuple(float(v) for v in ORACLE_ROWS[p])

        return predict

    return SimAdapter(
        protocol=agent_pb2.PROTOCOL_SBFT,
        simulator="sbft_sim",
        training_run=lambda seed: training_sim(np.random.default_rng(seed), seed),
        spec_run=spec_run,
        run_agent=run_agent,
        # run_agent's phase is the sim's, so a training schedule's phase 0 is
        # updated on; only decisions before 10 s are not.
        run_training=run_agent,
        to_timeout=to_timeout,
        from_timeout=from_timeout,
        # Sim phase 0 is the spec's clean warm-up.
        row_phase=lambda row: row[0] - 1 if row[0] >= 1 else -1,
        row_reward=lambda row: row[3],
        spec_phase_count=len(spec_phases()),
        oracle=oracle,
    )


def main() -> None:
    # Both are read by the forked generate() workers; declared here because argparse
    # reads TRAIN_RATE_TPS below for its defaults.
    global NO_PROPOSAL_FAULTS, TRAIN_RATE_TPS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="generate randomized episodes and fit the prior")
    t.add_argument("--runs", type=int, default=4000)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    t.add_argument("--out", default=DEFAULT_PRIOR_PATH)
    t.add_argument("--reward-design", default="reward_tps_latency")
    t.add_argument("--no-proposal-faults", action="store_true",
                   help="train on network-delay phases only; the prior never "
                        "sees a delayed or silent proposer")
    t.add_argument("--target-floor", type=float, default=-2.0)
    t.add_argument("--rate", type=float, default=TRAIN_RATE_TPS,
                   help=f"client rate (req/s) the load draw is centred on; each run draws "
                        f"load_tps over {LOAD_DRAW_FACTORS[0]}-{LOAD_DRAW_FACTORS[1]} x it "
                        f"(default {TRAIN_RATE_TPS:.0f}, the rate the CloudLab blocks run at)")
    t.add_argument("--hold-episodes", type=int, default=1,
                   help="fit the mean reward of holding each action this many episodes")
    t.add_argument("--save-data", default=None, help="also save the generated (x, y) here")
    t.add_argument("--hidden", default="128,128,64", help="MLP hidden layer sizes")
    t.add_argument("--policy-prior", default=None,
                   help="drive each run with the deployed model on this prior and sample its probes "
                        "(on-policy data); default: random actions")
    t.add_argument("--probe", type=float, default=0.35,
                   help="with --policy-prior: chance that a decision starts a held probe")
    f = sub.add_parser("fit", help="fit a prior from data saved by train --save-data")
    f.add_argument("--data", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--seed", type=int, default=1)
    f.add_argument("--hidden", default="128,128,64")
    f.add_argument("--target-floor", type=float, default=-2.0)
    c = sub.add_parser("check", help="fixed (T, W) pairs on the spec at the host constants")
    c.add_argument("--seeds", type=int, default=3)
    c.add_argument("--workers", type=int, default=1)
    c.add_argument("--mode", choices=("spec", "cal"), default="spec",
                   help="spec: the whole 1510 s spec per pair, phase interiors; cal: each phase alone "
                        "for 130 s from 10 s, episodes 3-5, as the calibration runs")
    c.add_argument("--pairs", default="ladder", help="ladder | all | T:W,T:W,...")
    c.add_argument("--out", default=None)
    e = sub.add_parser("eval", help="SimReturnGPModel against the oracle on the spec")
    e.add_argument("--seeds", type=int, default=3)
    e.add_argument("--prior", default=DEFAULT_PRIOR_PATH)
    e.add_argument("--rate", type=float, default=TRAIN_RATE_TPS,
                   help=f"client rate (req/s) to score the prior at (default "
                        f"{TRAIN_RATE_TPS:.0f}); check stays at {MEASURED_RATE_TPS:.0f}, the "
                        f"rate of the measured fixture it compares against")
    args = parser.parse_args()
    from reward_utils import set_reward_design

    if args.cmd == "train":
        NO_PROPOSAL_FAULTS = bool(args.no_proposal_faults)  # inherited by the forked workers
        if args.rate <= 0:
            raise ValueError("--rate must be positive")
        TRAIN_RATE_TPS = float(args.rate)  # likewise
        if not math.isfinite(args.target_floor):
            raise ValueError("SimReturnGPModel needs a finite --target-floor")
        check_model_spec()
        set_reward_design(args.reward_design)
        t0 = time.time()
        if args.hold_episodes < 1:
            raise ValueError("--hold-episodes must be at least 1")
        if args.policy_prior is not None and not 0.0 < args.probe <= 1.0:
            raise ValueError("--probe must be in (0, 1]")
        x, y = generate(args.runs, args.seed, args.workers, args.reward_design, args.hold_episodes,
                        args.policy_prior, args.probe)
        t_gen = time.time() - t0
        if args.save_data:
            np.savez(args.save_data, x=x, y=y, hold_episodes=np.asarray(args.hold_episodes),
                     reward_design=np.asarray(args.reward_design), runs=np.asarray(args.runs),
                     seed=np.asarray(args.seed),
                     policy_prior=np.asarray(args.policy_prior or ""), probe=np.asarray(args.probe),
                     **SIM_WINDOWS.arrays())
        hidden = tuple(int(v) for v in args.hidden.split(","))
        rmse = train_prior(x, y, args.out, args.seed, args.reward_design, args.target_floor,
                           args.hold_episodes, hidden, args.rate)
        print(f"samples={len(x)} input_dim={x.shape[1]} runs={args.runs} hold={args.hold_episodes} workers={args.workers} "
              f"rate={args.rate:.0f} load_draw={LOAD_DRAW_FACTORS[0] * args.rate:.0f}-{LOAD_DRAW_FACTORS[1] * args.rate:.0f} "
              f"gen_s={t_gen:.1f} reward_mean={y.mean():.4f} reward_std={y.std():.4f} "
              f"floored_std={np.maximum(y, args.target_floor).std():.4f} "
              f"floored_share={(y < args.target_floor).mean():.3f} holdout_rmse={rmse:.4f} -> {args.out}")
        return
    if args.cmd == "fit":
        if not math.isfinite(args.target_floor):
            raise ValueError("SimReturnGPModel needs a finite --target-floor")
        check_model_spec()
        data = np.load(args.data)
        check_prior_windows(args.data, SIM_WINDOWS)
        x, y = data["x"], data["y"]
        hidden = tuple(int(v) for v in args.hidden.split(","))
        t0 = time.time()
        rmse = train_prior(x, y, args.out, args.seed, str(data["reward_design"]), args.target_floor,
                           int(data["hold_episodes"]), hidden)
        print(f"fit samples={len(x)} hidden={hidden} seed={args.seed} fit_s={time.time() - t0:.0f} "
              f"holdout_rmse={rmse:.4f} -> {args.out}")
        return
    if args.cmd == "check":
        if args.pairs == "ladder":
            pairs = LADDER_PAIRS
        elif args.pairs == "all":
            pairs = LADDER_PAIRS + OFF_LADDER_PAIRS
        else:
            pairs = [tuple(int(v) for v in s.split(":")) for s in args.pairs.split(",")]
        print(run_check(args.seeds, args.workers, args.mode, pairs, args.out))
        return
    check_model_spec()
    print(run_eval(args.seeds, args.prior, args.rate))


if __name__ == "__main__":
    main()
