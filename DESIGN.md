# Design of the adaptive-timeout experiments

This document gives the background needed to write about this project: the problem, the systems and the fault model, how the learned timeout policies work, how the simulators and baselines are built and calibrated, and how runs are measured and compared. It deliberately contains no results; those are in `RESULTS.md`. Terms used throughout (election timeout T, fast-path wait W, proposal delay D, straggler S, hop model, storm) are defined in `CLAUDE.md`. Dates in parentheses mark when a design decision was taken. It describes the setup as of 2026-10-18.

## 1. Problem

Leader-based Byzantine fault-tolerant (BFT) protocols rely on timeouts for liveness. A replica that sees no progress within its timeout suspects the leader and starts a view change (or round change) to replace it. The timeout faces a trade-off:

- **Too short**: honest but slow rounds expire. On a high-latency link the protocol elects leader after leader without committing (a storm), losing throughput.
- **Too long**: a faulty or slow leader keeps its role. Under a leader that withholds or delays its proposals, every round pays the leader's delay, so latency rises with the timeout.

The right timeout therefore depends on conditions that change at run time: network delay, and whether the current leader misbehaves. Deployed systems use a fixed timeout, sometimes with exponential backoff. This project asks whether a learning agent running beside each replica can set the timeout online from the replicas' own performance reports, and how close it comes to an oracle that knows the fault schedule.

Three properties shape the design:

- **Separability.** Each fault phase is built so that a different timeout is best in it. A policy that never changes its timeout can't do well everywhere.
- **Observability.** The agent sees only what replicas measure about themselves: throughput, latencies, view changes, phase timings. Whether a fault can be detected depends on what the report contains (section 3.2).
- **Uniformity.** The same pipeline, reward, judge and calibration method apply to all six protocols, so the comparison across protocols is fair (Kevin, 2026-10-09).

## 2. Systems

Six protocols run as four replicas (f = 1) plus one load client, on five CloudLab machines (one per replica, one for the client):

| Protocol | Implementation | Timeout(s) the agent sets | Load |
|---|---|---|---|
| PBFT | CppBedrock | election timeout T (request timer) | open loop, 20,000 req/s, 512 B |
| SBFT | CppBedrock | T, and the fast-path wait W | open loop, 20,000 req/s, 512 B |
| SmartBFT | Hyperledger Labs SmartBFT (smallbank example) | request timeout T | closed loop, 512 terminals |
| Tendermint | CometBFT | propose timeout (prevote and precommit follow as T/3) | open loop, 5,000 tx/s, 512 B |
| HotStuff | Rust autobahn fork, HotStuff baseline | pacemaker timeout | open loop, 105,000 tx/s, 512 B |
| Autobahn | Rust autobahn fork | slot timeout (fast-path wait fixed at 50 ms) | open loop, 140,000 tx/s, 512 B |

- **Offered rates.** These started from a base-capacity rule: 0.8 x the highest rate the base network sustains, rounded down to 5,000 (2026-10-11). HotStuff and Autobahn use the rule rates. For PBFT, SBFT and Tendermint the rule rates (25,000, 40,000 and 15,000) couldn't be sustained in the delay phases, so they were lowered to the rates above.
- **SmartBFT** runs closed loop, so its throughput is the number of terminals divided by latency.
- **Requests** are 512-byte no-ops everywhere (2026-10-11), so the protocols differ in consensus, not in application work.

**Engine changes.** The CppBedrock engines were changed for this project:
- Timer semantics: lowering the timeout dates deadlines from the change, not from each request's arrival.
- Several liveness defects were fixed, so that elections reflect the timeout and not bugs: view-change deadlock, backoff cap and backoff reset, stale batch bodies, relay flood.
- A replica whose request timer expires alone no longer leaves the view at once (which parked it for the rest of a run): it broadcasts a suspicion and leaves once f+1 replicas suspect the leader, as SmartBFT's complaint does (5c0a3cf, 2026-10-19). A suspicion stands until the view moves, as a PBFT ViewChange does, even if the request it was about executes later (de44bee, 2026-10-20): the first version let it lapse two intervals after it was sent, dated from receipt at each receiver, so a Suspect delayed 1.4 s in an overflowed queue was counted after its sender had dropped it, and the receiver left the view alone and sat out 91 s.
- The backup's request timer follows PBFT (relay-on-skip, 2026-10-20): it watches the oldest request the replica holds from its arrival and moves only when that request executes. The client broadcasts and each replica sheds on its own occupancy, so a backup can hold a request the primary never admitted; a backup relays such a request to the primary once per view, when a batch executes whose oldest request the backup admitted arrived later (the evidence that the primary has ordered past it), and times it from the relay. Nothing else moves the deadline, so a primary that orders everything late is replaced. Relaying on receipt, as PBFT's backup does, would send the primary every request three more times; the evidence-triggered relay sends one request per executed batch. The two earlier rules that excused a passed-by request on any execution (03708eb, 8157c1a) left every follower blind to a late primary after an overflow.
- **Backoff.** A lone suspicion re-arms the request timer at the timeout in force; it takes no backoff step. Leaving the view does (one's own f+1 suspicions, joining f+1 ViewChanges, or a NewView wait that expires), and from then the request timer and the NewView wait run at T x 2^backoff, capped at max(T, 30 s). The backoff returns to 0 once, in normal operation, a request that arrived after the new view's install executes, that is, once the inherited backlog has drained; the policy's T then applies again. So a timeout below a phase's floor does not stall: from the lowest rung (250 ms) the 5+150 need (about 620 ms for PBFT, 900 ms plus W for SBFT's slow path) is reached within two or three failed views, far below the cap, and the run shows as a storm (section 1), view after view with partial throughput, until the policy raises T. These rules are the engine's, not the agent's; the simulators model them.
- SmartBFT's smallbank example reports a phase field added on 2026-10-18 (section 3.2).
- The autobahn fork counts transactions rather than batch digests.
- **HotStuff and Autobahn pacemakers** have no backoff: every reset of the round timer (a timeout, a timeout certificate, a round or slot advance) takes the agent's current timeout, so consecutive timeouts run at the same delay, bounded only by the action range (100-3000 ms). A timeout below the 5+150 floor (about 530 ms on HotStuff, 1000 ms on Autobahn) therefore storms round after round until the policy raises it; the oracle's ladder and the heuristic's signal do, and that is what the 5+150 phases test.

## 3. The control loop

### 3.1 Episodes

Time is divided into 20 s episodes, identical across protocols (Kevin, 2026-10-09):

| Window | Length | Role |
|---|---|---|
| feature window | 7 s | its report is what the agent decides from |
| reply wait | 2 s | the agent decides; the new timeout applies at the end |
| warm-up | 4 s | lets the system settle under the new timeout |
| reward window | 7 s | its report scores the decision |

Replicas, agents, fault injector and client share one start time, so episode boundaries, fault onsets and client measurements line up. An episode whose apply instant falls at a fault onset has a feature window from the previous phase and a reward window in the new one. So a policy that reacts to what it sees always enters a new phase one episode late. Only the oracle, which knows the schedule, switches exactly at the onset.

### 3.2 Reports

Each replica measures itself over each window and sends a signed report. The core fields are shared:
- committed transactions and consensus instances, and throughput;
- consensus latency (mean, p50, p95);
- batch sizes and inter-commit gaps;
- leader changes and view changes, and the timeout in force.

Tendermint reports the same latency, batch and gap statistics plus its round-timeout violation rate, and has no view-change counters (a round change is not an election). Autobahn has no regency field. The view-change count is the number of elections in the trailing episode period (20 s) ending at the report on CppBedrock and SmartBFT, so a leader re-pin that lands between windows is still counted; HotStuff and Autobahn count round timeouts per window (2026-10-19, restart D14). What "consensus latency" measures differs by engine:

| Protocol | Consensus latency | Extra fields |
|---|---|---|
| PBFT | oldest request's arrival at the replica to execution, per batch | mean latency of the pre-prepare, prepare and commit phases |
| SBFT | as PBFT | SBFT phase latencies, W, fast/slow-path counts, the leader's measured gap between the (2f+1)-th and the n-th prepare (the 3rd and 4th with n = 4) |
| SmartBFT | each request's submission at the replica to delivery | proposal-to-decision time at followers |
| Tendermint | start of the height's first round to commit | propose, prevote and precommit step latencies; share of heights needing more than one round |
| HotStuff | block first seen to commit | the next leader's block receipt to its QC (proposal_to_qc_ms) |
| Autobahn | header first processed to commit | a replica's first receipt of another replica's Prepare to the slot's commit (prepare_to_commit_ms) |

**Why phase fields matter.** A leader that holds its proposal adds the hold to every request's consensus latency, but the hold happens before the proposal is sent. So phase latencies measured from the proposal's arrival don't include it. A slower link, by contrast, raises both. Reporting both lets a policy tell "the leader is slow" (lower the timeout to depose it) from "the network is slow" (raise the timeout to avoid storms).

Without a phase field the two can look identical. On SmartBFT before 2026-10-18, a 300 ms hold on a +50 ms link and a +150 ms link with no hold produced the same values in every field at a high timeout:

| field | hold on +50 | +150 link |
|---|---|---|
| latency (ms) | 971 | 976 |
| throughput (req/s) | 512 | 512 |
| commit gap (ms) | 489 | 489 |
| view changes | 0 | 0 |

A policy then can't learn to depose the leader in one and keep it in the other. The follower-measured proposal-to-decision time (112 ms against 306 ms in that example) resolves it.

The protocols differ in how a hold shows up:
- PBFT and SBFT report phase latencies (added to PBFT's model inputs on 2026-10-14).
- Tendermint's propose step contains the hold directly.
- HotStuff and Autobahn report a field the hold never enters (proposal_to_qc_ms, prepare_to_commit_ms; models read it since 2026-10-19).

**Recovery, state transfer and catch-up.** A replica that fell behind (a pause, an exile, a view it sat out) commits a backlog in a burst when it catches up. Counted naively, that burst is a throughput spike with either no latency or a very long one, and a window of silence precedes it. What each engine reports (2026-10-21):
- **PBFT and SBFT**: a recovery response carries a certified checkpoint snapshot plus committed proofs for at most one consensus window (16 sequences) past it. The snapshot is installed without executing the batches it covers, so they produce no consensus sample: neither throughput nor latency. Requests the snapshot proves executed are answered and dropped from the pool. The proofs past the checkpoint execute through the normal path and count: their transactions enter the throughput of the window in which they execute; a batch whose proposal this replica accepted live keeps its arrival stamp and contributes its latency, a body fetched from a peer has no stamp and contributes none. The burst is therefore bounded by one window.
- **SmartBFT**: a state transfer replays the ordered decisions through the same application path as delivery, but the replay returns no sample to the learning manager, so nothing of it enters throughput or latency; the replayed requests are completed in the pool so they are not proposed again.
- **Tendermint**: a height's consensus latency is dated from this replica's own entry into round 0 of that height, so a height obtained without entering its rounds (block sync) contributes no latency sample; its transactions count.
- **HotStuff and Autobahn**: a block or header committed as an ancestor obtained by sync has no first-seen stamp, so it contributes no latency sample; its transactions count, in the window in which the chain commits.

So a catch-up burst inflates a replica's own throughput by at most the backlog that executes through the normal path, and never produces a short latency. In sharing mode it does not reach the model at all: the agent reads the 2f+1 median across replicas (section 3.3), one recovering replica is a minority of any batch, and the measured-only rule keeps a replica with no latency samples from dragging the latency median to zero. The burst still shows in that replica's own `local_feature_total_transactions` column, which the validator reads to flag a stuck replica. The judge is client-side (section 7.1), so the results do not depend on replica-side accounting of recovery either. In local mode the recovering replica's agent would see the burst as its own feature window, which is one reason the evaluation shares.

### 3.3 Data sharing: agreement among agents

Each replica has an agent of its own, on the replica's host. The agent runs in one of two modes.

- **Local** (sharing off): the agent decides from its own replica's report and trains on its own replica's reward. Nothing crosses hosts, and a faulty replica or agent can only affect the timeout played at its own replica. The replicas can then disagree about the timeout, and so about when to elect.
- **Sharing** (every learning row of the evaluation): the four agents agree on one set of 2f+1 reports per episode, aggregate it by one deterministic rule, and so play the same timeout. One replica's view can be incomplete or misleading (a replica that is itself overloaded sees high latency while the system is fine; a leader may delay proposals to some replicas only), and in learning mode the replicas must play the same timeout or they disagree about when to elect. The oracle and the fixed-timeout calibration runs skip the exchange: each agent plays the schedule.

**Episodes are wall-clock, not sequence-numbered.** Reports used to be grouped by the consensus sequence number at which a window ended, which made the agents agree on a window before they could agree on its reports. Now the harness fixes one start instant before launching anything and passes anchor = start + warm-up to every replica and agent. Episode e applies at anchor + P(e-1), P = 20 s; its feature window opens 9 s before the apply instant (7 s feature window, 2 s reply wait) and its reward window closes 11 s after it, where the next feature window opens (section 3.1). The episode number is thus a function of the shared clock, the same on every host, and needs no agreement. The hosts' clocks are chrony-synchronized to CloudLab's NTP server (offsets of microseconds on clnode381, checked 2026-10-21), and each replica logs its lag at the apply instant against its own clock: 0-1 ms over the PBFT oracle row of 2026-10-21, p95 1 ms and at most 86 ms over the SmartBFT GP row of 2026-10-20. Reports still carry the window's first and last sequence numbers (`start_tick`, `report_seq`) with `window_consensus_count` 0 marking a wall-clock window; the agent rejects a report whose end precedes its start and otherwise only echoes them. A replica that starts late joins at the first episode whose feature window has not opened yet, under the same numbering; one that overruns its slot drops that episode's reward (a near-empty reward window would score as a stall) and skips to the next episode whose window is still ahead.

**The exchange, per episode e.**
1. The replica sends its feature report for e, with the reward for e-1 attached, to its local agent. The agent checks the protocol, the window, and that e is neither older than its current episode nor sealed; stamps the report with its node id; signs it (ECDSA P-256 over the serialized report; off in the harness runs, see the limits below); stores it and broadcasts it to the three other agents. A peer verifies the signature when signing is on, drops reports for past or sealed episodes, and keeps only the first report a node sends for an episode (a follow-up exists only because the sender is still waiting for a decision, and first reports are the most time-aligned).
2. Holding reports from 2f+1 = 3 distinct nodes for e, the agent seals e, waits 100 ms more for a fourth, then submits the set (3 or 4 reports) to its local report-chain node.
3. The report chain is a four-node SmartBFT instance of its own, one node per replica host with its own write-ahead log, which orders one batch per episode: every node's submission for e carries the same request id (the episode number), so the first batch the chain orders is the agreed one and the others are evicted as duplicates. Every chain node validates a batch before voting on it: the protocol, at least 2f+1 reports, every report's episode equal to the batch's, no two reports from one node. A leader that stalls is replaced by the chain's own view change (the 424 batches of the SmartBFT GP row of 2026-10-20 were all ordered in view 0). Delivery is monotone in the episode number.
4. The chain delivers the agreed batch to every agent. Each agent aggregates it by the rules below, applies the agreed reward for e-1 to its model, runs the model on the agreed features and stores the decision for e. The replica polls for the decision during the reply wait and applies it at the apply instant; with no decision by then it keeps the timeout it has (it happened in none of the rows checked). A poll for e is answered with the newest decision at or after e, so a replica whose own chain node lagged catches up instead of waiting for an episode that will never be ordered.

Everything an agent learns from is the agreed batch: the same features, the same reward and the same model seed on every node (the learners of section 5.2 take no random exploration step), so the four decisions agree without any vote on the decision itself. A run's dataset therefore holds one row per episode, and only the replica's own `local_feature_total_transactions` column differs between the four copies. The 2 s reply wait is the budget for the whole exchange (broadcast, collect wait, ordering, delivery, the model). It runs once per 20 s episode, so its cost is decoupled from the request path.

**Aggregation.** A per-field median over the batch (integer fields rounded), which tolerates one faulty or lagging reporter. Fields that some replicas leave empty need their own rule:
- **Measured-only fields**: every latency, inter-commit gap, batch-size and hold-free field is the median over the reports that measured it (> 0), on all six protocols (2026-10-19, restart). With an empty value in a plain median, a field read zero exactly in storms (SmartBFT's follower-only proposal-to-decision time, 2026-10-18).
- **SBFT leader fields**: the leader-only prepare-gap fields come from the replica that led the most sequences (the lowest node id on a tie, so every agent picks the same one).
- **Reward**: the rewards for e-1 ride on the reports for e. They are applied only when at least f+1 reports of the batch carry a reward for the same episode and all of them name the same timeout as the one played (a row labelled with an action no node ran would mislabel the training), and the reward report is the median over those carriers.

**Faulty agents.** The fault model allows one faulty host (f = 1), and its replica, agent and report-chain node are faulty together: the three share a host and a network namespace and trust each other. The exchange tolerates that host as follows.
- *Silence, lateness, or a clock off by more than the reply wait*: a batch forms from any 2f+1 nodes, so the three honest reports suffice; a report arriving after the seal is dropped; the late replica still receives the decision, or keeps its timeout. NTP keeps the real skew far below the budget. A replica that is recovering (section 3.2) is handled the same way: silent while it catches up, a minority of the batch when it reports its burst.
- *Manipulated values (data poisoning)*: with at most f faulty reports among 2f+1 or more, the per-field median lies in the range of the honest values (of three values with one faulty, the median is an honest one; the median of four is the mean of the two middle values, both bounded by honest ones). A faulty node is counted once (duplicate node ids fail the chain's check, and an agent keeps one report per node), cannot replace its report after seeing the others' (first report wins), and with signing on cannot speak for another node. A manipulated reward is bounded the same way when all 2f+1 carriers report it.
- *Equivocation*: a faulty agent may send different reports to different peers, so peers hold different sets; the chain orders one batch and everyone aggregates that one.
- *Spoofed episodes*: a report for a far-future episode waits in its bucket and never reaches 2f+1 by itself; a report for a past or sealed episode is dropped.
- *A faulty report-chain leader* can pick which of the valid batches for e to order (each holds at least f+1 honest reports) or stall, in which case the chain's view change replaces it; it cannot order a batch with fewer than 2f+1 reports or with duplicate nodes, because every chain node re-checks the proposal.

**What the exchange does not cover.**
- Signing is implemented (per-node ECDSA keys) but switched off in the harness runs (`--signing false`), so the chain checks structure only and authenticity rests on the private overlay between the hosts. The evaluation injects no agent fault: the specification schema has `dirtyReport` (a node scales its report fields by random factors of 0.4-2.0 before sharing) and `ignoreReport` (a node submits exactly 2f+1 reports and drops the rest), and every evaluation specification leaves both empty. The robustness above is by construction, not measured.
- A median of two values is their mean. Where fewer than three reports measured a field, one faulty value moves the aggregate: SmartBFT's proposal-to-decision time is measured by followers only, so a three-report batch that includes the leader holds two measurements; SBFT's prepare-gap fields come from the one report claiming to have led the most sequences; a reward carried by only f+1 = 2 reports (one honest node overran and dropped its reward) is the mean of the two.
- A carrier naming a different timeout skips that episode's reward update on every node: a faulty node can withhold an episode of learning, though not poison it.

### 3.4 Reward

Every protocol uses the same reward: `R = throughput / N - mean consensus latency (s)` over the reward window, where N is the protocol's offered rate (SmartBFT 10,000). One percent of the offered rate trades for 10 ms of latency (Kevin, 2026-10-07, 10-09 and 10-11). A window in which nothing commits, or that commits without a latency sample, would report zero latency, so it is taken as if its latency were the full window length (7 s), in reward and feature windows on all six protocols (2026-10-19, restart D15), and a stall can never look fast.

## 4. Fault model

### 4.1 Faults

Each protocol has a fixed failure specification: a warm-up, then a sequence of phases, normally 300 s each. The fault types are:

- **Link delay**: extra one-way delay on every replica's outgoing traffic (tc netem), on top of a 5 ms base. The three link levels are base (5 ms one way), 5+50 and 5+150. Jitter is 0 in every specification (2026-10-19, restart D3; SmartBFT and Tendermint had 20% until then).
- **Leader proposal hold**: the leader holds each proposal for D ms before sending it. In CppBedrock and SmartBFT the hold follows whoever leads at each 60 s tick. After a deposition the next leader runs clean until the next tick, when the new leader is held. In Tendermint one fixed replica holds its proposals (every fourth height), long enough that they never arrive in time. In HotStuff and Autobahn one replica holds its proposals until just before its own timeout (T - X), so the hold grows with the timeout.
- **Straggler** (SBFT only): everything one replica sends is slowed by an extra S ms, its TCP acknowledgements and flow-control updates included, so its inbound bulk transfers slow down too. That stretches the gap between the (2f+1)-th and the n-th prepare (the 3rd and 4th here), which the fast-path wait W races against: the leader starts the wait at 2f+1 prepares and takes the fast path if all n arrive within it, for any n. At the step to S 450 (p6) the straggler can fall behind on receiving the leader's messages and expire alone. Until 2026-10-19 a lone expiry left the view, and a lone view change never gets f+1 support (standard PBFT), so the straggler stayed out of the view and SBFT lost the all-n fast path for the rest of the run (5 of 9 five-machine oracle draws, about -0.03 in p7); since 5c0a3cf a lone expiry only suspects the leader (section 2), the straggler keeps voting, and the fast path is lost only while its prepares arrive after the wait.
- **Vote withholding** (Autobahn only, all phases): one replica sends its votes 10 s late. Healthy Autobahn has no timeout floor (its fast path recovers from early timeouts). With one voter missing, every quorum needs all three honest replicas, which turns link delay into a soft floor.

Crash faults and corrupted agent reports are supported but not used in the current specifications.

### 4.2 Design principles

The specifications are designed so that each of three timeout rungs is the best choice in some phase:

1. **One rung per link level.** Each link level has a floor below which healthy rounds time out and the protocol storms. For PBFT and the SBFT fast path the floor follows the hop model, about 100 + 3h ms, where h is the one-way delay and 100 ms is the proposal interval. The SBFT slow path's floor is about 125 + 5h + W ms. The rung sits about 1.1x above the floor.
2. **Attacks punish timeouts that are too high.** A leader hold makes a timeout above it pay the hold on every round, while a timeout below it deposes the leader. Attacks are strongest on the base link, weaker at 5+50 and absent at 5+150, where only the link punishes timeouts that are too low (Kevin's attack-strength rule).
3. **A target winner pattern.**
   - PBFT and SmartBFT: phase 1 rung 1, phase 2 rung 1 or 2, phase 3 rung 2 or 3, phase 4 rung 2, phase 5 rung 3.
   - Tendermint, HotStuff and Autobahn: phase 1 rung 1, phase 3 rung 2, phase 4 rung 3.
   - Every specification ends with a return to the base link.
4. **The oracle never switches before a fault starts.** It may switch later, or play two rungs within a phase.

### 4.3 Phases per protocol

**PBFT** (rungs 250 / 550 / 1200 ms)

| Phase | Fault | Oracle | Why |
|---|---|---|---|
| p0 | base | 250 | all rungs tie |
| p1 | hold 300 ms | 250 | 250 deposes the held leader; 550 and 1200 keep it |
| p2 | hold 550 ms | 250 | 250 and 550 depose; 1200 pays the hold |
| p3 | +50 | 550 | 250 is close to the floor and loses on latency; 550 and 1200 tie |
| p4 | +50, hold 500 ms | 550 | 550 deposes; 1200 pays |
| p5 | +150 | 1200 | 550 storms |
| p6 | base | 250 | |

**SBFT** (T rungs 250 / 650 / 1600 ms x W 10 / 150 ms; straggler on one replica)

| Phase | Fault | Oracle (T, W) | Why |
|---|---|---|---|
| p0 | base | 250, 10 | |
| p1 | hold 300 | 250, 10 | 250 deposes |
| p2 | +50 | 650, 10 | 250 is below the floor |
| p3 | +50, straggler 60, hold 550 | 650, 150 | 650 deposes; W 150 catches the 60 ms straggler |
| p4 | +50, straggler 300, hold 350 | 650, 10 | no allowed W catches 300 ms, so the slow path at the smallest W wins |
| p5 | +150, straggler 60 | 1600, 150 | request ages at the onset reach 1100-1200 ms (the ladder's top rung; 1300 until 2026-10-19, restart D4) |
| p6 | +150, straggler 450 | 1600, 10 | healthy request ages reach about 1300 ms |
| p7 | base | 1600 for 20 s, then 250 | batches still in flight from 5+150 would trigger elections at 250 |

**SmartBFT** (rungs 300 / 800 / 1800 ms)

| Phase | Fault | Oracle |
|---|---|---|
| p0 | base | 300 |
| p1 | hold 300 | 300 (800 and 1800 keep the held leader) |
| p2 | hold 600 | 300 (1800 keeps the held leader) |
| p3 | +50 | 800 (300 storms) |
| p4 | +50, hold 300 | 800 (1800 keeps the held leader) |
| p5 | +150 | 1800 (only 1800 survives) |
| p6 | base | 300 (the rungs are within 0.03; Kevin chose 300) |

**Tendermint** (propose-timeout rungs 100 / 200 / 400 ms)

| Phase | Fault | Oracle |
|---|---|---|
| p0 | base | 100 |
| p1 | one proposer withholds (every 4th height) | 100 (the fastest escape from the withheld heights) |
| p2 | +50 | 200 (100 is below the floor) |
| p3 | +50, withholding proposer | 200 |
| p4 | +150 | 400 (every rung is overloaded; 400 delivers about 0.68 of the offered rate) |
| p5 | base | 100 |

**HotStuff** (rungs 100 / 300 / 700 ms; one replica holds until T - X)

| Phase | Fault | Oracle |
|---|---|---|
| p0 | base | 100 |
| p1 | hold until T - 100 | 100 (the hold vanishes at 100) |
| p2 | +50 | 300 |
| p3 | +50, hold until T - 400 | 300 (the hold vanishes at 300) |
| p4 | +150 | 700 (overloaded at every rung) |
| p5 | base | 100 |

**Autobahn** (rungs 250 / 500 / 1250 ms; one replica withholds votes throughout and holds proposals until T - X)

| Phase | Fault | Oracle |
|---|---|---|
| p0 | base | 250 |
| p1 | hold until T - 190 | 250 |
| p2 | +50 | 500 |
| p3 | +50, hold until T - 690 | 500 |
| p4 | +150 | 500 for 20 s, then 1250 (500 recovers the slots stalled at the onset faster) |
| p5 | base | 250 |

## 5. Policies compared

### 5.1 Action spaces

- **Discrete**: the three rungs (SBFT: three T rungs x two W values, six arms).
- **Continuous**: any timeout in 100-3000 ms, the same range for every protocol and every continuous model (2026-10-19, restart D10a; SmartBFT had 4000 and HotStuff 2000, and LinUCB and PPO spanned only the ladder). On SBFT every continuous model also chooses W in 10-500 ms (D10c): the GP over a (T, W) grid, LinUCB over a sum of a T basis (9 log-spaced centres) and a W basis (4 centres), PPO with a second Beta head for W. Until then LinUCB and PPO set W to T/8.
- **Opening**: every model, the heuristic and every engine start at the ladder's middle rung, and on SBFT at the minimum wait (D10b; the untrained GP opened mid-grid, above Tendermint's top rung, and the engines at 8000, 5000 or 300 ms).

### 5.2 The 26 configurations

Each protocol is evaluated on 26 configurations, one run each, on two sets of five machines that pull rows from one queue per protocol; the table shows the set of each row (2026-10-19, restart). The rows run in the order and under the oracle gate of section 7.2.

- **Oracle**: the schedule in section 4.3, the best rung per phase from calibration runs, switching at onsets.
- **Heuristic**: T = k x a latency signal from the report (section 6.3), a deterministic baseline built only from calibration knowledge.
- **24 learned configurations**, every combination of:
  - action space: continuous or discrete;
  - learner: GP, LinUCB or PPO;
  - start: pretrained (a prior learned in the protocol's simulator) or untrained;
  - latency floor: with or without.

**The learners:**
- **GP**: a Gaussian-process bandit over (report context, timeout). Its prior mean is a neural network trained in the simulator. Online, a GP residual corrects that prior from the run's own rewards, and the decision is the timeout with the best pessimistic estimate (mean minus one standard deviation). Its noise and signal constants are in reward units on every protocol. Only a few fields enter the GP's similarity measure (p50 latency, throughput, and on PBFT, SBFT and SmartBFT view changes); the prior reads all context fields. Fields that follow the model's own choice (HotStuff's and Autobahn's round-timeout counts, SBFT's path counts) are read by the prior only (2026-10-19, restart D9). No model reads the timeout in force, which only repeats its own last action (2026-10-19, restart D9a). The discrete GP scores only the rungs.
- **LinUCB**: a linear contextual bandit, over the arms or over a basis of timeouts, discount 0.95 and no random exploration. It fits rewards centred on their running mean (D9d). Its pretrained prior is a ridge regression fit to simulated experience.
- **PPO**: a policy-gradient learner, pretrained by on-policy training in the simulator, with no random exploration share.
- **Latency floor**: forbids timeouts below the part of the latency the timer must cover (section 6.4).

Every learner keeps learning online from its run's rewards.

## 6. Simulators and calibration

### 6.1 Simulators

Each protocol has an analytic simulator whose only job is to produce training experience for the priors. The simulators are built from each protocol's mechanics, not fit to the evaluation runs:
- **Common mechanics**: hop counts, batching, the proposal interval, timer rules, view-change cost, backoff.
- **Per protocol**:
  - SBFT models four replicas with separate timers, the fast and slow paths, stragglers and congestion windows.
  - SmartBFT models its closed-loop client.
  - Tendermint models round changes, a capped mempool and per-node timer spread.
  - HotStuff models TCP/Nagle bistability and its mempool cap.
  - Autobahn models lanes, slots and the effect of withheld votes.
- **Same interface as the real system.** Each simulator runs the same 20 s episodes and emits reports with the same fields and empty-window rule, so a policy reads simulated and real reports the same way.

**What the simulators don't model** is part of the design boundary. Examples:
- CppBedrock's overtaken re-arm in PBFT (one shared pool executes in arrival order, so it cannot fire there; SBFT models it);
- SmartBFT's state transfer, exile and split-vote deadlock;
- Tendermint's lock logic;
- Autobahn's mempool and the stall after the 5+150 onset.

**Training never uses the evaluation specification.** Each simulated training run draws its own constants and a random sequence of phases, from one recipe shared by all six (learning_agent/sim_common.py; 2026-10-19, restart D11):
- offered load 0.5-1.5 x N; the first fault at 40-400 s and 40-400 s between faults;
- extra link delay log-uniform 10-250 ms; a leader hold (or a relative hold's offset X) log-uniform 50-1500 ms; base delay 2-10 ms; no jitter;
- a quarter of the runs start at the opening action, the rest anywhere in the range; a backlog in 30% of runs where a simulator models one;
- every constant not fitted on the blind cells drawn 0.5-2x around a fitted or cited value.

Each simulator keeps its protocol's own fault kinds and mix: SBFT also draws stragglers and dead peers, Tendermint keeps a silent proposer (a hold around 10 s) in 40% of its delayed-proposer phases, Autobahn withholds votes in most phases. So the priors learn the general relationship between report, timeout and reward, rather than memorising the six phases.

### 6.2 Fitting the simulators: blind calibration

The constants that mechanics don't fix are fitted to measurements, such as per-request processing cost, transfer time, view-change cost and TCP effects. The method is the same for all six protocols:

1. **A blind specification.** It has the same fault kinds as the real specification but none of its values, by one rule for all six (scripts/cloudlab/tmp/cal/blindall/blind_rule.py, seed 1019; 2026-10-19, restart D2): extra link delays of 20-150 ms, every other magnitude (holds, relative-hold offsets, straggler lags) from 0.4x the smallest to 1.6x the largest real value, on a 5 ms step, draws of one kind stratified, and no draw within 10% of a real value; 60 s of base, then 120 s per fault phase, then 60 s of recovery. The fit therefore can't be tuned to the phases the policies are evaluated on.
2. **Constant-timeout runs** of the blind specification at 6-8 timeouts spanning the ladder, with one or two draws each. A cell is a scheduled-mode run whose schedule is one constant row (`-10,T` then `0,T`; for SBFT a (T, W) pair), so the agent only applies the CSV and the per-episode dataset is written; there is one such file per protocol and rung (60 for the single-timeout protocols, 36 for SBFT's T x W grid), and PBFT's p5/p6 re-measurement used prefix schedules that play the oracle up to the phase's onset and then hold a constant (section 6.2, step 6). These files are calibration tooling, not part of the experiment: they are untracked, and because every file under `learning_agent/config` enters the freeze hash they stay in place until the last row has run.
3. **Targets per (phase, timeout) cell**: mean throughput, median p50 and median mean latency, taken per window as the median over replicas, with the first episode of each phase dropped.
4. **Loss**: the mean over cells of `((tps_sim - tps) / (N/10))^2 + ln(latency ratio)^2 + ln(p50 ratio)^2`. Throughput errors are measured in tenths of the offered rate, the same rule for all six (2026-10-13).
5. **Search and install**: a random search of about 1,500 candidates per seed over several seeds. The candidate with the lowest blind loss is installed.
6. **Real-spec check**: the real specification, compressed to 120 s phases, is used only as a check: does the simulator rank the rungs per phase as the measurements do? Per phase, the regret is the measured score of the measured best rung minus that of the simulator's pick; a mean regret over the phases of at most 0.05 passes (restart S3). A phase whose outcome depends on the state a run brings into it is measured from the oracle's state, with prefix schedules (the oracle's rungs up to the phase, then the candidate), as the oracle itself is judged (section 6.5): PBFT's constant T 1200 cell carried its own p4 backlog (a held leader that T 1200 never deposes) into the 5+150 onset and scored -4.2 in p5 where the oracle's path scores +0.40, failing the check at 0.60 until p5 and p6 were re-measured that way (0.0055; 2026-10-21).

### 6.3 The heuristic baseline

The heuristic sets T = k x a latency statistic over recent committing windows, clamped to the action range:

| Protocol | Signal | k |
|---|---|---|
| PBFT | sum of the three phase latencies (minimum over 4 windows, hold-free) | 8.24 |
| SBFT | sum of the three phase latencies (minimum over 4 windows), with a slow-path lower bound and W from the measured prepare gap | 4.29 |
| SmartBFT | proposal-to-decision time at followers (hold-free) | 5.22 |
| Tendermint | p50 consensus latency | 0.53 |
| HotStuff | the next leader's block receipt to its QC (proposal_to_qc_ms, hold-free) | 4.29 |
| Autobahn | a replica's first receipt of another's Prepare to the slot's commit (prepare_to_commit_ms, hold-free) | 1.23 |

PBFT and SBFT were refitted on the restart-cal7 calibration of the fixed engine (2026-10-21, section 8); SBFT runs under restart-v2, PBFT under restart-v3 (its p5/p6 real-spec targets re-measured from the oracle's state before its ranking check passed).

Two extra rules:
- **Storms**: three or more view changes in a window double T, and T can't fall back until the latency estimate drops by a third.
- **Windows with no commits** (or commits but no measurement of the signal), once the run has committed: T = k x the feature window's length (7000 ms), clamped to the range, on all six (2026-10-19, restart D5/D15; such windows kept the timeout until then, and HotStuff's heuristic stalled through 5+150 at the timeout it brought). Windows before the first commit keep the opening timeout.

The signals follow the mechanics:
- The minimum over windows lets one healthy window reset an estimate inflated by an attack.
- SBFT's phase sum excludes a held leader's hold, so the heuristic deposes it (2026-10-14).
- Tendermint uses the median because its delayed proposer slows only every fourth height.

**Calibration of k.** k is fitted on the restart's blind cells (2026-10-20), on both the end-to-end and the hold-free signal where a protocol has both, and one signal is kept per protocol: the hold-free one unless the end-to-end one has the strictly better blind score (restart D6). Autobahn's end-to-end mean was ahead by 0.004, but Kevin chose its hold-free signal so that every protocol with one uses it (2026-10-20). It comes from the same blind runs:
1. For each candidate k, find where the rule would settle in each phase: the fixed point T = k x latency(T).
2. Score that point with the client judge.
3. Score each k by the mean over the blind phases (RUNSPEC answer 12; the median used until then let HotStuff and Tendermint fit k = 0.01, T 100 everywhere, since the phases T 100 breaks are fewer than half), with k at least the latency floor's share c when the signal is the floor's (section 6.4; it binds on no protocol).
4. Take the low edge of the plateau of k values within 0.01 of the best score, rounded up to 2 decimals.

The fit replays the rule on the 2f+1 median aggregate the live rule reads (SmartBFT's script since 2026-10-19, PBFT's and SBFT's since 2026-10-21; until then those two replayed replica 0's own report, and a run in which replica 0 alone idled for an episode stopped the fit).

For the four protocols of the first pass the two-signal margins are 0.001-0.004, inside the plateau tolerance, so the signal choice is decided by noise-sized differences. PBFT and SBFT were refitted on their restart-cal7 calibration (2026-10-21). PBFT: the hold-free phase sum, k 8.24, blind +0.756 and real-spec +0.723, against the p95 refit (k 0.22: +0.752, +0.555) and the previous p95 k 1.21 (+0.716, +0.734); Kevin chose the hold-free signal, as for Autobahn. SBFT (one signal, the phase sum): k 4.29, blind +0.581 and real-spec +0.486, against +0.490 and +0.362 for the previous 1.76; the jump comes from reading the aggregate, whose phase sum is smaller than replica 0's own. A p95 refit for SBFT (k 1.53) would score +0.546 blind and +0.510 real-spec, inside the margin Kevin accepts for keeping the hold-free signal. HotStuff's fitted k scores -0.130 on the real-spec check runs, against +0.274 for the previous 0.83: it lands T at 217-244 in the real spec's p2-p4 instead of about 300 (`scripts/cloudlab/tmp/cal/restart/s3/heur/SUMMARY.md`).

Link phases bound k from below and are well sampled, while attack phases bound it from above with few draws. The low edge stays clear of storms while leaning toward deposing held leaders. The constants never come from oracle or evaluation runs (Kevin, 2026-10-07/08).

### 6.4 The latency floor

The floor is a multiplier x a signal the leader's hold never enters, plus an offset, on the last committing window (2026-10-19, restart D8). Discrete learners can't play rungs below it, and continuous learners search only above it. The constants follow each protocol's mechanics, not a fit:

| Protocol | Floor | Why |
|---|---|---|
| PBFT, SBFT | 100 + 1.5 x (pre-prepare + prepare + commit latency) | the proposal interval plus 3 hops; the phase sum is 2 hops |
| SmartBFT | 3 x proposal-to-decision time | no pipelining: the in-flight decision plus its own, 3 hops each; the field is 2 |
| HotStuff | 2 x proposal-to-QC time | one round of three; the field is half a round |
| Autobahn | 1 x prepare-to-commit time | one slot of two; the field is the slot |
| Tendermint | 0.33 x p50 consensus latency | one step of three (unchanged) |

The floor removes timeouts that are certain to storm. Since it reads a hold-free signal, it never forbids the timeouts that would depose a held leader (the floor read the heuristic's signal until 2026-10-19, and SmartBFT's p95 latency carried the hold). On the archived oracle runs it never exceeds the oracle's timeout and it blocks the storming rungs. `--latency-floor-multiplier` scales the whole floor (1 as defined, 0 off).

### 6.5 Ladders and the oracle

- **Ladders** follow section 4.2: one rung just above each link level's floor. For CppBedrock the rungs track the hop model. For example, SBFT's middle rung is 650, because 550 sits exactly on the 5+50 slow-path bound with W 150.
- **Oracle schedules** come from constant-timeout runs of the compressed real specification, scored by the client judge, with at least two draws per rung (five for SBFT). Phases whose outcome depends on what came before were judged from the oracle's own state (prefix runs). The 20 s refinements at SBFT's p7 and Autobahn's p4 came from five-machine draws.
- **The oracle is restricted to the rungs.** A continuous policy can beat it in a phase whose best timeout lies between rungs.

### 6.6 Pretraining

All 30 priors (six protocols x the GP, discrete and continuous LinUCB, discrete and continuous PPO) are trained in one pass from the frozen code, on one recipe (2026-10-19, restart D11; scripts/cloudlab/tmp/cal/restart/pretrain_all.sh):
- **GP prior**: a network mapping (report context, timeout) to the reward, 4000 runs. Rewards are floored at -2, so collapses don't dominate the fit. Where the leader's hold is re-pinned every 60 s (PBFT, SBFT, SmartBFT) an action is held 3 episodes and the target is their mean reward, so the prior sees the held leader return; elsewhere 1 episode.
- **LinUCB prior**: a ridge regression on 4000 runs of an exploring behaviour policy.
- **PPO prior**: on-policy training over 2000 runs, without the floor; floor rows load the no-floor prior.

Priors record the protocol, their input columns, the action range their positions are normalised over, the episode windows, the reward definition and the code_hash of the tree that trained them, and won't load under different ones; the harness refuses a prior whose code_hash differs from FREEZE.

**Gate.** Before install every prior runs on the simulated real specification against the same model untrained and the simulated oracle, on 50 paired seeds. A prior fails when it is worse than untrained with a one-sided paired p < 0.05. A failing prior is reported with its rows, never retrained or tuned until it passes. SBFT's continuous PPO prior failed (pretrained +0.353 against untrained +0.399, p 0.000, the same under restart-v2 and restart-v3); its two pretrained rows run and carry the verdict in the table (Kevin, 2026-10-21). The 35 other priors pass.

## 7. Measurement and evaluation

### 7.1 The judge

Runs are scored from the client's own measurements, not from the agents' rewards:
- The client logs committed requests and the median latency each second.
- Each phase is scored from 20 s after its onset (skipping the transition) to the next onset, in 20 s chunks: `R = committed/s / N - latency (s)`.
- The latency is the commit-weighted mean of the per-second medians. A chunk with no commits scores 10 s.
- A phase's score is the mean of its chunks, and a run's score is the mean over phases.

The same judge picks the oracle's rungs and calibrates the heuristic, so every comparison uses one yardstick.

### 7.2 Runs

- **One run per configuration**: the 26 configurations above, each on the full specification on five machines. The oracle row is the latest run of its adopted schedule.
- **Order and the oracle gate** (Kevin, 2026-10-20): per protocol the oracle and the heuristic run first, then the 12 continuous rows, then the 12 discrete rows, and the discrete rows run only if the oracle is not beaten by the heuristic or a continuous row. Beaten by at most 0.05, the oracle reruns once (the row is always the latest draw); beaten by more, or still beaten after the rerun, the discrete rows are deferred and the fact is noted. SmartBFT's oracle (+0.362) was not beaten; Tendermint's (+0.587) was beaten by 0.003 and reran.
- **Finalized rows** live in `results/<protocol>/<run>/`, each with a `row.json` (configuration, host set, frozen code, prior, gate verdict, validator flags) written when the run was accepted; `results/RESULTS.md` and `results/plots/<protocol>.png` are generated from those directories alone.
- **Run length**: warm-up plus the last onset plus 300 s (about 30-40 minutes per run).
- **Validity**: a run counts when it ran to length at the adopted rate with 512 B requests, delivered at least three of the four agent datasets, applied the network faults correctly, and has a score for every phase.
- **Noise**: repeated runs of the same oracle schedule differ by up to about 0.01 in mean score and up to about 0.07 in a single phase. The single-phase spread is largest in phases that open with a stall or storm, and up to 0.025 in PBFT's base phases from run-to-run drift in base latency. Per-phase differences smaller than that aren't evidence either way.

### 7.3 Data collected

Every run keeps:
- each agent's per-episode dataset: every report field for the feature and reward windows, and the timeout applied;
- the client's per-second log;
- the run configuration (mode, rate, windows, model, prior, floor, code versions);
- the failure specification used;
- optionally, server, fault-injector and agent logs.

The datasets support offline analysis, for example scoring a prior's predictions against the reports a real run produced, without rerunning experiments.

## 8. Limitations and threats to validity

- **Single cluster, single draw.** All runs use two fixed sets of five CloudLab machines, with one run per configuration. Base latency drifts between runs (section 7.2).
- **The simulators are approximations.** They omit some engine rules and effects (section 6.1). For example, SmartBFT's simulated proposal-to-decision time reads a few ms high. Priors can only be as good as the simulator in the regimes that matter.
- **Calibration and evaluation ran on several freezes.** The four protocols of the first pass were calibrated on `restart-cal` and their rows ran on `restart-v1` (differing by the fitted values and by no model reading the timeout in force, D9a, which the fits do not use), then on `restart-v2` and `restart-v3` once PBFT and SBFT joined; every later freeze differs from the one before only in PBFT or SBFT files, the priors under each are deterministic repeats, and a row is accepted under the freeze it ran on (`ACCEPTED_FREEZES`, one list per protocol).
- **HotStuff and Autobahn attacks are relative to the timeout.** The hold shrinks to zero at the right rung, so the attack only punishes higher timeouts.
- **Observability is uneven.** Protocols whose reports separate a slow leader from a slow link (phase fields, step latencies, view-change effects) give learners information that SmartBFT lacked until 2026-10-18.
- **The oracle is the best rung, not the best timeout.** A continuous policy can beat it in a phase whose optimum lies between rungs.
- **PBFT and SBFT ran on a second freeze (Kevin, 2026-10-20).** Their engine changed after the four other protocols' calibration: a replica whose request timer expired alone left the view for good, parking it (in every SBFT T 1600 cell; on SBFT that also ends the fast path), and the fix that made a lone expiry only suspect the leader (5c0a3cf) exposed a second defect, a follower holding requests the leader shed at the pending cap never timing a late leader (PBFT T 800's held-leader phase fell from +0.89 to +0.14). Two fixes were rejected on measurement, one that unwatched every passed-by request at once (storms after view changes) and one that had the leader step down on its own expiry (election cascades on inherited backlogs); the PBFT relay rule of section 2 replaced them. A third change made a suspicion stand until the view moves (section 2). PBFT and SBFT were calibrated on `restart-cal7` (the relay rule and standing suspicions); SBFT's refit was installed as `restart-v2` and PBFT's, after its p5/p6 targets were re-measured from the oracle's state (section 6.2), as `restart-v3`. Each differs from the freeze before only in CppBedrock, the two CppBedrock simulators and the PBFT/SBFT heuristic constants, so the other protocols' rows are byte-identical in every file their code paths read. Under the PBFT rule a timeout below a phase's floor storms, as the definitions require, where the blind followers of 5c0a3cf had hidden it: the low-T calibration cells changed accordingly.
- **Agent faults are not injected.** The sharing exchange (section 3.3) tolerates one faulty host by construction (2f+1 report sets, medians, one ordered batch per episode), but no evaluation run exercises it: report signing is off in the harness and the `dirtyReport`/`ignoreReport` injections are empty in every specification. Its robustness is argued, not measured.
- **Known engine defects, not fixed (Kevin, 2026-10-19).** On SmartBFT a replica more than one sequence behind can be exiled, and a same-sequence 2-2 split can occur (both upstream). The validator flags parked and stuck replicas per row, and a run with a split is rerun.

## Appendix: where things live

| Topic | Location |
|---|---|
| Failure specifications | `config/<protocol>/failure_spec.xml` |
| Oracle schedules | `learning_agent/config/schedule_<protocol>.csv` |
| Calibration cell schedules (constant and prefix; untracked, inside the freeze hash) | `learning_agent/config/cal_<protocol>_T<n>.csv`, `sbcal_T<n>_W<n>.csv`, `pbpre5_*`, `pbpre6_*`, `sbpre5_*`, `sbpre6_*` |
| Ladders, report field lists | `learning_agent/models.py` |
| Heuristic, GP model, latency floor | `learning_agent/continuous_bandit.py`, `learning_agent/latency_floor.py` |
| Report formats | `proto/agent.proto` |
| Data sharing (agents, aggregation, report chain, replica episode cycle) | `learning_agent/main.py`, `learning_agent/aggregator.py`, `SmartBFT/examples/sharing/`, `CppBedrock/src/core/agent/AgentClient.cpp` (the SmartBFT, CometBFT and Rust adapters follow the same grid) |
| Simulators | `learning_agent/<protocol>_sim.py` |
| Pretraining | `learning_agent/pretrain.py` (GP priors: the simulators' `train` command) |
| Fault injection | `scripts/apply_network_failures.py`, plus each engine's proposal-delay controller |
| Harnesses | `scripts/cloudlab/run_<protocol>.sh` |
| Calibration and restart tooling (blind specs, fits, judge, worker, validator, freezes) | `scripts/cloudlab/tmp/cal/` (`blindall/`, `pt/judge.py`, `restart/`) |
| Finalized rows, results table and plots | `results/<protocol>/`, `results/results_md.py`, `results/plot.py` (rows accepted by `scripts/cloudlab/tmp/cal/restart/finalize.py`) |
| Decision logs | `runs/<date>-<topic>/` |
