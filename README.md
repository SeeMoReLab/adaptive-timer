## Learning Agent

This learning agent signs local reports with ECDSA. With `--sharing true` it broadcasts to peers, collects reports until it reaches `2f+1` (out of `3f+1`) for an episode, waits briefly for late arrivals, and submits a batch to the consensus service. When consensus returns, the agent takes the median report and feeds it into the model to generate a timeout recommendation. With `--sharing false` it skips sharing/consensus and predicts locally.

### Generate Keys

Keys are required before starting agents. This writes PEM keys under `config/keys` based on `config/hosts.config`.

```bash
bash generate_keys.sh
```

Use `--force` to overwrite existing keys.
If `--key-dir` is omitted, the agent uses the built-in default key pair. Signing is enabled by default and can be disabled with `--signing false`.

### Run Agents

Start one agent per node id listed in `config/hosts.config`:

```bash
python learning_agent/main.py --node-id 0
python learning_agent/main.py --node-id 1
```

To run with sharing enabled:

```bash
python learning_agent/main.py --node-id 0 --sharing true
```

To use a scheduled timeout model from a CSV (`time_s,timeout_ms`):
Allowed timeout values are `200`, `500`, and `2000` milliseconds.

```bash
python learning_agent/main.py \
  --node-id 0 \
  --sharing false \
  --model-type scheduled \
  --schedule-file learning_agent/config/schedule.csv
```

To log reward-labeled samples for offline train/test (generic, any model type):

```bash
python learning_agent/main.py \
  --node-id 0 \
  --dataset-csv scripts/tmp/reward_dataset_node0.csv
```

The dataset contains one completed row per episode: decision-time metrics use
the `feature_` prefix, reward-window metrics use `reward_`, and `reward` is the
scalar supplied to the model.

### Send a Report

The client sends a randomized report for the chosen node and episode. It reads the target address from `config/hosts.config`. Use `GetTimeout` to poll for status.

```bash
python learning_agent/client.py --node-id 0 --episode 5
```

You can pass `--seed` to make the random report deterministic.

### Regenerate Protobufs

If you change `proto/agent.proto`, regenerate Python stubs:

```bash
bash compile_proto.sh
```

### Run Agent Simulation

To generate synthetic dataset from the observed environment dataset:

```bash
python learning_agent/simulator.py \
  --protocol smartbft \
  --model cmab \
  --data-dir learning_agent/data/smartbft_wallclock \
  --num-episodes 500 \
  --phases "5;0;0;5;1;2;3;4;5;3;0;5;0;3;4;2;1;5;4;0" \
  --experiment-name full-sequence \
  --verbose \
  --reward-scaler quadratic \
```

With $N$ agent observations located in ./data/smartbft_wallclock, each agent will have 1,000 episodes for each phase.

## CppBedrock (PBFT, SBFT) Run

The CppBedrock replicas (`CppBedrock/` submodule) run under the same CloudLab harness shape as SmartBFT and the Rust protocols: per-replica network namespaces, one network fault applier per replica, one learning agent per replica, a synchronized start time, and a benchmark client on node0 that prints per-second Monitor lines. `scripts/cloudlab/setup.sh` installs the build dependencies (cmake, gRPC, protobuf, yaml-cpp, nlohmann-json, OpenSSL).

```bash
bash scripts/cloudlab/run_pbft.sh ~/.ssh/<cloudlab_private_key> normal -t 300 --rate 1000 --log
bash scripts/cloudlab/run_sbft.sh ~/.ssh/<cloudlab_private_key> scheduled -t 600 --rate 1000 --log --dataset
```

Both runners source `scripts/cloudlab/lib_bedrock.sh`; `run_pbft.sh --help` lists the options (client target mode, client and proposal signing, initial election and slow-path timeouts, learning windows). The client broadcasts every request to every replica by default so that each replica runs the PBFT request timer over it; a leader that fails to get a request executed within the election timeout is replaced through a PBFT view change (see `CppBedrock/adaptive_timers.MD`). Failure specs live in `config/pbft/failure_spec.xml` and `config/sbft/failure_spec.xml`: the `<network>` section is applied by `scripts/apply_network_failures.py`, the `<pbft>` or `<sbft>` section is read by the replicas for proposal-delay injection (same semantics as the SmartBFT and Rust controllers, including `interval` and the `leader` window), and the `<agent>` section by the learning agents. Schedules for scheduled mode are `learning_agent/config/schedule_pbft.csv` (`time_s,timeout_ms`) and `schedule_sbft.csv` (`time_s,election_timeout_ms,slow_path_timeout_ms`). SBFT reports which requests took the fast path and which the slow path (the slow-path timeout is the second knob the agent tunes).

Timeout sweeps in scheduled mode, one CloudLab run per value with combined agent datasets, use `scripts/cloudlab/run_pbft_sweep.sh <key> -d <seconds> -t '300;1000;3000'` and `run_sbft_sweep.sh <key> -d <seconds> -t '1000;3000' -s '20;100'` (every election and slow-path timeout pair).

Outputs land in `scripts/cloudlab/tmp/runs/<protocol>/<run_id>/` (`<protocol>_output.txt`, `client.log`, `committee.json`, `run.env`, and with `--log` the server, applier, and agent logs). Plot a run with:

```bash
python3 scripts/cloudlab/plot_pbft.py --input scripts/cloudlab/tmp/runs/pbft/<run_id> --output plot.png
```

For a local loopback run without the harness see `CppBedrock/adaptive_timers.MD`.

## SmartBFT Run

These scripts run SmallBank SmartBFT with:
- per-replica network namespaces,
- one network fault applier process per replica (`scripts/apply_network_failures.py`),
- synchronized start time for client and failure schedule.

Failure spec source is `config/smartbft/failure_spec.xml` (packaged to `library/config/failure_spec.xml`).

### CloudLab

1. Extract SSH targets from manifest (example username: `myuser`):

```bash
python scripts/cloudlab/extract_ssh_targets.py \
  --manifest scripts/cloudlab/cloudlab.manifest \
  --username myuser
```

This writes `scripts/cloudlab/tmp/cloudlab_ssh_targets.txt` by default.

2. Install dependencies and collect CloudLab node IP mapping:

```bash
bash scripts/cloudlab/setup.sh ~/.ssh/<cloudlab_private_key>
```

This writes `scripts/cloudlab/tmp/cloudlab_ip.txt`.

3. Run SmartBFT:

```bash
bash scripts/cloudlab/run_smartbft.sh ~/.ssh/<cloudlab_private_key> learning --model cmab -t 3610
```

Outputs:
- local collected client output: `scripts/cloudlab/tmp/output.txt`
- remote per-replica applier logs: `~/library/applier_<replica_id>.log` on server nodes

Notes:
- CloudLab run also requires passwordless `sudo -n` on target nodes (for namespaces/routes/tc).
- The runner handles namespace setup, per-replica applier launch, server/client launch, and teardown.
