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
  --protocol pbft \
  --model cmab \
  --data-dir learning_agent/data/pbft \
  --num-episodes 500 \
  --phases "5;1;2;3;4;5;0;2;1;5;4;2;1" \
  --verbose
```
With $N$ agent observations located in ./data/pbft, each agent will have 1,000 episodes for each phase.

## PBFT Run

These scripts run SmallBank PBFT with:
- per-replica network namespaces,
- one network fault applier process per replica (`scripts/apply_network_failures.py`),
- synchronized start time for client and failure schedule.

Failure spec source is `config/bft-smart/failure_spec.xml` (packaged to `library/config/failure_spec.xml`).

### Local (Single Machine)

Requirements:
- Ubuntu with `sudo -n` available (passwordless sudo),
- `tmux`, `python3`, `iproute2`, `tc`,
- Java/Gradle toolchain for `scripts/prepare_pbft.sh`.

Run:

```bash
bash scripts/local/run_pbft.sh -f 1 -t 60
```

Optional mode (currently informational only):

```bash
bash scripts/local/run_pbft.sh normal -f 1 -t 120
```

Outputs:
- client output: `scripts/local/tmp/output.txt`
- staged runtime library: `scripts/local/tmp/library`
- per-replica applier logs: `scripts/local/tmp/library/applier_<replica_id>.log`

The script always tears down tmux sessions and namespaces on exit.

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

3. Run PBFT:

```bash
bash scripts/cloudlab/run_pbft.sh ~/.ssh/<cloudlab_private_key> -f 1 -t 60
```

Optional mode (currently informational only):

```bash
bash scripts/cloudlab/run_pbft.sh ~/.ssh/<cloudlab_private_key> normal -f 1 -t 120
```

Outputs:
- local collected client output: `scripts/cloudlab/tmp/output.txt`
- remote per-replica applier logs: `~/library/applier_<replica_id>.log` on server nodes

Notes:
- CloudLab run also requires passwordless `sudo -n` on target nodes (for namespaces/routes/tc).
- The runner handles namespace setup, per-replica applier launch, server/client launch, and teardown.

TODO:

- Test feeding data to linear model
- Make systems.config generation script and read from `consensus.config` and `data_exchange.config` instead of `system.config`
- Try running end to end
