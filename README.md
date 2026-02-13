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
  --schedule-file learning_agent/examples/timeout_schedule.csv
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

TODO:

- Test feeding data to linear model
- Make systems.config generation script and read from `consensus.config` and `data_exchange.config` instead of `system.config`
- Try running end to end
