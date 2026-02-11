## Learning Agent
This learning agent signs local reports with ECDSA. In `data-share` mode it broadcasts to peers, collects reports until it reaches `2f+1` (out of `3f+1`) for an episode, waits briefly for late arrivals, and submits a batch to the consensus service. When consensus returns, the agent takes the median report and feeds it into the model to generate a timeout recommendation. In `local-only` mode it skips sharing/consensus and predicts locally.

### Generate Keys
Keys are required before starting agents. This writes PEM keys under `config/keys` based on `config/hosts.config`.

```bash
bash generate_keys.sh
```

Use `--force` to overwrite existing keys.

### Run Agents
Start one agent per node id listed in `config/hosts.config`:

```bash
python learning_agent/main.py --node-id 0
python learning_agent/main.py --node-id 1
```

To run in local-only mode:

```bash
python learning_agent/main.py --node-id 0 --mode local-only
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
