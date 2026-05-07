#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

COMETBFT_SRC="${REPO_ROOT}/cometbft"
LEARNING_AGENT_DIR="${REPO_ROOT}/learning_agent"
SCHEDULE_TENDERMINT="${LEARNING_AGENT_DIR}/config/schedule_tendermint.csv"
FAILURE_SPEC="${REPO_ROOT}/config/tendermint/failure_spec.xml"
MASTER_CONFIG="${REPO_ROOT}/config/tendermint/config.toml"

MODE="normal"
F=1
EXPERIMENT_TIME_S=60
LOG_ENABLED=0
DATASET_ENABLED=0
AGENT_MODEL_TYPE="random"
AGENT_MODEL_SET=0

LOCAL_TMP_DIR="${SCRIPT_DIR}/tmp"
RUN_ROOT="${LOCAL_TMP_DIR}/tendermint_local"
OUTPUT_LOCAL="${LOCAL_TMP_DIR}/tendermint_output.txt"
LOGS_LOCAL_DIR="${LOCAL_TMP_DIR}/tendermint_logs"

LOAD_CONNECTIONS=5
LOAD_RATE=10
LOAD_SEND_PERIOD_MS=100
LOAD_SIZE=500
POST_PEER_CONNECT_DELAY=5
START_ALIGN_DELAY_MS=15000

usage() {
  cat <<EOF
Usage: $0 [normal|learning|sharing|scheduled] [--dataset] [--log] [--model <random|linear|cmab>] [-f <fault_tolerance>] [-t <experiment_seconds>]

Arguments:
  mode              Optional; one of normal|learning|sharing|scheduled (default: normal).

Flags:
  --dataset         Collect per-replica learning-agent datasets as scripts/local/tmp/agentN.csv.
  --log             Keep per-node and client logs under scripts/local/tmp/tendermint_logs/.
  --model           Learning-agent model for learning/sharing modes: random|linear|cmab (default: random).
  -f <int>          Fault tolerance f. Number of nodes is derived as 3f+1 (default: 1).
  -t <int>          Load-generation duration in seconds (default: 60).
EOF
}

mode_enables_learning_agent() {
  case "$1" in
    learning|sharing|scheduled)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

normalize_agent_model_type() {
  local model_lower
  model_lower="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  case "${model_lower}" in
    random|linear|cmab)
      printf '%s\n' "${model_lower}"
      ;;
    *)
      return 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -f)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for -f." >&2
        usage >&2
        exit 2
      fi
      F="${2}"
      shift 2
      ;;
    -t)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for -t." >&2
        usage >&2
        exit 2
      fi
      EXPERIMENT_TIME_S="${2}"
      shift 2
      ;;
    --log)
      LOG_ENABLED=1
      shift
      ;;
    --dataset)
      DATASET_ENABLED=1
      shift
      ;;
    --model)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --model." >&2
        usage >&2
        exit 2
      fi
      AGENT_MODEL_TYPE="${2}"
      AGENT_MODEL_SET=1
      shift 2
      ;;
    normal|learning|sharing|scheduled)
      MODE="$1"
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "${F}" =~ ^[0-9]+$ ]] || (( F < 1 )); then
  echo "Invalid -f value: ${F}. It must be an integer >= 1." >&2
  exit 2
fi
if ! [[ "${EXPERIMENT_TIME_S}" =~ ^[0-9]+$ ]] || (( EXPERIMENT_TIME_S < 1 )); then
  echo "Invalid -t value: ${EXPERIMENT_TIME_S}. It must be an integer >= 1." >&2
  exit 2
fi
if [[ "${MODE}" != "normal" && "${MODE}" != "learning" && "${MODE}" != "sharing" && "${MODE}" != "scheduled" ]]; then
  echo "Unsupported mode: ${MODE}. Supported: normal, learning, sharing, scheduled." >&2
  exit 2
fi
if ! mode_enables_learning_agent "${MODE}"; then
  LEARNING_AGENT_ENABLED=0
else
  LEARNING_AGENT_ENABLED=1
fi
if ! AGENT_MODEL_TYPE="$(normalize_agent_model_type "${AGENT_MODEL_TYPE}")"; then
  echo "Invalid --model value: ${AGENT_MODEL_TYPE}. Use one of random|linear|cmab." >&2
  exit 2
fi
if (( AGENT_MODEL_SET == 1 )) && [[ "${MODE}" != "learning" && "${MODE}" != "sharing" ]]; then
  echo "--model is only supported with learning or sharing mode." >&2
  exit 2
fi
if (( DATASET_ENABLED == 1 )) && (( LEARNING_AGENT_ENABLED == 0 )); then
  echo "--dataset requires a learning-agent mode: learning|sharing|scheduled." >&2
  exit 2
fi
if [[ ! -d "${COMETBFT_SRC}" ]]; then
  echo "Missing cometbft directory: ${COMETBFT_SRC}" >&2
  exit 1
fi
if [[ ! -f "${FAILURE_SPEC}" ]]; then
  echo "Missing failure spec: ${FAILURE_SPEC}" >&2
  exit 1
fi
if [[ ! -f "${MASTER_CONFIG}" ]]; then
  echo "Missing master config: ${MASTER_CONFIG}" >&2
  exit 1
fi
if (( LEARNING_AGENT_ENABLED == 1 )); then
  if [[ ! -f "${LEARNING_AGENT_DIR}/main.py" ]]; then
    echo "Missing learning agent main: ${LEARNING_AGENT_DIR}/main.py" >&2
    exit 1
  fi
fi
if [[ "${MODE}" == "scheduled" ]]; then
  if [[ ! -f "${SCHEDULE_TENDERMINT}" ]]; then
    echo "Missing Tendermint schedule file: ${SCHEDULE_TENDERMINT}" >&2
    exit 1
  fi
fi

for cmd in go tmux curl python3 awk sed; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Required command not found: ${cmd}" >&2
    exit 1
  fi
done

num_nodes=$((F * 3 + 1))
run_root="${RUN_ROOT}"
src_dir="${COMETBFT_SRC}"
testnet_dir="${run_root}/mytestnet"
logs_dir="${run_root}/logs"
output_file="${run_root}/tendermint_output.txt"
results_csv="${run_root}/results.csv"
client_log="${logs_dir}/client.log"
agent_hosts_config="${run_root}/agent_hosts.config"
schedule_file="${SCHEDULE_TENDERMINT}"
buildvcs_flag="-buildvcs=false"

p2p_port() { echo $((26656 + $1 * 3)); }
rpc_port() { echo $((26657 + $1 * 3)); }
abci_port() { echo $((26658 + $1 * 3)); }
agent_port() { echo $((50000 + $1)); }
load_endpoints_csv() {
  local endpoints=()
  local i rpc_p
  for ((i = 0; i < num_nodes; i++)); do
    rpc_p="$(rpc_port "${i}")"
    endpoints+=("ws://127.0.0.1:${rpc_p}/websocket")
  done
  local IFS=,
  echo "${endpoints[*]}"
}

sed_in_place() {
  local expr="$1"
  local file="$2"
  if [[ "${OSTYPE:-}" == darwin* ]]; then
    sed -i '' "${expr}" "${file}"
  else
    sed -i "${expr}" "${file}"
  fi
}

apply_master_config() {
  local master_config="$1"
  local node_config="$2"
  python3 - "$master_config" "$node_config" <<'PY'
import re
import sys

master_path = sys.argv[1]
node_path = sys.argv[2]

section_re = re.compile(r'^\s*\[([^\]]+)\]\s*$')
assign_re = re.compile(r'^([A-Za-z0-9_.-]+)\s*=\s*(.*)$')
line_assign_re = re.compile(r'^(\s*)([A-Za-z0-9_.-]+)(\s*=\s*)(.*?)(\s*(?:#.*)?)\n?$')

def collect_assignments(path):
    assignments = {}
    section = ""
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            sm = section_re.match(line)
            if sm:
                section = sm.group(1).strip()
                continue
            am = assign_re.match(line)
            if am:
                key = am.group(1)
                value = am.group(2)
                assignments[(section, key)] = value
    return assignments

master_values = collect_assignments(master_path)

out_lines = []
section = ""
with open(node_path, "r", encoding="utf-8") as fh:
    for raw in fh:
        stripped = raw.strip()
        sm = section_re.match(stripped)
        if sm:
            section = sm.group(1).strip()
            out_lines.append(raw)
            continue

        lm = line_assign_re.match(raw)
        if not lm:
            out_lines.append(raw)
            continue

        indent, key, equals, _, trailing = lm.groups()
        replacement = master_values.get((section, key))
        if replacement is None:
            out_lines.append(raw)
            continue
        out_lines.append(f"{indent}{key}{equals}{replacement}{trailing}\n")

with open(node_path, "w", encoding="utf-8") as fh:
    fh.writelines(out_lines)
PY
}

kill_run_sessions() {
  sessions="$(tmux ls -F '#S' 2>/dev/null || true)"
  if [[ -z "${sessions}" ]]; then
    return 0
  fi
  while read -r session_name; do
    [[ -z "${session_name}" ]] && continue
    if [[ "${session_name}" == ll_tm_tendermint_node_* || "${session_name}" == ll_tm_abci_* || "${session_name}" == ll_tm_tendermint_agent_* ]]; then
      tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
    fi
  done <<< "${sessions}"
}

collect_run_pids() {
  ps ax -o pid= -o command= | awk -v root="${testnet_dir}" '
    /cometbft node/ && index($0, root "/node") { print $1 }
  '
  for ((k = 0; k < num_nodes; k++)); do
    local port
    port="$(abci_port "${k}")"
    ps ax -o pid= -o command= | awk -v addr="tcp://127.0.0.1:""${port}" '
      /abci-cli kvstore/ && index($0, addr) { print $1 }
    '
  done
}

force_stop_run_processes() {
  local pids
  pids="$(collect_run_pids | awk '!seen[$1]++')"
  if [[ -z "${pids}" ]]; then
    return 0
  fi

  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    kill "${pid}" >/dev/null 2>&1 || true
  done <<< "${pids}"

  for _ in $(seq 1 6); do
    sleep 1
    pids="$(collect_run_pids | awk '!seen[$1]++')"
    [[ -z "${pids}" ]] && return 0
  done

  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    kill -9 "${pid}" >/dev/null 2>&1 || true
  done <<< "${pids}"
}

run_report_with_retry() {
  local max_attempts=30
  local attempt=1
  local tmp_out=""
  while true; do
    tmp_out="$(mktemp)"
    if report "$@" >"${tmp_out}" 2>&1; then
      cat "${tmp_out}"
      rm -f "${tmp_out}"
      return 0
    fi

    if grep -q "resource temporarily unavailable" "${tmp_out}"; then
      if (( attempt == 1 )); then
        echo "Report DB is still locked; waiting for node shutdown to complete..."
      fi
      rm -f "${tmp_out}"
      if (( attempt >= max_attempts )); then
        echo "Report failed after ${max_attempts} attempts waiting for DB lock release."
        return 1
      fi
      sleep 1
      attempt=$((attempt + 1))
      continue
    fi

    cat "${tmp_out}"
    rm -f "${tmp_out}"
    return 1
  done
}

append_node_phase_commit_summary() {
  local out_file="$1"
  local node_logs_dir="$2"
  local nodes="$3"
  python3 - "$node_logs_dir" "$nodes" >> "${out_file}" <<'PY'
import datetime
import os
import re
import sys

logs_dir = sys.argv[1]
num_nodes = int(sys.argv[2])

ts_re = re.compile(r'^[A-Z]\[(?:(\d{4}-\d{2}-\d{2})\|)?(\d{2}:\d{2}:\d{2}\.\d{3})\]')
new_height_re = re.compile(r'phase enter\s+module=consensus\s+height=(\d+)\s+round=\d+\s+phase=RoundStepNewHeight\b')
new_round_re = re.compile(r'phase enter\s+module=consensus\s+height=(\d+)\s+round=\d+\s+phase=RoundStepNewRound\b')
finalize_re = re.compile(r'finalizing commit of block\s+module=consensus\s+height=(\d+)\b.*\bnum_txs=(\d+)\b')
committed_re = re.compile(r'committed state\s+module=state\s+height=(\d+)\b')

def parse_ts(line):
    m = ts_re.search(line)
    if not m:
        return None
    date_part = m.group(1)
    time_part = m.group(2)
    if date_part:
        return datetime.datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S.%f")
    return datetime.datetime.strptime(time_part, "%H:%M:%S.%f")

print("==> Per-height per-node commit summary (RoundStepNewRound -> committed state):")
per_node = []
all_heights = set()

for node_idx in range(num_nodes):
    path = os.path.join(logs_dir, f"node_{node_idx}.log")
    if not os.path.isfile(path):
        per_node.append({
            "rows": {},
            "committed": {},
            "new_height_start": {},
        })
        continue

    new_height_start = {}
    new_round_start = {}
    finalized_txs = {}
    committed = {}

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            ts = parse_ts(line)
            if ts is None:
                continue

            m_nh = new_height_re.search(line)
            if m_nh:
                h = int(m_nh.group(1))
                if h not in new_height_start:
                    new_height_start[h] = ts
                continue

            m_nr = new_round_re.search(line)
            if m_nr:
                h = int(m_nr.group(1))
                if h not in new_round_start:
                    new_round_start[h] = ts
                continue

            m_fc = finalize_re.search(line)
            if m_fc:
                h = int(m_fc.group(1))
                txs = int(m_fc.group(2))
                if h not in finalized_txs:
                    finalized_txs[h] = txs
                continue

            m_cs = committed_re.search(line)
            if m_cs:
                h = int(m_cs.group(1))
                if h not in committed:
                    committed[h] = ts

    node_rows = {}
    for h, commit_ts in committed.items():
        start_ts = new_round_start.get(h)
        txs = finalized_txs.get(h, "-")
        if start_ts is None:
            lat = "NA"
        else:
            latency_ms = int(round((commit_ts - start_ts).total_seconds() * 1000.0))
            if latency_ms < 0:
                # Handle midnight rollover for time-only logs.
                if latency_ms < -12 * 60 * 60 * 1000:
                    latency_ms += 24 * 60 * 60 * 1000
                else:
                    latency_ms = 0
            lat = str(latency_ms)
        node_rows[h] = (lat, str(txs))
        all_heights.add(h)
    per_node.append({
        "rows": node_rows,
        "committed": committed,
        "new_height_start": new_height_start,
    })

if not all_heights:
    print("  no committed heights found")
else:
    heights = sorted(all_heights)
    headers = ["height", "prev_h_commit_to_new_h_ms"]
    for i in range(num_nodes):
        headers.append(f"n{i}_lat_ms")
    headers.extend(["lat_min_ms", "lat_max_ms", "lat_avg_ms"])
    for i in range(num_nodes):
        headers.append(f"n{i}_txs")

    rows = []
    for h in heights:
        row = [str(h)]

        gap_values = []
        if h > 1:
            for i in range(num_nodes):
                prev_commit = per_node[i]["committed"].get(h - 1)
                curr_new_height = per_node[i]["new_height_start"].get(h)
                if prev_commit is None or curr_new_height is None:
                    continue
                gap_ms = int(round((curr_new_height - prev_commit).total_seconds() * 1000.0))
                if gap_ms < 0:
                    if gap_ms < -12 * 60 * 60 * 1000:
                        gap_ms += 24 * 60 * 60 * 1000
                    else:
                        gap_ms = 0
                gap_values.append(gap_ms)
        if gap_values:
            row.append(f"{(sum(gap_values) / len(gap_values)):.1f}")
        else:
            row.append("-")

        lat_cells = []
        tx_cells = []
        lat_values = []
        for i in range(num_nodes):
            item = per_node[i]["rows"].get(h)
            if item is None:
                lat_cells.append("-")
                tx_cells.append("-")
            else:
                lat, txs = item
                lat_cells.append(lat)
                tx_cells.append(txs)
                try:
                    lat_values.append(int(lat))
                except ValueError:
                    pass

        if lat_values:
            lat_min = str(min(lat_values))
            lat_max = str(max(lat_values))
            lat_avg = f"{(sum(lat_values) / len(lat_values)):.1f}"
        else:
            lat_min = "-"
            lat_max = "-"
            lat_avg = "-"

        row.extend(lat_cells)
        row.extend([lat_min, lat_max, lat_avg])
        row.extend(tx_cells)
        rows.append(row)

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            if len(cell) > widths[idx]:
                widths[idx] = len(cell)

    def fmt_row(cells):
        return " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells)))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))
PY
}

cleanup() {
  local exit_code=$?
  kill_run_sessions
  force_stop_run_processes
  if [[ -n "${output_file:-}" && -f "${output_file}" ]]; then
    mkdir -p "${LOCAL_TMP_DIR}" >/dev/null 2>&1 || true
    cp -f "${output_file}" "${OUTPUT_LOCAL}" >/dev/null 2>&1 || true
  fi
  return "${exit_code}"
}
trap cleanup EXIT INT TERM

start_unix_ms="$(( $(python3 -c 'import time; print(int(time.time() * 1000))') + START_ALIGN_DELAY_MS ))"

mkdir -p "${LOCAL_TMP_DIR}"
rm -rf "${run_root}"
mkdir -p "${run_root}" "${logs_dir}"
: > "${output_file}"

echo "==> mode=${MODE}, f=${F}, n=${num_nodes}"
echo "==> synced start-unix-ms: ${start_unix_ms}"
echo "==> run root: ${run_root}"

cd "${src_dir}"
export PATH="$HOME/go/bin:${PATH}"

if (( LEARNING_AGENT_ENABLED == 1 )); then
  if ! python3 -c "import grpc, torch, sklearn, cryptography" >/dev/null 2>&1; then
    echo "Learning-agent mode requires grpc/torch/sklearn/cryptography in python3." >&2
    echo "Run scripts/cloudlab/setup.sh or install learning_agent requirements." >&2
    exit 1
  fi
fi

echo "==> Installing required CLIs (abci-cli/load/report)..."
go install "${buildvcs_flag}" ./abci/cmd/abci-cli
go install "${buildvcs_flag}" ./test/loadtime/cmd/load
go install "${buildvcs_flag}" ./test/loadtime/cmd/report

echo "==> Building cometbft binary..."
GOFLAGS="${GOFLAGS:-} ${buildvcs_flag}" make build

cometbft_bin="${src_dir}/build/cometbft"
if [[ ! -x "${cometbft_bin}" ]]; then
  echo "Missing built cometbft binary: ${cometbft_bin}" >&2
  exit 1
fi

echo "==> Generating local testnet config (n=${num_nodes})..."
"${cometbft_bin}" testnet \
  --v "${num_nodes}" \
  --o "${testnet_dir}" \
  --populate-persistent-peers \
  --starting-ip-address 127.0.0.1

echo "==> Applying master Tendermint config: ${MASTER_CONFIG}"

for ((i = 0; i < num_nodes; i++)); do
  config_file="${testnet_dir}/node${i}/config/config.toml"
  if [[ ! -f "${config_file}" ]]; then
    echo "Missing config file: ${config_file}" >&2
    exit 1
  fi
  apply_master_config "${MASTER_CONFIG}" "${config_file}"
  for ((j = 1; j < num_nodes; j++)); do
    sed_in_place "s|@127.0.0.$((j + 1)):26656|@127.0.0.1:$(p2p_port "${j}")|g" "${config_file}"
  done
  if (( LEARNING_AGENT_ENABLED == 1 )); then
    agent_p="$(agent_port "${i}")"
    sed_in_place "s|adaptive_timer_addr = \".*\"|adaptive_timer_addr = \"127.0.0.1:${agent_p}\"|" "${config_file}"
    sed_in_place "s|adaptive_timer_node_index = .*|adaptive_timer_node_index = ${i}|" "${config_file}"
  else
    sed_in_place "s|adaptive_timer_addr = \".*\"|adaptive_timer_addr = \"\"|" "${config_file}"
    sed_in_place "s|adaptive_timer_node_index = .*|adaptive_timer_node_index = 0|" "${config_file}"
  fi
done

if (( LEARNING_AGENT_ENABLED == 1 )); then
  : > "${agent_hosts_config}"
  for ((i = 0; i < num_nodes; i++)); do
    echo "${i} 127.0.0.1 0 0 $(agent_port "${i}") $(rpc_port "${i}")" >> "${agent_hosts_config}"
  done
fi

echo "==> Starting ${num_nodes} kvstore and CometBFT nodes in tmux..."
kill_run_sessions

if (( LEARNING_AGENT_ENABLED == 1 )); then
  echo "==> Starting learning agents (mode=${MODE})..."
  agent_mode_args=""
  if [[ "${MODE}" == "sharing" ]]; then
    agent_mode_args="--sharing true --model-type ${AGENT_MODEL_TYPE}"
  elif [[ "${MODE}" == "learning" ]]; then
    agent_mode_args="--model-type ${AGENT_MODEL_TYPE}"
  elif [[ "${MODE}" == "scheduled" ]]; then
    agent_mode_args="--model-type scheduled --schedule-file \"${schedule_file}\""
  fi
  for ((i = 0; i < num_nodes; i++)); do
    agent_session="ll_tm_tendermint_agent_${i}"
    agent_log="${logs_dir}/agent_${i}.log"
    dataset_args=""
    if (( DATASET_ENABLED == 1 )); then
      rm -f "${run_root}/agent${i}.csv"
      dataset_args="--dataset-csv \"${run_root}/agent${i}.csv\""
    fi
    tmux new-session -d -s "${agent_session}" \
      "python3 \"${LEARNING_AGENT_DIR}/main.py\" --node-id \"${i}\" --protocol tendermint --hosts-config \"${agent_hosts_config}\" --failure-spec \"${FAILURE_SPEC}\" --injection-start-unix-ms \"${start_unix_ms}\" ${agent_mode_args} ${dataset_args} > \"${agent_log}\" 2>&1"
  done
  sleep 1
  for ((i = 0; i < num_nodes; i++)); do
    agent_session="ll_tm_tendermint_agent_${i}"
    if ! tmux has-session -t "${agent_session}" 2>/dev/null; then
      echo "Learning-agent session exited early: ${agent_session}" >&2
      tail -n 80 "${logs_dir}/agent_${i}.log" >&2 || true
      exit 1
    fi
  done
fi

for ((i = 0; i < num_nodes; i++)); do
  abci_session="ll_tm_abci_${i}"
  node_session="ll_tm_tendermint_node_${i}"
  abci_log="${logs_dir}/abci_${i}.log"
  node_log="${logs_dir}/node_${i}.log"
  abci_p="$(abci_port "${i}")"
  p2p_p="$(p2p_port "${i}")"
  rpc_p="$(rpc_port "${i}")"
  node_home="${testnet_dir}/node${i}"

  tmux new-session -d -s "${abci_session}" \
    "abci-cli kvstore --address tcp://127.0.0.1:${abci_p} > \"${abci_log}\" 2>&1"
  tmux new-session -d -s "${node_session}" \
    "COMETBFT_NODE_INDEX=${i} COMETBFT_PROPOSE_DELAY_FAILURE_SPEC=\"${FAILURE_SPEC}\" COMETBFT_PROPOSE_DELAY_FAULTY_NODES=${F} COMETBFT_PROPOSE_DELAY_START_UNIX_MS=${start_unix_ms} \"${cometbft_bin}\" node --home \"${node_home}\" --proxy_app tcp://127.0.0.1:${abci_p} --p2p.laddr tcp://0.0.0.0:${p2p_p} --rpc.laddr tcp://0.0.0.0:${rpc_p} > \"${node_log}\" 2>&1"
done

echo "==> Waiting for nodes to be ready..."
for ((i = 0; i < num_nodes; i++)); do
  rpc_p="$(rpc_port "${i}")"
  ready=0
  for attempt in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${rpc_p}/status" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if (( ready == 0 )); then
    echo "Node ${i} failed to become ready on RPC port ${rpc_p}." >&2
    exit 1
  fi
done

expected_peers=$((num_nodes - 1))
rpc0="$(rpc_port 0)"
load_endpoints="$(load_endpoints_csv)"
echo "==> Waiting for peers to connect (expecting ${expected_peers})..."
for attempt in $(seq 1 30); do
  peers="$(curl -sf "http://127.0.0.1:${rpc0}/net_info" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["n_peers"])' 2>/dev/null || echo "0")"
  if [[ "${peers}" =~ ^[0-9]+$ ]] && (( peers >= expected_peers )); then
    break
  fi
  sleep 2
done

if (( POST_PEER_CONNECT_DELAY > 0 )); then
  sleep "${POST_PEER_CONNECT_DELAY}"
fi

{
  echo "==> mode=${MODE}"
  echo "==> f=${F} n=${num_nodes}"
  echo "==> post-peer-connect wait: ${POST_PEER_CONNECT_DELAY}s"
  echo "==> synced start-unix-ms: ${start_unix_ms}"
  echo "==> load: duration=${EXPERIMENT_TIME_S}s, connections=${LOAD_CONNECTIONS}, rate=${LOAD_RATE}, send-period=${LOAD_SEND_PERIOD_MS}ms, size=${LOAD_SIZE}B"
  echo "==> client log: ${client_log}"
} > "${output_file}"

load \
  --endpoints "${load_endpoints}" \
  --start-unix-ms "${start_unix_ms}" \
  --connections "${LOAD_CONNECTIONS}" \
  --rate "${LOAD_RATE}" \
  --send-period "${LOAD_SEND_PERIOD_MS}ms" \
  --time "${EXPERIMENT_TIME_S}" \
  --size "${LOAD_SIZE}" 2>&1 | tee "${client_log}" | awk '/^-- Monitor / { print }' >> "${output_file}"

echo "" >> "${output_file}"
echo "==> Stopping nodes..." >> "${output_file}"
kill_run_sessions
force_stop_run_processes
sleep 2

report_state_log="${logs_dir}/node_0.log"

{
  echo ""
  echo "==> Latency report:"
  if [[ -f "${report_state_log}" ]]; then
    echo "Using state log for wall-clock latency: ${report_state_log}"
    run_report_with_retry --data-dir "${testnet_dir}/node0/data" --database-type goleveldb --state-log "${report_state_log}" --csv "${results_csv}"
    run_report_with_retry --data-dir "${testnet_dir}/node0/data" --database-type goleveldb --state-log "${report_state_log}"
  else
    echo "State log for node0 not found; falling back to chain block timestamp latency."
    run_report_with_retry --data-dir "${testnet_dir}/node0/data" --database-type goleveldb --csv "${results_csv}"
    run_report_with_retry --data-dir "${testnet_dir}/node0/data" --database-type goleveldb
  fi
  echo ""
  echo "==> Throughput:"
  awk -F',' '
    NR==1 { next }
    NR==2 { first=$2; last=$2; count=1; next }
    { if ($2+0 > last+0) last=$2; count++ }
    END {
      duration_s = (last - first) / 1e9
      if (duration_s > 0)
        printf "Total tx: %d\nDuration: %.2fs\nThroughput: %.2f tx/sec\n", count, duration_s, count/duration_s
      else
        print "Not enough data"
    }
  ' "${results_csv}"
  echo ""
  append_node_phase_commit_summary "${output_file}" "${logs_dir}" "${num_nodes}"
} >> "${output_file}" 2>&1

mkdir -p "${LOCAL_TMP_DIR}"
cp -f "${output_file}" "${OUTPUT_LOCAL}"
if (( LOG_ENABLED == 1 )); then
  rm -rf "${LOGS_LOCAL_DIR}"
  mkdir -p "${LOGS_LOCAL_DIR}"
  cp -a "${logs_dir}/." "${LOGS_LOCAL_DIR}/"
  rm -f "${LOGS_LOCAL_DIR}"/abci_*.log
else
  rm -f "${results_csv}"
fi
if (( DATASET_ENABLED == 1 )); then
  echo "Collecting learning-agent dataset CSVs..."
  for ((i = 0; i < num_nodes; i++)); do
    src_csv="${run_root}/agent${i}.csv"
    dst_csv="${LOCAL_TMP_DIR}/agent${i}.csv"
    rm -f "${dst_csv}"
    if [[ -f "${src_csv}" ]]; then
      cp -f "${src_csv}" "${dst_csv}"
      echo "  - copied ${dst_csv}"
    else
      echo "  - missing dataset for node ${i}: ${src_csv}" >&2
    fi
  done
fi

echo "Done. Client output saved to ${OUTPUT_LOCAL}."
if (( LOG_ENABLED == 1 )); then
  echo "Logs saved to ${LOGS_LOCAL_DIR} (including client.log)."
fi
