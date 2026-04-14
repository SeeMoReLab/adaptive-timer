#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TARGETS_FILE_DEFAULT="${REPO_ROOT}/scripts/cloudlab/tmp/cloudlab_ssh_targets.txt"
TARGETS_FILE="${TARGETS_FILE_DEFAULT}"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
MODE="normal"
F=1
EXPERIMENT_TIME_S=60
LOG_ENABLED=0
DATASET_ENABLED=0
AGENT_MODEL_TYPE="random"
AGENT_MODEL_SET=0
LOAD_CONNECTIONS=5
LOAD_RATE=20
LOAD_SEND_PERIOD_MS=100

LOCAL_TMP_DIR="${REPO_ROOT}/scripts/cloudlab/tmp"
OUTPUT_LOCAL="${LOCAL_TMP_DIR}/tendermint_output.txt"
LOGS_LOCAL_DIR="${LOCAL_TMP_DIR}/tendermint_logs"
LOCAL_PAYLOAD="${REPO_ROOT}/scripts/tmp/tendermint_cloudlab_payload.tar.gz"
APPLIER_LOCAL_PATH="${REPO_ROOT}/scripts/apply_network_failures.py"

usage() {
  cat <<EOF
Usage: $0 <private_key_path> [normal|learning|sharing|scheduled] [--dataset] [--log] [--model <random|linear|cmab>] [-f <fault_tolerance>] [-t <experiment_seconds>] [targets_file]

Arguments:
  private_key_path  Local SSH private key for CloudLab nodes.
  mode              Optional; one of normal|learning|sharing|scheduled (default: normal).
  targets_file      Optional path to cloudlab SSH targets file.

Flags:
  --dataset         Collect per-replica learning-agent dataset CSVs as agentN.csv.
  --log             Fetch remote Tendermint, ABCI, and client logs into scripts/cloudlab/tmp/tendermint_logs/.
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

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  echo "Missing required private_key_path." >&2
  usage >&2
  exit 2
fi

LOCAL_SSH_KEY_PATH="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ "${TARGETS_FILE}" != "${TARGETS_FILE_DEFAULT}" ]]; then
        echo "Unexpected extra positional argument: $1" >&2
        usage >&2
        exit 2
      fi
      TARGETS_FILE="$1"
      shift
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
  echo "Mode '${MODE}' is not implemented. Supported modes: normal, learning, sharing, scheduled." >&2
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
LOCAL_SSH_KEY_PATH="${LOCAL_SSH_KEY_PATH/#\~/$HOME}"
if [[ ! -f "${LOCAL_SSH_KEY_PATH}" ]]; then
  echo "Local SSH key not found: ${LOCAL_SSH_KEY_PATH}" >&2
  exit 1
fi
SSH_OPTS+=(-i "${LOCAL_SSH_KEY_PATH}" -o IdentitiesOnly=yes)

if [[ ! -f "${TARGETS_FILE}" ]]; then
  echo "Targets file not found: ${TARGETS_FILE}" >&2
  exit 1
fi

if [[ ! -d "${REPO_ROOT}/cometbft" ]]; then
  echo "Local cometbft directory not found: ${REPO_ROOT}/cometbft" >&2
  exit 1
fi
if [[ ! -f "${REPO_ROOT}/config/tendermint/config.toml" ]]; then
  echo "Master Tendermint config not found: ${REPO_ROOT}/config/tendermint/config.toml" >&2
  exit 1
fi
if [[ ! -f "${APPLIER_LOCAL_PATH}" ]]; then
  echo "Network applier script not found: ${APPLIER_LOCAL_PATH}" >&2
  exit 1
fi
if (( LEARNING_AGENT_ENABLED == 1 )); then
  if [[ ! -d "${REPO_ROOT}/learning_agent" ]]; then
    echo "Local learning_agent directory not found: ${REPO_ROOT}/learning_agent" >&2
    exit 1
  fi
fi
if [[ "${MODE}" == "scheduled" ]]; then
  if [[ ! -f "${REPO_ROOT}/learning_agent/config/schedule_tendermint.csv" ]]; then
    echo "Missing Tendermint schedule file: ${REPO_ROOT}/learning_agent/config/schedule_tendermint.csv" >&2
    exit 1
  fi
fi

targets_nodes=()
targets_hosts=()
targets_users=()
while read -r node host user _; do
  [[ -z "${node:-}" ]] && continue
  [[ "${node:0:1}" == "#" ]] && continue
  targets_nodes+=("${node}")
  targets_hosts+=("${host}")
  targets_users+=("${user}")
done < "${TARGETS_FILE}"

if (( ${#targets_nodes[@]} < 1 )); then
  echo "No targets found in ${TARGETS_FILE}" >&2
  exit 1
fi

remote_host=""
remote_user=""
for i in "${!targets_nodes[@]}"; do
  if [[ "${targets_nodes[$i]}" == "node0" ]]; then
    remote_host="${targets_hosts[$i]}"
    remote_user="${targets_users[$i]}"
    break
  fi
done
if [[ -z "${remote_host}" || -z "${remote_user}" ]]; then
  remote_host="${targets_hosts[0]}"
  remote_user="${targets_users[0]}"
fi

mkdir -p "${LOCAL_TMP_DIR}" "$(dirname "${LOCAL_PAYLOAD}")"
rm -f "${LOCAL_PAYLOAD}" "${OUTPUT_LOCAL}"
: > "${OUTPUT_LOCAL}"
LOCAL_NUM_NODES=$((F * 3 + 1))
if (( DATASET_ENABLED == 1 )); then
  rm -f "${LOCAL_TMP_DIR}"/agent*.csv
fi
if (( LOG_ENABLED == 1 )); then
  rm -rf "${LOGS_LOCAL_DIR}"
  mkdir -p "${LOGS_LOCAL_DIR}"
fi

echo "Creating CometBFT payload tarball..."
if (( LEARNING_AGENT_ENABLED == 1 )); then
  COPYFILE_DISABLE=1 tar -C "${REPO_ROOT}" -czf "${LOCAL_PAYLOAD}" \
    --exclude='cometbft/mytestnet' \
    --exclude='cometbft/logs' \
    --exclude='cometbft/build' \
    --exclude='learning_agent/__pycache__' \
    cometbft \
    config/tendermint \
    learning_agent
else
  COPYFILE_DISABLE=1 tar -C "${REPO_ROOT}" -czf "${LOCAL_PAYLOAD}" \
    --exclude='cometbft/mytestnet' \
    --exclude='cometbft/logs' \
    --exclude='cometbft/build' \
    cometbft \
    config/tendermint
fi

REMOTE_PAYLOAD="/tmp/tendermint_cloudlab_payload_$$.tar.gz"
REMOTE_APPLIER="/tmp/tendermint_apply_network_failures_$$.py"

echo "Copying payload to ${remote_user}@${remote_host}..."
scp "${SSH_OPTS[@]}" "${LOCAL_PAYLOAD}" "${remote_user}@${remote_host}:${REMOTE_PAYLOAD}"
scp "${SSH_OPTS[@]}" "${APPLIER_LOCAL_PATH}" "${remote_user}@${remote_host}:${REMOTE_APPLIER}"

echo "Running remote CometBFT experiment on ${remote_host} (mode=${MODE}, f=${F}, t=${EXPERIMENT_TIME_S}s)..."
remote_failed=0
if ! ssh "${SSH_OPTS[@]}" "${remote_user}@${remote_host}" "bash -s" -- "${REMOTE_PAYLOAD}" "${REMOTE_APPLIER}" "${F}" "${EXPERIMENT_TIME_S}" "${MODE}" "${LOG_ENABLED}" "${DATASET_ENABLED}" "${LOAD_CONNECTIONS}" "${LOAD_RATE}" "${LOAD_SEND_PERIOD_MS}" "${AGENT_MODEL_TYPE}" <<'REMOTE'
set -euo pipefail

payload_path="$1"
applier_path="$2"
fault_tolerance="$3"
experiment_time_s="$4"
mode="$5"
log_enabled="$6"
dataset_enabled="$7"
load_connections="$8"
load_rate="$9"
load_send_period_ms="${10}"
agent_model_type="${11}"

if [[ "${mode}" != "normal" && "${mode}" != "learning" && "${mode}" != "sharing" && "${mode}" != "scheduled" ]]; then
  echo "Unsupported mode on remote: ${mode} (supported: normal, learning, sharing, scheduled)" >&2
  exit 2
fi
if ! [[ "${fault_tolerance}" =~ ^[0-9]+$ ]] || (( fault_tolerance < 1 )); then
  echo "Invalid fault_tolerance: ${fault_tolerance}" >&2
  exit 2
fi
if ! [[ "${experiment_time_s}" =~ ^[0-9]+$ ]] || (( experiment_time_s < 1 )); then
  echo "Invalid experiment_time_s: ${experiment_time_s}" >&2
  exit 2
fi
if ! [[ "${dataset_enabled}" =~ ^[0-9]+$ ]] || (( dataset_enabled < 0 || dataset_enabled > 1 )); then
  echo "Invalid dataset_enabled: ${dataset_enabled}" >&2
  exit 2
fi
if ! [[ "${load_connections}" =~ ^[0-9]+$ ]] || (( load_connections < 1 )); then
  echo "Invalid load_connections: ${load_connections}" >&2
  exit 2
fi
if ! [[ "${load_rate}" =~ ^[0-9]+$ ]] || (( load_rate < 1 )); then
  echo "Invalid load_rate: ${load_rate}" >&2
  exit 2
fi
if ! [[ "${load_send_period_ms}" =~ ^[0-9]+$ ]] || (( load_send_period_ms < 1 )); then
  echo "Invalid load_send_period_ms: ${load_send_period_ms}" >&2
  exit 2
fi
if [[ "${agent_model_type}" != "random" && "${agent_model_type}" != "linear" && "${agent_model_type}" != "cmab" ]]; then
  echo "Invalid agent_model_type: ${agent_model_type}. Use random|linear|cmab." >&2
  exit 2
fi
if [[ "${mode}" != "learning" && "${mode}" != "sharing" && "${agent_model_type}" != "random" ]]; then
  echo "agent_model_type is only supported with learning/sharing mode." >&2
  exit 2
fi
if (( dataset_enabled == 1 )) && [[ "${mode}" != "learning" && "${mode}" != "sharing" && "${mode}" != "scheduled" ]]; then
  echo "dataset_enabled requires learning-agent mode." >&2
  exit 2
fi

num_nodes=$((fault_tolerance * 3 + 1))
learning_agent_enabled=0
if [[ "${mode}" == "learning" || "${mode}" == "sharing" || "${mode}" == "scheduled" ]]; then
  learning_agent_enabled=1
fi

run_root="$HOME/tendermint_cloudlab"
src_dir="${run_root}/cometbft"
learning_agent_dir="${run_root}/learning_agent"
master_config="${run_root}/config/tendermint/config.toml"
testnet_dir="${run_root}/mytestnet"
logs_dir="${run_root}/logs"
output_file="$HOME/tendermint_output.txt"
results_csv="$HOME/results.csv"
client_log="${logs_dir}/client.log"
agent_hosts_config="${run_root}/agent_hosts.config"
schedule_file="${learning_agent_dir}/config/schedule_tendermint.csv"
applier_hosts_config="${run_root}/applier_hosts.config"
applier_script="${run_root}/apply_network_failures.py"

load_size=500
post_peer_connect_delay=5
start_align_delay_ms=15000
buildvcs_flag="-buildvcs=false"
failure_spec="${run_root}/config/tendermint/failure_spec.xml"
report_state_log="${logs_dir}/node_0.log"
overlay_a=10
overlay_b=251
overlay_subnet=1
overlay_mask=24
overlay_host_offset=10
overlay_bridge_name="br-tendermint"
overlay_bridge_ip="${overlay_a}.${overlay_b}.${overlay_subnet}.1"
max_overlay_nodes=$((255 - overlay_host_offset))
run_as_user="$(id -un)"
run_as_home="${HOME}"

# Ensure caller can always try to fetch a run log, even on early failures.
: > "${output_file}"
{
  echo "==> remote run initialized"
  echo "==> mode=${mode}"
  echo "==> f=${fault_tolerance} n=${num_nodes}"
  echo "==> duration=${experiment_time_s}s"
} >> "${output_file}"

p2p_port() { echo $((26656 + $1 * 3)); }
rpc_port() { echo $((26657 + $1 * 3)); }
abci_port() { echo $((26658 + $1 * 3)); }
agent_port() { echo $((50000 + $1)); }
node_ip() { echo "${overlay_a}.${overlay_b}.${overlay_subnet}.$((overlay_host_offset + $1))"; }
ns_name() { echo "tm-r$1"; }
ns_host_if() { echo "tm-vh$1"; }
ns_inner_if() { echo "tm-vn$1"; }
load_endpoints_csv() {
  local endpoints=()
  local i rpc_p ip
  for ((i = 0; i < num_nodes; i++)); do
    rpc_p="$(rpc_port "${i}")"
    ip="$(node_ip "${i}")"
    endpoints+=("ws://${ip}:${rpc_p}/websocket")
  done
  local IFS=,
  echo "${endpoints[*]}"
}

write_applier_hosts_config() {
  : > "${applier_hosts_config}"
  local i ip
  for ((i = 0; i < num_nodes; i++)); do
    ip="$(node_ip "${i}")"
    echo "${i} ${ip}" >> "${applier_hosts_config}"
  done
}

apply_master_config() {
  local master_cfg="$1"
  local node_cfg="$2"
  python3 - "$master_cfg" "$node_cfg" <<'PY'
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
  if ! command -v tmux >/dev/null 2>&1; then
    return 0
  fi
  sessions="$(tmux ls -F '#S' 2>/dev/null || true)"
  if [[ -z "${sessions}" ]]; then
    return 0
  fi
  while read -r session_name; do
    [[ -z "${session_name}" ]] && continue
    if [[ "${session_name}" == tm_tendermint_node_* || "${session_name}" == tm_abci_* || "${session_name}" == tm_tendermint_agent_* || "${session_name}" == tm_applier_* ]]; then
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

setup_network_namespaces() {
  if ! sudo -n true >/dev/null 2>&1; then
    echo "Namespace setup requires passwordless sudo on remote host." >&2
    echo "Verify with: ssh <node> 'sudo -n true'" >&2
    return 1
  fi

  existing_ns="$(ip netns list | awk '/^tm-r[0-9]+/ {print $1}')"
  if [[ -n "${existing_ns}" ]]; then
    printf '%s\n' "${existing_ns}" | while read -r name; do
      [[ -z "${name}" ]] && continue
      sudo -n ip netns del "${name}" >/dev/null 2>&1 || true
    done
  fi

  stale_veths="$(ip -o link show | awk -F': ' '{print $2}' | sed 's/@.*//' | awk '/^tm-vh[0-9]+$/')"
  if [[ -n "${stale_veths}" ]]; then
    printf '%s\n' "${stale_veths}" | while read -r if_name; do
      [[ -z "${if_name}" ]] && continue
      sudo -n ip link del "${if_name}" >/dev/null 2>&1 || true
    done
  fi

  sudo -n ip link set "${overlay_bridge_name}" down >/dev/null 2>&1 || true
  sudo -n ip link del "${overlay_bridge_name}" type bridge >/dev/null 2>&1 || true
  sudo -n ip link add name "${overlay_bridge_name}" type bridge
  sudo -n ip addr replace "${overlay_bridge_ip}/${overlay_mask}" dev "${overlay_bridge_name}"
  sudo -n ip link set "${overlay_bridge_name}" up
  sudo -n sysctl -w net.ipv4.ip_forward=1 >/dev/null

  local i ns host_if inner_if ip
  for ((i = 0; i < num_nodes; i++)); do
    ns="$(ns_name "${i}")"
    host_if="$(ns_host_if "${i}")"
    inner_if="$(ns_inner_if "${i}")"
    ip="$(node_ip "${i}")"

    sudo -n ip netns del "${ns}" >/dev/null 2>&1 || true
    sudo -n ip link del "${host_if}" >/dev/null 2>&1 || true
    sudo -n ip netns add "${ns}"
    sudo -n ip link add "${host_if}" type veth peer name "${inner_if}"
    sudo -n ip link set "${inner_if}" netns "${ns}"
    sudo -n ip link set "${host_if}" master "${overlay_bridge_name}"
    sudo -n ip link set "${host_if}" up
    sudo -n ip netns exec "${ns}" ip link set lo up
    sudo -n ip netns exec "${ns}" ip addr replace "${ip}/${overlay_mask}" dev "${inner_if}"
    sudo -n ip netns exec "${ns}" ip link set "${inner_if}" up
    sudo -n ip netns exec "${ns}" ip route replace default via "${overlay_bridge_ip}"
  done
}

teardown_network_namespaces() {
  local existing_ns
  existing_ns="$(ip netns list | awk '/^tm-r[0-9]+/ {print $1}')"
  if [[ -n "${existing_ns}" ]]; then
    printf '%s\n' "${existing_ns}" | while read -r name; do
      [[ -z "${name}" ]] && continue
      sudo -n ip netns del "${name}" >/dev/null 2>&1 || true
    done
  fi

  sudo -n ip link set "${overlay_bridge_name}" down >/dev/null 2>&1 || true
  sudo -n ip link del "${overlay_bridge_name}" type bridge >/dev/null 2>&1 || true
}

start_network_appliers() {
  local i ns iface session applier_log
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found; cannot start network appliers." >&2
    return 1
  fi
  if [[ ! -f "${applier_script}" ]]; then
    echo "Missing network applier script: ${applier_script}" >&2
    return 1
  fi
  if [[ ! -f "${applier_hosts_config}" ]]; then
    echo "Missing applier hosts config: ${applier_hosts_config}" >&2
    return 1
  fi

  for ((i = 0; i < num_nodes; i++)); do
    ns="$(ns_name "${i}")"
    iface="$(ns_inner_if "${i}")"
    session="tm_applier_${i}"
    applier_log="${logs_dir}/applier_${i}.log"
    tmux kill-session -t "${session}" >/dev/null 2>&1 || true
    tmux new-session -d -s "${session}" \
      "cd \"${run_root}\" && python3 \"${applier_script}\" --failure-spec \"${failure_spec}\" --hosts-config \"${applier_hosts_config}\" --replica-id ${i} --netns ${ns} --interface ${iface} --start-unix-ms ${start_unix_ms} > \"${applier_log}\" 2>&1"
  done
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
  kill_run_sessions
  force_stop_run_processes
  teardown_network_namespaces
  rm -f "${payload_path}"
  rm -f "${applier_path}"
}
trap cleanup EXIT

for cmd in go tmux curl python3 tar awk sed ip sudo; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Required command not found on remote host: ${cmd}" >&2
    exit 1
  fi
done

start_unix_ms="$(( $(python3 -c 'import time; print(int(time.time() * 1000))') + start_align_delay_ms ))"

rm -rf "${run_root}"
mkdir -p "${run_root}" "${logs_dir}"
tar -xzf "${payload_path}" -C "${run_root}"
cp -a "${applier_path}" "${applier_script}"
chmod +x "${applier_script}"

if [[ ! -d "${src_dir}" ]]; then
  echo "Extracted cometbft directory missing: ${src_dir}" >&2
  exit 1
fi
if [[ ! -f "${master_config}" ]]; then
  echo "Extracted master config missing: ${master_config}" >&2
  exit 1
fi
if (( learning_agent_enabled == 1 )); then
  if [[ ! -f "${learning_agent_dir}/main.py" ]]; then
    echo "Extracted learning_agent missing: ${learning_agent_dir}/main.py" >&2
    exit 1
  fi
  if [[ ! -f "${schedule_file}" ]]; then
    echo "Missing Tendermint schedule file: ${schedule_file}" >&2
    exit 1
  fi
  if ! python3 -c "import grpc, torch, sklearn, cryptography" >/dev/null 2>&1; then
    echo "Learning-agent mode requires grpc/torch/sklearn/cryptography in python3." >&2
    echo "Run scripts/cloudlab/setup.sh first." >&2
    exit 1
  fi
fi

cd "${src_dir}"
export PATH="$HOME/go/bin:${PATH}"

echo "Installing required CLIs (abci-cli/load/report)..."
go install "${buildvcs_flag}" ./abci/cmd/abci-cli
go install "${buildvcs_flag}" ./test/loadtime/cmd/load
go install "${buildvcs_flag}" ./test/loadtime/cmd/report

echo "Building cometbft binary..."
GOFLAGS="${GOFLAGS:-} ${buildvcs_flag}" make build

cometbft_bin="${src_dir}/build/cometbft"
if [[ ! -x "${cometbft_bin}" ]]; then
  echo "Missing built cometbft binary: ${cometbft_bin}" >&2
  exit 1
fi

if [[ ! -f "${failure_spec}" ]]; then
  echo "Missing failure spec: ${failure_spec}" >&2
  exit 1
fi
if (( num_nodes > max_overlay_nodes )); then
  echo "Too many nodes (${num_nodes}) for overlay host offset ${overlay_host_offset} in /24 subnet." >&2
  exit 1
fi

echo "Setting up per-node network namespaces..."
setup_network_namespaces

echo "Generating local testnet config (n=${num_nodes})..."
"${cometbft_bin}" testnet \
  --v "${num_nodes}" \
  --o "${testnet_dir}" \
  --populate-persistent-peers \
  --starting-ip-address "$(node_ip 0)"

echo "Applying master Tendermint config: ${master_config}"

echo "Applying master Tendermint config: ${master_config}"

for ((i = 0; i < num_nodes; i++)); do
  config_file="${testnet_dir}/node${i}/config/config.toml"
  if [[ ! -f "${config_file}" ]]; then
    echo "Missing config file: ${config_file}" >&2
    exit 1
  fi
  apply_master_config "${master_config}" "${config_file}"
  for ((j = 1; j < num_nodes; j++)); do
    sed -i "s|@$(node_ip "${j}"):26656|@$(node_ip "${j}"):$(p2p_port "${j}")|g" "${config_file}"
  done
  if (( learning_agent_enabled == 1 )); then
    agent_p="$(agent_port "${i}")"
    sed -i "s|adaptive_timer_addr = \".*\"|adaptive_timer_addr = \"$(node_ip "${i}"):${agent_p}\"|" "${config_file}"
    sed -i "s|adaptive_timer_node_index = .*|adaptive_timer_node_index = ${i}|" "${config_file}"
  else
    sed -i "s|adaptive_timer_addr = \".*\"|adaptive_timer_addr = \"\"|" "${config_file}"
    sed -i "s|adaptive_timer_node_index = .*|adaptive_timer_node_index = 0|" "${config_file}"
  fi
done

if (( learning_agent_enabled == 1 )); then
  : > "${agent_hosts_config}"
  for ((i = 0; i < num_nodes; i++)); do
    echo "${i} $(node_ip "${i}") 0 0 $(agent_port "${i}") $(rpc_port "${i}")" >> "${agent_hosts_config}"
  done
fi
echo "Starting ${num_nodes} kvstore and CometBFT nodes in tmux..."
kill_run_sessions
if (( learning_agent_enabled == 1 )); then
  agent_mode_args=""
  if [[ "${mode}" == "sharing" ]]; then
    agent_mode_args="--sharing true --model-type ${agent_model_type}"
  elif [[ "${mode}" == "learning" ]]; then
    agent_mode_args="--model-type ${agent_model_type}"
  elif [[ "${mode}" == "scheduled" ]]; then
    agent_mode_args="--model-type scheduled --schedule-file \"${schedule_file}\""
  fi
  for ((i = 0; i < num_nodes; i++)); do
    agent_session="tm_tendermint_agent_${i}"
    ns="$(ns_name "${i}")"
    agent_log="${logs_dir}/agent_${i}.log"
    dataset_args=""
    if (( dataset_enabled == 1 )); then
      rm -f "${run_root}/agent${i}.csv"
      dataset_args="--dataset-csv \"${run_root}/agent${i}.csv\""
    fi
    tmux new-session -d -s "${agent_session}" \
      "sudo -n ip netns exec ${ns} sudo -n -u ${run_as_user} env HOME=${run_as_home} PATH=${run_as_home}/go/bin:${PATH} PYTHONDONTWRITEBYTECODE=1 bash -lc 'cd \"${run_root}\" && exec python3 \"${learning_agent_dir}/main.py\" --node-id \"${i}\" --protocol tendermint --hosts-config \"${agent_hosts_config}\" --failure-spec \"${failure_spec}\" --injection-start-unix-ms \"${start_unix_ms}\" ${agent_mode_args} ${dataset_args}' > \"${agent_log}\" 2>&1"
  done
  sleep 1
  for ((i = 0; i < num_nodes; i++)); do
    agent_session="tm_tendermint_agent_${i}"
    if ! tmux has-session -t "${agent_session}" 2>/dev/null; then
      echo "Learning-agent session exited early: ${agent_session}" >&2
      tail -n 80 "${logs_dir}/agent_${i}.log" >&2 || true
      exit 1
    fi
  done
fi
for ((i = 0; i < num_nodes; i++)); do
  abci_session="tm_abci_${i}"
  node_session="tm_tendermint_node_${i}"
  ns="$(ns_name "${i}")"
  abci_log="${logs_dir}/abci_${i}.log"
  node_log="${logs_dir}/node_${i}.log"
  abci_p="$(abci_port "${i}")"
  p2p_p="$(p2p_port "${i}")"
  rpc_p="$(rpc_port "${i}")"
  node_home="${testnet_dir}/node${i}"

  tmux new-session -d -s "${abci_session}" \
    "sudo -n ip netns exec ${ns} sudo -n -u ${run_as_user} env HOME=${run_as_home} PATH=${run_as_home}/go/bin:${PATH} bash -lc 'cd \"${src_dir}\" && exec abci-cli kvstore --address tcp://127.0.0.1:${abci_p}' > \"${abci_log}\" 2>&1"
  tmux new-session -d -s "${node_session}" \
    "sudo -n ip netns exec ${ns} sudo -n -u ${run_as_user} env HOME=${run_as_home} PATH=${run_as_home}/go/bin:${PATH} COMETBFT_NODE_INDEX=${i} COMETBFT_PROPOSE_DELAY_FAILURE_SPEC=\"${failure_spec}\" COMETBFT_PROPOSE_DELAY_FAULTY_NODES=${fault_tolerance} COMETBFT_PROPOSE_DELAY_START_UNIX_MS=${start_unix_ms} bash -lc 'cd \"${src_dir}\" && exec \"${cometbft_bin}\" node --home \"${node_home}\" --proxy_app tcp://127.0.0.1:${abci_p} --p2p.laddr tcp://0.0.0.0:${p2p_p} --rpc.laddr tcp://0.0.0.0:${rpc_p}' > \"${node_log}\" 2>&1"
done

write_applier_hosts_config
echo "Starting per-replica network failure appliers in tmux..."
start_network_appliers

for ((i = 0; i < num_nodes; i++)); do
  ip="$(node_ip "${i}")"
  rpc_p="$(rpc_port "${i}")"
  ready=0
  for attempt in $(seq 1 60); do
    if curl -sf "http://${ip}:${rpc_p}/status" >/dev/null 2>&1; then
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
rpc0_ip="$(node_ip 0)"
load_endpoints="$(load_endpoints_csv)"
for attempt in $(seq 1 30); do
  peers="$(curl -sf "http://${rpc0_ip}:${rpc0}/net_info" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["n_peers"])' 2>/dev/null || echo "0")"
  if [[ "${peers}" =~ ^[0-9]+$ ]] && (( peers >= expected_peers )); then
    break
  fi
  sleep 2
done

if (( post_peer_connect_delay > 0 )); then
  sleep "${post_peer_connect_delay}"
fi

{
  echo "==> mode=${mode}"
  echo "==> f=${fault_tolerance} n=${num_nodes}"
  echo "==> post-peer-connect wait: ${post_peer_connect_delay}s"
  echo "==> synced start-unix-ms: ${start_unix_ms}"
  echo "==> load: duration=${experiment_time_s}s, connections=${load_connections}, rate=${load_rate}, send-period=${load_send_period_ms}ms, size=${load_size}B"
  echo "==> client log: ${client_log}"
} > "${output_file}"

load \
  --endpoints "${load_endpoints}" \
  --start-unix-ms "${start_unix_ms}" \
  --connections "${load_connections}" \
  --rate "${load_rate}" \
  --send-period "${load_send_period_ms}ms" \
  --time "${experiment_time_s}" \
  --size "${load_size}" 2>&1 | tee "${client_log}" | awk '/^-- Monitor / { print }' >> "${output_file}"

echo "" >> "${output_file}"
echo "==> Stopping nodes..." >> "${output_file}"
kill_run_sessions
force_stop_run_processes
sleep 2

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

if (( log_enabled == 0 )); then
  rm -f "${results_csv}"
fi

echo "Done. Output written to ${output_file}"
REMOTE
then
  remote_failed=1
fi

echo "Fetching tendermint output to ${OUTPUT_LOCAL}..."
fetched_output=0
for remote_output_path in \
  "tendermint_output.txt" \
  "~/tendermint_output.txt" \
  "/home/${remote_user}/tendermint_output.txt" \
  "tendermint_cloudlab/tendermint_output.txt"; do
  if scp "${SSH_OPTS[@]}" "${remote_user}@${remote_host}:${remote_output_path}" "${OUTPUT_LOCAL}" >/dev/null 2>&1; then
    fetched_output=1
    break
  fi
done
if (( fetched_output == 0 )); then
  {
    echo "Failed to fetch remote output file from ${remote_host}."
    echo "Tried: tendermint_output.txt, ~/tendermint_output.txt, /home/${remote_user}/tendermint_output.txt, tendermint_cloudlab/tendermint_output.txt"
    echo "remote_failed=${remote_failed}"
  } > "${OUTPUT_LOCAL}"
  if (( remote_failed == 0 )); then
    echo "Failed to fetch tendermint output from ${remote_host}." >&2
    exit 1
  fi
fi

if (( LOG_ENABLED == 1 )); then
  echo "Fetching logs to ${LOGS_LOCAL_DIR}..."
  if ! scp -r "${SSH_OPTS[@]}" "${remote_user}@${remote_host}:tendermint_cloudlab/logs/." "${LOGS_LOCAL_DIR}/"; then
    echo "Failed to fetch some/all logs from ${remote_host}." >&2
  fi
fi
if (( DATASET_ENABLED == 1 )); then
  echo "Fetching learning-agent datasets to ${LOCAL_TMP_DIR}..."
  for ((i = 0; i < LOCAL_NUM_NODES; i++)); do
    local_csv="${LOCAL_TMP_DIR}/agent${i}.csv"
    rm -f "${local_csv}"
    if ! scp "${SSH_OPTS[@]}" "${remote_user}@${remote_host}:tendermint_cloudlab/agent${i}.csv" "${local_csv}"; then
      echo "Missing or failed to fetch agent${i}.csv from ${remote_host}." >&2
    fi
  done
fi

if (( remote_failed == 1 )); then
  echo "Remote run failed. Retrieved output file may contain failure details: ${OUTPUT_LOCAL}" >&2
  exit 1
fi

echo "Done. Client output saved to ${OUTPUT_LOCAL}."
if (( LOG_ENABLED == 1 )); then
  echo "Logs saved to ${LOGS_LOCAL_DIR} (including client.log)."
fi
if (( DATASET_ENABLED == 1 )); then
  echo "Datasets saved to ${LOCAL_TMP_DIR}/agentN.csv."
fi
if (( DATASET_ENABLED == 1 )); then
  echo "Datasets saved to ${LOCAL_TMP_DIR}/agentN.csv."
fi
