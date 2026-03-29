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

LOCAL_TMP_DIR="${REPO_ROOT}/scripts/cloudlab/tmp"
OUTPUT_LOCAL="${LOCAL_TMP_DIR}/tendermint_output.txt"
LOGS_LOCAL_DIR="${LOCAL_TMP_DIR}/tendermint_logs"
LOCAL_PAYLOAD="${REPO_ROOT}/scripts/tmp/tendermint_cloudlab_payload.tar.gz"

usage() {
  cat <<EOF
Usage: $0 <private_key_path> [normal|heuristic|learning|sharing|scheduled] [--log] [-f <fault_tolerance>] [-t <experiment_seconds>] [targets_file]

Arguments:
  private_key_path  Local SSH private key for CloudLab nodes.
  mode              Optional; one of normal|heuristic|learning|sharing|scheduled (default: normal).
  targets_file      Optional path to cloudlab SSH targets file.

Flags:
  --log             Fetch remote Tendermint and ABCI logs into scripts/cloudlab/tmp/tendermint_logs/.
  -f <int>          Fault tolerance f. Number of nodes is derived as 3f+1 (default: 1).
  -t <int>          Load-generation duration in seconds (default: 60).
EOF
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
    normal|heuristic|learning|sharing|scheduled)
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

if [[ "${MODE}" != "normal" ]]; then
  echo "Mode '${MODE}' is not implemented yet. Supported mode right now: normal." >&2
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

if [[ ! -d "${REPO_ROOT}/tendermint" ]]; then
  echo "Local tendermint directory not found: ${REPO_ROOT}/tendermint" >&2
  exit 1
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
if (( LOG_ENABLED == 1 )); then
  rm -rf "${LOGS_LOCAL_DIR}"
  mkdir -p "${LOGS_LOCAL_DIR}"
fi

echo "Creating Tendermint payload tarball..."
tar -C "${REPO_ROOT}" -czf "${LOCAL_PAYLOAD}" \
  --exclude='tendermint/mytestnet' \
  --exclude='tendermint/logs' \
  --exclude='tendermint/build' \
  tendermint

REMOTE_PAYLOAD="/tmp/tendermint_cloudlab_payload_$$.tar.gz"

echo "Copying payload to ${remote_user}@${remote_host}..."
scp "${SSH_OPTS[@]}" "${LOCAL_PAYLOAD}" "${remote_user}@${remote_host}:${REMOTE_PAYLOAD}"

echo "Running remote Tendermint experiment on ${remote_host} (mode=${MODE}, f=${F}, t=${EXPERIMENT_TIME_S}s)..."
remote_failed=0
if ! ssh "${SSH_OPTS[@]}" "${remote_user}@${remote_host}" "bash -s" -- "${REMOTE_PAYLOAD}" "${F}" "${EXPERIMENT_TIME_S}" "${MODE}" "${LOG_ENABLED}" <<'REMOTE'
set -euo pipefail

payload_path="$1"
fault_tolerance="$2"
experiment_time_s="$3"
mode="$4"
log_enabled="$5"

if [[ "${mode}" != "normal" ]]; then
  echo "Unsupported mode on remote: ${mode}" >&2
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

num_nodes=$((fault_tolerance * 3 + 1))
maverick_nodes=0

run_root="$HOME/tendermint_cloudlab"
src_dir="${run_root}/tendermint"
testnet_dir="${run_root}/mytestnet"
logs_dir="${run_root}/logs"
output_file="$HOME/tendermint_output.txt"
results_csv="$HOME/results.csv"

load_rate=200
load_connections=1
load_size=500
post_peer_connect_delay=5
buildvcs_flag="-buildvcs=false"

# Ensure caller can always try to fetch a run log, even on early failures.
: > "${output_file}"

p2p_port() { echo $((26656 + $1 * 3)); }
rpc_port() { echo $((26657 + $1 * 3)); }
abci_port() { echo $((26658 + $1 * 3)); }

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
    if [[ "${session_name}" == tm_tendermint_node_* || "${session_name}" == tm_abci_* ]]; then
      tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
    fi
  done <<< "${sessions}"
}

cleanup() {
  kill_run_sessions
  rm -f "${payload_path}"
}
trap cleanup EXIT

for cmd in go tmux curl python3 tar awk sed; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Required command not found on remote host: ${cmd}" >&2
    exit 1
  fi
done

rm -rf "${run_root}"
mkdir -p "${run_root}" "${logs_dir}"
tar -xzf "${payload_path}" -C "${run_root}"

if [[ ! -d "${src_dir}" ]]; then
  echo "Extracted tendermint directory missing: ${src_dir}" >&2
  exit 1
fi

cd "${src_dir}"
export PATH="$HOME/go/bin:${PATH}"

echo "Installing required CLIs (abci-cli/load/report)..."
go install "${buildvcs_flag}" ./abci/cmd/abci-cli
go install "${buildvcs_flag}" ./test/loadtime/cmd/load
go install "${buildvcs_flag}" ./test/loadtime/cmd/report

echo "Building tendermint binary..."
GOFLAGS="${GOFLAGS:-} ${buildvcs_flag}" make build

tendermint_bin="${src_dir}/build/tendermint"
if [[ ! -x "${tendermint_bin}" ]]; then
  echo "Missing built tendermint binary: ${tendermint_bin}" >&2
  exit 1
fi

echo "Generating local testnet config (n=${num_nodes}, maverick=${maverick_nodes})..."
"${tendermint_bin}" testnet \
  --v "${num_nodes}" \
  --o "${testnet_dir}" \
  --populate-persistent-peers \
  --starting-ip-address 127.0.0.1

for ((i = 0; i < num_nodes; i++)); do
  config_file="${testnet_dir}/node${i}/config/config.toml"
  if [[ ! -f "${config_file}" ]]; then
    echo "Missing config file: ${config_file}" >&2
    exit 1
  fi
  for ((j = 1; j < num_nodes; j++)); do
    sed -i "s|@127.0.0.$((j + 1)):26656|@127.0.0.1:$(p2p_port "${j}")|g" "${config_file}"
  done
done

echo "Starting ${num_nodes} kvstore and Tendermint nodes in tmux..."
kill_run_sessions
for ((i = 0; i < num_nodes; i++)); do
  abci_session="tm_abci_${i}"
  node_session="tm_tendermint_node_${i}"
  abci_log="${logs_dir}/abci_${i}.log"
  node_log="${logs_dir}/node_${i}.log"
  abci_p="$(abci_port "${i}")"
  p2p_p="$(p2p_port "${i}")"
  rpc_p="$(rpc_port "${i}")"
  node_home="${testnet_dir}/node${i}"

  tmux new-session -d -s "${abci_session}" \
    "abci-cli kvstore --address tcp://127.0.0.1:${abci_p} > \"${abci_log}\" 2>&1"
  tmux new-session -d -s "${node_session}" \
    "\"${tendermint_bin}\" node --home \"${node_home}\" --proxy_app tcp://127.0.0.1:${abci_p} --p2p.laddr tcp://0.0.0.0:${p2p_p} --rpc.laddr tcp://0.0.0.0:${rpc_p} > \"${node_log}\" 2>&1"
done

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
for attempt in $(seq 1 30); do
  peers="$(curl -sf "http://127.0.0.1:${rpc0}/net_info" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["n_peers"])' 2>/dev/null || echo "0")"
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
  echo "==> load: rate=${load_rate} tx/s, duration=${experiment_time_s}s, connections=${load_connections}, size=${load_size}B"
} > "${output_file}"

load \
  --endpoints "ws://127.0.0.1:${rpc0}/websocket" \
  --broadcast-tx-method async \
  --connections "${load_connections}" \
  --rate "${load_rate}" \
  --time "${experiment_time_s}" \
  --size "${load_size}" >> "${output_file}" 2>&1

echo "" >> "${output_file}"
echo "==> Stopping nodes..." >> "${output_file}"
kill_run_sessions
sleep 2

{
  echo ""
  echo "==> Latency report:"
  report --data-dir "${testnet_dir}/node0/data" --database-type goleveldb --csv "${results_csv}"
  report --data-dir "${testnet_dir}/node0/data" --database-type goleveldb
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
if ! scp "${SSH_OPTS[@]}" "${remote_user}@${remote_host}:tendermint_output.txt" "${OUTPUT_LOCAL}"; then
  if (( remote_failed == 1 )); then
    echo "Remote run failed before tendermint_output.txt could be fetched from ${remote_host}." >&2
  else
    echo "Failed to fetch tendermint_output.txt from ${remote_host}." >&2
    exit 1
  fi
fi

if (( LOG_ENABLED == 1 )); then
  echo "Fetching logs to ${LOGS_LOCAL_DIR}..."
  if ! scp -r "${SSH_OPTS[@]}" "${remote_user}@${remote_host}:tendermint_cloudlab/logs/." "${LOGS_LOCAL_DIR}/"; then
    echo "Failed to fetch some/all logs from ${remote_host}." >&2
  fi
fi

if (( remote_failed == 1 )); then
  echo "Remote run failed. Retrieved output file may contain failure details: ${OUTPUT_LOCAL}" >&2
  exit 1
fi

echo "Done. Client output saved to ${OUTPUT_LOCAL}."
if (( LOG_ENABLED == 1 )); then
  echo "Logs saved to ${LOGS_LOCAL_DIR}."
fi
