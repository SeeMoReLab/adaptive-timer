#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TARGETS_FILE_DEFAULT="${REPO_ROOT}/scripts/cloudlab/tmp/cloudlab_ssh_targets.txt"
SMARTBFT_SRC="${REPO_ROOT}/SmartBFT"
LEARNING_AGENT_SRC_DIR="${REPO_ROOT}/learning_agent"
APPLIER_LOCAL_PATH="${REPO_ROOT}/scripts/apply_network_failures.py"
CONFIG_DEFAULT="${REPO_ROOT}/config/pbft/smallbank.xml"
FAILURE_SPEC_DEFAULT="${REPO_ROOT}/config/pbft/failure_spec.xml"
LOCAL_TMP_DIR="${REPO_ROOT}/scripts/cloudlab/tmp"
LOCAL_TAR="${LOCAL_TMP_DIR}/smartbft_payload.tar.gz"
OUTPUT_LOCAL="${LOCAL_TMP_DIR}/smartbft_output.txt"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)

MODE="normal"
F=1
NODES_SET=0
NODES=4
TARGETS_FILE="${TARGETS_FILE_DEFAULT}"
EXPERIMENT_TIME_S=300
CLIENT_START_DELAY_S=10
START_OFFSET_AFTER_CLIENT_LAUNCH_S=20
DATASET_ENABLED=0
LOG_ENABLED=0
VERBOSE_ENABLED=0
AGENT_MODEL_TYPE="random"
AGENT_MODEL_SET=0

BATCH_SIZE=200
BATCH_TIMEOUT="50ms"
REQUEST_TIMEOUT="30s"
SUBMIT_MODE="broadcast"
CREATE_WORKERS=""
SERVER_PORT_BASE=19100
AGENT_PORT=15501
LEARNING_NODE_ID=1
INITIAL_TIMEOUT="5s"
REQUEST_TIMEOUT_BACKOFF_MAX="10s"
CONFIG_PATH="${CONFIG_DEFAULT}"
FAILURE_SPEC_PATH="${FAILURE_SPEC_DEFAULT}"
FAILURE_SPEC_ENABLED=1

OVERLAY_OCTET_A=10
OVERLAY_OCTET_B=252
OVERLAY_MASK_BITS=24
OVERLAY_BRIDGE_NAME="br-smartbft"
OVERLAY_HOST_OFFSET=10
ARTIFACT_FETCH_MAX_PARALLEL=8

usage() {
  cat <<EOF
Usage: $0 <private_key_path> [normal|heuristic|learning|sharing|scheduled] [--dataset] [--log] [--model <random|linear|cmab>] [-f <fault_tolerance>] [-t <experiment_seconds>] [targets_file]

Arguments:
  private_key_path  Local SSH private key for CloudLab nodes.
  mode              Optional; one of normal|heuristic|learning|sharing|scheduled (default: normal).
  targets_file      Optional path to cloudlab SSH targets file.

Flags:
  --dataset                         Compatibility flag; agent modes collect the agent CSV automatically.
  --log                             Download client, server, applier, and agent logs.
  --verbose                         Enable SmartBFT debug logs on server replicas.
  --model <random|linear|cmab>      Learning-agent model for learning mode (default: random).
  -f <int>                          Fault tolerance f; nodes defaults to 3f+1 (default: 1).
  -t <int>                          Seconds to run after benchmark start (default: 300).

SmartBFT options:
  --nodes <int>                     Override node count (default: 3f+1).
  --config <path>                   SmallBank XML config (default: config/pbft/smallbank.xml).
  --failure-spec <path>             failure_spec.xml for proposal and network delay.
  --no-failure-spec                 Disable proposal delay and network delay appliers.
  --batch-size <int>                SmartBFT batch size (default: 200).
  --batch-timeout <duration>        SmartBFT batch timeout (default: 50ms).
  --request-timeout <duration>      Client request timeout (default: 30s).
  --submit-mode <broadcast|leader>  Client submit mode (default: broadcast).
  --create-workers <int>            Account creation workers (default: first workload phase terminals).
  --client-start-delay <seconds>    Wait after servers are ready before launching client (default: 10).
  --start-offset <seconds>          Delay from client launch to benchmark/failure start (default: 20).
  --port-base <int>                 Base server port; node i listens on base+i (default: 19100).

Learning-agent options:
  --agent-port <port>               Learning-agent port inside each replica namespace (default: 15501).
  --learning-node-id <id>           Deprecated; SmartBFT CloudLab now starts one agent per node.
  --initial-timeout <duration>      Initial PBFT timeout value reported by SmartBFT (default: 5s).

Notes:
  Request timeout backoff is enabled by this script with max effective timeout 10s.
EOF
}

mode_enables_learning_agent() {
  case "$1" in
    heuristic|learning|sharing|scheduled) return 0 ;;
    *) return 1 ;;
  esac
}

normalize_agent_model_type() {
  local model_lower
  model_lower="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  case "${model_lower}" in
    random|linear|cmab) printf '%s\n' "${model_lower}" ;;
    *) return 1 ;;
  esac
}

absolute_path() {
  local path="$1"
  local dir
  local base
  if [[ "${path}" == /* ]]; then
    printf '%s\n' "${path}"
    return
  fi
  dir="$(dirname "${path}")"
  base="$(basename "${path}")"
  printf '%s/%s\n' "$(cd "${dir}" && pwd -P)" "${base}"
}

is_ipv4_literal() {
  local value="$1"
  [[ "${value}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]
}

write_filtered_output() {
  local source_log="$1"
  local output_path="$2"
  awk '
    /^Monitor duration=/ {
      seen_monitor = 1
      print
      next
    }
    seen_monitor == 0 {
      print
    }
  ' "${source_log}" > "${output_path}"
}

queue_artifact_download() {
  local node="$1"
  local host="$2"
  local user="$3"
  local remote_check_path="$4"
  local remote_copy_path="$5"
  local local_path="$6"
  local artifact_label="$7"
  local artifact_kind="$8"
  local missing_ok="${9:-0}"

  rm -f "${local_path}"
  while true; do
    running_jobs="$(jobs -pr | wc -l | tr -d '[:space:]')"
    [[ -n "${running_jobs}" ]] || running_jobs=0
    if (( running_jobs < ARTIFACT_FETCH_MAX_PARALLEL )); then
      break
    fi
    sleep 0.05
  done

  (
    if ssh "${SSH_OPTS[@]}" "${user}@${host}" "test -f ${remote_check_path}"; then
      if scp "${SSH_OPTS[@]}" "${user}@${host}:${remote_copy_path}" "${local_path}"; then
        echo "  - copied ${artifact_label} from ${node}"
      else
        echo "  - failed to copy ${artifact_label} from ${node}" >&2
        exit 1
      fi
    else
      echo "  - missing ${artifact_label} on ${node}" >&2
      (( missing_ok == 1 )) && exit 0
      exit 1
    fi
  ) &

  artifact_fetch_pids+=("$!")
  artifact_fetch_kinds+=("${artifact_kind}")
}

collect_server_endpoints() {
  local hosts_file="$1"
  awk '
    /^[[:space:]]*#/ || NF == 0 { next }
    NF >= 3 { print $1, $2, $3 }
  ' "${hosts_file}"
}

wait_for_server_endpoints_from_node0_or_fail() {
  local node0_host="$1"
  local node0_user="$2"
  local hosts_file="$3"
  local deadline_epoch_s="$4"
  local endpoints
  local last_unready=""

  endpoints="$(collect_server_endpoints "${hosts_file}")"
  if [[ -z "${endpoints}" ]]; then
    echo "No server endpoints found in ${hosts_file}" >&2
    return 1
  fi

  while true; do
    local all_ready=1
    local unready=""
    while read -r node_id node_ip node_port; do
      [[ -z "${node_id}" ]] && continue
      if ! ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" \
        "timeout 1 bash -c 'exec 3<>/dev/tcp/${node_ip}/${node_port}'" >/dev/null 2>&1; then
        all_ready=0
        unready+="${node_id}@${node_ip}:${node_port} "
      fi
    done <<< "${endpoints}"

    if (( all_ready == 1 )); then
      return 0
    fi

    last_unready="${unready}"
    if (( $(date +%s) >= deadline_epoch_s )); then
      break
    fi
    sleep 1
  done

  echo "Timed out waiting for SmartBFT server ports from ${node0_host}." >&2
  [[ -z "${last_unready}" ]] || echo "Still unreachable: ${last_unready}" >&2
  return 1
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
      F="${2:-}"
      shift 2
      ;;
    -t)
      EXPERIMENT_TIME_S="${2:-}"
      shift 2
      ;;
    --dataset)
      DATASET_ENABLED=1
      shift
      ;;
    --log)
      LOG_ENABLED=1
      shift
      ;;
    --verbose)
      VERBOSE_ENABLED=1
      shift
      ;;
    --model)
      AGENT_MODEL_TYPE="${2:-}"
      AGENT_MODEL_SET=1
      shift 2
      ;;
    --nodes)
      NODES="${2:-}"
      NODES_SET=1
      shift 2
      ;;
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --failure-spec)
      FAILURE_SPEC_PATH="${2:-}"
      FAILURE_SPEC_ENABLED=1
      shift 2
      ;;
    --no-failure-spec)
      FAILURE_SPEC_ENABLED=0
      shift
      ;;
    --batch-size)
      BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --batch-timeout)
      BATCH_TIMEOUT="${2:-}"
      shift 2
      ;;
    --request-timeout)
      REQUEST_TIMEOUT="${2:-}"
      shift 2
      ;;
    --submit-mode)
      SUBMIT_MODE="${2:-}"
      shift 2
      ;;
    --create-workers)
      CREATE_WORKERS="${2:-}"
      shift 2
      ;;
    --client-start-delay)
      CLIENT_START_DELAY_S="${2:-}"
      shift 2
      ;;
    --start-offset)
      START_OFFSET_AFTER_CLIENT_LAUNCH_S="${2:-}"
      shift 2
      ;;
    --port-base)
      SERVER_PORT_BASE="${2:-}"
      shift 2
      ;;
    --agent-port)
      AGENT_PORT="${2:-}"
      shift 2
      ;;
    --learning-node-id)
      LEARNING_NODE_ID="${2:-}"
      shift 2
      ;;
    --initial-timeout)
      INITIAL_TIMEOUT="${2:-}"
      shift 2
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
if (( NODES_SET == 0 )); then
  NODES="$((3 * F + 1))"
fi
for value_name in NODES EXPERIMENT_TIME_S BATCH_SIZE SERVER_PORT_BASE AGENT_PORT; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "Invalid ${value_name}: ${value}" >&2
    exit 2
  fi
done
for value_name in CLIENT_START_DELAY_S START_OFFSET_AFTER_CLIENT_LAUNCH_S; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "Invalid ${value_name}: ${value}" >&2
    exit 2
  fi
done
if (( NODES < 4 )); then
  echo "Invalid node count: ${NODES}. SmartBFT SmallBank requires at least 4 nodes." >&2
  exit 2
fi
if (( SERVER_PORT_BASE + NODES > 65535 )); then
  echo "Invalid --port-base ${SERVER_PORT_BASE} for ${NODES} nodes." >&2
  exit 2
fi
case "${SUBMIT_MODE}" in
  broadcast|leader) ;;
  *) echo "Invalid --submit-mode value: ${SUBMIT_MODE}. Expected broadcast or leader." >&2; exit 2 ;;
esac
if [[ -n "${CREATE_WORKERS}" ]] && (! [[ "${CREATE_WORKERS}" =~ ^[0-9]+$ ]] || (( CREATE_WORKERS < 1 ))); then
  echo "Invalid --create-workers value: ${CREATE_WORKERS}." >&2
  exit 2
fi
if ! AGENT_MODEL_TYPE="$(normalize_agent_model_type "${AGENT_MODEL_TYPE}")"; then
  echo "Invalid --model value: ${AGENT_MODEL_TYPE}. Use one of random|linear|cmab." >&2
  exit 2
fi
if (( AGENT_MODEL_SET == 1 )) && [[ "${MODE}" != "learning" ]]; then
  echo "--model is only supported with learning mode for SmartBFT CloudLab runs." >&2
  exit 2
fi
if ! mode_enables_learning_agent "${MODE}"; then
  LEARNING_AGENT_ENABLED=0
else
  LEARNING_AGENT_ENABLED=1
fi
if (( DATASET_ENABLED == 1 )) && (( LEARNING_AGENT_ENABLED == 0 )); then
  echo "--dataset requires heuristic|learning|sharing|scheduled mode." >&2
  exit 2
fi
if (( LEARNING_AGENT_ENABLED == 1 )); then
  DATASET_ENABLED=1
fi
if [[ "${MODE}" == "sharing" ]]; then
  echo "SmartBFT sharing mode starts one learning agent per node." >&2
fi

LOCAL_SSH_KEY_PATH="${LOCAL_SSH_KEY_PATH/#\~/$HOME}"
CONFIG_PATH="$(absolute_path "${CONFIG_PATH}")"
FAILURE_SPEC_PATH="$(absolute_path "${FAILURE_SPEC_PATH}")"

if [[ ! -f "${LOCAL_SSH_KEY_PATH}" ]]; then
  echo "Local SSH key not found: ${LOCAL_SSH_KEY_PATH}" >&2
  exit 1
fi
SSH_OPTS+=(-i "${LOCAL_SSH_KEY_PATH}" -o IdentitiesOnly=yes)
if [[ ! -d "${SMARTBFT_SRC}" ]]; then
  echo "SmartBFT directory not found: ${SMARTBFT_SRC}" >&2
  exit 1
fi
if [[ ! -d "${LEARNING_AGENT_SRC_DIR}" ]]; then
  echo "learning_agent directory not found: ${LEARNING_AGENT_SRC_DIR}" >&2
  exit 1
fi
if [[ ! -f "${APPLIER_LOCAL_PATH}" ]]; then
  echo "Network applier script not found: ${APPLIER_LOCAL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "SmallBank config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if (( FAILURE_SPEC_ENABLED == 1 )) && [[ ! -f "${FAILURE_SPEC_PATH}" ]]; then
  echo "Failure spec not found: ${FAILURE_SPEC_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TARGETS_FILE}" ]]; then
  echo "Targets file not found: ${TARGETS_FILE}" >&2
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

machine_count="${#targets_nodes[@]}"
if (( machine_count < 1 )); then
  echo "No targets found in ${TARGETS_FILE}" >&2
  exit 1
fi

node0_host=""
node0_user=""
server_nodes=()
server_hosts=()
server_users=()
for i in "${!targets_nodes[@]}"; do
  node="${targets_nodes[$i]}"
  host="${targets_hosts[$i]}"
  user="${targets_users[$i]}"
  if [[ "${node}" == "node0" ]]; then
    node0_host="${host}"
    node0_user="${user}"
    if (( machine_count == 1 )); then
      server_nodes+=("${node}")
      server_hosts+=("${host}")
      server_users+=("${user}")
    fi
  else
    server_nodes+=("${node}")
    server_hosts+=("${host}")
    server_users+=("${user}")
  fi
done
if [[ -z "${node0_host}" || -z "${node0_user}" ]]; then
  echo "node0 entry not found in ${TARGETS_FILE}" >&2
  exit 1
fi
if (( ${#server_nodes[@]} < 1 )); then
  echo "No server nodes found from targets file: ${TARGETS_FILE}" >&2
  exit 1
fi

if (( machine_count == 1 )); then
  replicas_per_server_machine="${NODES}"
else
  server_machine_count=$((machine_count - 1))
  if (( NODES % server_machine_count != 0 )); then
    echo "Replica layout mismatch: nodes=${NODES} not divisible by server machine count=${server_machine_count}" >&2
    exit 1
  fi
  replicas_per_server_machine=$((NODES / server_machine_count))
fi
server_machine_count="${#server_nodes[@]}"

echo "Resolving server gateway IPs..."
server_gateway_ips=()
for i in "${!server_nodes[@]}"; do
  node="${server_nodes[$i]}"
  host="${server_hosts[$i]}"
  user="${server_users[$i]}"
  gateway_ip=""
  if is_ipv4_literal "${host}"; then
    gateway_ip="${host}"
  else
    gateway_ip="$(ssh "${SSH_OPTS[@]}" "${user}@${host}" "hostname -I | awk '{print \$1}'" 2>/dev/null || true)"
  fi
  if ! is_ipv4_literal "${gateway_ip}"; then
    echo "Failed to resolve gateway IP for ${node} (${user}@${host})." >&2
    exit 1
  fi
  server_gateway_ips+=("${gateway_ip}")
  echo "  - ${node}: ${gateway_ip}"
done

node0_advertise_ip=""
if is_ipv4_literal "${node0_host}"; then
  node0_advertise_ip="${node0_host}"
else
  node0_advertise_ip="$(ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" "hostname -I | awk '{print \$1}'" 2>/dev/null || true)"
fi
if ! is_ipv4_literal "${node0_advertise_ip}"; then
  echo "Failed to resolve node0 advertise IP." >&2
  exit 1
fi
echo "Client reply advertise IP: ${node0_advertise_ip}"

mkdir -p "${LOCAL_TMP_DIR}"
HOSTS_CONFIG_LOCAL="${LOCAL_TMP_DIR}/smartbft_hosts.config"
FAILURE_HOSTS_CONFIG_LOCAL="${LOCAL_TMP_DIR}/smartbft_hosts.failure.config"
: > "${HOSTS_CONFIG_LOCAL}"
: > "${FAILURE_HOSTS_CONFIG_LOCAL}"
for ((replica_id = 0; replica_id < NODES; replica_id++)); do
  machine_idx=$((replica_id / replicas_per_server_machine))
  slot=$((replica_id % replicas_per_server_machine))
  node_id=$((replica_id + 1))
  ip="${OVERLAY_OCTET_A}.${OVERLAY_OCTET_B}.$((machine_idx + 1)).$((OVERLAY_HOST_OFFSET + slot))"
  port=$((SERVER_PORT_BASE + node_id))
  printf '%d %s %d\n' "${node_id}" "${ip}" "${port}" >> "${HOSTS_CONFIG_LOCAL}"
  printf '%d %s %d\n' "${replica_id}" "${ip}" "${port}" >> "${FAILURE_HOSTS_CONFIG_LOCAL}"
done

echo "Selected mode: ${MODE}"
echo "SmartBFT nodes: ${NODES} (f=${F}, replicas_per_server_machine=${replicas_per_server_machine})"
echo "Submit mode: ${SUBMIT_MODE}"
echo "Batch size: ${BATCH_SIZE}; batch timeout: ${BATCH_TIMEOUT}; request timeout: ${REQUEST_TIMEOUT}"
echo "Request timeout backoff: enabled (max=${REQUEST_TIMEOUT_BACKOFF_MAX})"
echo "SmartBFT verbose server logs: ${VERBOSE_ENABLED}"
echo "Learning-agent orchestration: ${LEARNING_AGENT_ENABLED} (dataset=${DATASET_ENABLED}, log=${LOG_ENABLED})"
if [[ "${MODE}" == "learning" ]]; then
  echo "Learning-agent model: ${AGENT_MODEL_TYPE}"
fi

echo "Creating SmartBFT CloudLab payload tarball..."
rm -f "${LOCAL_TAR}"
tar -czf "${LOCAL_TAR}" \
  --exclude='SmartBFT/.git' \
  --exclude='SmartBFT/build' \
  -C "${REPO_ROOT}" SmartBFT config learning_agent scripts/apply_network_failures.py

echo "Copying payload to CloudLab nodes and building SmartBFT SmallBank..."
pids=()
nodes=()
for i in "${!targets_nodes[@]}"; do
  node="${targets_nodes[$i]}"
  host="${targets_hosts[$i]}"
  user="${targets_users[$i]}"
  (
    remote_tar="/tmp/smartbft_payload_$$.tar.gz"
    remote_config="/tmp/smartbft_smallbank_$$.xml"
    remote_failure_spec=""
    scp "${SSH_OPTS[@]}" "${LOCAL_TAR}" "${user}@${host}:${remote_tar}"
    scp "${SSH_OPTS[@]}" "${HOSTS_CONFIG_LOCAL}" "${user}@${host}:/tmp/smartbft_hosts_$$.config"
    scp "${SSH_OPTS[@]}" "${FAILURE_HOSTS_CONFIG_LOCAL}" "${user}@${host}:/tmp/smartbft_hosts_failure_$$.config"
    scp "${SSH_OPTS[@]}" "${CONFIG_PATH}" "${user}@${host}:${remote_config}"
    if (( FAILURE_SPEC_ENABLED == 1 )); then
      remote_failure_spec="/tmp/smartbft_failure_spec_$$.xml"
      scp "${SSH_OPTS[@]}" "${FAILURE_SPEC_PATH}" "${user}@${host}:${remote_failure_spec}"
    fi
    ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- "${remote_tar}" "/tmp/smartbft_hosts_$$.config" "/tmp/smartbft_hosts_failure_$$.config" "${remote_config}" "${remote_failure_spec}" <<'REMOTE'
set -euo pipefail

tar_path="$1"
hosts_path="$2"
failure_hosts_path="$3"
config_path="$4"
failure_spec_path="$5"
run_root="$HOME/smartbft_cloudlab"

remove_path_with_sudo_fallback() {
  local target="$1"
  if [[ ! -e "${target}" ]]; then
    return 0
  fi
  if rm -rf "${target}" 2>/dev/null; then
    return 0
  fi
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo -n rm -rf "${target}"
    return 0
  fi
  echo "Failed to remove ${target} on $(hostname)." >&2
  return 1
}

remove_path_with_sudo_fallback "${run_root}"
mkdir -p "${run_root}"
tar -xzf "${tar_path}" -C "${run_root}"
mkdir -p "${run_root}/bin" "${run_root}/logs" "${run_root}/wal"
cp "${hosts_path}" "${run_root}/hosts.config"
cp "${failure_hosts_path}" "${run_root}/hosts.failure.config"
mkdir -p "${run_root}/config/pbft"
cp "${config_path}" "${run_root}/config/pbft/smallbank.xml"
if [[ -n "${failure_spec_path}" ]]; then
  cp "${failure_spec_path}" "${run_root}/config/pbft/failure_spec.xml"
fi
chmod +x "${run_root}/scripts/apply_network_failures.py"
rm -f "${tar_path}" "${hosts_path}" "${failure_hosts_path}" "${config_path}"
if [[ -n "${failure_spec_path}" ]]; then
  rm -f "${failure_spec_path}"
fi

export PATH="$HOME/go/bin:${PATH}"
if ! command -v go >/dev/null 2>&1; then
  echo "go not found on $(hostname)" >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found on $(hostname)" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on $(hostname)" >&2
  exit 1
fi

cd "${run_root}/SmartBFT"
echo "Using Go toolchain: $(go version)"
GOCACHE="${run_root}/gocache" go build -o "${run_root}/bin/smartbft-smallbank" ./examples/smallbank
REMOTE
  ) &
  pids+=("$!")
  nodes+=("${node}")
done

copy_failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "  - ${nodes[$i]}: ready"
  else
    echo "  - ${nodes[$i]}: setup failed" >&2
    copy_failed=1
  fi
done
rm -f "${LOCAL_TAR}"
if (( copy_failed != 0 )); then
  echo "Payload setup failed on one or more nodes." >&2
  exit 1
fi

echo "Checking passwordless sudo availability..."
sudo_check_failed=0
for i in "${!targets_nodes[@]}"; do
  node="${targets_nodes[$i]}"
  host="${targets_hosts[$i]}"
  user="${targets_users[$i]}"
  if ssh "${SSH_OPTS[@]}" "${user}@${host}" "sudo -n true" >/dev/null 2>&1; then
    echo "  - ${node}: sudo -n OK"
  else
    echo "  - ${node}: sudo -n FAILED" >&2
    sudo_check_failed=1
  fi
done
if (( sudo_check_failed != 0 )); then
  echo "Namespace setup requires passwordless sudo on all targets." >&2
  exit 1
fi

echo "Setting up SmartBFT per-replica network namespaces..."
ns_setup_failed=0
for i in "${!server_nodes[@]}"; do
  node="${server_nodes[$i]}"
  host="${server_hosts[$i]}"
  user="${server_users[$i]}"
  machine_idx="${i}"
  if ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- \
    "${replicas_per_server_machine}" \
    "${machine_idx}" \
    "${server_machine_count}" \
    "${OVERLAY_OCTET_A}" \
    "${OVERLAY_OCTET_B}" \
    "${OVERLAY_MASK_BITS}" \
    "${OVERLAY_BRIDGE_NAME}" \
    "${OVERLAY_HOST_OFFSET}" \
    "${server_gateway_ips[@]}" <<'REMOTE'
set -euo pipefail

replicas_per_machine="$1"
machine_idx="$2"
server_machine_count="$3"
overlay_a="$4"
overlay_b="$5"
overlay_mask="$6"
bridge_name="$7"
host_offset="$8"
shift 8
server_gateway_ips=("$@")

bridge_ip="${overlay_a}.${overlay_b}.$((machine_idx + 1)).1"
local_subnet="${overlay_a}.${overlay_b}.$((machine_idx + 1)).0/${overlay_mask}"

existing_ns="$(ip netns list | awk '/^smartbft-r[0-9]+/ {print $1}')"
if [[ -n "${existing_ns}" ]]; then
  printf '%s\n' "${existing_ns}" | while read -r ns_name; do
    [[ -z "${ns_name}" ]] && continue
    sudo -n ip netns del "${ns_name}" >/dev/null 2>&1 || true
  done
fi
stale_veths="$(ip -o link show | awk -F': ' '{print $2}' | sed 's/@.*//' | awk '/^sbh[0-9]+$/')"
if [[ -n "${stale_veths}" ]]; then
  printf '%s\n' "${stale_veths}" | while read -r if_name; do
    [[ -z "${if_name}" ]] && continue
    sudo -n ip link del "${if_name}" >/dev/null 2>&1 || true
  done
fi

sudo -n ip link set "${bridge_name}" down >/dev/null 2>&1 || true
sudo -n ip link del "${bridge_name}" type bridge >/dev/null 2>&1 || true
sudo -n ip link add name "${bridge_name}" type bridge
sudo -n ip addr replace "${bridge_ip}/${overlay_mask}" dev "${bridge_name}"
sudo -n ip link set "${bridge_name}" up
sudo -n sysctl -w net.ipv4.ip_forward=1 >/dev/null

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  ns_name="smartbft-r${replica_id}"
  host_if="sbh${replica_id}"
  ns_if="sbn${replica_id}"
  ns_ip="${overlay_a}.${overlay_b}.$((machine_idx + 1)).$((host_offset + n))"

  sudo -n ip netns del "${ns_name}" >/dev/null 2>&1 || true
  sudo -n ip link del "${host_if}" >/dev/null 2>&1 || true
  sudo -n ip netns add "${ns_name}"
  sudo -n ip link add "${host_if}" type veth peer name "${ns_if}"
  sudo -n ip link set "${ns_if}" netns "${ns_name}"
  sudo -n ip link set "${host_if}" master "${bridge_name}"
  sudo -n ip link set "${host_if}" up

  sudo -n ip netns exec "${ns_name}" ip link set lo up
  sudo -n ip netns exec "${ns_name}" ip addr replace "${ns_ip}/${overlay_mask}" dev "${ns_if}"
  sudo -n ip netns exec "${ns_name}" ip link set "${ns_if}" up
  sudo -n ip netns exec "${ns_name}" ip route replace default via "${bridge_ip}"
done

for ((peer_idx = 0; peer_idx < server_machine_count; peer_idx++)); do
  if (( peer_idx == machine_idx )); then
    continue
  fi
  subnet="${overlay_a}.${overlay_b}.$((peer_idx + 1)).0/${overlay_mask}"
  gw_ip="${server_gateway_ips[$peer_idx]}"
  sudo -n ip route replace "${subnet}" via "${gw_ip}"
done

echo "Configured ${replicas_per_machine} SmartBFT namespaces on $(hostname) in ${local_subnet}."
REMOTE
  then
    echo "  - ${node}: namespaces ready"
  else
    echo "  - ${node}: namespace setup failed" >&2
    ns_setup_failed=1
  fi
done
if (( ns_setup_failed != 0 )); then
  echo "Namespace setup failed on one or more server nodes." >&2
  exit 1
fi

if (( machine_count > 1 )); then
  echo "Configuring client host routes on node0..."
  if ! ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" "bash -s" -- \
    "${server_machine_count}" \
    "${OVERLAY_OCTET_A}" \
    "${OVERLAY_OCTET_B}" \
    "${OVERLAY_MASK_BITS}" \
    "${server_nodes[@]}" \
    "${server_hosts[@]}" <<'REMOTE'
set -euo pipefail

server_machine_count="$1"
overlay_a="$2"
overlay_b="$3"
overlay_mask="$4"
shift 4

server_nodes=()
for ((i = 0; i < server_machine_count; i++)); do
  server_nodes+=("$1")
  shift
done
server_hosts=()
for ((i = 0; i < server_machine_count; i++)); do
  server_hosts+=("$1")
  shift
done

is_ipv4_literal() {
  local value="$1"
  [[ "${value}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]
}

for ((peer_idx = 0; peer_idx < server_machine_count; peer_idx++)); do
  peer_node="${server_nodes[$peer_idx]}"
  peer_host="${server_hosts[$peer_idx]}"
  subnet="${overlay_a}.${overlay_b}.$((peer_idx + 1)).0/${overlay_mask}"
  gw_ip="$(getent ahostsv4 "${peer_node}" 2>/dev/null | awk 'NR==1 {print $1}' || true)"
  if ! is_ipv4_literal "${gw_ip}"; then
    if is_ipv4_literal "${peer_host}"; then
      gw_ip="${peer_host}"
    else
      gw_ip="$(getent ahostsv4 "${peer_host}" 2>/dev/null | awk 'NR==1 {print $1}' || true)"
    fi
  fi
  if ! is_ipv4_literal "${gw_ip}"; then
    echo "Failed to resolve gateway IP on node0 for ${peer_node}/${peer_host}" >&2
    exit 1
  fi
  if ! sudo -n ip route replace "${subnet}" via "${gw_ip}"; then
    route_get="$(ip -4 route get "${gw_ip}" 2>/dev/null | head -n1 || true)"
    route_dev="$(awk '{for (i = 1; i <= NF; i++) if ($i == "dev") { print $(i + 1); exit }}' <<< "${route_get}")"
    [[ -n "${route_dev}" ]] && sudo -n ip route replace "${subnet}" via "${gw_ip}" dev "${route_dev}" onlink && continue
    echo "Failed route replace on node0: subnet=${subnet} gw=${gw_ip}" >&2
    exit 1
  fi
done
REMOTE
  then
    echo "  - node0: failed to configure client routes" >&2
    exit 1
  fi
  echo "  - node0: client routes configured"
fi

START_UNIX_MS="$(( ($(date +%s) + CLIENT_START_DELAY_S + START_OFFSET_AFTER_CLIENT_LAUNCH_S) * 1000 ))"
echo "Using shared start_unix_ms=${START_UNIX_MS}"

if (( FAILURE_SPEC_ENABLED == 1 )); then
  echo "Starting SmartBFT network failure appliers..."
  applier_start_failed=0
  for i in "${!server_nodes[@]}"; do
    node="${server_nodes[$i]}"
    host="${server_hosts[$i]}"
    user="${server_users[$i]}"
    machine_idx="${i}"
    if ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- "${replicas_per_server_machine}" "${machine_idx}" "${START_UNIX_MS}" <<'REMOTE'
set -euo pipefail

replicas_per_machine="$1"
machine_idx="$2"
start_unix_ms="$3"
run_root="$HOME/smartbft_cloudlab"

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  ns_name="smartbft-r${replica_id}"
  interface_name="sbn${replica_id}"
  session_name="smartbft_applier_${replica_id}"
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  tmux new-session -d -s "${session_name}" \
    "cd \"${run_root}\" && python3 \"${run_root}/scripts/apply_network_failures.py\" --failure-spec \"${run_root}/config/pbft/failure_spec.xml\" --hosts-config \"${run_root}/hosts.failure.config\" --replica-id ${replica_id} --netns ${ns_name} --interface ${interface_name} --start-unix-ms ${start_unix_ms} > \"${run_root}/logs/applier_${replica_id}.log\" 2>&1"
done
REMOTE
    then
      echo "  - ${node}: appliers started"
    else
      echo "  - ${node}: failed to start appliers" >&2
      applier_start_failed=1
    fi
  done
  if (( applier_start_failed != 0 )); then
    echo "Applier start failed on one or more nodes." >&2
    exit 1
  fi
else
  echo "Failure spec disabled; not starting network appliers."
fi

if (( LEARNING_AGENT_ENABLED == 1 )); then
  echo "Starting SmartBFT learning agents..."
  agent_start_failed=0
  agent_start_pids=()
  agent_start_nodes=()
  for i in "${!server_nodes[@]}"; do
    node="${server_nodes[$i]}"
    host="${server_hosts[$i]}"
    user="${server_users[$i]}"
    machine_idx="${i}"
    (
      ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- \
        "${replicas_per_server_machine}" \
        "${machine_idx}" \
        "${START_UNIX_MS}" \
        "${MODE}" \
        "${DATASET_ENABLED}" \
        "${AGENT_MODEL_TYPE}" \
        "${AGENT_PORT}" <<'REMOTE'
set -euo pipefail

replicas_per_machine="$1"
machine_idx="$2"
start_unix_ms="$3"
mode="$4"
dataset_enabled="$5"
model_type="$6"
agent_port="$7"
run_root="$HOME/smartbft_cloudlab"

agent_model_args=""
if [[ "${mode}" == "learning" || "${mode}" == "sharing" ]]; then
  agent_model_args="--model-type ${model_type}"
elif [[ "${mode}" == "scheduled" ]]; then
  agent_model_args="--model-type scheduled --schedule-file learning_agent/config/schedule_pbft.csv"
fi

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  ns_name="smartbft-r${replica_id}"
  session_name="smartbft_agent_${replica_id}"
  agent_log="${run_root}/logs/agent_${replica_id}.log"
  dataset_args=""
  if [[ "${dataset_enabled}" == "1" ]]; then
    rm -f "${run_root}/agent${replica_id}.csv"
    dataset_args="--dataset-csv ${run_root}/agent${replica_id}.csv"
  fi
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  tmux new-session -d -s "${session_name}" \
    "sudo -n ip netns exec ${ns_name} bash -lc 'cd \"${run_root}\" && exec python3 learning_agent/main.py --node-id ${replica_id} --protocol pbft --host 127.0.0.1 --port ${agent_port} --signing false ${agent_model_args} ${dataset_args} --failure-spec \"${run_root}/config/pbft/failure_spec.xml\" --injection-start-unix-ms ${start_unix_ms}' > \"${agent_log}\" 2>&1"
done

sleep 2
for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  session_name="smartbft_agent_${replica_id}"
  agent_log="${run_root}/logs/agent_${replica_id}.log"
  if ! tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "Learning-agent session exited early: ${session_name}" >&2
    tail -n 80 "${agent_log}" >&2 || true
    exit 1
  fi
done
REMOTE
    ) &
    agent_start_pids+=("$!")
    agent_start_nodes+=("${node}")
  done
  for i in "${!agent_start_pids[@]}"; do
    if wait "${agent_start_pids[$i]}"; then
      echo "  - ${agent_start_nodes[$i]}: learning agents started"
    else
      echo "  - ${agent_start_nodes[$i]}: failed to start learning agents" >&2
      agent_start_failed=1
    fi
  done
  if (( agent_start_failed != 0 )); then
    echo "SmartBFT learning-agent start failed on one or more nodes." >&2
    exit 1
  fi
fi

echo "Starting SmartBFT server replicas..."
server_start_failed=0
server_start_pids=()
server_start_nodes=()
for i in "${!server_nodes[@]}"; do
  node="${server_nodes[$i]}"
  host="${server_hosts[$i]}"
  user="${server_users[$i]}"
  machine_idx="${i}"
  (
    ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- \
      "${replicas_per_server_machine}" \
      "${machine_idx}" \
      "${START_UNIX_MS}" \
	      "${FAILURE_SPEC_ENABLED}" \
	      "${LEARNING_AGENT_ENABLED}" \
	      "${AGENT_PORT}" \
	      "${BATCH_SIZE}" \
	      "${BATCH_TIMEOUT}" \
	      "${REQUEST_TIMEOUT}" \
	      "${INITIAL_TIMEOUT}" \
	      "${REQUEST_TIMEOUT_BACKOFF_MAX}" \
	      "${VERBOSE_ENABLED}" <<'REMOTE'
set -euo pipefail

replicas_per_machine="$1"
machine_idx="$2"
start_unix_ms="$3"
failure_enabled="$4"
learning_enabled="$5"
agent_port="$6"
batch_size="$7"
batch_timeout="$8"
request_timeout="$9"
initial_timeout="${10}"
backoff_max="${11}"
verbose_enabled="${12}"
run_root="$HOME/smartbft_cloudlab"

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  node_id=$((replica_id + 1))
  ns_name="smartbft-r${replica_id}"
  session_name="smartbft_server_${node_id}"
  server_log="${run_root}/logs/server_${node_id}.log"
  data_dir="${run_root}/wal/node_${node_id}"
  args=(
    "${run_root}/bin/smartbft-smallbank"
    "--role" "server"
    "--node-id" "${node_id}"
    "--hosts-config" "${run_root}/hosts.config"
    "--batch-size" "${batch_size}"
    "--batch-timeout" "${batch_timeout}"
    "--request-timeout" "${request_timeout}"
    "--request-timeout-backoff"
    "--request-timeout-backoff-max" "${backoff_max}"
    "--data-dir" "${data_dir}"
    "--keep-data"
  )
	  if [[ "${failure_enabled}" == "1" ]]; then
	    args+=("--failure-spec" "${run_root}/config/pbft/failure_spec.xml" "--failure-start-unix-ms" "${start_unix_ms}")
	  fi
	  if [[ "${learning_enabled}" == "1" ]]; then
	    args+=("--learning" "--agent-target" "127.0.0.1:${agent_port}" "--learning-node-id" "${node_id}" "--learning-initial-election-timeout" "${initial_timeout}")
	  fi
	  if [[ "${verbose_enabled}" == "1" ]]; then
	    args+=("--verbose")
	  fi
  printf -v server_cmd '%q ' "${args[@]}"
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  : > "${server_log}"
  tmux new-session -d -s "${session_name}" \
    "sudo -n ip netns exec ${ns_name} bash -lc 'exec ${server_cmd}' > \"${server_log}\" 2>&1"
done
sleep 2
for ((n = 0; n < replicas_per_machine; n++)); do
  node_id=$((machine_idx * replicas_per_machine + n + 1))
  session_name="smartbft_server_${node_id}"
  if ! tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "SmartBFT server session exited early: ${session_name}" >&2
    tail -n 100 "${run_root}/logs/server_${node_id}.log" >&2 || true
    exit 1
  fi
done
REMOTE
  ) &
  server_start_pids+=("$!")
  server_start_nodes+=("${node}")
done
for i in "${!server_start_pids[@]}"; do
  if wait "${server_start_pids[$i]}"; then
    echo "  - ${server_start_nodes[$i]}: servers started"
  else
    echo "  - ${server_start_nodes[$i]}: failed to start servers" >&2
    server_start_failed=1
  fi
done
if (( server_start_failed != 0 )); then
  echo "SmartBFT server start failed on one or more nodes." >&2
  exit 1
fi

client_delay_deadline="$(( $(date +%s) + CLIENT_START_DELAY_S ))"
if ! wait_for_server_endpoints_from_node0_or_fail "${node0_host}" "${node0_user}" "${HOSTS_CONFIG_LOCAL}" "${client_delay_deadline}"; then
  exit 1
fi
remaining_client_delay="$((client_delay_deadline - $(date +%s)))"
if (( remaining_client_delay > 0 )); then
  echo "Waiting ${remaining_client_delay}s before starting SmartBFT client..."
  sleep "${remaining_client_delay}"
fi

echo "Starting SmartBFT client on node0..."
remote_create_workers="${CREATE_WORKERS:-__default__}"
ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" "bash -s" -- \
  "${START_UNIX_MS}" \
  "${SUBMIT_MODE}" \
  "${REQUEST_TIMEOUT}" \
  "${remote_create_workers}" \
  "${node0_advertise_ip}" <<'REMOTE'
set -euo pipefail

start_unix_ms="$1"
submit_mode="$2"
request_timeout="$3"
create_workers="$4"
reply_advertise="$5"
if [[ "${create_workers}" == "__default__" ]]; then
  create_workers=""
fi
run_root="$HOME/smartbft_cloudlab"
args=(
  "${run_root}/bin/smartbft-smallbank"
  "--role" "client"
  "--hosts-config" "${run_root}/hosts.config"
  "--config" "${run_root}/config/pbft/smallbank.xml"
  "--request-timeout" "${request_timeout}"
  "--submit-mode" "${submit_mode}"
  "--client-reply-listen" "0.0.0.0:0"
  "--client-reply-advertise-host" "${reply_advertise}"
  "--create"
  "--execute"
  "--start-unix-ms" "${start_unix_ms}"
)
if [[ -n "${create_workers}" ]]; then
  args+=("--create-workers" "${create_workers}")
fi
printf -v client_cmd '%q ' "${args[@]}"
tmux kill-session -t smartbft_client >/dev/null 2>&1 || true
: > "${run_root}/logs/client.log"
tmux new-session -d -s smartbft_client \
  "cd \"${run_root}\" && exec ${client_cmd} > \"${run_root}/logs/client.log\" 2>&1"
REMOTE

EXPERIMENT_END_UNIX_MS="$((START_UNIX_MS + EXPERIMENT_TIME_S * 1000))"
EXPERIMENT_END_EPOCH_S="$((EXPERIMENT_END_UNIX_MS / 1000))"
echo "Experiment deadline unix_ms=${EXPERIMENT_END_UNIX_MS} (start_unix_ms + ${EXPERIMENT_TIME_S}s)"
while true; do
  now_epoch_s="$(date +%s)"
  remaining_experiment_s="$((EXPERIMENT_END_EPOCH_S - now_epoch_s))"
  if (( remaining_experiment_s <= 0 )); then
    echo "Experiment deadline reached."
    break
  fi
  remaining_minutes="$(((remaining_experiment_s + 59) / 60))"
  echo "Time left until experiment deadline: ${remaining_experiment_s}s (~${remaining_minutes}m)"
  sleep_s=300
  (( remaining_experiment_s < sleep_s )) && sleep_s="${remaining_experiment_s}"
  sleep "${sleep_s}"
done

echo "Stopping SmartBFT tmux sessions..."
stop_failed=0
for i in "${!targets_nodes[@]}"; do
  node="${targets_nodes[$i]}"
  host="${targets_hosts[$i]}"
  user="${targets_users[$i]}"
  if ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" <<'REMOTE'
set -euo pipefail
if ! command -v tmux >/dev/null 2>&1; then
  exit 0
fi
sessions="$(tmux ls -F '#S' 2>/dev/null || true)"
[[ -z "${sessions}" ]] && exit 0
printf '%s\n' "${sessions}" | while read -r session_name; do
  [[ -z "${session_name}" ]] && continue
  if [[ "${session_name}" == smartbft_server_* || "${session_name}" == smartbft_applier_* || "${session_name}" == smartbft_agent_* || "${session_name}" == smartbft_client ]]; then
    tmux send-keys -t "${session_name}" C-c >/dev/null 2>&1 || true
  fi
done
sleep 3
printf '%s\n' "${sessions}" | while read -r session_name; do
  [[ -z "${session_name}" ]] && continue
  if [[ "${session_name}" == smartbft_server_* || "${session_name}" == smartbft_applier_* || "${session_name}" == smartbft_agent_* || "${session_name}" == smartbft_client ]]; then
    tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  fi
done
REMOTE
  then
    echo "  - ${node}: stopped"
  else
    echo "  - ${node}: failed to stop cleanly" >&2
    stop_failed=1
  fi
done

echo "Tearing down SmartBFT namespace setup..."
teardown_failed=0
for i in "${!server_nodes[@]}"; do
  node="${server_nodes[$i]}"
  host="${server_hosts[$i]}"
  user="${server_users[$i]}"
  machine_idx="${i}"
  if ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- \
    "${machine_idx}" "${server_machine_count}" "${OVERLAY_OCTET_A}" "${OVERLAY_OCTET_B}" "${OVERLAY_MASK_BITS}" "${OVERLAY_BRIDGE_NAME}" <<'REMOTE'
set -euo pipefail
machine_idx="$1"
server_machine_count="$2"
overlay_a="$3"
overlay_b="$4"
overlay_mask="$5"
bridge_name="$6"
for ((peer_idx = 0; peer_idx < server_machine_count; peer_idx++)); do
  (( peer_idx == machine_idx )) && continue
  subnet="${overlay_a}.${overlay_b}.$((peer_idx + 1)).0/${overlay_mask}"
  sudo -n ip route del "${subnet}" >/dev/null 2>&1 || true
done
existing_ns="$(ip netns list | awk '/^smartbft-r[0-9]+/ {print $1}')"
if [[ -n "${existing_ns}" ]]; then
  printf '%s\n' "${existing_ns}" | while read -r ns_name; do
    [[ -z "${ns_name}" ]] && continue
    sudo -n ip netns del "${ns_name}" >/dev/null 2>&1 || true
  done
fi
sudo -n ip link set "${bridge_name}" down >/dev/null 2>&1 || true
sudo -n ip link del "${bridge_name}" type bridge >/dev/null 2>&1 || true
REMOTE
  then
    echo "  - ${node}: namespace teardown complete"
  else
    echo "  - ${node}: namespace teardown failed" >&2
    teardown_failed=1
  fi
done
if (( machine_count > 1 )); then
  if ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" "bash -s" -- \
    "${server_machine_count}" "${OVERLAY_OCTET_A}" "${OVERLAY_OCTET_B}" "${OVERLAY_MASK_BITS}" <<'REMOTE'
set -euo pipefail
server_machine_count="$1"
overlay_a="$2"
overlay_b="$3"
overlay_mask="$4"
for ((peer_idx = 0; peer_idx < server_machine_count; peer_idx++)); do
  subnet="${overlay_a}.${overlay_b}.$((peer_idx + 1)).0/${overlay_mask}"
  sudo -n ip route del "${subnet}" >/dev/null 2>&1 || true
done
REMOTE
  then
    echo "  - node0: client routes cleared"
  else
    echo "  - node0: failed to clear client routes" >&2
    teardown_failed=1
  fi
fi

mkdir -p "$(dirname "${OUTPUT_LOCAL}")"
artifact_fetch_pids=()
artifact_fetch_kinds=()
output_fetch_failed=0
log_fetch_failed=0
dataset_fetch_failed=0

CLIENT_LOG_LOCAL="${LOCAL_TMP_DIR}/smartbft_client.log"
echo "Fetching SmartBFT client log..."
queue_artifact_download "node0" "${node0_host}" "${node0_user}" \
  "\$HOME/smartbft_cloudlab/logs/client.log" \
  "smartbft_cloudlab/logs/client.log" \
  "${CLIENT_LOG_LOCAL}" \
  "smartbft client.log" \
  "output"

if (( LOG_ENABLED == 1 )); then
  LOGS_LOCAL_DIR="${LOCAL_TMP_DIR}/smartbft_logs"
  rm -rf "${LOGS_LOCAL_DIR}"
  mkdir -p "${LOGS_LOCAL_DIR}/servers" "${LOGS_LOCAL_DIR}/appliers" "${LOGS_LOCAL_DIR}/agents"
  for i in "${!server_nodes[@]}"; do
    node="${server_nodes[$i]}"
    host="${server_hosts[$i]}"
    user="${server_users[$i]}"
    machine_idx="${i}"
    for ((n = 0; n < replicas_per_server_machine; n++)); do
      replica_id=$((machine_idx * replicas_per_server_machine + n))
      node_id=$((replica_id + 1))
      queue_artifact_download "${node}" "${host}" "${user}" \
        "\$HOME/smartbft_cloudlab/logs/server_${node_id}.log" \
        "smartbft_cloudlab/logs/server_${node_id}.log" \
        "${LOGS_LOCAL_DIR}/servers/server_${node_id}.log" \
        "server_${node_id}.log" \
        "log"
      if (( FAILURE_SPEC_ENABLED == 1 )); then
        queue_artifact_download "${node}" "${host}" "${user}" \
          "\$HOME/smartbft_cloudlab/logs/applier_${replica_id}.log" \
          "smartbft_cloudlab/logs/applier_${replica_id}.log" \
          "${LOGS_LOCAL_DIR}/appliers/applier_${replica_id}.log" \
          "applier_${replica_id}.log" \
          "log" \
          1
      fi
      if (( LEARNING_AGENT_ENABLED == 1 )); then
        queue_artifact_download "${node}" "${host}" "${user}" \
          "\$HOME/smartbft_cloudlab/logs/agent_${replica_id}.log" \
          "smartbft_cloudlab/logs/agent_${replica_id}.log" \
          "${LOGS_LOCAL_DIR}/agents/agent_${replica_id}.log" \
          "agent_${replica_id}.log" \
          "log" \
          1
      fi
    done
  done
fi

if (( LEARNING_AGENT_ENABLED == 1 )) && (( DATASET_ENABLED == 1 )); then
  for i in "${!server_nodes[@]}"; do
    node="${server_nodes[$i]}"
    host="${server_hosts[$i]}"
    user="${server_users[$i]}"
    machine_idx="${i}"
    for ((n = 0; n < replicas_per_server_machine; n++)); do
      replica_id=$((machine_idx * replicas_per_server_machine + n))
      queue_artifact_download "${node}" "${host}" "${user}" \
        "\$HOME/smartbft_cloudlab/agent${replica_id}.csv" \
        "smartbft_cloudlab/agent${replica_id}.csv" \
        "${LOCAL_TMP_DIR}/agent${replica_id}.csv" \
        "agent${replica_id}.csv" \
        "dataset" \
        1
    done
  done
fi

for i in "${!artifact_fetch_pids[@]}"; do
  if ! wait "${artifact_fetch_pids[$i]}"; then
    case "${artifact_fetch_kinds[$i]}" in
      output) output_fetch_failed=1 ;;
      dataset) dataset_fetch_failed=1 ;;
      *) log_fetch_failed=1 ;;
    esac
  fi
done
if (( output_fetch_failed != 0 )); then
  echo "Failed to fetch SmartBFT client log." >&2
  exit 1
fi
write_filtered_output "${CLIENT_LOG_LOCAL}" "${OUTPUT_LOCAL}"
echo "Filtered SmartBFT output saved to ${OUTPUT_LOCAL}"

if (( log_fetch_failed != 0 )); then
  echo "Some SmartBFT logs could not be fetched." >&2
fi
if (( dataset_fetch_failed != 0 )); then
  echo "Some SmartBFT dataset files could not be fetched." >&2
fi
if (( LEARNING_AGENT_ENABLED == 1 )) && (( DATASET_ENABLED == 1 )); then
  saved_agent_datasets=()
  for ((replica_id = 0; replica_id < NODES; replica_id++)); do
    agent_dataset_local="${LOCAL_TMP_DIR}/agent${replica_id}.csv"
    if [[ -f "${agent_dataset_local}" ]]; then
      saved_agent_datasets+=("${agent_dataset_local}")
    fi
  done
  if (( ${#saved_agent_datasets[@]} > 0 )); then
    echo "Learning-agent datasets saved to:"
    printf '  %s\n' "${saved_agent_datasets[@]}"
  else
    echo "Learning-agent datasets not found." >&2
  fi
fi
if (( stop_failed != 0 )); then
  echo "Some nodes reported tmux stop issues; output was still collected." >&2
fi
if (( teardown_failed != 0 )); then
  echo "Some namespace teardown steps failed; manual cleanup may be required." >&2
fi
