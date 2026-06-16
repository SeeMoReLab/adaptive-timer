#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PREPARE_SCRIPT="${REPO_ROOT}/scripts/prepare_pbft.sh"
TARGETS_FILE_DEFAULT="${REPO_ROOT}/scripts/cloudlab/tmp/cloudlab_ssh_targets.txt"
LIBRARY_DIR="${REPO_ROOT}/scripts/tmp/library"
LEARNING_AGENT_SRC_DIR="${REPO_ROOT}/learning_agent"
PAYLOAD_DIR="${REPO_ROOT}/scripts/tmp/cloudlab_payload"
LOCAL_ZIP="${REPO_ROOT}/scripts/tmp/library.zip"
APPLIER_LOCAL_PATH="${REPO_ROOT}/scripts/apply_network_failures.py"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
MODE="normal"
F=1
TARGETS_FILE="${TARGETS_FILE_DEFAULT}"
EXPERIMENT_TIME_S=300
CLIENT_START_DELAY_S=10
START_OFFSET_AFTER_CLIENT_LAUNCH_S=20
OUTPUT_LOCAL="${REPO_ROOT}/scripts/cloudlab/tmp/pbft_output.txt"
DATASET_ENABLED=0
LOG_ENABLED=0
TRIM_OUTPUT=0
OVERLAY_OCTET_A=10
OVERLAY_OCTET_B=250
OVERLAY_MASK_BITS=24
OVERLAY_BRIDGE_NAME="br-pbft"
OVERLAY_HOST_OFFSET=10
LOG_LEVEL="INFO"
AGENT_MODEL_TYPE="random"
AGENT_MODEL_SET=0
ARTIFACT_FETCH_MAX_PARALLEL=8

mode_enables_learning_agent() {
  case "$1" in
    heuristic|learning|sharing|scheduled)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

normalize_log_level() {
  local level_upper
  level_upper="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
  case "${level_upper}" in
    TRACE|DEBUG|INFO|WARN|ERROR)
      printf '%s\n' "${level_upper}"
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

configure_logback_level() {
  local logback_file="$1"
  local level="$2"
  local tmp_file

  if [[ ! -f "${logback_file}" ]]; then
    echo "logback.xml not found: ${logback_file}" >&2
    return 1
  fi

  tmp_file="$(mktemp)"
  awk -v level="${level}" '
    {
      if ($0 ~ /^[[:space:]]*<root[[:space:]]+level="/) {
        gsub(/level="[^"]+"/, "level=\"" level "\"")
      } else if ($0 ~ /^[[:space:]]*<logger[[:space:]]+name="bftsmart"[[:space:]]+level="/) {
        gsub(/level="[^"]+"/, "level=\"" level "\"")
      } else if ($0 ~ /^[[:space:]]*<logger[[:space:]]+name="bftsmart\.proxy"[[:space:]]+level="/) {
        gsub(/level="[^"]+"/, "level=\"" level "\"")
      }
      print
    }
  ' "${logback_file}" > "${tmp_file}"
  mv "${tmp_file}" "${logback_file}"
}

trim_client_output_file() {
  local output_file="$1"
  local header_lines="$2"
  local tmp_file

  if [[ ! -f "${output_file}" ]]; then
    echo "Cannot trim missing output file: ${output_file}" >&2
    return 1
  fi

  tmp_file="$(mktemp)"
  awk -v header_lines="${header_lines}" '
    NR <= header_lines { print; next }
    /-- Monitor/ { print }
  ' "${output_file}" > "${tmp_file}"
  mv "${tmp_file}" "${output_file}"
}

rewrite_hosts_for_namespaces() {
  local hosts_file="$1"
  local replicas_per_machine="$2"
  local tmp_hosts

  if [[ ! -f "${hosts_file}" ]]; then
    echo "Hosts file not found for namespace rewrite: ${hosts_file}" >&2
    return 1
  fi

  tmp_hosts="$(mktemp)"
  awk \
    -v rpm="${replicas_per_machine}" \
    -v oa="${OVERLAY_OCTET_A}" \
    -v ob="${OVERLAY_OCTET_B}" \
    -v host_offset="${OVERLAY_HOST_OFFSET}" '
    BEGIN {
      OFS = " "
      if (rpm < 1) {
        print "Invalid replicas_per_machine for rewrite: " rpm > "/dev/stderr"
        exit 1
      }
    }
    /^[[:space:]]*#/ || NF == 0 { print; next }
    NF < 6 { print; next }
    {
      replica_id = $1 + 0
      machine_idx = int(replica_id / rpm)
      slot = replica_id % rpm
      ip = sprintf("%d.%d.%d.%d", oa, ob, machine_idx + 1, host_offset + slot)
      $2 = ip
      print $1, $2, $3, $4, $5, $6
    }
  ' "${hosts_file}" > "${tmp_hosts}"
  mv "${tmp_hosts}" "${hosts_file}"
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
    while read -r replica_id replica_ip replica_port; do
      [[ -z "${replica_id}" ]] && continue
      if ! ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" \
        "timeout 1 bash -c 'exec 3<>/dev/tcp/${replica_ip}/${replica_port}'" >/dev/null 2>&1; then
        all_ready=0
        unready+="${replica_id}@${replica_ip}:${replica_port} "
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

  echo "Timed out waiting for server ports to become reachable from ${node0_host}." >&2
  if [[ -n "${last_unready}" ]]; then
    echo "Still unreachable: ${last_unready}" >&2
  fi
  return 1
}

dump_remote_server_logs() {
  local lines="${1:-80}"
  for i in "${!server_nodes[@]}"; do
    local node="${server_nodes[$i]}"
    local host="${server_hosts[$i]}"
    local user="${server_users[$i]}"
    local machine_idx="${i}"
    echo "===== ${node} server logs =====" >&2
    ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- "${replicas_per_server_machine}" "${machine_idx}" "${lines}" <<'REMOTE' >&2 || true
set -euo pipefail

replicas_per_machine="$1"
machine_idx="$2"
lines="$3"

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  log_path="$HOME/library/server_${replica_id}.log"
  echo "--- replica ${replica_id} (${log_path}) ---"
  if [[ -f "${log_path}" ]]; then
    tail -n "${lines}" "${log_path}" || true
  else
    echo "(missing log file)"
  fi
done
REMOTE
  done
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

  # Avoid overwhelming sshd (CloudLab nodes may reset connections when too many
  # concurrent SSH/SCP handshakes are opened at once).
  while true; do
    running_jobs="$(jobs -pr | wc -l | tr -d '[:space:]')"
    if [[ -z "${running_jobs}" ]]; then
      running_jobs=0
    fi
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
      if (( missing_ok == 1 )); then
        exit 0
      fi
      exit 1
    fi
  ) &

  artifact_fetch_pids+=("$!")
  artifact_fetch_kinds+=("${artifact_kind}")
}

usage() {
  cat <<EOF
Usage: $0 <private_key_path> [normal|heuristic|learning|sharing|scheduled] [--dataset] [--log] [--trim-output] [--log-level <TRACE|DEBUG|INFO|WARN|ERROR>] [--model <random|linear|cmab>] [-f <fault_tolerance>] [-t <experiment_seconds>] [targets_file]

Arguments:
  private_key_path  Local SSH private key for CloudLab nodes.
  mode              Optional; one of normal|heuristic|learning|sharing|scheduled (default: normal).
  targets_file      Optional path to cloudlab SSH targets file.

Flags:
  --dataset         Collect per-replica learning-agent dataset CSVs as agentN.csv.
  --log             Download per-replica logs (server, applier, and mode-specific dataexchange/agent logs) to scripts/cloudlab/tmp/.
  --trim-output     Post-process pbft_output.txt: keep lines 1..19, then only lines containing "-- Monitor".
  --log-level       Logging level applied to root, bftsmart and bftsmart.proxy in staged logback.xml (default: INFO).
  --model           Learning-agent model for learning/sharing modes: random|linear|cmab (cmab uses Thompson sampling; default: random).
  -f <int>          Fault tolerance f passed to prepare_pbft.sh (default: 1).
  -t <int>          Seconds to run experiment before stopping tmux sessions (default: 60).
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
    --dataset)
      DATASET_ENABLED=1
      shift
      ;;
    --log)
      LOG_ENABLED=1
      shift
      ;;
    --trim-output)
      TRIM_OUTPUT=1
      shift
      ;;
    --log-level)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --log-level." >&2
        usage >&2
        exit 2
      fi
      LOG_LEVEL="${2}"
      shift 2
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
if ! LOG_LEVEL="$(normalize_log_level "${LOG_LEVEL}")"; then
  echo "Invalid --log-level value: ${LOG_LEVEL}. Use one of TRACE|DEBUG|INFO|WARN|ERROR." >&2
  exit 2
fi
if ! AGENT_MODEL_TYPE="$(normalize_agent_model_type "${AGENT_MODEL_TYPE}")"; then
  echo "Invalid --model value: ${AGENT_MODEL_TYPE}. Use one of random|linear|cmab." >&2
  exit 2
fi
if (( AGENT_MODEL_SET == 1 )) && [[ "${MODE}" != "learning" && "${MODE}" != "sharing" ]]; then
  echo "--model is only supported with learning or sharing mode." >&2
  exit 2
fi

LOCAL_SSH_KEY_PATH="${LOCAL_SSH_KEY_PATH/#\~/$HOME}"
if [[ ! -f "${LOCAL_SSH_KEY_PATH}" ]]; then
  echo "Local SSH key not found: ${LOCAL_SSH_KEY_PATH}" >&2
  exit 1
fi
SSH_OPTS+=(-i "${LOCAL_SSH_KEY_PATH}" -o IdentitiesOnly=yes)

if [[ ! -x "${PREPARE_SCRIPT}" ]]; then
  echo "prepare_pbft.sh not found or not executable: ${PREPARE_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${APPLIER_LOCAL_PATH}" ]]; then
  echo "Network applier script not found: ${APPLIER_LOCAL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TARGETS_FILE}" ]]; then
  echo "Targets file not found: ${TARGETS_FILE}" >&2
  exit 1
fi

if ! mode_enables_learning_agent "${MODE}"; then
  LEARNING_AGENT_ENABLED=0
else
  LEARNING_AGENT_ENABLED=1
fi
echo "Selected mode: ${MODE}"
if [[ "${MODE}" == "learning" || "${MODE}" == "sharing" ]]; then
  echo "Learning-agent model: ${AGENT_MODEL_TYPE}"
fi
echo "Learning-agent orchestration: ${LEARNING_AGENT_ENABLED} (dataset=${DATASET_ENABLED}, log=${LOG_ENABLED})"
echo "Preparing PBFT configs in cloudlab mode with f=${F}..."
"${PREPARE_SCRIPT}" cloudlab -f "${F}"

if [[ ! -d "${LIBRARY_DIR}" ]]; then
  echo "Prepared library directory not found: ${LIBRARY_DIR}" >&2
  exit 1
fi
configure_logback_level "${LIBRARY_DIR}/config/logback.xml" "${LOG_LEVEL}"
echo "Configured staged logback level: ${LOG_LEVEL}"
if [[ ! -d "${LEARNING_AGENT_SRC_DIR}" ]]; then
  echo "learning_agent directory not found: ${LEARNING_AGENT_SRC_DIR}" >&2
  exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "zip command not found on local machine." >&2
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

NUM_SERVERS="$(
  awk -F= '
    /^[[:space:]]*system\.servers\.num[[:space:]]*=/ {
      gsub(/[[:space:]]/, "", $2)
      print $2
      exit
    }
  ' "${LIBRARY_DIR}/smallbank/system.config"
)"
if ! [[ "${NUM_SERVERS}" =~ ^[0-9]+$ ]] || (( NUM_SERVERS < 1 )); then
  echo "Unable to determine system.servers.num from ${LIBRARY_DIR}/smallbank/system.config" >&2
  exit 1
fi

if (( machine_count == 1 )); then
  replicas_per_server_machine="${NUM_SERVERS}"
else
  server_machine_count=$((machine_count - 1))
  if (( NUM_SERVERS % server_machine_count != 0 )); then
    echo "Replica layout mismatch: system.servers.num=${NUM_SERVERS} not divisible by server machine count=${server_machine_count}" >&2
    exit 1
  fi
  replicas_per_server_machine=$((NUM_SERVERS / server_machine_count))
fi

if (( ${#server_nodes[@]} < 1 )); then
  echo "No server nodes found from targets file: ${TARGETS_FILE}" >&2
  exit 1
fi

is_ipv4_literal() {
  local value="$1"
  [[ "${value}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]
}

echo "Resolving server gateway IPs for cross-node routes..."
server_gateway_ips=()
for i in "${!server_nodes[@]}"; do
  node="${server_nodes[$i]}"
  host="${server_hosts[$i]}"
  user="${server_users[$i]}"
  gateway_ip=""

  if is_ipv4_literal "${host}"; then
    gateway_ip="${host}"
  else
    gateway_ip="$(
      ssh "${SSH_OPTS[@]}" "${user}@${host}" \
        "hostname -I | awk '{print \$1}'" 2>/dev/null || true
    )"
  fi

  if ! is_ipv4_literal "${gateway_ip}"; then
    echo "Failed to resolve gateway IP for ${node} (${user}@${host})." >&2
    exit 1
  fi

  server_gateway_ips+=("${gateway_ip}")
  echo "  - ${node}: ${gateway_ip}"
done

echo "Rewriting hosts configs for per-replica namespace IPs..."
rewrite_hosts_for_namespaces "${LIBRARY_DIR}/smallbank/hosts.config" "${replicas_per_server_machine}"
rewrite_hosts_for_namespaces "${LIBRARY_DIR}/dataexchange/hosts.config" "${replicas_per_server_machine}"

mkdir -p "$(dirname "${LOCAL_ZIP}")"
rm -f "${LOCAL_ZIP}"
rm -rf "${PAYLOAD_DIR}"
mkdir -p "${PAYLOAD_DIR}"
trap 'rm -f "${LOCAL_ZIP}"; rm -rf "${PAYLOAD_DIR}"' EXIT

echo "Creating temporary CloudLab payload zip (library + learning_agent)..."
cp -a "${LIBRARY_DIR}" "${PAYLOAD_DIR}/library"
cp -a "${LEARNING_AGENT_SRC_DIR}" "${PAYLOAD_DIR}/learning_agent"
(
  cd "${PAYLOAD_DIR}"
  zip -qr "${LOCAL_ZIP}" library learning_agent
)

echo "Copying payload zip to all CloudLab nodes and materializing per-role directories (parallel)..."
pids=()
nodes=()

for i in "${!targets_nodes[@]}"; do
  node="${targets_nodes[$i]}"
  host="${targets_hosts[$i]}"
  user="${targets_users[$i]}"

  create_client=0
  replica_count="${replicas_per_server_machine}"
  if [[ "${node}" == "node0" ]]; then
    create_client=1
    if (( machine_count > 1 )); then
      replica_count=0
    fi
  fi

  (
    remote_zip="/tmp/library_payload_$$.zip"
    remote_applier="/tmp/apply_network_failures_$$.py"
    scp "${SSH_OPTS[@]}" "${LOCAL_ZIP}" "${user}@${host}:${remote_zip}"
    scp "${SSH_OPTS[@]}" "${APPLIER_LOCAL_PATH}" "${user}@${host}:${remote_applier}"

    ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- "${remote_zip}" "${remote_applier}" "${replica_count}" "${create_client}" <<'REMOTE'
set -euo pipefail

zip_path="$1"
applier_path="$2"
replica_count="$3"
create_client="$4"
tmp_extract_dir="$HOME/.library_extract_$$"

cleanup() {
  rm -f "$zip_path"
  rm -f "$applier_path"
  rm -rf "$tmp_extract_dir"
}
trap cleanup EXIT

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
  echo "Failed to remove ${target} on $(hostname); permission denied and passwordless sudo is unavailable." >&2
  return 1
}

shopt -s nullglob
stale_paths=("$HOME/library" "$HOME/learning_agent" "$HOME"/smallbank[0-9]* "$HOME"/dataexchange[0-9]*)
shopt -u nullglob
for stale_path in "${stale_paths[@]}"; do
  remove_path_with_sudo_fallback "${stale_path}"
done

mkdir -p "$tmp_extract_dir"

if command -v unzip >/dev/null 2>&1; then
  unzip -q "$zip_path" -d "$tmp_extract_dir"
elif command -v python3 >/dev/null 2>&1; then
  python3 - "$zip_path" "$tmp_extract_dir" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as zf:
    zf.extractall(sys.argv[2])
PY
else
  echo "Neither unzip nor python3 found on $(hostname)" >&2
  exit 1
fi

library_template="$tmp_extract_dir/library"
learning_agent_template="$tmp_extract_dir/learning_agent"
if [[ ! -d "$library_template" ]]; then
  echo "library directory not found after extraction on $(hostname): $library_template" >&2
  exit 1
fi
if [[ ! -d "$learning_agent_template" ]]; then
  echo "learning_agent directory not found after extraction on $(hostname): $learning_agent_template" >&2
  exit 1
fi

cp -a "$library_template" "$HOME/library"
cp -a "$learning_agent_template" "$HOME/learning_agent"
cp -a "$applier_path" "$HOME/library/apply_network_failures.py"
chmod +x "$HOME/library/apply_network_failures.py"

if [[ ! -d "$HOME/library/smallbank" || ! -d "$HOME/library/dataexchange" ]]; then
  echo "smallbank/dataexchange directories missing under $HOME/library on $(hostname)" >&2
  exit 1
fi
if [[ ! -f "$HOME/learning_agent/main.py" ]]; then
  echo "learning_agent/main.py missing under $HOME/learning_agent on $(hostname)" >&2
  exit 1
fi
if [[ ! -f "$HOME/learning_agent/requirements.txt" ]]; then
  echo "learning_agent/requirements.txt missing under $HOME/learning_agent on $(hostname)" >&2
  exit 1
fi

if (( create_client == 1 )); then
  rm -rf "$HOME/library/client"
  cp -a "$HOME/library/smallbank" "$HOME/library/client"
fi

for ((idx = 0; idx < replica_count; idx++)); do
  cp -a "$HOME/library/smallbank" "$HOME/library/smallbank${idx}"
  cp -a "$HOME/library/dataexchange" "$HOME/library/dataexchange${idx}"
done
REMOTE
  ) &

  pids+=("$!")
  nodes+=("${node}")
done

copy_failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "  - ${nodes[$i]}: copied"
  else
    echo "  - ${nodes[$i]}: copy failed" >&2
    copy_failed=1
  fi
done

if [[ "${copy_failed}" -ne 0 ]]; then
  echo "Copy failed on one or more nodes." >&2
  exit 1
fi

rm -f "${LOCAL_ZIP}"

echo "Checking passwordless sudo availability for namespace setup..."
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
  echo "Verify with: ssh <node> 'sudo -n true'" >&2
  exit 1
fi

echo "Setting up per-replica network namespaces on server nodes..."
ns_setup_failed=0
server_machine_count="${#server_nodes[@]}"
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

# Ensure local hostname resolves without DNS (required by Java InetAddress.getLocalHost()).
local_fqdn="$(hostname -f 2>/dev/null || hostname)"
local_short="$(hostname -s 2>/dev/null || hostname)"
if ! getent hosts "${local_fqdn}" >/dev/null 2>&1; then
  if ! grep -Eq "(^|[[:space:]])${local_fqdn}([[:space:]]|$)" /etc/hosts; then
    echo "127.0.1.1 ${local_fqdn} ${local_short}" | sudo -n tee -a /etc/hosts >/dev/null
  fi
fi

# Clean stale namespaces from previous runs.
existing_ns="$(ip netns list | awk '/^pbft-r[0-9]+/ {print $1}')"
if [[ -n "${existing_ns}" ]]; then
  printf '%s\n' "${existing_ns}" | while read -r ns_name; do
    [[ -z "${ns_name}" ]] && continue
    sudo -n ip netns del "${ns_name}" >/dev/null 2>&1 || true
  done
fi

# Clean stale host-side veth interfaces from prior failed runs.
stale_veths="$(ip -o link show | awk -F': ' '{print $2}' | sed 's/@.*//' | awk '/^vh[0-9]+$/')"
if [[ -n "${stale_veths}" ]]; then
  printf '%s\n' "${stale_veths}" | while read -r if_name; do
    [[ -z "${if_name}" ]] && continue
    sudo -n ip link del "${if_name}" >/dev/null 2>&1 || true
  done
fi

# Recreate bridge for deterministic state.
sudo -n ip link set "${bridge_name}" down >/dev/null 2>&1 || true
sudo -n ip link del "${bridge_name}" type bridge >/dev/null 2>&1 || true
sudo -n ip link add name "${bridge_name}" type bridge
sudo -n ip addr replace "${bridge_ip}/${overlay_mask}" dev "${bridge_name}"
sudo -n ip link set "${bridge_name}" up
sudo -n sysctl -w net.ipv4.ip_forward=1 >/dev/null

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  ns_name="pbft-r${replica_id}"
  host_if="vh${replica_id}"
  ns_if="vn${replica_id}"
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

# Route to other namespace subnets through corresponding server hosts.
for ((peer_idx = 0; peer_idx < server_machine_count; peer_idx++)); do
  if (( peer_idx == machine_idx )); then
    continue
  fi
  subnet="${overlay_a}.${overlay_b}.$((peer_idx + 1)).0/${overlay_mask}"
  gw_ip="${server_gateway_ips[$peer_idx]}"
  if ! sudo -n ip route replace "${subnet}" via "${gw_ip}"; then
    echo "Failed route replace on $(hostname): subnet=${subnet} gw=${gw_ip}" >&2
    exit 1
  fi
done

echo "Configured ${replicas_per_machine} namespaces on $(hostname) in ${local_subnet}."
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
  if ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" "bash -s" -- \
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
  if [[ $# -lt 1 ]]; then
    echo "Invalid route setup arguments on node0." >&2
    exit 1
  fi
  server_nodes+=("$1")
  shift
done

server_hosts=()
for ((i = 0; i < server_machine_count; i++)); do
  if [[ $# -lt 1 ]]; then
    echo "Invalid route setup arguments on node0." >&2
    exit 1
  fi
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
  gw_ip=""

  # Prefer CloudLab node aliases (node1, node2, ...) because they typically
  # resolve to the experiment-network address visible from node0.
  if [[ -n "${peer_node}" ]]; then
    gw_ip="$(getent ahostsv4 "${peer_node}" 2>/dev/null | awk 'NR==1 {print $1}' || true)"
  fi
  if ! is_ipv4_literal "${gw_ip}"; then
    if is_ipv4_literal "${peer_host}"; then
      gw_ip="${peer_host}"
    else
      gw_ip="$(getent ahostsv4 "${peer_host}" 2>/dev/null | awk 'NR==1 {print $1}' || true)"
    fi
  fi

  if ! is_ipv4_literal "${gw_ip}"; then
    echo "Failed to resolve gateway IP on node0 for peer=${peer_node} host=${peer_host}" >&2
    exit 1
  fi

  # Retry with explicit device+onlink when strict nexthop validation rejects
  # the plain route form on some CloudLab network layouts.
  if ! sudo -n ip route replace "${subnet}" via "${gw_ip}"; then
    route_get="$(ip -4 route get "${gw_ip}" 2>/dev/null | head -n1 || true)"
    route_dev="$(awk '{for (i = 1; i <= NF; i++) if ($i == "dev") { print $(i + 1); exit }}' <<< "${route_get}")"
    if [[ -n "${route_dev}" ]] && sudo -n ip route replace "${subnet}" via "${gw_ip}" dev "${route_dev}" onlink; then
      continue
    fi
    echo "Failed route replace on node0: subnet=${subnet} gw=${gw_ip} peer=${peer_node} host=${peer_host}" >&2
    exit 1
  fi
done
REMOTE
  then
    echo "  - node0: client routes configured"
  else
    echo "  - node0: failed to configure client routes" >&2
    exit 1
  fi
fi

# Shared experiment/failure start time:
# - wait 10s after server launch before starting client
# - then start benchmark/failures at +10s after client launch
START_UNIX_MS="$(( ($(date +%s) + CLIENT_START_DELAY_S + START_OFFSET_AFTER_CLIENT_LAUNCH_S) * 1000 ))"
echo "Using shared start_unix_ms=${START_UNIX_MS}"

echo "Starting network failure appliers in tmux..."
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

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found on $(hostname)" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on $(hostname)" >&2
  exit 1
fi
if [[ ! -f "$HOME/library/apply_network_failures.py" ]]; then
  echo "applier missing on $(hostname): $HOME/library/apply_network_failures.py" >&2
  exit 1
fi
failure_spec_path="$HOME/library/config/failure_spec.xml"
if [[ -f "${failure_spec_path}" ]]; then
  echo "failure spec found on $(hostname): ${failure_spec_path}"
else
  echo "failure spec not found on $(hostname): ${failure_spec_path}" >&2
fi

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  ns_name="pbft-r${replica_id}"
  interface_name="vn${replica_id}"
  session_name="pbft_applier_${replica_id}"
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  tmux new-session -d -s "${session_name}" \
    "cd \"$HOME/library\" && python3 \"$HOME/library/apply_network_failures.py\" --failure-spec \"$HOME/library/config/failure_spec.xml\" --hosts-config \"$HOME/library/smallbank/hosts.config\" --replica-id ${replica_id} --netns ${ns_name} --interface ${interface_name} --start-unix-ms ${start_unix_ms} > \"$HOME/library/applier_${replica_id}.log\" 2>&1"
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

if [[ "${MODE}" == "sharing" ]]; then
  echo "Starting data-exchange PBFT servers in tmux (parallel)..."
  dataexchange_start_failed=0
  dataexchange_start_pids=()
  dataexchange_start_nodes=()
  for i in "${!server_nodes[@]}"; do
    node="${server_nodes[$i]}"
    host="${server_hosts[$i]}"
    user="${server_users[$i]}"
    machine_idx="${i}"

    (
      ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- "${replicas_per_server_machine}" "${machine_idx}" "${OVERLAY_OCTET_A}" "${OVERLAY_OCTET_B}" "${OVERLAY_HOST_OFFSET}" <<'REMOTE'
set -euo pipefail

replicas_per_machine="$1"
machine_idx="$2"
overlay_a="$3"
overlay_b="$4"
overlay_host_offset="$5"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found on $(hostname)" >&2
  exit 1
fi

if [[ ! -d "$HOME/library" ]]; then
  echo "library directory missing on $(hostname): $HOME/library" >&2
  exit 1
fi

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  ns_name="pbft-r${replica_id}"
  session_name="pbft_dataexchange_${replica_id}"
  dataexchange_log="$HOME/library/dataexchange_${replica_id}.log"
  bind_ip="${overlay_a}.${overlay_b}.$((machine_idx + 1)).$((overlay_host_offset + n))"
  config_path="$HOME/library/dataexchange${n}/system.config"
  if ! ip netns list | awk '{print $1}' | grep -qx "${ns_name}"; then
    echo "Namespace missing on $(hostname): ${ns_name}" >&2
    exit 1
  fi
  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config file on $(hostname): ${config_path}" >&2
    exit 1
  fi
  if grep -Eq '^[[:space:]]*system\.communication\.bindaddress[[:space:]]*=' "${config_path}"; then
    sed -E -i "s|^[[:space:]]*system\.communication\.bindaddress[[:space:]]*=.*$|system.communication.bindaddress = ${bind_ip}|" "${config_path}"
  else
    echo "system.communication.bindaddress = ${bind_ip}" >> "${config_path}"
  fi
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  : > "${dataexchange_log}"
  tmux new-session -d -s "${session_name}" \
    "sudo -n ip netns exec ${ns_name} bash -lc 'cd \"$HOME/library\" && exec bash smartrun.sh bftsmart.demo.dataexchange.DataExchangeServer ${replica_id} dataexchange${n}' > \"${dataexchange_log}\" 2>&1"
done

sleep 2
for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  session_name="pbft_dataexchange_${replica_id}"
  if ! tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "Data-exchange server session exited early on $(hostname): ${session_name}" >&2
    tail -n 80 "$HOME/library/dataexchange_${replica_id}.log" >&2 || true
    exit 1
  fi
done
REMOTE
    ) &
    dataexchange_start_pids+=("$!")
    dataexchange_start_nodes+=("${node}")
  done

  for i in "${!dataexchange_start_pids[@]}"; do
    pid="${dataexchange_start_pids[$i]}"
    node="${dataexchange_start_nodes[$i]}"
    if wait "${pid}"; then
      echo "  - ${node}: data-exchange servers started"
    else
      echo "  - ${node}: failed to start data-exchange servers" >&2
      dataexchange_start_failed=1
    fi
  done
  if (( dataexchange_start_failed != 0 )); then
    echo "Data-exchange server start failed on one or more nodes." >&2
    exit 1
  fi
fi

if (( LEARNING_AGENT_ENABLED == 1 )); then
  echo "Starting learning-agent replicas in tmux (parallel)..."
  agent_start_failed=0
  agent_start_pids=()
  agent_start_nodes=()
  for i in "${!server_nodes[@]}"; do
    node="${server_nodes[$i]}"
    host="${server_hosts[$i]}"
    user="${server_users[$i]}"
    machine_idx="${i}"

    (
      ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- "${replicas_per_server_machine}" "${machine_idx}" "${START_UNIX_MS}" "${MODE}" "${DATASET_ENABLED}" "${AGENT_MODEL_TYPE}" <<'REMOTE'
set -euo pipefail

replicas_per_machine="$1"
machine_idx="$2"
start_unix_ms="$3"
mode="$4"
dataset_enabled="$5"
model_type="$6"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found on $(hostname)" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on $(hostname)" >&2
  exit 1
fi
if [[ ! -f "$HOME/learning_agent/main.py" ]]; then
  echo "learning agent missing on $(hostname): $HOME/learning_agent/main.py" >&2
  exit 1
fi
if [[ ! -f "$HOME/library/smallbank/hosts.config" ]]; then
  echo "hosts config missing on $(hostname): $HOME/library/smallbank/hosts.config" >&2
  exit 1
fi
if [[ "${mode}" == "scheduled" ]] && [[ ! -f "$HOME/learning_agent/config/schedule_pbft.csv" ]]; then
  echo "Scheduled mode requires $HOME/learning_agent/config/schedule_pbft.csv on $(hostname)" >&2
  exit 1
fi

agent_mode_args=""
if [[ "${mode}" == "sharing" ]]; then
  agent_mode_args="--sharing true --model-type ${model_type}"
elif [[ "${mode}" == "learning" ]]; then
  agent_mode_args="--model-type ${model_type}"
elif [[ "${mode}" == "scheduled" ]]; then
  agent_mode_args="--model-type scheduled --schedule-file learning_agent/config/schedule_pbft.csv"
fi

remove_path_with_sudo_fallback() {
  local target="$1"
  if rm -rf "${target}" 2>/dev/null; then
    return 0
  fi
  if [[ ! -e "${target}" ]]; then
    return 0
  fi
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo -n rm -rf "${target}"
    return 0
  fi
  echo "Failed to remove ${target} on $(hostname); permission denied and passwordless sudo is unavailable." >&2
  return 1
}

run_as_user="$(id -un)"
run_as_home="${HOME}"

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  ns_name="pbft-r${replica_id}"
  session_name="pbft_agent_${replica_id}"
  agent_log="$HOME/library/agent_${replica_id}.log"
  dataset_args=""
  if [[ "${dataset_enabled}" == "1" ]]; then
    remove_path_with_sudo_fallback "$HOME/agent${replica_id}.csv"
    dataset_args="--dataset-csv agent${replica_id}.csv"
  fi
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  tmux new-session -d -s "${session_name}" \
    "sudo -n ip netns exec ${ns_name} sudo -n -u ${run_as_user} env HOME=${run_as_home} PYTHONDONTWRITEBYTECODE=1 bash -lc 'cd \"${run_as_home}\" && exec python3 learning_agent/main.py --node-id ${replica_id} --hosts-config \"${run_as_home}/library/smallbank/hosts.config\" --failure-spec \"${run_as_home}/library/config/failure_spec.xml\" --injection-start-unix-ms ${start_unix_ms} ${agent_mode_args} ${dataset_args}' > \"${agent_log}\" 2>&1"
done

sleep 2
for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  session_name="pbft_agent_${replica_id}"
  if ! tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "Learning-agent session exited early on $(hostname): ${session_name}" >&2
    tail -n 60 "$HOME/library/agent_${replica_id}.log" >&2 || true
    exit 1
  fi
done
REMOTE
    ) &
    agent_start_pids+=("$!")
    agent_start_nodes+=("${node}")
  done
  for i in "${!agent_start_pids[@]}"; do
    pid="${agent_start_pids[$i]}"
    node="${agent_start_nodes[$i]}"
    if wait "${pid}"; then
      echo "  - ${node}: learning agents started"
    else
      echo "  - ${node}: failed to start learning agents" >&2
      agent_start_failed=1
    fi
  done
  if (( agent_start_failed != 0 )); then
    echo "Learning-agent start failed on one or more nodes." >&2
    exit 1
  fi
fi

echo "Starting PBFT server replicas in tmux (parallel)..."
start_failed=0
server_start_pids=()
server_start_nodes=()

for i in "${!server_nodes[@]}"; do
  node="${server_nodes[$i]}"
  host="${server_hosts[$i]}"
  user="${server_users[$i]}"
  machine_idx="${i}"

  (
    ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- "${replicas_per_server_machine}" "${machine_idx}" "${START_UNIX_MS}" "${LEARNING_AGENT_ENABLED}" "${OVERLAY_OCTET_A}" "${OVERLAY_OCTET_B}" "${OVERLAY_HOST_OFFSET}" <<'REMOTE'
set -euo pipefail

replicas_per_machine="$1"
machine_idx="$2"
start_unix_ms="$3"
learning_enabled="$4"
overlay_a="$5"
overlay_b="$6"
overlay_host_offset="$7"
server_learning_flag=""
if [[ "${learning_enabled}" == "1" ]]; then
  server_learning_flag="--learning"
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found on $(hostname)" >&2
  exit 1
fi

if [[ ! -d "$HOME/library" ]]; then
  echo "library directory missing on $(hostname): $HOME/library" >&2
  exit 1
fi

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  ns_name="pbft-r${replica_id}"
  session_name="pbft_server_${replica_id}"
  server_log="$HOME/library/server_${replica_id}.log"
  bind_ip="${overlay_a}.${overlay_b}.$((machine_idx + 1)).$((overlay_host_offset + n))"
  config_path="$HOME/library/smallbank${n}/system.config"
  if ! ip netns list | awk '{print $1}' | grep -qx "${ns_name}"; then
    echo "Namespace missing on $(hostname): ${ns_name}" >&2
    exit 1
  fi
  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config file on $(hostname): ${config_path}" >&2
    exit 1
  fi
  if grep -Eq '^[[:space:]]*system\.communication\.bindaddress[[:space:]]*=' "${config_path}"; then
    sed -E -i "s|^[[:space:]]*system\.communication\.bindaddress[[:space:]]*=.*$|system.communication.bindaddress = ${bind_ip}|" "${config_path}"
  else
    echo "system.communication.bindaddress = ${bind_ip}" >> "${config_path}"
  fi
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  : > "${server_log}"
  tmux new-session -d -s "${session_name}" \
    "sudo -n ip netns exec ${ns_name} bash -lc 'cd \"$HOME/library\" && exec bash smartrun.sh bftsmart.demo.smallbank.SmallBankServer ${replica_id} --config-dir smallbank${n} --failure-start-unix-ms ${start_unix_ms} --failure-spec config/failure_spec.xml ${server_learning_flag}' > \"${server_log}\" 2>&1"
done

sleep 2
for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  session_name="pbft_server_${replica_id}"
  if ! tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "Server session exited early on $(hostname): ${session_name}" >&2
    tail -n 80 "$HOME/library/server_${replica_id}.log" >&2 || true
    exit 1
  fi
done
REMOTE
  ) &
  server_start_pids+=("$!")
  server_start_nodes+=("${node}")
done

for i in "${!server_start_pids[@]}"; do
  pid="${server_start_pids[$i]}"
  node="${server_start_nodes[$i]}"
  if wait "${pid}"; then
    echo "  - ${node}: servers started"
  else
    echo "  - ${node}: failed to start servers" >&2
    start_failed=1
  fi
done

if (( start_failed != 0 )); then
  echo "Server start failed on one or more nodes." >&2
  dump_remote_server_logs 120
  exit 1
fi

# Start readiness probing window from "now" so it cannot expire before first probe.
client_delay_deadline="$(( $(date +%s) + CLIENT_START_DELAY_S ))"
if ! wait_for_server_endpoints_from_node0_or_fail "${node0_host}" "${node0_user}" "${LIBRARY_DIR}/smallbank/hosts.config" "${client_delay_deadline}"; then
  dump_remote_server_logs 120
  exit 1
fi

echo "Waiting ${CLIENT_START_DELAY_S}s before starting client..."
remaining_client_delay="$((client_delay_deadline - $(date +%s)))"
if (( remaining_client_delay > 0 )); then
  sleep "${remaining_client_delay}"
fi

echo "Starting client on node0 in tmux..."
ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" "bash -s" -- "${START_UNIX_MS}" <<'REMOTE'
set -euo pipefail

start_unix_ms="$1"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found on $(hostname)" >&2
  exit 1
fi

if [[ ! -d "$HOME/library/client" ]]; then
  echo "client directory missing on $(hostname): $HOME/library/client" >&2
  exit 1
fi

tmux kill-session -t pbft_client >/dev/null 2>&1 || true
tmux new-session -d -s pbft_client \
  "cd \"$HOME/library\" && bash smartrun.sh bftsmart.demo.smallbank.SmallBankClient --config-dir client --create --execute --start-unix-ms ${start_unix_ms} > pbft_output.txt 2>&1"
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
  if (( remaining_experiment_s < sleep_s )); then
    sleep_s="${remaining_experiment_s}"
  fi
  sleep "${sleep_s}"
done

echo "Stopping PBFT tmux sessions on all nodes..."
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
if [[ -z "${sessions}" ]]; then
  exit 0
fi

printf '%s\n' "${sessions}" | while read -r session_name; do
  [[ -z "${session_name}" ]] && continue
  if [[ "${session_name}" == pbft_server_* || "${session_name}" == pbft_dataexchange_* || "${session_name}" == pbft_applier_* || "${session_name}" == pbft_agent_* || "${session_name}" == pbft_client ]]; then
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

echo "Tearing down namespace network setup..."
teardown_failed=0
server_machine_count="${#server_nodes[@]}"
for i in "${!server_nodes[@]}"; do
  node="${server_nodes[$i]}"
  host="${server_hosts[$i]}"
  user="${server_users[$i]}"
  machine_idx="${i}"

  if ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- \
    "${machine_idx}" \
    "${server_machine_count}" \
    "${OVERLAY_OCTET_A}" \
    "${OVERLAY_OCTET_B}" \
    "${OVERLAY_MASK_BITS}" \
    "${OVERLAY_BRIDGE_NAME}" <<'REMOTE'
set -euo pipefail

machine_idx="$1"
server_machine_count="$2"
overlay_a="$3"
overlay_b="$4"
overlay_mask="$5"
bridge_name="$6"

# Drop routes to peer namespace subnets.
for ((peer_idx = 0; peer_idx < server_machine_count; peer_idx++)); do
  if (( peer_idx == machine_idx )); then
    continue
  fi
  subnet="${overlay_a}.${overlay_b}.$((peer_idx + 1)).0/${overlay_mask}"
  sudo -n ip route del "${subnet}" >/dev/null 2>&1 || true
done

# Delete replica namespaces.
existing_ns="$(ip netns list | awk '/^pbft-r[0-9]+/ {print $1}')"
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
    "${server_machine_count}" \
    "${OVERLAY_OCTET_A}" \
    "${OVERLAY_OCTET_B}" \
    "${OVERLAY_MASK_BITS}" <<'REMOTE'
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

OUTPUT_LOCAL_DIR="$(dirname "${OUTPUT_LOCAL}")"
mkdir -p "${OUTPUT_LOCAL_DIR}"

artifact_fetch_pids=()
artifact_fetch_kinds=()
output_fetch_failed=0
server_log_fetch_failed=0
dataexchange_log_fetch_failed=0
agent_log_fetch_failed=0
dataset_fetch_failed=0
applier_log_fetch_failed=0

echo "Fetching artifacts in parallel..."
echo "  - client output: ${OUTPUT_LOCAL}"
queue_artifact_download \
  "node0" "${node0_host}" "${node0_user}" \
  "\$HOME/library/pbft_output.txt" \
  "library/pbft_output.txt" \
  "${OUTPUT_LOCAL}" \
  "pbft_output.txt" \
  "pbft_output"

if (( LOG_ENABLED == 1 )); then
  SERVER_LOGS_LOCAL_DIR="${REPO_ROOT}/scripts/cloudlab/tmp/server_logs"
  APPLIER_LOGS_LOCAL_DIR="${REPO_ROOT}/scripts/cloudlab/tmp/applier_logs"
  echo "  - server logs: ${SERVER_LOGS_LOCAL_DIR}"
  echo "  - applier logs: ${APPLIER_LOGS_LOCAL_DIR}"
  rm -rf "${SERVER_LOGS_LOCAL_DIR}"
  rm -rf "${APPLIER_LOGS_LOCAL_DIR}"
  mkdir -p "${SERVER_LOGS_LOCAL_DIR}"
  mkdir -p "${APPLIER_LOGS_LOCAL_DIR}"
  for i in "${!server_nodes[@]}"; do
    node="${server_nodes[$i]}"
    host="${server_hosts[$i]}"
    user="${server_users[$i]}"
    machine_idx="${i}"
    for ((n = 0; n < replicas_per_server_machine; n++)); do
      replica_id=$((machine_idx * replicas_per_server_machine + n))
      remote_log_path="\$HOME/library/server_${replica_id}.log"
      remote_log_rel="library/server_${replica_id}.log"
      local_log="${SERVER_LOGS_LOCAL_DIR}/server_${replica_id}.log"
      queue_artifact_download \
        "${node}" "${host}" "${user}" \
        "${remote_log_path}" \
        "${remote_log_rel}" \
        "${local_log}" \
        "server_${replica_id}.log" \
        "server_log"

      remote_applier_log_path="\$HOME/library/applier_${replica_id}.log"
      remote_applier_log_rel="library/applier_${replica_id}.log"
      local_applier_log="${APPLIER_LOGS_LOCAL_DIR}/applier_${replica_id}.log"
      queue_artifact_download \
        "${node}" "${host}" "${user}" \
        "${remote_applier_log_path}" \
        "${remote_applier_log_rel}" \
        "${local_applier_log}" \
        "applier_${replica_id}.log" \
        "applier_log"
    done
  done

  if [[ "${MODE}" == "sharing" ]]; then
    DATAEXCHANGE_LOGS_LOCAL_DIR="${REPO_ROOT}/scripts/cloudlab/tmp/dataexchange_logs"
    echo "  - data-exchange logs: ${DATAEXCHANGE_LOGS_LOCAL_DIR}"
    rm -rf "${DATAEXCHANGE_LOGS_LOCAL_DIR}"
    mkdir -p "${DATAEXCHANGE_LOGS_LOCAL_DIR}"
    for i in "${!server_nodes[@]}"; do
      node="${server_nodes[$i]}"
      host="${server_hosts[$i]}"
      user="${server_users[$i]}"
      machine_idx="${i}"
      for ((n = 0; n < replicas_per_server_machine; n++)); do
        replica_id=$((machine_idx * replicas_per_server_machine + n))
        remote_log_path="\$HOME/library/dataexchange_${replica_id}.log"
        remote_log_rel="library/dataexchange_${replica_id}.log"
        local_log="${DATAEXCHANGE_LOGS_LOCAL_DIR}/dataexchange_${replica_id}.log"
        queue_artifact_download \
          "${node}" "${host}" "${user}" \
          "${remote_log_path}" \
          "${remote_log_rel}" \
          "${local_log}" \
          "dataexchange_${replica_id}.log" \
          "dataexchange_log"
      done
    done
  fi

  if [[ "${MODE}" == "learning" || "${MODE}" == "scheduled" || "${MODE}" == "sharing" ]]; then
    AGENT_LOGS_LOCAL_DIR="${REPO_ROOT}/scripts/cloudlab/tmp/agent_logs"
    echo "  - learning-agent logs: ${AGENT_LOGS_LOCAL_DIR}"
    rm -rf "${AGENT_LOGS_LOCAL_DIR}"
    mkdir -p "${AGENT_LOGS_LOCAL_DIR}"
    for i in "${!server_nodes[@]}"; do
      node="${server_nodes[$i]}"
      host="${server_hosts[$i]}"
      user="${server_users[$i]}"
      machine_idx="${i}"
      for ((n = 0; n < replicas_per_server_machine; n++)); do
        replica_id=$((machine_idx * replicas_per_server_machine + n))
        remote_log_path="\$HOME/library/agent_${replica_id}.log"
        remote_log_rel="library/agent_${replica_id}.log"
        local_log="${AGENT_LOGS_LOCAL_DIR}/agent_${replica_id}.log"
        queue_artifact_download \
          "${node}" "${host}" "${user}" \
          "${remote_log_path}" \
          "${remote_log_rel}" \
          "${local_log}" \
          "agent_${replica_id}.log" \
          "agent_log"
      done
    done
  fi
fi

if (( LEARNING_AGENT_ENABLED == 1 )) && (( DATASET_ENABLED == 1 )); then
  echo "  - learning-agent datasets: ${OUTPUT_LOCAL_DIR}"
  for i in "${!server_nodes[@]}"; do
    node="${server_nodes[$i]}"
    host="${server_hosts[$i]}"
    user="${server_users[$i]}"
    machine_idx="${i}"
    for ((n = 0; n < replicas_per_server_machine; n++)); do
      replica_id=$((machine_idx * replicas_per_server_machine + n))
      remote_csv_path="\$HOME/agent${replica_id}.csv"
      remote_csv_rel="agent${replica_id}.csv"
      local_csv="${OUTPUT_LOCAL_DIR}/agent${replica_id}.csv"
      queue_artifact_download \
        "${node}" "${host}" "${user}" \
        "${remote_csv_path}" \
        "${remote_csv_rel}" \
        "${local_csv}" \
        "agent${replica_id}.csv" \
        "dataset" \
        1
    done
  done
fi

for i in "${!artifact_fetch_pids[@]}"; do
  if ! wait "${artifact_fetch_pids[$i]}"; then
    case "${artifact_fetch_kinds[$i]}" in
      output)
        output_fetch_failed=1
        ;;
      server_log)
        server_log_fetch_failed=1
        ;;
      applier_log)
        applier_log_fetch_failed=1
        ;;
      dataexchange_log)
        dataexchange_log_fetch_failed=1
        ;;
      agent_log)
        agent_log_fetch_failed=1
        ;;
      dataset)
        dataset_fetch_failed=1
        ;;
    esac
  fi
done

if (( output_fetch_failed != 0 )); then
  echo "Failed to fetch client output to ${OUTPUT_LOCAL}." >&2
  exit 1
fi

if (( TRIM_OUTPUT == 1 )); then
  echo "Trimming ${OUTPUT_LOCAL}: preserving first 19 lines, then only '-- Monitor' lines..."
  trim_client_output_file "${OUTPUT_LOCAL}" 19
fi

if (( server_log_fetch_failed != 0 )); then
  echo "Some server logs could not be fetched." >&2
fi
if (( applier_log_fetch_failed != 0 )); then
  echo "Some applier logs could not be fetched." >&2
fi
if (( dataexchange_log_fetch_failed != 0 )); then
  echo "Some data-exchange logs could not be fetched." >&2
fi
if (( agent_log_fetch_failed != 0 )); then
  echo "Some learning-agent logs could not be fetched." >&2
fi
if (( dataset_fetch_failed != 0 )); then
  echo "Some dataset files could not be fetched." >&2
fi

if (( stop_failed != 0 )); then
  echo "Some nodes reported tmux stop issues; output was still collected." >&2
fi
if (( teardown_failed != 0 )); then
  echo "Some nodes reported namespace teardown issues." >&2
fi

echo "Done. Client output saved to ${OUTPUT_LOCAL}."
