#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PREPARE_SCRIPT="${REPO_ROOT}/scripts/prepare_pbft.sh"
TARGETS_FILE_DEFAULT="${REPO_ROOT}/scripts/cloudlab/tmp/cloudlab_ssh_targets.txt"
LIBRARY_DIR="${REPO_ROOT}/scripts/tmp/library"
LOCAL_ZIP="${REPO_ROOT}/scripts/tmp/library.zip"
APPLIER_LOCAL_PATH="${REPO_ROOT}/scripts/apply_network_failures.py"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
MODE="normal"
F=1
TARGETS_FILE="${TARGETS_FILE_DEFAULT}"
EXPERIMENT_TIME_S=60
CLIENT_START_DELAY_S=10
START_OFFSET_AFTER_CLIENT_LAUNCH_S=10
OUTPUT_LOCAL="${REPO_ROOT}/scripts/cloudlab/tmp/output.txt"
OVERLAY_OCTET_A=10
OVERLAY_OCTET_B=250
OVERLAY_MASK_BITS=24
OVERLAY_BRIDGE_NAME="br-pbft"
OVERLAY_HOST_OFFSET=10

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

usage() {
  cat <<EOF
Usage: $0 <private_key_path> [normal|heuristic|learning|sharing] [-f <fault_tolerance>] [-t <experiment_seconds>] [targets_file]

Arguments:
  private_key_path  Local SSH private key for CloudLab nodes.
  mode              Optional; one of normal|heuristic|learning|sharing (default: normal).
  targets_file      Optional path to cloudlab SSH targets file.

Flags:
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
    normal|heuristic|learning|sharing)
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

echo "Selected mode: ${MODE} (no mode-specific behavior yet)"
echo "Preparing PBFT configs in cloudlab mode with f=${F}..."
"${PREPARE_SCRIPT}" cloudlab -f "${F}"

if [[ ! -d "${LIBRARY_DIR}" ]]; then
  echo "Prepared library directory not found: ${LIBRARY_DIR}" >&2
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

echo "Rewriting hosts configs for per-replica namespace IPs..."
rewrite_hosts_for_namespaces "${LIBRARY_DIR}/smallbank/hosts.config" "${replicas_per_server_machine}"
rewrite_hosts_for_namespaces "${LIBRARY_DIR}/dataexchange/hosts.config" "${replicas_per_server_machine}"

mkdir -p "$(dirname "${LOCAL_ZIP}")"
rm -f "${LOCAL_ZIP}"
trap 'rm -f "${LOCAL_ZIP}"' EXIT

echo "Creating temporary library zip..."
(
  cd "$(dirname "${LIBRARY_DIR}")"
  zip -qr "${LOCAL_ZIP}" "$(basename "${LIBRARY_DIR}")"
)

echo "Copying library zip to all CloudLab nodes and materializing per-role directories (parallel)..."
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

shopt -s nullglob
rm -rf "$HOME/library"
rm -rf "$HOME"/smallbank[0-9]*
rm -rf "$HOME"/dataexchange[0-9]*
shopt -u nullglob

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

template_dir="$tmp_extract_dir/library"
if [[ ! -d "$template_dir" ]]; then
  echo "library directory not found after extraction on $(hostname): $template_dir" >&2
  exit 1
fi

cp -a "$template_dir" "$HOME/library"
cp -a "$applier_path" "$HOME/library/apply_network_failures.py"
chmod +x "$HOME/library/apply_network_failures.py"

if [[ ! -d "$HOME/library/smallbank" || ! -d "$HOME/library/dataexchange" ]]; then
  echo "smallbank/dataexchange directories missing under $HOME/library on $(hostname)" >&2
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
    "${server_nodes[@]}" <<'REMOTE'
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
server_node_names=("$@")

bridge_ip="${overlay_a}.${overlay_b}.$((machine_idx + 1)).1"
local_subnet="${overlay_a}.${overlay_b}.$((machine_idx + 1)).0/${overlay_mask}"

# Clean stale namespaces from previous runs.
existing_ns="$(ip netns list | awk '/^pbft-r[0-9]+/ {print $1}')"
if [[ -n "${existing_ns}" ]]; then
  printf '%s\n' "${existing_ns}" | while read -r ns_name; do
    [[ -z "${ns_name}" ]] && continue
    sudo -n ip netns del "${ns_name}" >/dev/null 2>&1 || true
  done
fi

# Recreate bridge for deterministic state.
sudo -n ip link set "${bridge_name}" down >/dev/null 2>&1 || true
sudo -n ip link del "${bridge_name}" type bridge >/dev/null 2>&1 || true
sudo -n ip link add name "${bridge_name}" type bridge
sudo -n ip addr add "${bridge_ip}/${overlay_mask}" dev "${bridge_name}"
sudo -n ip link set "${bridge_name}" up
sudo -n sysctl -w net.ipv4.ip_forward=1 >/dev/null

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  ns_name="pbft-r${replica_id}"
  host_if="vh${replica_id}"
  ns_if="vn${replica_id}"
  ns_ip="${overlay_a}.${overlay_b}.$((machine_idx + 1)).$((host_offset + n))"

  sudo -n ip netns add "${ns_name}"
  sudo -n ip link add "${host_if}" type veth peer name "${ns_if}"
  sudo -n ip link set "${ns_if}" netns "${ns_name}"
  sudo -n ip link set "${host_if}" master "${bridge_name}"
  sudo -n ip link set "${host_if}" up

  sudo -n ip netns exec "${ns_name}" ip link set lo up
  sudo -n ip netns exec "${ns_name}" ip addr add "${ns_ip}/${overlay_mask}" dev "${ns_if}"
  sudo -n ip netns exec "${ns_name}" ip link set "${ns_if}" up
  sudo -n ip netns exec "${ns_name}" ip route replace default via "${bridge_ip}"
done

# Route to other namespace subnets through corresponding server hosts.
for ((peer_idx = 0; peer_idx < server_machine_count; peer_idx++)); do
  if (( peer_idx == machine_idx )); then
    continue
  fi
  subnet="${overlay_a}.${overlay_b}.$((peer_idx + 1)).0/${overlay_mask}"
  gw_name="${server_node_names[$peer_idx]}"
  gw_ip="$(getent ahostsv4 "${gw_name}" | awk 'NR==1 {print $1}')"
  if [[ -z "${gw_ip}" ]]; then
    gw_ip="${gw_name}"
  fi
  sudo -n ip route replace "${subnet}" via "${gw_ip}"
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
  if ! ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" "bash -s" -- \
    "${server_machine_count}" \
    "${OVERLAY_OCTET_A}" \
    "${OVERLAY_OCTET_B}" \
    "${OVERLAY_MASK_BITS}" \
    "${server_nodes[@]}" <<'REMOTE'
set -euo pipefail

server_machine_count="$1"
overlay_a="$2"
overlay_b="$3"
overlay_mask="$4"
shift 4
server_node_names=("$@")

for ((peer_idx = 0; peer_idx < server_machine_count; peer_idx++)); do
  subnet="${overlay_a}.${overlay_b}.$((peer_idx + 1)).0/${overlay_mask}"
  gw_name="${server_node_names[$peer_idx]}"
  gw_ip="$(getent ahostsv4 "${gw_name}" | awk 'NR==1 {print $1}')"
  if [[ -z "${gw_ip}" ]]; then
    gw_ip="${gw_name}"
  fi
  sudo -n ip route replace "${subnet}" via "${gw_ip}"
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

echo "Starting PBFT server replicas in tmux..."
start_failed=0
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

if [[ ! -d "$HOME/library" ]]; then
  echo "library directory missing on $(hostname): $HOME/library" >&2
  exit 1
fi

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  ns_name="pbft-r${replica_id}"
  session_name="pbft_server_${replica_id}"
  if ! ip netns list | awk '{print $1}' | grep -qx "${ns_name}"; then
    echo "Namespace missing on $(hostname): ${ns_name}" >&2
    exit 1
  fi
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  tmux new-session -d -s "${session_name}" \
    "sudo -n ip netns exec ${ns_name} bash -lc 'cd \"$HOME/library\" && bash smartrun.sh bftsmart.demo.smallbank.SmallBankServer ${replica_id} --config-dir smallbank${n} --failure-start-unix-ms ${start_unix_ms} --failure-spec config/failure_spec.xml'"
done
REMOTE
  then
    echo "  - ${node}: servers started"
  else
    echo "  - ${node}: failed to start servers" >&2
    start_failed=1
  fi
done

if (( start_failed != 0 )); then
  echo "Server start failed on one or more nodes." >&2
  exit 1
fi

echo "Waiting ${CLIENT_START_DELAY_S}s before starting client..."
sleep "${CLIENT_START_DELAY_S}"

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
  "cd \"$HOME/library\" && bash smartrun.sh bftsmart.demo.smallbank.SmallBankClient --config-dir client --create --execute --start-unix-ms ${start_unix_ms} > output.txt 2>&1"
REMOTE

echo "Experiment running for ${EXPERIMENT_TIME_S}s..."
sleep "${EXPERIMENT_TIME_S}"

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
  if [[ "${session_name}" == pbft_server_* || "${session_name}" == pbft_applier_* || "${session_name}" == pbft_client ]]; then
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

mkdir -p "$(dirname "${OUTPUT_LOCAL}")"
rm -f "${OUTPUT_LOCAL}"
echo "Fetching client output to ${OUTPUT_LOCAL}..."
scp "${SSH_OPTS[@]}" "${node0_user}@${node0_host}:~/library/output.txt" "${OUTPUT_LOCAL}"

if (( stop_failed != 0 )); then
  echo "Some nodes reported tmux stop issues; output was still collected." >&2
fi
if (( teardown_failed != 0 )); then
  echo "Some nodes reported namespace teardown issues." >&2
fi

echo "Done. Client output saved to ${OUTPUT_LOCAL}."
