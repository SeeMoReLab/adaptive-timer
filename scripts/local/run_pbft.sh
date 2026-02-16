#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PREPARE_SCRIPT="${REPO_ROOT}/scripts/prepare_pbft.sh"
PREPARED_LIBRARY_DIR="${REPO_ROOT}/scripts/tmp/library"
LOCAL_TMP_DIR="${REPO_ROOT}/scripts/local/tmp"
LOCAL_LIBRARY_DIR="${LOCAL_TMP_DIR}/library"
OUTPUT_LOCAL="${LOCAL_TMP_DIR}/output.txt"
APPLIER_PATH="${REPO_ROOT}/scripts/apply_network_failures.py"

MODE="normal"
F=1
EXPERIMENT_TIME_S=60
CLIENT_START_DELAY_S=10
START_OFFSET_AFTER_CLIENT_LAUNCH_S=10

OVERLAY_OCTET_A=10
OVERLAY_OCTET_B=251
OVERLAY_MASK_BITS=24
OVERLAY_BRIDGE_NAME="br-pbft-local"
OVERLAY_HOST_OFFSET=10

cleanup_done=0
cleanup() {
  if (( cleanup_done == 1 )); then
    return
  fi
  cleanup_done=1

  set +e
  if command -v tmux >/dev/null 2>&1; then
    sessions="$(tmux ls -F '#S' 2>/dev/null || true)"
    if [[ -n "${sessions}" ]]; then
      while read -r session_name; do
        [[ -z "${session_name}" ]] && continue
        if [[ "${session_name}" == pbft_server_* || "${session_name}" == pbft_applier_* || "${session_name}" == pbft_client ]]; then
          tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
        fi
      done <<< "${sessions}"
    fi
  fi

  if sudo -n true >/dev/null 2>&1; then
    existing_ns="$(ip netns list | awk '/^pbft-r[0-9]+/ {print $1}')"
    if [[ -n "${existing_ns}" ]]; then
      while read -r ns_name; do
        [[ -z "${ns_name}" ]] && continue
        sudo -n ip netns del "${ns_name}" >/dev/null 2>&1 || true
      done <<< "${existing_ns}"
    fi
    sudo -n ip link set "${OVERLAY_BRIDGE_NAME}" down >/dev/null 2>&1 || true
    sudo -n ip link del "${OVERLAY_BRIDGE_NAME}" type bridge >/dev/null 2>&1 || true
  fi

  set -e
}
trap cleanup EXIT

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
      slot = replica_id % rpm
      ip = sprintf("%d.%d.1.%d", oa, ob, host_offset + slot)
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

wait_for_server_endpoints_or_fail() {
  local hosts_file="$1"
  local deadline_epoch_s="$2"
  local endpoints
  local last_unready=""

  endpoints="$(collect_server_endpoints "${hosts_file}")"
  if [[ -z "${endpoints}" ]]; then
    echo "No server endpoints found in ${hosts_file}" >&2
    return 1
  fi

  while (( $(date +%s) < deadline_epoch_s )); do
    local all_ready=1
    local unready=""
    while read -r replica_id replica_ip replica_port; do
      [[ -z "${replica_id}" ]] && continue
      if ! timeout 1 bash -c "exec 3<>/dev/tcp/${replica_ip}/${replica_port}" >/dev/null 2>&1; then
        all_ready=0
        unready+="${replica_id}@${replica_ip}:${replica_port} "
      fi
    done <<< "${endpoints}"

    if (( all_ready == 1 )); then
      return 0
    fi

    last_unready="${unready}"
    sleep 1
  done

  echo "Timed out waiting for server ports to become reachable." >&2
  if [[ -n "${last_unready}" ]]; then
    echo "Still unreachable: ${last_unready}" >&2
  fi
  return 1
}

usage() {
  cat <<EOF
Usage: $0 [normal|heuristic|learning|sharing] [-f <fault_tolerance>] [-t <experiment_seconds>]

Arguments:
  mode              Optional; one of normal|heuristic|learning|sharing (default: normal).

Flags:
  -f <int>          Fault tolerance f passed to prepare_pbft.sh (default: 1).
  -t <int>          Seconds to run experiment before stopping tmux sessions (default: 60).
EOF
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
    normal|heuristic|learning|sharing)
      MODE="$1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
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

if [[ ! -x "${PREPARE_SCRIPT}" ]]; then
  echo "prepare_pbft.sh not found or not executable: ${PREPARE_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${APPLIER_PATH}" ]]; then
  echo "Network applier script not found: ${APPLIER_PATH}" >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux command not found." >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 command not found." >&2
  exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "timeout command not found (coreutils)." >&2
  exit 1
fi
if ! sudo -n true >/dev/null 2>&1; then
  echo "This script requires passwordless sudo (sudo -n)." >&2
  exit 1
fi

echo "Selected mode: ${MODE} (no mode-specific behavior yet)"
echo "Preparing PBFT configs in local mode with f=${F}..."
"${PREPARE_SCRIPT}" local -f "${F}"

if [[ ! -d "${PREPARED_LIBRARY_DIR}" ]]; then
  echo "Prepared library directory not found: ${PREPARED_LIBRARY_DIR}" >&2
  exit 1
fi

echo "Staging local runtime library under ${LOCAL_LIBRARY_DIR}..."
mkdir -p "${LOCAL_TMP_DIR}"
rm -rf "${LOCAL_LIBRARY_DIR}"
cp -a "${PREPARED_LIBRARY_DIR}" "${LOCAL_LIBRARY_DIR}"

NUM_SERVERS="$(
  awk -F= '
    /^[[:space:]]*system\.servers\.num[[:space:]]*=/ {
      gsub(/[[:space:]]/, "", $2)
      print $2
      exit
    }
  ' "${LOCAL_LIBRARY_DIR}/smallbank/system.config"
)"
if ! [[ "${NUM_SERVERS}" =~ ^[0-9]+$ ]] || (( NUM_SERVERS < 1 )); then
  echo "Unable to determine system.servers.num from ${LOCAL_LIBRARY_DIR}/smallbank/system.config" >&2
  exit 1
fi

replicas_per_machine="${NUM_SERVERS}"

echo "Rewriting hosts configs for local namespace IPs..."
rewrite_hosts_for_namespaces "${LOCAL_LIBRARY_DIR}/smallbank/hosts.config" "${replicas_per_machine}"
rewrite_hosts_for_namespaces "${LOCAL_LIBRARY_DIR}/dataexchange/hosts.config" "${replicas_per_machine}"

echo "Materializing per-replica config directories..."
rm -rf "${LOCAL_LIBRARY_DIR}/client"
shopt -s nullglob
rm -rf "${LOCAL_LIBRARY_DIR}"/smallbank[0-9]* "${LOCAL_LIBRARY_DIR}"/dataexchange[0-9]*
shopt -u nullglob
cp -a "${LOCAL_LIBRARY_DIR}/smallbank" "${LOCAL_LIBRARY_DIR}/client"
for ((idx = 0; idx < replicas_per_machine; idx++)); do
  cp -a "${LOCAL_LIBRARY_DIR}/smallbank" "${LOCAL_LIBRARY_DIR}/smallbank${idx}"
  cp -a "${LOCAL_LIBRARY_DIR}/dataexchange" "${LOCAL_LIBRARY_DIR}/dataexchange${idx}"
done

echo "Configuring local namespaces..."
cleanup
cleanup_done=0

bridge_ip="${OVERLAY_OCTET_A}.${OVERLAY_OCTET_B}.1.1"
sudo -n ip link add name "${OVERLAY_BRIDGE_NAME}" type bridge
sudo -n ip addr add "${bridge_ip}/${OVERLAY_MASK_BITS}" dev "${OVERLAY_BRIDGE_NAME}"
sudo -n ip link set "${OVERLAY_BRIDGE_NAME}" up
sudo -n sysctl -w net.ipv4.ip_forward=1 >/dev/null

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id="${n}"
  ns_name="pbft-r${replica_id}"
  host_if="vh${replica_id}"
  ns_if="vn${replica_id}"
  ns_ip="${OVERLAY_OCTET_A}.${OVERLAY_OCTET_B}.1.$((OVERLAY_HOST_OFFSET + n))"

  sudo -n ip netns add "${ns_name}"
  sudo -n ip link add "${host_if}" type veth peer name "${ns_if}"
  sudo -n ip link set "${ns_if}" netns "${ns_name}"
  sudo -n ip link set "${host_if}" master "${OVERLAY_BRIDGE_NAME}"
  sudo -n ip link set "${host_if}" up

  sudo -n ip netns exec "${ns_name}" ip link set lo up
  sudo -n ip netns exec "${ns_name}" ip addr add "${ns_ip}/${OVERLAY_MASK_BITS}" dev "${ns_if}"
  sudo -n ip netns exec "${ns_name}" ip link set "${ns_if}" up
  sudo -n ip netns exec "${ns_name}" ip route replace default via "${bridge_ip}"
done

# Shared experiment/failure start time:
# - wait 10s after server launch before starting client
# - then start benchmark/failures at +10s after client launch
START_UNIX_MS="$(( ($(date +%s) + CLIENT_START_DELAY_S + START_OFFSET_AFTER_CLIENT_LAUNCH_S) * 1000 ))"
echo "Using shared start_unix_ms=${START_UNIX_MS}"

echo "Starting network failure appliers..."
applier_start_failed=0
for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id="${n}"
  ns_name="pbft-r${replica_id}"
  interface_name="vn${replica_id}"
  session_name="pbft_applier_${replica_id}"
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  tmux new-session -d -s "${session_name}" \
    "cd \"${LOCAL_LIBRARY_DIR}\" && python3 \"${APPLIER_PATH}\" --failure-spec \"${LOCAL_LIBRARY_DIR}/config/failure_spec.xml\" --hosts-config \"${LOCAL_LIBRARY_DIR}/smallbank/hosts.config\" --replica-id ${replica_id} --netns ${ns_name} --interface ${interface_name} --start-unix-ms ${START_UNIX_MS} > \"${LOCAL_LIBRARY_DIR}/applier_${replica_id}.log\" 2>&1"
done
sleep 1
for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id="${n}"
  session_name="pbft_applier_${replica_id}"
  if ! tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "Applier session exited early: ${session_name}" >&2
    echo "Last applier log lines (replica ${replica_id}):" >&2
    tail -n 40 "${LOCAL_LIBRARY_DIR}/applier_${replica_id}.log" >&2 || true
    applier_start_failed=1
  fi
done
if (( applier_start_failed != 0 )); then
  echo "One or more appliers failed to stay up." >&2
  exit 1
fi

echo "Starting PBFT servers..."
server_start_failed=0
server_launch_epoch_s="$(date +%s)"
for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id="${n}"
  ns_name="pbft-r${replica_id}"
  session_name="pbft_server_${replica_id}"
  server_log="${LOCAL_LIBRARY_DIR}/server_${replica_id}.log"
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  : > "${server_log}"
  tmux new-session -d -s "${session_name}" \
    "sudo -n ip netns exec ${ns_name} bash -lc 'cd \"${LOCAL_LIBRARY_DIR}\" && exec bash smartrun.sh bftsmart.demo.smallbank.SmallBankServer ${replica_id} --config-dir smallbank${n} --failure-start-unix-ms ${START_UNIX_MS} --failure-spec config/failure_spec.xml' > \"${server_log}\" 2>&1"
done
sleep 2
for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id="${n}"
  session_name="pbft_server_${replica_id}"
  if ! tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "Server session exited early: ${session_name}" >&2
    echo "Last server log lines (replica ${replica_id}):" >&2
    tail -n 60 "${LOCAL_LIBRARY_DIR}/server_${replica_id}.log" >&2 || true
    server_start_failed=1
  fi
done
if (( server_start_failed != 0 )); then
  echo "One or more PBFT servers failed to stay up." >&2
  exit 1
fi

client_delay_deadline="$((server_launch_epoch_s + CLIENT_START_DELAY_S))"
if ! wait_for_server_endpoints_or_fail "${LOCAL_LIBRARY_DIR}/smallbank/hosts.config" "${client_delay_deadline}"; then
  for ((n = 0; n < replicas_per_machine; n++)); do
    replica_id="${n}"
    echo "Last server log lines (replica ${replica_id}):" >&2
    tail -n 80 "${LOCAL_LIBRARY_DIR}/server_${replica_id}.log" >&2 || true
  done
  exit 1
fi

echo "Waiting ${CLIENT_START_DELAY_S}s before starting client..."
remaining_client_delay="$((client_delay_deadline - $(date +%s)))"
if (( remaining_client_delay > 0 )); then
  sleep "${remaining_client_delay}"
fi

echo "Starting client..."
tmux kill-session -t pbft_client >/dev/null 2>&1 || true
tmux new-session -d -s pbft_client \
  "cd \"${LOCAL_LIBRARY_DIR}\" && bash smartrun.sh bftsmart.demo.smallbank.SmallBankClient --config-dir client --create --execute --start-unix-ms ${START_UNIX_MS} > output.txt 2>&1"

echo "Experiment running for ${EXPERIMENT_TIME_S}s..."
sleep "${EXPERIMENT_TIME_S}"

echo "Stopping tmux sessions..."
sessions="$(tmux ls -F '#S' 2>/dev/null || true)"
if [[ -n "${sessions}" ]]; then
  while read -r session_name; do
    [[ -z "${session_name}" ]] && continue
    if [[ "${session_name}" == pbft_server_* || "${session_name}" == pbft_applier_* || "${session_name}" == pbft_client ]]; then
      tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
    fi
  done <<< "${sessions}"
fi

mkdir -p "${LOCAL_TMP_DIR}"
rm -f "${OUTPUT_LOCAL}"
if [[ -f "${LOCAL_LIBRARY_DIR}/output.txt" ]]; then
  cp -a "${LOCAL_LIBRARY_DIR}/output.txt" "${OUTPUT_LOCAL}"
  echo "Done. Client output saved to ${OUTPUT_LOCAL}."
else
  echo "Client output not found: ${LOCAL_LIBRARY_DIR}/output.txt" >&2
fi
