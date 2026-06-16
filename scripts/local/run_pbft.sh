#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PREPARE_SCRIPT="${REPO_ROOT}/scripts/prepare_pbft.sh"
PREPARED_LIBRARY_DIR="${REPO_ROOT}/scripts/tmp/library"
LOCAL_TMP_DIR="${REPO_ROOT}/scripts/local/tmp"
LOCAL_LIBRARY_DIR="${LOCAL_TMP_DIR}/library"
LEARNING_AGENT_SRC_DIR="${REPO_ROOT}/learning_agent"
OUTPUT_LOCAL="${LOCAL_TMP_DIR}/output.txt"
APPLIER_PATH="${REPO_ROOT}/scripts/apply_network_failures.py"
RUN_AS_USER="${SUDO_USER:-${USER:-}}"
RUN_AS_HOME="${HOME}"
if [[ -n "${RUN_AS_USER}" ]]; then
  resolved_home="$(getent passwd "${RUN_AS_USER}" 2>/dev/null | cut -d: -f6 || true)"
  if [[ -n "${resolved_home}" ]]; then
    RUN_AS_HOME="${resolved_home}"
  fi
fi
USER_PYENV_ROOT="${PYENV_ROOT:-${RUN_AS_HOME}/.pyenv}"

MODE="normal"
F=1
EXPERIMENT_TIME_S=300
CLIENT_START_DELAY_S=10
START_OFFSET_AFTER_CLIENT_LAUNCH_S=10
DATASET_ENABLED=0
LOG_ENABLED=0
AGENT_MODEL_TYPE="random"
AGENT_MODEL_SET=0

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
        if [[ "${session_name}" == pbft_server_* || "${session_name}" == pbft_applier_* || "${session_name}" == pbft_agent_* || "${session_name}" == pbft_client ]]; then
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
    stale_veths="$(
      ip -o link show \
        | awk -F': ' '{print $2}' \
        | sed 's/@.*//' \
        | awk '/^vh[0-9]+$/ || /^vn[0-9]+$/'
    )"
    if [[ -n "${stale_veths}" ]]; then
      while read -r if_name; do
        [[ -z "${if_name}" ]] && continue
        sudo -n ip link del "${if_name}" >/dev/null 2>&1 || true
      done <<< "${stale_veths}"
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
Usage: $0 [normal|heuristic|learning|sharing|scheduled] [--dataset] [--log] [--model <random|linear|cmab>] [-f <fault_tolerance>] [-t <experiment_seconds>]

Arguments:
  mode              Optional; one of normal|heuristic|learning|sharing|scheduled|cmab (default: normal).

Flags:
  --dataset         Collect per-replica learning-agent dataset CSVs as agentN.csv.
  --log             Collect per-replica server logs under scripts/local/tmp/server_logs/.
  --model           Learning-agent model for learning/sharing modes: random|linear|cmab (cmab uses Thompson sampling; default: random).
  -f <int>          Fault tolerance f passed to prepare_pbft.sh (default: 1).
  -t <int>          Seconds to run experiment before stopping tmux sessions (default: 300).
EOF
}

mode_enables_learning_agent() {
  case "$1" in
    heuristic|learning|sharing|scheduled|cmab)
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
    --dataset)
      DATASET_ENABLED=1
      shift
      ;;
    --log)
      LOG_ENABLED=1
      shift
      ;;
    normal|heuristic|learning|sharing|scheduled|cmab)
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
  echo "--dataset requires a learning-agent mode: heuristic|learning|sharing|scheduled." >&2
  exit 2
fi
SERVER_LEARNING_FLAG=""
AGENT_SHARING_FLAG=""
AGENT_MODEL_ARGS=""
AGENT_PYTHON_BIN="python3"
if (( LEARNING_AGENT_ENABLED == 1 )); then
  SERVER_LEARNING_FLAG="--learning"
  # Resolve a concrete interpreter path before entering sudo/netns.
  if [[ -x "${USER_PYENV_ROOT}/versions/timer/bin/python3" ]]; then
    AGENT_PYTHON_BIN="${USER_PYENV_ROOT}/versions/timer/bin/python3"
  elif [[ -x "${USER_PYENV_ROOT}/versions/timer/bin/python" ]]; then
    AGENT_PYTHON_BIN="${USER_PYENV_ROOT}/versions/timer/bin/python"
  elif command -v pyenv >/dev/null 2>&1; then
    pyenv_timer_python="$(PYENV_VERSION=timer pyenv which python3 2>/dev/null || true)"
    if [[ -n "${pyenv_timer_python}" && -x "${pyenv_timer_python}" ]]; then
      AGENT_PYTHON_BIN="${pyenv_timer_python}"
    fi
  fi
fi
if [[ "${MODE}" == "sharing" ]]; then
  AGENT_SHARING_FLAG="--sharing true"
  AGENT_MODEL_ARGS="--model-type ${AGENT_MODEL_TYPE}"
elif [[ "${MODE}" == "learning" ]]; then
  AGENT_MODEL_ARGS="--model-type ${AGENT_MODEL_TYPE}"
fi
if [[ "${MODE}" == "scheduled" ]]; then
  AGENT_MODEL_ARGS="--model-type scheduled --schedule-file learning_agent/config/schedule_pbft.csv"
fi
if [[ "${MODE}" == "cmab" ]]; then
  AGENT_MODEL_ARGS="--model-type cmab"
fi

if (( LEARNING_AGENT_ENABLED == 1 )); then
  if ! "${AGENT_PYTHON_BIN}" -c "import grpc" >/dev/null 2>&1; then
    echo "Selected learning-agent interpreter is missing grpc: ${AGENT_PYTHON_BIN}" >&2
    echo "Resolver context: RUN_AS_USER=${RUN_AS_USER:-<unset>}, RUN_AS_HOME=${RUN_AS_HOME}, USER_PYENV_ROOT=${USER_PYENV_ROOT}" >&2
    echo "Candidate timer interpreters checked:" >&2
    echo "  - ${USER_PYENV_ROOT}/versions/timer/bin/python3" >&2
    echo "  - ${USER_PYENV_ROOT}/versions/timer/bin/python" >&2
    echo "Install dependencies into that interpreter and retry." >&2
    exit 1
  fi
  echo "Learning-agent interpreter: ${AGENT_PYTHON_BIN}"
fi

echo "Selected mode: ${MODE}"
if [[ "${MODE}" == "learning" || "${MODE}" == "sharing" ]]; then
  echo "Learning-agent model: ${AGENT_MODEL_TYPE}"
fi
echo "Learning-agent orchestration: ${LEARNING_AGENT_ENABLED} (dataset=${DATASET_ENABLED}, log=${LOG_ENABLED})"
echo "Preparing PBFT configs in local mode with f=${F}..."
"${PREPARE_SCRIPT}" local -f "${F}"

if [[ ! -d "${PREPARED_LIBRARY_DIR}" ]]; then
  echo "Prepared library directory not found: ${PREPARED_LIBRARY_DIR}" >&2
  exit 1
fi
if (( LEARNING_AGENT_ENABLED == 1 )) && [[ ! -d "${LEARNING_AGENT_SRC_DIR}" ]]; then
  echo "learning_agent directory not found: ${LEARNING_AGENT_SRC_DIR}" >&2
  exit 1
fi

echo "Staging local runtime library under ${LOCAL_LIBRARY_DIR}..."
mkdir -p "${LOCAL_TMP_DIR}"
rm -rf "${LOCAL_LIBRARY_DIR}"
cp -a "${PREPARED_LIBRARY_DIR}" "${LOCAL_LIBRARY_DIR}"
if (( LEARNING_AGENT_ENABLED == 1 )); then
  rm -rf "${LOCAL_LIBRARY_DIR}/learning_agent"
  cp -a "${LEARNING_AGENT_SRC_DIR}" "${LOCAL_LIBRARY_DIR}/learning_agent"
fi
if [[ "${MODE}" == "scheduled" ]] && [[ ! -f "${LOCAL_LIBRARY_DIR}/learning_agent/config/schedule_pbft.csv" ]]; then
  echo "Scheduled mode requires ${LOCAL_LIBRARY_DIR}/learning_agent/config/schedule_pbft.csv" >&2
  exit 1
fi

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
sudo -n ip link set "${OVERLAY_BRIDGE_NAME}" down >/dev/null 2>&1 || true
sudo -n ip link del "${OVERLAY_BRIDGE_NAME}" type bridge >/dev/null 2>&1 || true
sudo -n ip link add name "${OVERLAY_BRIDGE_NAME}" type bridge
sudo -n ip addr replace "${bridge_ip}/${OVERLAY_MASK_BITS}" dev "${OVERLAY_BRIDGE_NAME}"
sudo -n ip link set "${OVERLAY_BRIDGE_NAME}" up
sudo -n sysctl -w net.ipv4.ip_forward=1 >/dev/null

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id="${n}"
  ns_name="pbft-r${replica_id}"
  host_if="vh${replica_id}"
  ns_if="vn${replica_id}"
  ns_ip="${OVERLAY_OCTET_A}.${OVERLAY_OCTET_B}.1.$((OVERLAY_HOST_OFFSET + n))"

  sudo -n ip netns del "${ns_name}" >/dev/null 2>&1 || true
  sudo -n ip link del "${host_if}" >/dev/null 2>&1 || true
  sudo -n ip link del "${ns_if}" >/dev/null 2>&1 || true
  sudo -n ip netns add "${ns_name}"
  sudo -n ip link add "${host_if}" type veth peer name "${ns_if}"
  sudo -n ip link set "${ns_if}" netns "${ns_name}"
  sudo -n ip link set "${host_if}" master "${OVERLAY_BRIDGE_NAME}"
  sudo -n ip link set "${host_if}" up

  sudo -n ip netns exec "${ns_name}" ip link set lo up
  sudo -n ip netns exec "${ns_name}" ip addr replace "${ns_ip}/${OVERLAY_MASK_BITS}" dev "${ns_if}"
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

if (( LEARNING_AGENT_ENABLED == 1 )); then
  echo "Starting learning-agent replicas (parallel)..."
  agent_start_failed=0
  agent_start_pids=()
  agent_start_replicas=()
  for ((n = 0; n < replicas_per_machine; n++)); do
    (
      replica_id="${n}"
      ns_name="pbft-r${replica_id}"
      session_name="pbft_agent_${replica_id}"
      agent_log="${LOCAL_LIBRARY_DIR}/agent_${replica_id}.log"
      dataset_args=""
      if (( DATASET_ENABLED == 1 )); then
        rm -f "${LOCAL_LIBRARY_DIR}/agent${replica_id}.csv"
        dataset_args="--dataset-csv agent${replica_id}.csv"
      fi
      tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
      : > "${agent_log}"
      tmux new-session -d -s "${session_name}" \
        "sudo -n ip netns exec ${ns_name} bash -lc 'cd \"${LOCAL_LIBRARY_DIR}\" && echo \"[DEBUG] netns launch context\" && echo \"[DEBUG] whoami:\" && id -un && echo \"[DEBUG] pwd:\" && pwd && echo \"[DEBUG] python_bin: ${AGENT_PYTHON_BIN}\" && \"${AGENT_PYTHON_BIN}\" -c \"import sys; print(\\\"[DEBUG] sys.executable=\\\" + sys.executable)\" && \"${AGENT_PYTHON_BIN}\" -c \"import grpc; print(\\\"[DEBUG] grpc import ok\\\")\" && exec \"${AGENT_PYTHON_BIN}\" learning_agent/main.py --node-id ${replica_id} --protocol pbft --hosts-config \"${LOCAL_LIBRARY_DIR}/smallbank/hosts.config\" --failure-spec \"${LOCAL_LIBRARY_DIR}/config/failure_spec.xml\" --injection-start-unix-ms ${START_UNIX_MS} ${AGENT_SHARING_FLAG} ${AGENT_MODEL_ARGS} ${dataset_args}' > \"${agent_log}\" 2>&1"
    ) &
    agent_start_pids+=("$!")
    agent_start_replicas+=("${n}")
  done
  for i in "${!agent_start_pids[@]}"; do
    if ! wait "${agent_start_pids[$i]}"; then
      echo "Failed to launch learning-agent session for replica ${agent_start_replicas[$i]}" >&2
      agent_start_failed=1
    fi
  done
  sleep 2
  for ((n = 0; n < replicas_per_machine; n++)); do
    replica_id="${n}"
    session_name="pbft_agent_${replica_id}"
    if ! tmux has-session -t "${session_name}" 2>/dev/null; then
      echo "Learning-agent session exited early: ${session_name}" >&2
      echo "Last learning-agent log lines (replica ${replica_id}):" >&2
      tail -n 60 "${LOCAL_LIBRARY_DIR}/agent_${replica_id}.log" >&2 || true
      agent_start_failed=1
    fi
  done
  if (( agent_start_failed != 0 )); then
    echo "One or more learning agents failed to stay up." >&2
    exit 1
  fi
fi

echo "Starting PBFT servers (parallel)..."
server_start_failed=0
server_launch_epoch_s="$(date +%s)"
server_start_pids=()
server_start_replicas=()
for ((n = 0; n < replicas_per_machine; n++)); do
  (
    replica_id="${n}"
    ns_name="pbft-r${replica_id}"
    session_name="pbft_server_${replica_id}"
    server_log="${LOCAL_LIBRARY_DIR}/server_${replica_id}.log"
    tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
    : > "${server_log}"
    tmux new-session -d -s "${session_name}" \
      "sudo -n ip netns exec ${ns_name} bash -lc 'cd \"${LOCAL_LIBRARY_DIR}\" && exec bash smartrun.sh bftsmart.demo.smallbank.SmallBankServer ${replica_id} --config-dir smallbank${n} --failure-start-unix-ms ${START_UNIX_MS} --failure-spec config/failure_spec.xml ${SERVER_LEARNING_FLAG}' > \"${server_log}\" 2>&1"
  ) &
  server_start_pids+=("$!")
  server_start_replicas+=("${n}")
done
for i in "${!server_start_pids[@]}"; do
  if ! wait "${server_start_pids[$i]}"; then
    echo "Failed to launch PBFT server session for replica ${server_start_replicas[$i]}" >&2
    server_start_failed=1
  fi
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

echo "Stopping tmux sessions..."
sessions="$(tmux ls -F '#S' 2>/dev/null || true)"
if [[ -n "${sessions}" ]]; then
  while read -r session_name; do
    [[ -z "${session_name}" ]] && continue
    if [[ "${session_name}" == pbft_server_* || "${session_name}" == pbft_applier_* || "${session_name}" == pbft_agent_* || "${session_name}" == pbft_client ]]; then
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

if (( LOG_ENABLED == 1 )); then
  SERVER_LOGS_LOCAL_DIR="${LOCAL_TMP_DIR}/server_logs"
  echo "Collecting server logs to ${SERVER_LOGS_LOCAL_DIR}..."
  rm -rf "${SERVER_LOGS_LOCAL_DIR}"
  mkdir -p "${SERVER_LOGS_LOCAL_DIR}"
  log_collect_failed=0
  for ((n = 0; n < replicas_per_machine; n++)); do
    replica_id="${n}"
    src_log="${LOCAL_LIBRARY_DIR}/server_${replica_id}.log"
    dst_log="${SERVER_LOGS_LOCAL_DIR}/server_${replica_id}.log"
    rm -f "${dst_log}"
    if [[ -f "${src_log}" ]]; then
      cp -a "${src_log}" "${dst_log}"
      echo "  - copied server_${replica_id}.log"
    else
      echo "  - missing server_${replica_id}.log: ${src_log}" >&2
      log_collect_failed=1
    fi
  done
  if (( log_collect_failed != 0 )); then
    echo "Some server logs could not be collected." >&2
  fi
fi

if (( LEARNING_AGENT_ENABLED == 1 )) && (( DATASET_ENABLED == 1 )); then
  echo "Collecting learning-agent dataset CSVs..."
  for ((n = 0; n < replicas_per_machine; n++)); do
    replica_id="${n}"
    src_csv="${LOCAL_LIBRARY_DIR}/agent${replica_id}.csv"
    dst_csv="${LOCAL_TMP_DIR}/agent${replica_id}.csv"
    rm -f "${dst_csv}"
    if [[ -f "${src_csv}" ]]; then
      cp -a "${src_csv}" "${dst_csv}"
      echo "  - copied ${dst_csv}"
    else
      echo "  - missing dataset for replica ${replica_id}: ${src_csv}" >&2
    fi
  done
fi
