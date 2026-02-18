#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PREPARE_SCRIPT="${REPO_ROOT}/scripts/prepare_pbft.sh"
PREPARED_LIBRARY_DIR="${REPO_ROOT}/scripts/tmp/library"
LOCAL_TMP_DIR="${REPO_ROOT}/scripts/local/tmp"
LOCAL_LIBRARY_DIR="${LOCAL_TMP_DIR}/library_debug"
LEARNING_AGENT_SRC_DIR="${REPO_ROOT}/learning_agent"

MODE="normal"
F=1
DATASET_ENABLED=0
SERVER_READY_TIMEOUT_S=30
LAUNCHER_STAGGER_S=0.2
AUTO_START_DELAY_S=30
START_UNIX_MS=""
TERMINAL_BACKEND=""

usage() {
  cat <<EOF
Usage: $0 [normal|heuristic|learning|sharing|scheduled] [--dataset] [-f <fault_tolerance>] [--start-unix-ms <epoch_ms>]

Arguments:
  mode              Optional; one of normal|heuristic|learning|sharing|scheduled (default: normal).

Flags:
  --dataset         Save per-replica learning-agent datasets (agentN.csv) under ${LOCAL_LIBRARY_DIR}.
  -f <int>          Fault tolerance f passed to prepare_pbft.sh (default: 1).
  --start-unix-ms   Absolute epoch milliseconds for benchmark start (default: now + ${AUTO_START_DELAY_S}s).

Notes:
  - This debug runner launches each process in a separate terminal.
  - macOS is supported via Terminal.app (osascript).
  - Linux is supported when one of gnome-terminal/x-terminal-emulator/xterm is available.
  - No network namespace or failure injection is used; all replicas use 127.0.0.1 with per-replica ports.
EOF
}

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

quote_cmd() {
  local quoted=""
  local token
  for token in "$@"; do
    local q
    printf -v q '%q' "${token}"
    quoted+="${q} "
  done
  printf '%s' "${quoted% }"
}

escape_for_applescript() {
  local text="$1"
  text="${text//\\/\\\\}"
  text="${text//\"/\\\"}"
  text="${text//$'\n'/\\n}"
  printf '%s' "${text}"
}

rewrite_hosts_for_localhost() {
  local hosts_file="$1"
  local tmp_hosts

  if [[ ! -f "${hosts_file}" ]]; then
    echo "Hosts file not found for localhost rewrite: ${hosts_file}" >&2
    return 1
  fi

  tmp_hosts="$(mktemp)"
  awk '
    BEGIN { OFS = " " }
    /^[[:space:]]*#/ || NF == 0 { print; next }
    NF < 6 { print; next }
    {
      $2 = "127.0.0.1"
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
      if ! bash -c "exec 3<>/dev/tcp/${replica_ip}/${replica_port}" >/dev/null 2>&1; then
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

detect_terminal_backend() {
  local os_name
  os_name="$(uname -s)"
  case "${os_name}" in
    Darwin)
      if ! command -v osascript >/dev/null 2>&1; then
        echo "osascript not found; cannot open Terminal.app windows." >&2
        exit 1
      fi
      TERMINAL_BACKEND="macos-terminal"
      ;;
    Linux)
      if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
        echo "No graphical display detected (DISPLAY/WAYLAND_DISPLAY unset)." >&2
        exit 1
      fi
      if command -v gnome-terminal >/dev/null 2>&1; then
        TERMINAL_BACKEND="gnome-terminal"
      elif command -v x-terminal-emulator >/dev/null 2>&1; then
        TERMINAL_BACKEND="x-terminal-emulator"
      elif command -v xterm >/dev/null 2>&1; then
        TERMINAL_BACKEND="xterm"
      else
        echo "No supported terminal emulator found (need gnome-terminal, x-terminal-emulator, or xterm)." >&2
        exit 1
      fi
      ;;
    *)
      echo "Unsupported OS for debug launcher: ${os_name}" >&2
      exit 1
      ;;
  esac
}

launch_in_terminal() {
  local title="$1"
  local launcher_script="$2"
  local launch_cmd
  local escaped

  case "${TERMINAL_BACKEND}" in
    macos-terminal)
      launch_cmd="$(quote_cmd bash "${launcher_script}")"
      escaped="$(escape_for_applescript "${launch_cmd}")"
      osascript \
        -e 'tell application "Terminal" to activate' \
        -e "tell application \"Terminal\" to do script \"${escaped}\"" >/dev/null
      ;;
    gnome-terminal)
      launch_cmd="$(quote_cmd bash "${launcher_script}")"
      gnome-terminal -- bash -lc "${launch_cmd}" >/dev/null 2>&1 &
      ;;
    x-terminal-emulator)
      x-terminal-emulator -e bash "${launcher_script}" >/dev/null 2>&1 &
      ;;
    xterm)
      xterm -T "${title}" -e bash "${launcher_script}" >/dev/null 2>&1 &
      ;;
    *)
      echo "Unknown terminal backend: ${TERMINAL_BACKEND}" >&2
      exit 1
      ;;
  esac
}

write_launcher_script() {
  local launcher_script="$1"
  local title="$2"
  local log_path="$3"
  local command_str="$4"
  local q_library
  local q_log

  printf -v q_library '%q' "${LOCAL_LIBRARY_DIR}"
  printf -v q_log '%q' "${log_path}"

  cat > "${launcher_script}" <<EOF
#!/usr/bin/env bash
set -u
printf '\\033]0;%s\\007' "${title}"
cd ${q_library}
: > ${q_log}
echo "[${title}] Starting..."
set -o pipefail
{
${command_str}
} > >(tee -a ${q_log}) 2>&1
status=\$?
echo
echo "[${title}] Exit code: \${status}"
echo "[${title}] Log: ${log_path}"
EOF
  chmod +x "${launcher_script}"
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
    --dataset)
      DATASET_ENABLED=1
      shift
      ;;
    --start-unix-ms)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --start-unix-ms." >&2
        usage >&2
        exit 2
      fi
      START_UNIX_MS="${2}"
      shift 2
      ;;
    normal|heuristic|learning|sharing|scheduled)
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
if [[ -n "${START_UNIX_MS}" ]]; then
  if ! [[ "${START_UNIX_MS}" =~ ^[0-9]+$ ]]; then
    echo "Invalid --start-unix-ms value: ${START_UNIX_MS}. It must be an integer >= 0." >&2
    exit 2
  fi
fi

if [[ ! -x "${PREPARE_SCRIPT}" ]]; then
  echo "prepare_pbft.sh not found or not executable: ${PREPARE_SCRIPT}" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 command not found." >&2
  exit 1
fi

if ! mode_enables_learning_agent "${MODE}"; then
  LEARNING_AGENT_ENABLED=0
else
  LEARNING_AGENT_ENABLED=1
fi

AGENT_PYTHON_BIN="python3"
AGENT_MODE_ARGS=()
SERVER_MODE_ARGS=()
if (( LEARNING_AGENT_ENABLED == 1 )); then
  SERVER_MODE_ARGS+=(--learning)
  if [[ -x "${PYENV_ROOT:-${HOME}/.pyenv}/versions/timer/bin/python3" ]]; then
    AGENT_PYTHON_BIN="${PYENV_ROOT:-${HOME}/.pyenv}/versions/timer/bin/python3"
  elif [[ -x "${PYENV_ROOT:-${HOME}/.pyenv}/versions/timer/bin/python" ]]; then
    AGENT_PYTHON_BIN="${PYENV_ROOT:-${HOME}/.pyenv}/versions/timer/bin/python"
  elif command -v pyenv >/dev/null 2>&1; then
    pyenv_timer_python="$(PYENV_VERSION=timer pyenv which python3 2>/dev/null || true)"
    if [[ -n "${pyenv_timer_python}" && -x "${pyenv_timer_python}" ]]; then
      AGENT_PYTHON_BIN="${pyenv_timer_python}"
    fi
  fi

  if ! "${AGENT_PYTHON_BIN}" -c "import grpc" >/dev/null 2>&1; then
    echo "Selected learning-agent interpreter is missing grpc: ${AGENT_PYTHON_BIN}" >&2
    exit 1
  fi

  if [[ "${MODE}" == "sharing" ]]; then
    AGENT_MODE_ARGS+=(--sharing true)
  fi
fi

echo "Selected mode: ${MODE}"
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

echo "Staging debug runtime under ${LOCAL_LIBRARY_DIR}..."
mkdir -p "${LOCAL_TMP_DIR}"
rm -rf "${LOCAL_LIBRARY_DIR}"
cp -a "${PREPARED_LIBRARY_DIR}" "${LOCAL_LIBRARY_DIR}"
if (( LEARNING_AGENT_ENABLED == 1 )); then
  rm -rf "${LOCAL_LIBRARY_DIR}/learning_agent"
  cp -a "${LEARNING_AGENT_SRC_DIR}" "${LOCAL_LIBRARY_DIR}/learning_agent"
fi

if [[ "${MODE}" == "scheduled" ]]; then
  schedule_csv="${LOCAL_LIBRARY_DIR}/learning_agent/config/schedule.csv"
  if [[ ! -f "${schedule_csv}" ]]; then
    echo "Scheduled mode requires ${schedule_csv}" >&2
    exit 1
  fi
  AGENT_MODE_ARGS+=(--model-type scheduled --schedule-file "${schedule_csv}")
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

echo "Rewriting hosts configs to localhost..."
rewrite_hosts_for_localhost "${LOCAL_LIBRARY_DIR}/smallbank/hosts.config"
rewrite_hosts_for_localhost "${LOCAL_LIBRARY_DIR}/dataexchange/hosts.config"

echo "Materializing per-replica config directories..."
rm -rf "${LOCAL_LIBRARY_DIR}/client"
shopt -s nullglob
rm -rf "${LOCAL_LIBRARY_DIR}"/smallbank[0-9]* "${LOCAL_LIBRARY_DIR}"/dataexchange[0-9]*
shopt -u nullglob
cp -a "${LOCAL_LIBRARY_DIR}/smallbank" "${LOCAL_LIBRARY_DIR}/client"
for ((n = 0; n < NUM_SERVERS; n++)); do
  cp -a "${LOCAL_LIBRARY_DIR}/smallbank" "${LOCAL_LIBRARY_DIR}/smallbank${n}"
  cp -a "${LOCAL_LIBRARY_DIR}/dataexchange" "${LOCAL_LIBRARY_DIR}/dataexchange${n}"
done

detect_terminal_backend
echo "Terminal backend: ${TERMINAL_BACKEND}"

if [[ -z "${START_UNIX_MS}" ]]; then
  START_UNIX_MS="$(( ($(date +%s) + AUTO_START_DELAY_S) * 1000 ))"
fi
echo "Using shared start_unix_ms=${START_UNIX_MS}"

LAUNCHER_DIR="${LOCAL_TMP_DIR}/launchers_debug"
rm -rf "${LAUNCHER_DIR}"
mkdir -p "${LAUNCHER_DIR}"

if (( LEARNING_AGENT_ENABLED == 1 )); then
  echo "Launching learning-agent processes..."
  for ((n = 0; n < NUM_SERVERS; n++)); do
    replica_id="${n}"
    agent_log="${LOCAL_LIBRARY_DIR}/agent_${replica_id}.log"
    launcher_script="${LAUNCHER_DIR}/agent_${replica_id}.sh"
    dataset_args=()
    if (( DATASET_ENABLED == 1 )); then
      rm -f "${LOCAL_LIBRARY_DIR}/agent${replica_id}.csv"
      dataset_args+=(--dataset-csv "${LOCAL_LIBRARY_DIR}/agent${replica_id}.csv")
    fi
    agent_cmd=(
      "${AGENT_PYTHON_BIN}"
      learning_agent/main.py
      --node-id "${replica_id}"
      --hosts-config "${LOCAL_LIBRARY_DIR}/smallbank/hosts.config"
      --injection-start-unix-ms "${START_UNIX_MS}"
    )
    if (( ${#AGENT_MODE_ARGS[@]} > 0 )); then
      agent_cmd+=("${AGENT_MODE_ARGS[@]}")
    fi
    if (( ${#dataset_args[@]} > 0 )); then
      agent_cmd+=("${dataset_args[@]}")
    fi
    write_launcher_script \
      "${launcher_script}" \
      "PBFT Agent ${replica_id}" \
      "${agent_log}" \
      "$(quote_cmd "${agent_cmd[@]}")"
    launch_in_terminal "PBFT Agent ${replica_id}" "${launcher_script}"
    sleep "${LAUNCHER_STAGGER_S}"
  done
fi

echo "Launching PBFT server processes..."
for ((n = 0; n < NUM_SERVERS; n++)); do
  replica_id="${n}"
  server_log="${LOCAL_LIBRARY_DIR}/server_${replica_id}.log"
  launcher_script="${LAUNCHER_DIR}/server_${replica_id}.sh"
  server_cmd=(
    bash
    smartrun.sh
    bftsmart.demo.smallbank.SmallBankServer
    "${replica_id}"
    --config-dir "smallbank${n}"
  )
  if (( ${#SERVER_MODE_ARGS[@]} > 0 )); then
    server_cmd+=("${SERVER_MODE_ARGS[@]}")
  fi
  write_launcher_script \
    "${launcher_script}" \
    "PBFT Server ${replica_id}" \
    "${server_log}" \
    "$(quote_cmd "${server_cmd[@]}")"
  launch_in_terminal "PBFT Server ${replica_id}" "${launcher_script}"
  sleep "${LAUNCHER_STAGGER_S}"
done

echo "Waiting for servers to become reachable..."
server_ready_deadline="$(( $(date +%s) + SERVER_READY_TIMEOUT_S ))"
if ! wait_for_server_endpoints_or_fail "${LOCAL_LIBRARY_DIR}/smallbank/hosts.config" "${server_ready_deadline}"; then
  echo "Servers were launched, but not all ports became reachable in time." >&2
  exit 1
fi

echo "Launching PBFT client..."
client_log="${LOCAL_LIBRARY_DIR}/output.txt"
client_launcher="${LAUNCHER_DIR}/client.sh"
client_cmd=(
  bash
  smartrun.sh
  bftsmart.demo.smallbank.SmallBankClient
  --config-dir client
  --create
  --execute
  --start-unix-ms "${START_UNIX_MS}"
)
write_launcher_script \
  "${client_launcher}" \
  "PBFT Client" \
  "${client_log}" \
  "$(quote_cmd "${client_cmd[@]}")"
launch_in_terminal "PBFT Client" "${client_launcher}"

echo "Launched all processes in separate terminals."
echo "Logs are under: ${LOCAL_LIBRARY_DIR}"
echo "Hosts config: ${LOCAL_LIBRARY_DIR}/smallbank/hosts.config"
