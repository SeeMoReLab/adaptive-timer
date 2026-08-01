#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SMARTBFT_DIR="${REPO_ROOT}/SmartBFT"
LEARNING_AGENT_SRC_DIR="${REPO_ROOT}/learning_agent"
CONFIG_DEFAULT="${REPO_ROOT}/config/pbft/smallbank.xml"
FAILURE_SPEC_DEFAULT="${REPO_ROOT}/config/pbft/failure_spec.xml"
LOCAL_TMP_DIR="${REPO_ROOT}/scripts/local/tmp/smartbft"
OUTPUT_LOCAL="${LOCAL_TMP_DIR}/output.txt"

RUN_AS_USER="${SUDO_USER:-${USER:-}}"
RUN_AS_HOME="${HOME}"
if [[ -n "${RUN_AS_USER}" ]] && command -v getent >/dev/null 2>&1; then
  resolved_home="$(getent passwd "${RUN_AS_USER}" 2>/dev/null | cut -d: -f6 || true)"
  if [[ -n "${resolved_home}" ]]; then
    RUN_AS_HOME="${resolved_home}"
  fi
fi
USER_PYENV_ROOT="${PYENV_ROOT:-${RUN_AS_HOME}/.pyenv}"

MODE="normal"
F=1
NODES_SET=0
NODES=4
EXPERIMENT_TIME_S=300
START_DELAY_S=10

CONFIG_PATH="${CONFIG_DEFAULT}"
FAILURE_SPEC_PATH="${FAILURE_SPEC_DEFAULT}"
FAILURE_SPEC_ENABLED=1
DATASET_ENABLED=0
LOG_ENABLED=0
AGENT_MODEL_TYPE="random"
AGENT_MODEL_SET=0
AGENT_HOST="127.0.0.1"
AGENT_PORT=15501
LEARNING_NODE_ID=1
SERVER_PORT_BASE=19100

BATCH_SIZE=200
BATCH_TIMEOUT="50ms"
REQUEST_TIMEOUT="30s"
SUBMIT_MODE="broadcast"
CREATE_WORKERS=""
INITIAL_TIMEOUT="5s"
LEARNING_WINDOW_MODE="consensus"
LEARNING_FEATURE_DURATION="10s"
LEARNING_REPLY_WAIT="2s"
LEARNING_WARMUP_DURATION="3s"
LEARNING_REWARD_DURATION="5s"
BACKOFF_ENABLED=0
REQUEST_TIMEOUT_BACKOFF_MAX="10s"

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
      signaled=0
      while read -r session_name; do
        [[ -z "${session_name}" ]] && continue
        if [[ "${session_name}" == smartbft_agent_* || "${session_name}" == smartbft_server_* || "${session_name}" == smartbft_client ]]; then
          tmux send-keys -t "${session_name}" C-c >/dev/null 2>&1 || true
          signaled=1
        fi
      done <<< "${sessions}"
      if (( signaled == 1 )); then
        sleep 3
      fi
      while read -r session_name; do
        [[ -z "${session_name}" ]] && continue
        if [[ "${session_name}" == smartbft_agent_* || "${session_name}" == smartbft_server_* || "${session_name}" == smartbft_client ]]; then
          tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
        fi
      done <<< "${sessions}"
    fi
  fi
  set -e
}
trap cleanup EXIT

usage() {
  cat <<EOF
Usage: $0 [normal|heuristic|learning|sharing|scheduled] [--dataset] [--log] [--backoff] [--model <random|linear|cmab|cmab_old|continuous>] [-f <fault_tolerance>] [-t <experiment_seconds>]

Arguments:
  mode              Optional; one of normal|heuristic|learning|sharing|scheduled (default: normal).

Flags:
  --dataset         Compatibility flag; agent modes collect one agent CSV per node.
  --log             Keep/copy SmartBFT benchmark and learning-agent logs.
  --backoff         Enable request timeout backoff with a 10s maximum.
  --model           Learning-agent model for learning mode: random|linear|cmab|cmab_old|continuous (default: random).
  -f <int>          Fault tolerance f; nodes defaults to 3f+1 (default: 1).
  -t <int>          Seconds to run after benchmark start before stopping tmux sessions (default: 300).

SmartBFT options:
  --nodes <int>                         Override node count (default: 3f+1).
  --config <path>                       SmallBank XML config (default: config/pbft/smallbank.xml).
  --failure-spec <path>                 failure_spec.xml for proposal delay (default: config/pbft/failure_spec.xml).
  --no-failure-spec                     Do not pass a failure spec to SmartBFT.
  --batch-size <int>                    SmartBFT batch size (default: 500).
  --batch-timeout <duration>            SmartBFT batch timeout (default: 50ms).
  --request-timeout <duration>          Client request timeout (default: 30s).
  --submit-mode <broadcast|leader>      Client submit mode (default: broadcast).
  --create-workers <int>                Account creation workers (default: first workload phase terminals).
  --start-delay <seconds>               Delay before workload/failure start (default: 10).
  --port-base <int>                     Base server port; node i listens on base+i (default: 19100).

Learning-agent options:
  --agent-host <host>                   Learning-agent bind/connect host (default: 127.0.0.1).
  --agent-port <port>                   Base learning-agent port; node i uses base+i-1 (default: 15501).
  --learning-node-id <id>               Deprecated; SmartBFT local now starts one agent per node.
  --initial-timeout <duration>          Initial PBFT timeout value reported by SmartBFT (default: 5s).
  --learning-window-mode <mode>         Episode boundaries: consensus|wall-clock (default: consensus).
  --learning-feature-duration <duration>
                                         Feature window in wall-clock mode (default: 10s).
  --learning-reply-wait <duration>      Recommendation wait in wall-clock mode (default: 2s).
  --learning-warmup-duration <duration> Post-apply warm-up in wall-clock mode (default: 3s).
  --learning-reward-duration <duration> Reward window in wall-clock mode (default: 5s).

Notes:
  SmartBFT SmallBank runs one server process per replica plus one client process.
  Request timeout backoff is disabled unless --backoff is supplied.
  Logs are written as server_N.log, client.log, and agent_REPLICA.log.
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

normalize_agent_model_type() {
  local model_lower
  model_lower="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  case "${model_lower}" in
    random|linear|cmab|cmab_old|continuous)
      printf '%s\n' "${model_lower}"
      ;;
    *)
      return 1
      ;;
  esac
}

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local deadline_epoch_s="$3"

  while (( $(date +%s) < deadline_epoch_s )); do
    if bash -c "exec 3<>/dev/tcp/${host}/${port}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
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

write_filtered_output() {
  local source_log="$1"
  local output_path="$2"

  awk '
    /Monitor duration=/ {
      seen_monitor = 1
      print
      next
    }
    seen_monitor == 0 {
      print
    }
  ' "${source_log}" > "${output_path}"
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
    --backoff)
      BACKOFF_ENABLED=1
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
    --nodes)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --nodes." >&2
        usage >&2
        exit 2
      fi
      NODES="${2}"
      NODES_SET=1
      shift 2
      ;;
    --config)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --config." >&2
        usage >&2
        exit 2
      fi
      CONFIG_PATH="${2}"
      shift 2
      ;;
    --failure-spec)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --failure-spec." >&2
        usage >&2
        exit 2
      fi
      FAILURE_SPEC_PATH="${2}"
      FAILURE_SPEC_ENABLED=1
      shift 2
      ;;
    --no-failure-spec)
      FAILURE_SPEC_ENABLED=0
      shift
      ;;
    --batch-size)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --batch-size." >&2
        usage >&2
        exit 2
      fi
      BATCH_SIZE="${2}"
      shift 2
      ;;
    --batch-timeout)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --batch-timeout." >&2
        usage >&2
        exit 2
      fi
      BATCH_TIMEOUT="${2}"
      shift 2
      ;;
    --request-timeout)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --request-timeout." >&2
        usage >&2
        exit 2
      fi
      REQUEST_TIMEOUT="${2}"
      shift 2
      ;;
    --submit-mode)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --submit-mode." >&2
        usage >&2
        exit 2
      fi
      SUBMIT_MODE="${2}"
      shift 2
      ;;
    --create-workers)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --create-workers." >&2
        usage >&2
        exit 2
      fi
      CREATE_WORKERS="${2}"
      shift 2
      ;;
    --start-delay)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --start-delay." >&2
        usage >&2
        exit 2
      fi
      START_DELAY_S="${2}"
      shift 2
      ;;
    --port-base)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --port-base." >&2
        usage >&2
        exit 2
      fi
      SERVER_PORT_BASE="${2}"
      shift 2
      ;;
    --agent-host)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --agent-host." >&2
        usage >&2
        exit 2
      fi
      AGENT_HOST="${2}"
      shift 2
      ;;
    --agent-port)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --agent-port." >&2
        usage >&2
        exit 2
      fi
      AGENT_PORT="${2}"
      shift 2
      ;;
    --learning-node-id)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --learning-node-id." >&2
        usage >&2
        exit 2
      fi
      LEARNING_NODE_ID="${2}"
      shift 2
      ;;
    --initial-timeout)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --initial-timeout." >&2
        usage >&2
        exit 2
      fi
      INITIAL_TIMEOUT="${2}"
      shift 2
      ;;
    --learning-window-mode)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --learning-window-mode." >&2
        usage >&2
        exit 2
      fi
      LEARNING_WINDOW_MODE="${2}"
      shift 2
      ;;
    --learning-feature-duration)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --learning-feature-duration." >&2
        usage >&2
        exit 2
      fi
      LEARNING_FEATURE_DURATION="${2}"
      shift 2
      ;;
    --learning-reply-wait)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --learning-reply-wait." >&2
        usage >&2
        exit 2
      fi
      LEARNING_REPLY_WAIT="${2}"
      shift 2
      ;;
    --learning-warmup-duration)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --learning-warmup-duration." >&2
        usage >&2
        exit 2
      fi
      LEARNING_WARMUP_DURATION="${2}"
      shift 2
      ;;
    --learning-reward-duration)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --learning-reward-duration." >&2
        usage >&2
        exit 2
      fi
      LEARNING_REWARD_DURATION="${2}"
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
if ! [[ "${EXPERIMENT_TIME_S}" =~ ^[0-9]+$ ]] || (( EXPERIMENT_TIME_S < 1 )); then
  echo "Invalid -t value: ${EXPERIMENT_TIME_S}. It must be an integer >= 1." >&2
  exit 2
fi
if ! [[ "${START_DELAY_S}" =~ ^[0-9]+$ ]]; then
  echo "Invalid --start-delay value: ${START_DELAY_S}. It must be an integer >= 0." >&2
  exit 2
fi
if (( NODES_SET == 0 )); then
  NODES="$((3 * F + 1))"
fi
case "${LEARNING_WINDOW_MODE}" in
  consensus|wall-clock) ;;
  *) echo "Invalid --learning-window-mode value: ${LEARNING_WINDOW_MODE}. Expected consensus or wall-clock." >&2; exit 2 ;;
esac
if ! [[ "${NODES}" =~ ^[0-9]+$ ]] || (( NODES < 4 )); then
  echo "Invalid node count: ${NODES}. SmartBFT SmallBank requires at least 4 nodes." >&2
  exit 2
fi
if ! [[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]] || (( BATCH_SIZE < 1 )); then
  echo "Invalid --batch-size value: ${BATCH_SIZE}." >&2
  exit 2
fi
case "${SUBMIT_MODE}" in
  broadcast|leader)
    ;;
  *)
    echo "Invalid --submit-mode value: ${SUBMIT_MODE}. Expected broadcast or leader." >&2
    exit 2
    ;;
esac
if [[ -n "${CREATE_WORKERS}" ]]; then
  if ! [[ "${CREATE_WORKERS}" =~ ^[0-9]+$ ]] || (( CREATE_WORKERS < 1 )); then
    echo "Invalid --create-workers value: ${CREATE_WORKERS}." >&2
    exit 2
  fi
fi
if ! [[ "${AGENT_PORT}" =~ ^[0-9]+$ ]] || (( AGENT_PORT < 1 || AGENT_PORT + NODES - 1 > 65535 )); then
  echo "Invalid --agent-port value: ${AGENT_PORT}." >&2
  exit 2
fi
if ! [[ "${SERVER_PORT_BASE}" =~ ^[0-9]+$ ]] || (( SERVER_PORT_BASE < 1 || SERVER_PORT_BASE + NODES > 65535 )); then
  echo "Invalid --port-base value: ${SERVER_PORT_BASE} for ${NODES} nodes." >&2
  exit 2
fi
if ! mode_enables_learning_agent "${MODE}"; then
  LEARNING_AGENT_ENABLED=0
else
  LEARNING_AGENT_ENABLED=1
fi
CONFIG_PATH="$(absolute_path "${CONFIG_PATH}")"
FAILURE_SPEC_PATH="$(absolute_path "${FAILURE_SPEC_PATH}")"

if [[ ! -d "${SMARTBFT_DIR}" ]]; then
  echo "SmartBFT directory not found: ${SMARTBFT_DIR}" >&2
  exit 1
fi
if (( LEARNING_AGENT_ENABLED == 1 )) && [[ ! -d "${LEARNING_AGENT_SRC_DIR}" ]]; then
  echo "learning_agent directory not found: ${LEARNING_AGENT_SRC_DIR}" >&2
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
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux command not found." >&2
  exit 1
fi
if ! command -v go >/dev/null 2>&1; then
  echo "go command not found." >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 command not found." >&2
  exit 1
fi

if ! AGENT_MODEL_TYPE="$(normalize_agent_model_type "${AGENT_MODEL_TYPE}")"; then
  echo "Invalid --model value: ${AGENT_MODEL_TYPE}. Use one of random|linear|cmab|cmab_old|continuous." >&2
  exit 2
fi
if (( AGENT_MODEL_SET == 1 )) && [[ "${MODE}" != "learning" ]]; then
  echo "--model is only supported with learning mode for SmartBFT local runs." >&2
  exit 2
fi
if (( DATASET_ENABLED == 1 )) && (( LEARNING_AGENT_ENABLED == 0 )); then
  echo "--dataset requires a learning-agent mode: heuristic|learning|sharing|scheduled." >&2
  exit 2
fi
if (( LEARNING_AGENT_ENABLED == 1 )); then
  DATASET_ENABLED=1
fi
if [[ "${MODE}" == "sharing" ]]; then
  echo "SmartBFT sharing mode starts one learning agent per node." >&2
fi

AGENT_MODEL_ARGS=""
AGENT_PYTHON_BIN="python3"
if (( LEARNING_AGENT_ENABLED == 1 )); then
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

  if [[ "${MODE}" == "learning" ]]; then
    AGENT_MODEL_ARGS="--model-type ${AGENT_MODEL_TYPE}"
  elif [[ "${MODE}" == "scheduled" ]]; then
    AGENT_MODEL_ARGS="--model-type scheduled --schedule-file learning_agent/config/schedule_pbft.csv"
  fi

  agent_import_error="$(mktemp)"
  if ! (cd "${LEARNING_AGENT_SRC_DIR}" && "${AGENT_PYTHON_BIN}" -c "import grpc, torch, agent_pb2_grpc") > /dev/null 2> "${agent_import_error}"; then
    echo "Selected learning-agent interpreter cannot import grpc, torch, and generated agent stubs: ${AGENT_PYTHON_BIN}" >&2
    echo "Resolver context: RUN_AS_USER=${RUN_AS_USER:-<unset>}, RUN_AS_HOME=${RUN_AS_HOME}, USER_PYENV_ROOT=${USER_PYENV_ROOT}" >&2
    cat "${agent_import_error}" >&2
    rm -f "${agent_import_error}"
    echo "Install dependencies into that interpreter and retry." >&2
    exit 1
  fi
  rm -f "${agent_import_error}"
fi

mkdir -p "${LOCAL_TMP_DIR}"
rm -f "${OUTPUT_LOCAL}" "${LOCAL_TMP_DIR}/client.log" "${LOCAL_TMP_DIR}/smartbft.log" "${LOCAL_TMP_DIR}"/server_*.log "${LOCAL_TMP_DIR}"/agent_*.log "${LOCAL_TMP_DIR}"/agent*.csv
rm -rf "${LOCAL_TMP_DIR}/wal"
cleanup
cleanup_done=0

HOSTS_CONFIG="${LOCAL_TMP_DIR}/hosts.config"
: > "${HOSTS_CONFIG}"
for ((n = 1; n <= NODES; n++)); do
  printf '%d 127.0.0.1 %d\n' "${n}" "$((SERVER_PORT_BASE + n))" >> "${HOSTS_CONFIG}"
done

START_UNIX_MS="$(( ($(date +%s) + START_DELAY_S) * 1000 ))"
EXPERIMENT_END_UNIX_MS="$((START_UNIX_MS + EXPERIMENT_TIME_S * 1000))"
EXPERIMENT_END_EPOCH_S="$((EXPERIMENT_END_UNIX_MS / 1000))"

echo "Selected mode: ${MODE}"
echo "SmartBFT nodes: ${NODES} (f=${F})"
echo "SmartBFT hosts config: ${HOSTS_CONFIG}"
echo "SmallBank config: ${CONFIG_PATH}"
echo "Submit mode: ${SUBMIT_MODE}"
echo "SmartBFT batch size: ${BATCH_SIZE}"
echo "SmartBFT batch timeout: ${BATCH_TIMEOUT}"
echo "SmartBFT request timeout: ${REQUEST_TIMEOUT}"
if (( BACKOFF_ENABLED == 1 )); then
  echo "SmartBFT request timeout backoff: enabled (max=${REQUEST_TIMEOUT_BACKOFF_MAX})"
else
  echo "SmartBFT request timeout backoff: disabled"
fi
if (( FAILURE_SPEC_ENABLED == 1 )); then
  echo "Failure spec: ${FAILURE_SPEC_PATH}"
else
  echo "Failure spec: disabled"
fi
echo "Learning-agent orchestration: ${LEARNING_AGENT_ENABLED} (dataset=${DATASET_ENABLED}, log=${LOG_ENABLED})"
if [[ "${MODE}" == "learning" ]]; then
  echo "Learning-agent model: ${AGENT_MODEL_TYPE}"
fi
echo "Using shared start_unix_ms=${START_UNIX_MS}"

if (( LEARNING_AGENT_ENABLED == 1 )); then
  echo "Learning-agent interpreter: ${AGENT_PYTHON_BIN}"
  echo "Starting SmartBFT PBFT learning agents..."
  failure_agent_args=""
  if (( FAILURE_SPEC_ENABLED == 1 )); then
    failure_agent_args="--failure-spec \"${FAILURE_SPEC_PATH}\" --injection-start-unix-ms ${START_UNIX_MS}"
  fi
  for ((replica_id = 0; replica_id < NODES; replica_id++)); do
    agent_port=$((AGENT_PORT + replica_id))
    session_name="smartbft_agent_${replica_id}"
    agent_log="${LOCAL_TMP_DIR}/agent_${replica_id}.log"
    dataset_args=""
    if (( DATASET_ENABLED == 1 )); then
      dataset_args="--dataset-csv ${LOCAL_TMP_DIR}/agent${replica_id}.csv"
    fi
    tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
    : > "${agent_log}"
    if ! tmux new-session -d -s "${session_name}" \
      "cd \"${REPO_ROOT}\" && exec \"${AGENT_PYTHON_BIN}\" learning_agent/main.py --node-id ${replica_id} --protocol pbft --host \"${AGENT_HOST}\" --port ${agent_port} --signing false ${AGENT_MODEL_ARGS} ${dataset_args} ${failure_agent_args} > \"${agent_log}\" 2>&1"; then
      echo "Failed to start tmux session ${session_name}." >&2
      exit 1
    fi
  done

  for ((replica_id = 0; replica_id < NODES; replica_id++)); do
    agent_port=$((AGENT_PORT + replica_id))
    if ! wait_for_tcp "${AGENT_HOST}" "${agent_port}" "$(( $(date +%s) + 15 ))"; then
      echo "Timed out waiting for learning agent at ${AGENT_HOST}:${agent_port}" >&2
      echo "Last learning-agent log lines:" >&2
      tail -n 80 "${LOCAL_TMP_DIR}/agent_${replica_id}.log" >&2 || true
      exit 1
    fi
  done
fi

server_common_args=(
  "go" "run" "./examples/smallbank"
  "--role" "server"
  "--hosts-config" "${HOSTS_CONFIG}"
  "--batch-size" "${BATCH_SIZE}"
  "--batch-timeout" "${BATCH_TIMEOUT}"
  "--request-timeout" "${REQUEST_TIMEOUT}"
)
if (( BACKOFF_ENABLED == 1 )); then
  server_common_args+=(
    "--request-timeout-backoff"
    "--request-timeout-backoff-max" "${REQUEST_TIMEOUT_BACKOFF_MAX}"
  )
fi
if (( FAILURE_SPEC_ENABLED == 1 )); then
  server_common_args+=(
    "--failure-spec" "${FAILURE_SPEC_PATH}"
    "--failure-start-unix-ms" "${START_UNIX_MS}"
  )
fi
echo "Starting SmartBFT server replicas..."
for ((n = 1; n <= NODES; n++)); do
  session_name="smartbft_server_${n}"
  server_log="${LOCAL_TMP_DIR}/server_${n}.log"
  server_data_dir="${LOCAL_TMP_DIR}/wal/node_${n}"
  server_args=("${server_common_args[@]}" "--node-id" "${n}" "--data-dir" "${server_data_dir}")
  if (( LEARNING_AGENT_ENABLED == 1 )); then
    agent_port=$((AGENT_PORT + n - 1))
    server_args+=(
      "--learning"
      "--agent-target" "${AGENT_HOST}:${agent_port}"
      "--learning-node-id" "${n}"
      "--learning-initial-election-timeout" "${INITIAL_TIMEOUT}"
      "--learning-window-mode" "${LEARNING_WINDOW_MODE}"
      "--learning-feature-duration" "${LEARNING_FEATURE_DURATION}"
      "--learning-reply-wait" "${LEARNING_REPLY_WAIT}"
      "--learning-warmup-duration" "${LEARNING_WARMUP_DURATION}"
      "--learning-reward-duration" "${LEARNING_REWARD_DURATION}"
    )
  fi
  printf -v server_cmd '%q ' "${server_args[@]}"
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  : > "${server_log}"
  if ! tmux new-session -d -s "${session_name}" \
    "cd \"${SMARTBFT_DIR}\" && GOCACHE=\"${LOCAL_TMP_DIR}/gocache\" ${server_cmd} > \"${server_log}\" 2>&1"; then
    echo "Failed to start tmux session ${session_name}." >&2
    exit 1
  fi
done

echo "Waiting for SmartBFT server ports..."
for ((n = 1; n <= NODES; n++)); do
  port="$((SERVER_PORT_BASE + n))"
  if ! wait_for_tcp "127.0.0.1" "${port}" "$(( $(date +%s) + 30 ))"; then
    echo "Timed out waiting for smartbft_server_${n} on 127.0.0.1:${port}" >&2
    echo "Last server log lines:" >&2
    tail -n 80 "${LOCAL_TMP_DIR}/server_${n}.log" >&2 || true
    exit 1
  fi
done

client_args=(
  "go" "run" "./examples/smallbank"
  "--role" "client"
  "--hosts-config" "${HOSTS_CONFIG}"
  "--config" "${CONFIG_PATH}"
  "--request-timeout" "${REQUEST_TIMEOUT}"
  "--submit-mode" "${SUBMIT_MODE}"
  "--create"
  "--execute"
  "--start-unix-ms" "${START_UNIX_MS}"
)
if [[ -n "${CREATE_WORKERS}" ]]; then
  client_args+=("--create-workers" "${CREATE_WORKERS}")
fi

printf -v client_cmd '%q ' "${client_args[@]}"
echo "Starting SmartBFT client..."
tmux kill-session -t smartbft_client >/dev/null 2>&1 || true
: > "${LOCAL_TMP_DIR}/client.log"
if ! tmux new-session -d -s smartbft_client \
  "cd \"${SMARTBFT_DIR}\" && GOCACHE=\"${LOCAL_TMP_DIR}/gocache\" ${client_cmd} > \"${LOCAL_TMP_DIR}/client.log\" 2>&1"; then
  echo "Failed to start tmux session smartbft_client." >&2
  exit 1
fi

echo "SmartBFT server sessions: smartbft_server_1..smartbft_server_${NODES}"
echo "SmartBFT client session: smartbft_client"
if (( LEARNING_AGENT_ENABLED == 1 )); then
  last_agent_id=$((NODES - 1))
  echo "Learning-agent sessions: smartbft_agent_0..smartbft_agent_${last_agent_id}"
fi
echo "Logs:"
echo "  Client: ${LOCAL_TMP_DIR}/client.log"
echo "  Servers: ${LOCAL_TMP_DIR}/server_N.log"
if (( LEARNING_AGENT_ENABLED == 1 )); then
  echo "  Agents:   ${LOCAL_TMP_DIR}/agent_REPLICA.log"
fi
echo "Experiment deadline unix_ms=${EXPERIMENT_END_UNIX_MS} (start_unix_ms + ${EXPERIMENT_TIME_S}s)"

while true; do
  now_epoch_s="$(date +%s)"
  if ! tmux has-session -t smartbft_client 2>/dev/null; then
    echo "SmartBFT client session finished."
    break
  fi
  remaining_experiment_s="$((EXPERIMENT_END_EPOCH_S - now_epoch_s))"
  if (( remaining_experiment_s <= 0 )); then
    echo "Experiment deadline reached."
    break
  fi

  remaining_minutes="$(((remaining_experiment_s + 59) / 60))"
  echo "Time left until experiment deadline: ${remaining_experiment_s}s (~${remaining_minutes}m)"

  sleep_s=30
  if (( remaining_experiment_s < sleep_s )); then
    sleep_s="${remaining_experiment_s}"
  fi
  sleep "${sleep_s}"
done

echo "Stopping tmux sessions..."
cleanup
trap - EXIT

write_filtered_output "${LOCAL_TMP_DIR}/client.log" "${OUTPUT_LOCAL}"
echo "Done. Filtered SmartBFT client output saved to ${OUTPUT_LOCAL}."

if (( LOG_ENABLED == 1 )); then
  LOGS_LOCAL_DIR="${LOCAL_TMP_DIR}/logs"
  echo "Collecting logs to ${LOGS_LOCAL_DIR}..."
  rm -rf "${LOGS_LOCAL_DIR}"
  mkdir -p "${LOGS_LOCAL_DIR}"
  cp -a "${LOCAL_TMP_DIR}/client.log" "${LOGS_LOCAL_DIR}/client.log"
  for ((n = 1; n <= NODES; n++)); do
    if [[ -f "${LOCAL_TMP_DIR}/server_${n}.log" ]]; then
      cp -a "${LOCAL_TMP_DIR}/server_${n}.log" "${LOGS_LOCAL_DIR}/server_${n}.log"
    fi
  done
  if (( LEARNING_AGENT_ENABLED == 1 )); then
    for ((replica_id = 0; replica_id < NODES; replica_id++)); do
      if [[ -f "${LOCAL_TMP_DIR}/agent_${replica_id}.log" ]]; then
        cp -a "${LOCAL_TMP_DIR}/agent_${replica_id}.log" "${LOGS_LOCAL_DIR}/agent_${replica_id}.log"
      fi
    done
  fi
fi

if (( LEARNING_AGENT_ENABLED == 1 )) && (( DATASET_ENABLED == 1 )); then
  saved_agent_datasets=()
  for ((replica_id = 0; replica_id < NODES; replica_id++)); do
    if [[ -f "${LOCAL_TMP_DIR}/agent${replica_id}.csv" ]]; then
      saved_agent_datasets+=("${LOCAL_TMP_DIR}/agent${replica_id}.csv")
    fi
  done
  if (( ${#saved_agent_datasets[@]} > 0 )); then
    echo "Learning-agent datasets saved to:"
    printf '  %s\n' "${saved_agent_datasets[@]}"
  else
    echo "Learning-agent datasets not found." >&2
  fi
fi
