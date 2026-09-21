#!/usr/bin/env bash
# Shared CloudLab harness for the Rust BFT protocols (autobahn submodule).
# Not executable on its own: a thin per-protocol runner (run_hotstuff.sh,
# run_autobahn.sh) sets the protocol variables below and then sources this
# file, which defines and finally invokes rustbft_main "$@".
#
# The phase structure, file formats, and conventions deliberately mirror
# scripts/cloudlab/run_smartbft.sh so artifacts stay uniform across protocols:
# targets file (node host user; node0 = client machine), mkdir locks,
# timestamped artifact dir with run.env, payload tarball + remote build,
# per-replica network namespaces on the 10.252.x.y overlay, one shared
# START_UNIX_MS, tmux sessions, scp collection, Monitor-line filtering.
#
# Required variables from the sourcing script:
#   PROTOCOL_NAME         e.g. hotstuff
#   SUPPORTED_MODES       space-separated subset of: normal heuristic learning sharing scheduled
#   AGENT_PROTOCOL        --protocol value for learning_agent/main.py (when agent modes are supported)
#   SCHEDULE_FILE_REL     schedule CSV for scheduled mode, relative to repo root
#   FAILURE_SPEC_DEFAULT_REL  default failure spec, relative to repo root
#   NS_PREFIX             netns name prefix, e.g. hs-r  -> hs-r<replica_id>
#   HOST_IF_PREFIX        host veth prefix (<=12 chars incl. id), e.g. hsh
#   NS_IF_PREFIX          namespace veth prefix, e.g. hsn
#   BRIDGE_NAME           e.g. br-hotstuff

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INVOCATION_DIR="$(pwd -P)"
ORIGINAL_ARGS=("$@")

for required_var in PROTOCOL_NAME SUPPORTED_MODES AGENT_PROTOCOL SCHEDULE_FILE_REL FAILURE_SPEC_DEFAULT_REL NS_PREFIX HOST_IF_PREFIX NS_IF_PREFIX BRIDGE_NAME; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "lib_rustbft.sh: required variable ${required_var} is not set." >&2
    exit 2
  fi
done

TARGETS_FILE_DEFAULT="${REPO_ROOT}/scripts/cloudlab/tmp/cloudlab_ssh_targets.txt"
AUTOBAHN_SRC="${REPO_ROOT}/autobahn"
# baselines: one single-binary replica process. autobahn: a primary and a
# worker process per replica, built from the root workspace.
LAUNCH_KIND="${LAUNCH_KIND:-baselines}"
if [[ "${LAUNCH_KIND}" == "autobahn" ]]; then
  WORKSPACE_REL="autobahn"
  CLIENT_BIN_SRC="benchmark_client"
else
  WORKSPACE_REL="autobahn/baselines"
  CLIENT_BIN_SRC="client"
fi
LEARNING_AGENT_SRC_DIR="${REPO_ROOT}/learning_agent"
APPLIER_LOCAL_PATH="${REPO_ROOT}/scripts/apply_network_failures.py"
GEN_COMMITTEE_LOCAL_PATH="${REPO_ROOT}/scripts/cloudlab/gen_committee.py"
FAILURE_SPEC_DEFAULT="${REPO_ROOT}/${FAILURE_SPEC_DEFAULT_REL}"
LOCAL_TMP_DIR="${REPO_ROOT}/scripts/cloudlab/tmp"
RUNS_ROOT="${LOCAL_TMP_DIR}/runs/${PROTOCOL_NAME}"
RUN_ROOT_REMOTE_NAME="${PROTOCOL_NAME}_cloudlab"
OUTPUT_DIR=""

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)

MODE="normal"
F=1
NODES_SET=0
NODES=4
TARGETS_FILE="${TARGETS_FILE_DEFAULT}"
EXPERIMENT_TIME_S=300
CLIENT_START_DELAY_S=10
START_OFFSET_AFTER_CLIENT_LAUNCH_S=10
DATASET_ENABLED=0
LOG_ENABLED=0
VERBOSE_ENABLED=0
AGENT_MODEL_TYPE="random"
AGENT_MODEL_SET=0

TX_RATE=60000
TX_SIZE=512
REQUEST_TIMEOUT_MS=10000
CLIENT_RETRY_TIMEOUT_MS=1000
# Resolved after parsing: spread unless given explicitly.
CLIENT_TARGET_MODE=""
# Parallel client connections (independent senders) per replica front. One
# framed connection tops out around 10k tx/s, so leader mode needs several.
CLIENT_CONNECTIONS_PER_TARGET=4
MONITOR_INTERVAL_MS=1000
# Consensus/mempool knobs default to the checked-in template
# config/<protocol>/parameters.json; the flags below override individual
# fields and are only forwarded when explicitly given.
PARAMETERS_TEMPLATE_REL="config/${PROTOCOL_NAME}/parameters.json"
MAX_PAYLOAD_SIZE=""
MIN_BLOCK_DELAY_MS=""
FRONT_PORT=9000
MEMPOOL_PORT=9100
CONSENSUS_PORT=9200
AGENT_PORT=15501
INITIAL_TIMEOUT_MS=""
SCHEDULE_FILE_ARG=""
LEARNING_WINDOW_MODE="wall-clock"
LEARNING_FEATURE_DURATION_MS=10000
LEARNING_REPLY_WAIT_MS=2000
LEARNING_WARMUP_DURATION_MS=3000
LEARNING_REWARD_DURATION_MS=5000
FAILURE_SPEC_PATH="${FAILURE_SPEC_DEFAULT}"
FAILURE_SPEC_ENABLED=1

OVERLAY_OCTET_A=10
OVERLAY_OCTET_B=252
OVERLAY_MASK_BITS=24
OVERLAY_HOST_OFFSET=10
ARTIFACT_FETCH_MAX_PARALLEL=4
ARTIFACT_FETCH_MAX_ATTEMPTS=3
ARTIFACT_FETCH_RETRY_DELAY_S=1
LOCAL_RUN_LOCK_DIRS=("")

cleanup_local_run_locks() {
  local lock_dir
  for lock_dir in "${LOCAL_RUN_LOCK_DIRS[@]}"; do
    [[ -z "${lock_dir}" ]] && continue
    rmdir "${lock_dir}" 2>/dev/null || true
  done
}

acquire_local_run_lock() {
  local lock_dir="$1"
  local description="$2"
  local existing
  for existing in "${LOCAL_RUN_LOCK_DIRS[@]}"; do
    [[ -z "${existing}" ]] && continue
    if [[ "${existing}" == "${lock_dir}" ]]; then
      return 0
    fi
  done
  if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "Cannot start ${PROTOCOL_NAME} run: ${description} is already in use." >&2
    echo "Active-run lock: ${lock_dir}" >&2
    return 1
  fi
  LOCAL_RUN_LOCK_DIRS+=("${lock_dir}")
}

acquire_target_lock() {
  local user="$1"
  local host="$2"
  local lock_root="${LOCAL_TMP_DIR}/active-target-locks"
  local lock_key
  mkdir -p "${lock_root}"
  lock_key="$(printf '%s' "${host}" | cksum | awk '{print $1}')"
  acquire_local_run_lock "${lock_root}/target_${lock_key}.lock" "target ${user}@${host}"
}

trap cleanup_local_run_locks EXIT

usage() {
  cat <<EOF
Usage: $0 <private_key_path> [${SUPPORTED_MODES// /|}] [--dataset] [--log] [--output-dir <path>] [--model <name>] [-f <fault_tolerance>] [-t <experiment_seconds>] [targets_file]

Arguments:
  private_key_path  Local SSH private key for CloudLab nodes.
  mode              Optional; one of: ${SUPPORTED_MODES} (default: normal).
  targets_file      Optional path to cloudlab SSH targets file.

Flags:
  --dataset                         Agent modes collect the agent CSV automatically.
  --log                             Download client, server, applier, and agent logs.
  --output-dir <path>               Store artifacts here instead of a timestamped run directory.
  --verbose                         Enable debug logs on server replicas.
  --model <random|pavlov|linear|cmab|cmab_reset|claude|codex|continuous>
                                     Learning-agent model for learning mode (default: random).
  -f <int>                          Fault tolerance f; nodes defaults to 3f+1 (default: 1).
  -t <int>                          Seconds to run after benchmark start (default: 300).

${PROTOCOL_NAME} options:
  --nodes <int>                     Override node count (default: 3f+1).
  --failure-spec <path>             failure_spec.xml for proposal and network delay.
  --no-failure-spec                 Disable proposal delay and network delay appliers.
  --rate <int>                      Total client tx/s across all replicas (default: 1000).
  --tx-size <int>                   Transaction size in bytes, >= 9 (default: 512).
  --request-timeout <ms>            Client per-tx timeout in ms (default: 10000).
  --client-retry-timeout <ms>       Broadcast mode: re-broadcast an unacked tx after this many ms, doubling per retry (default: 1000).
  --client-target-mode <mode>       spread: rate split across all replicas (default); leader: all
                                     transactions go to the current leader, following leader hints;
                                     broadcast: like leader, plus shadow tx headers to every other
                                     replica so followers measure client latency. leader and
                                     broadcast need a stable-leader protocol, which none of the
                                     current Rust protocols is.
  --client-connections <int>        Parallel client connections (independent senders) per replica
                                     front (default: 4). One framed connection tops out around
                                     10k tx/s, so high rates in leader mode need several.
  --max-payload-size <bytes>        Override the mempool payload seal size (default: from config/${PROTOCOL_NAME}/parameters.json).
  --min-block-delay <ms>            Override the mempool/consensus min block delay (default: from config/${PROTOCOL_NAME}/parameters.json).
  --client-start-delay <seconds>    Wait after servers are ready before launching client (default: 10).
  --start-offset <seconds>          Delay from client launch to benchmark/failure start (default: 10).
  --front-port <port>               Client transactions port on each replica IP (default: 9000).

Learning-agent options:
  --agent-port <port>               Learning-agent port inside each replica namespace (default: 15501).
  --initial-timeout <ms>            Override the initial consensus timeout_delay (default: from config/${PROTOCOL_NAME}/parameters.json).
  --schedule <path>                 Schedule CSV for scheduled mode, inside the repo (default: ${SCHEDULE_FILE_REL}).
  --learning-window-mode <mode>     wall-clock only for Rust protocols (default: wall-clock).
  --learning-feature-duration <ms>  Feature window (default: 10000).
  --learning-reply-wait <ms>        Recommendation wait (default: 2000).
  --learning-warmup-duration <ms>   Post-apply warm-up (default: 3000).
  --learning-reward-duration <ms>   Reward window (default: 5000).
EOF
}

mode_enables_learning_agent() {
  case "$1" in
    heuristic|learning|sharing|scheduled) return 0 ;;
    *) return 1 ;;
  esac
}

mode_is_supported() {
  local candidate="$1"
  local mode
  for mode in ${SUPPORTED_MODES}; do
    [[ "${mode}" == "${candidate}" ]] && return 0
  done
  return 1
}

normalize_agent_model_type() {
  local model_lower
  model_lower="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  case "${model_lower}" in
    random|pavlov|linear|cmab|cmab_reset|continuous) printf '%s\n' "${model_lower}" ;;
    claude) printf '%s\n' "claude_v2" ;;
    codex) printf '%s\n' "codex_v1" ;;
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

queue_artifact_download() {
  local node="$1"
  local host="$2"
  local user="$3"
  local remote_copy_path="$4"
  local local_path="$5"
  local artifact_label="$6"
  local artifact_kind="$7"
  local missing_ok="${8:-0}"

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
    local attempt
    for ((attempt = 1; attempt <= ARTIFACT_FETCH_MAX_ATTEMPTS; attempt++)); do
      rm -f "${local_path}"
      if scp "${SSH_OPTS[@]}" "${user}@${host}:${remote_copy_path}" "${local_path}"; then
        echo "  - copied ${artifact_label} from ${node}"
        exit 0
      fi
      if (( attempt < ARTIFACT_FETCH_MAX_ATTEMPTS )); then
        sleep $((ARTIFACT_FETCH_RETRY_DELAY_S * attempt))
      fi
    done

    echo "  - missing or failed to copy ${artifact_label} from ${node}" >&2
    (( missing_ok == 1 )) && exit 0
    exit 1
  ) &

  artifact_fetch_pids+=("$!")
  artifact_fetch_kinds+=("${artifact_kind}")
}

wait_for_server_endpoints_from_node0_or_fail() {
  local node0_host="$1"
  local node0_user="$2"
  local endpoints_file="$3"
  local deadline_epoch_s="$4"
  local last_unready=""

  if [[ ! -s "${endpoints_file}" ]]; then
    echo "No server endpoints found in ${endpoints_file}" >&2
    return 1
  fi

  while true; do
    local all_ready=1
    local unready=""
    while read -r endpoint; do
      [[ -z "${endpoint}" ]] && continue
      local endpoint_ip="${endpoint%%:*}"
      local endpoint_port="${endpoint##*:}"
      if ! ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" \
        "timeout 1 bash -c 'exec 3<>/dev/tcp/${endpoint_ip}/${endpoint_port}'" >/dev/null 2>&1; then
        all_ready=0
        unready+="${endpoint} "
      fi
    done < "${endpoints_file}"

    if (( all_ready == 1 )); then
      return 0
    fi

    last_unready="${unready}"
    if (( $(date +%s) >= deadline_epoch_s )); then
      break
    fi
    sleep 1
  done

  echo "Timed out waiting for ${PROTOCOL_NAME} client ports from ${node0_host}." >&2
  [[ -z "${last_unready}" ]] || echo "Still unreachable: ${last_unready}" >&2
  return 1
}

rustbft_main() {
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
      -f) F="${2:-}"; shift 2 ;;
      -t) EXPERIMENT_TIME_S="${2:-}"; shift 2 ;;
      --dataset) DATASET_ENABLED=1; shift ;;
      --log) LOG_ENABLED=1; shift ;;
      --output-dir)
        if [[ $# -lt 2 || -z "${2}" ]]; then
          echo "Missing value for --output-dir." >&2
          usage >&2
          exit 2
        fi
        OUTPUT_DIR="${2}"
        shift 2
        ;;
      --verbose) VERBOSE_ENABLED=1; shift ;;
      --model) AGENT_MODEL_TYPE="${2:-}"; AGENT_MODEL_SET=1; shift 2 ;;
      --nodes) NODES="${2:-}"; NODES_SET=1; shift 2 ;;
      --failure-spec) FAILURE_SPEC_PATH="${2:-}"; FAILURE_SPEC_ENABLED=1; shift 2 ;;
      --no-failure-spec) FAILURE_SPEC_ENABLED=0; shift ;;
      --rate) TX_RATE="${2:-}"; shift 2 ;;
      --tx-size) TX_SIZE="${2:-}"; shift 2 ;;
      --request-timeout) REQUEST_TIMEOUT_MS="${2:-}"; shift 2 ;;
      --client-retry-timeout) CLIENT_RETRY_TIMEOUT_MS="${2:-}"; shift 2 ;;
      --client-target-mode) CLIENT_TARGET_MODE="${2:-}"; shift 2 ;;
      --client-connections) CLIENT_CONNECTIONS_PER_TARGET="${2:-}"; shift 2 ;;
      --max-payload-size) MAX_PAYLOAD_SIZE="${2:-}"; shift 2 ;;
      --min-block-delay) MIN_BLOCK_DELAY_MS="${2:-}"; shift 2 ;;
      --client-start-delay) CLIENT_START_DELAY_S="${2:-}"; shift 2 ;;
      --start-offset) START_OFFSET_AFTER_CLIENT_LAUNCH_S="${2:-}"; shift 2 ;;
      --front-port) FRONT_PORT="${2:-}"; shift 2 ;;
      --agent-port) AGENT_PORT="${2:-}"; shift 2 ;;
      --initial-timeout) INITIAL_TIMEOUT_MS="${2:-}"; shift 2 ;;
      --schedule) SCHEDULE_FILE_ARG="${2:-}"; shift 2 ;;
      --learning-window-mode) LEARNING_WINDOW_MODE="${2:-}"; shift 2 ;;
      --learning-feature-duration) LEARNING_FEATURE_DURATION_MS="${2:-}"; shift 2 ;;
      --learning-reply-wait) LEARNING_REPLY_WAIT_MS="${2:-}"; shift 2 ;;
      --learning-warmup-duration) LEARNING_WARMUP_DURATION_MS="${2:-}"; shift 2 ;;
      --learning-reward-duration) LEARNING_REWARD_DURATION_MS="${2:-}"; shift 2 ;;
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

  if ! mode_is_supported "${MODE}"; then
    echo "Mode '${MODE}' is not supported by run_${PROTOCOL_NAME}.sh yet (supported: ${SUPPORTED_MODES})." >&2
    exit 2
  fi
  if [[ -n "${SCHEDULE_FILE_ARG}" ]]; then
    if [[ "${MODE}" != "scheduled" ]]; then
      echo "--schedule is only meaningful in scheduled mode (mode is '${MODE}')." >&2
      exit 2
    fi
    # The schedule ships to the node inside the repo tarball, so it must live
    # under the repo root; store it relative to REPO_ROOT like the default.
    schedule_abs="$(cd "$(dirname "${SCHEDULE_FILE_ARG}")" 2>/dev/null && pwd)/$(basename "${SCHEDULE_FILE_ARG}")"
    if [[ ! -f "${schedule_abs}" ]]; then
      echo "Schedule file not found: ${SCHEDULE_FILE_ARG}" >&2
      exit 2
    fi
    if [[ "${schedule_abs}" != "${REPO_ROOT}/"* ]]; then
      echo "Schedule file must be inside the repo (${REPO_ROOT}): ${schedule_abs}" >&2
      exit 2
    fi
    SCHEDULE_FILE_REL="${schedule_abs#${REPO_ROOT}/}"
  fi
  if ! [[ "${F}" =~ ^[0-9]+$ ]] || (( F < 1 )); then
    echo "Invalid -f value: ${F}. It must be an integer >= 1." >&2
    exit 2
  fi
  if (( NODES_SET == 0 )); then
    NODES="$((3 * F + 1))"
  fi
  for value_name in NODES EXPERIMENT_TIME_S TX_RATE TX_SIZE REQUEST_TIMEOUT_MS CLIENT_RETRY_TIMEOUT_MS FRONT_PORT AGENT_PORT; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
      echo "Invalid ${value_name}: ${value}" >&2
      exit 2
    fi
  done
  # Parameter overrides are optional; when given they must be positive
  # integers (empty = take the template's value).
  for value_name in MAX_PAYLOAD_SIZE MIN_BLOCK_DELAY_MS INITIAL_TIMEOUT_MS; do
    value="${!value_name}"
    [[ -z "${value}" ]] && continue
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
      echo "Invalid ${value_name}: ${value}" >&2
      exit 2
    fi
  done
  if [[ ! -f "${REPO_ROOT}/${PARAMETERS_TEMPLATE_REL}" ]]; then
    echo "Parameters template not found: ${REPO_ROOT}/${PARAMETERS_TEMPLATE_REL}" >&2
    exit 1
  fi
  for value_name in CLIENT_START_DELAY_S START_OFFSET_AFTER_CLIENT_LAUNCH_S; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
      echo "Invalid ${value_name}: ${value}" >&2
      exit 2
    fi
  done
  if (( NODES < 4 )); then
    echo "Invalid node count: ${NODES}. BFT requires at least 4 nodes." >&2
    exit 2
  fi
  if (( TX_SIZE < 9 )); then
    echo "Invalid --tx-size ${TX_SIZE}; the ack header needs at least 9 bytes." >&2
    exit 2
  fi
  case "${LEARNING_WINDOW_MODE}" in
    wall-clock) ;;
    *) echo "Invalid --learning-window-mode value: ${LEARNING_WINDOW_MODE}. Rust protocols support wall-clock only." >&2; exit 2 ;;
  esac
  if [[ -z "${CLIENT_TARGET_MODE}" ]]; then
    CLIENT_TARGET_MODE="spread"
  fi
  case "${CLIENT_TARGET_MODE}" in
    spread) ;;
    # The leader-following modes need a stable leader to follow; every
    # current Rust protocol rotates leaders, so there is nothing to allow
    # here until a stable-leader protocol is added.
    leader|broadcast) echo "--client-target-mode ${CLIENT_TARGET_MODE} requires a stable-leader protocol; ${PROTOCOL_NAME} rotates leaders." >&2; exit 2 ;;
    *) echo "Invalid --client-target-mode value: ${CLIENT_TARGET_MODE}. Use spread, leader, or broadcast." >&2; exit 2 ;;
  esac
  if ! [[ "${CLIENT_CONNECTIONS_PER_TARGET}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid --client-connections value: ${CLIENT_CONNECTIONS_PER_TARGET}. Use a positive integer." >&2
    exit 2
  fi
  if ! AGENT_MODEL_TYPE="$(normalize_agent_model_type "${AGENT_MODEL_TYPE}")"; then
    echo "Invalid --model value. Use one of random|pavlov|linear|cmab|cmab_reset|claude|codex|continuous." >&2
    exit 2
  fi
  if (( AGENT_MODEL_SET == 1 )) && [[ "${MODE}" != "learning" && "${MODE}" != "sharing" ]]; then
    echo "--model is only supported with learning or sharing mode." >&2
    exit 2
  fi
  if ! mode_enables_learning_agent "${MODE}"; then
    LEARNING_AGENT_ENABLED=0
  else
    LEARNING_AGENT_ENABLED=1
  fi
  if (( DATASET_ENABLED == 1 )) && (( LEARNING_AGENT_ENABLED == 0 )); then
    echo "--dataset requires an agent mode (${SUPPORTED_MODES/normal /})." >&2
    exit 2
  fi
  if (( LEARNING_AGENT_ENABLED == 1 )); then
    DATASET_ENABLED=1
  fi
  if [[ "${MODE}" == "sharing" ]]; then
    echo "Sharing mode is not implemented for Rust protocols yet." >&2
    exit 2
  fi

  LOCAL_SSH_KEY_PATH="${LOCAL_SSH_KEY_PATH/#\~/$HOME}"
  FAILURE_SPEC_PATH="$(absolute_path "${FAILURE_SPEC_PATH}")"

  if [[ ! -f "${LOCAL_SSH_KEY_PATH}" ]]; then
    echo "Local SSH key not found: ${LOCAL_SSH_KEY_PATH}" >&2
    exit 1
  fi
  SSH_OPTS+=(-i "${LOCAL_SSH_KEY_PATH}" -o IdentitiesOnly=yes)
  if [[ ! -d "${AUTOBAHN_SRC}" ]]; then
    echo "autobahn submodule not found: ${AUTOBAHN_SRC}" >&2
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
  if (( FAILURE_SPEC_ENABLED == 1 )) && [[ ! -f "${FAILURE_SPEC_PATH}" ]]; then
    echo "Failure spec not found: ${FAILURE_SPEC_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${TARGETS_FILE}" ]]; then
    echo "Targets file not found: ${TARGETS_FILE}" >&2
    exit 1
  fi
  # The vendored proto must match the repo's source of truth; drift would
  # silently desynchronize the Rust agent client from learning_agent.
  if ! diff -q "${REPO_ROOT}/proto/agent.proto" "${AUTOBAHN_SRC}/proto/agent.proto" >/dev/null; then
    echo "proto drift: autobahn/proto/agent.proto differs from proto/agent.proto." >&2
    echo "Re-vendor the proto in the autobahn submodule before running." >&2
    exit 1
  fi

  while read -r node host user _; do
    [[ -z "${node:-}" ]] && continue
    [[ "${node:0:1}" == "#" ]] && continue
    if [[ -z "${host:-}" || -z "${user:-}" ]]; then
      continue
    fi
    acquire_target_lock "${user}" "${host}"
  done < "${TARGETS_FILE}"

  if [[ -z "${OUTPUT_DIR}" ]]; then
    mkdir -p "${RUNS_ROOT}"
    while true; do
      RUN_ID="$(date +%Y%m%d_%H%M%S)"
      ARTIFACT_DIR="${RUNS_ROOT}/${RUN_ID}"
      if mkdir "${ARTIFACT_DIR}" 2>/dev/null; then
        break
      fi
      if [[ ! -e "${ARTIFACT_DIR}" ]]; then
        echo "Failed to create run artifact directory: ${ARTIFACT_DIR}" >&2
        exit 1
      fi
      sleep 0.1
    done
  else
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="${OUTPUT_DIR/#\~/$HOME}"
    if [[ "${OUTPUT_DIR}" == /* ]]; then
      ARTIFACT_DIR="${OUTPUT_DIR}"
    else
      ARTIFACT_DIR="${INVOCATION_DIR}/${OUTPUT_DIR}"
    fi
    mkdir -p "${ARTIFACT_DIR}"
  fi
  ARTIFACT_DIR="$(cd "${ARTIFACT_DIR}" && pwd -P)"
  acquire_local_run_lock "${ARTIFACT_DIR}/.active-run.lock" "artifact directory ${ARTIFACT_DIR}"

  LOCAL_TAR="${ARTIFACT_DIR}/${PROTOCOL_NAME}_payload.tar.gz"
  OUTPUT_LOCAL="${ARTIFACT_DIR}/${PROTOCOL_NAME}_output.txt"
  cp "${TARGETS_FILE}" "${ARTIFACT_DIR}/cloudlab_ssh_targets.txt"
  if (( FAILURE_SPEC_ENABLED == 1 )); then
    cp "${FAILURE_SPEC_PATH}" "${ARTIFACT_DIR}/failure_spec.xml"
  fi
  if [[ "${MODE}" == "scheduled" ]]; then
    cp "${REPO_ROOT}/${SCHEDULE_FILE_REL}" "${ARTIFACT_DIR}/$(basename "${SCHEDULE_FILE_REL}")"
  fi
  {
    printf 'protocol=%q\n' "${PROTOCOL_NAME}"
    printf 'run_id=%q\n' "${RUN_ID}"
    printf 'artifact_dir=%q\n' "${ARTIFACT_DIR}"
    printf 'mode=%q\n' "${MODE}"
    printf 'fault_tolerance=%q\n' "${F}"
    printf 'nodes=%q\n' "${NODES}"
    printf 'experiment_time_s=%q\n' "${EXPERIMENT_TIME_S}"
    printf 'tx_rate=%q\n' "${TX_RATE}"
    printf 'tx_size=%q\n' "${TX_SIZE}"
    printf 'request_timeout_ms=%q\n' "${REQUEST_TIMEOUT_MS}"
    printf 'client_retry_timeout_ms=%q\n' "${CLIENT_RETRY_TIMEOUT_MS}"
    printf 'client_target_mode=%q\n' "${CLIENT_TARGET_MODE}"
    printf 'client_connections_per_target=%q\n' "${CLIENT_CONNECTIONS_PER_TARGET}"
    printf 'parameters_template=%q\n' "${PARAMETERS_TEMPLATE_REL}"
    printf 'max_payload_size=%q\n' "${MAX_PAYLOAD_SIZE:-from-template}"
    printf 'min_block_delay_ms=%q\n' "${MIN_BLOCK_DELAY_MS:-from-template}"
    printf 'initial_timeout_ms=%q\n' "${INITIAL_TIMEOUT_MS:-from-template}"
    printf 'agent_model=%q\n' "${AGENT_MODEL_TYPE}"
    printf 'learning_window_mode=%q\n' "${LEARNING_WINDOW_MODE}"
    printf 'dataset_enabled=%q\n' "${DATASET_ENABLED}"
    printf 'log_enabled=%q\n' "${LOG_ENABLED}"
    printf 'failure_spec_enabled=%q\n' "${FAILURE_SPEC_ENABLED}"
    printf 'targets_file=%q\n' "${TARGETS_FILE}"
    printf 'git_commit=%q\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'git_dirty=%q\n' "$([[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no 2>/dev/null)" ]] && echo 1 || echo 0)"
    printf 'autobahn_commit=%q\n' "$(git -C "${AUTOBAHN_SRC}" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'autobahn_dirty=%q\n' "$([[ -n "$(git -C "${AUTOBAHN_SRC}" status --porcelain --untracked-files=no 2>/dev/null)" ]] && echo 1 || echo 0)"
    printf 'command='
    printf '%q ' "$0" "${ORIGINAL_ARGS[@]}"
    printf '\n'
  } > "${ARTIFACT_DIR}/run.env"

  echo "Run artifacts: ${ARTIFACT_DIR}"

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

  # Per-replica overlay IPs, replica order; also the applier hosts config and
  # the client-endpoint readiness list.
  FAILURE_HOSTS_CONFIG_LOCAL="${ARTIFACT_DIR}/${PROTOCOL_NAME}_hosts.failure.config"
  AGENT_HOSTS_CONFIG_LOCAL="${ARTIFACT_DIR}/${PROTOCOL_NAME}_agent_hosts.config"
  CLIENT_TARGETS_LOCAL="${ARTIFACT_DIR}/client_targets.txt"
  : > "${FAILURE_HOSTS_CONFIG_LOCAL}"
  : > "${AGENT_HOSTS_CONFIG_LOCAL}"
  : > "${CLIENT_TARGETS_LOCAL}"
  replica_ips=()
  for ((replica_id = 0; replica_id < NODES; replica_id++)); do
    machine_idx=$((replica_id / replicas_per_server_machine))
    slot=$((replica_id % replicas_per_server_machine))
    ip="${OVERLAY_OCTET_A}.${OVERLAY_OCTET_B}.$((machine_idx + 1)).$((OVERLAY_HOST_OFFSET + slot))"
    replica_ips+=("${ip}")
    printf '%d %s %d\n' "${replica_id}" "${ip}" "${FRONT_PORT}" >> "${FAILURE_HOSTS_CONFIG_LOCAL}"
    printf '%d %s 0 0 %d 0\n' "${replica_id}" "${ip}" "${AGENT_PORT}" >> "${AGENT_HOSTS_CONFIG_LOCAL}"
    printf '%s:%d\n' "${ip}" "${FRONT_PORT}" >> "${CLIENT_TARGETS_LOCAL}"
  done
  replica_ips_csv="$(IFS=,; printf '%s' "${replica_ips[*]}")"
  client_targets_args="$(paste -sd' ' "${CLIENT_TARGETS_LOCAL}")"

  echo "Selected mode: ${MODE}"
  echo "${PROTOCOL_NAME} nodes: ${NODES} (f=${F}, replicas_per_server_machine=${replicas_per_server_machine})"
  echo "Client rate: ${TX_RATE} tx/s; tx size: ${TX_SIZE} B; request timeout: ${REQUEST_TIMEOUT_MS} ms; retry timeout: ${CLIENT_RETRY_TIMEOUT_MS} ms"
  echo "Learning-agent orchestration: ${LEARNING_AGENT_ENABLED} (dataset=${DATASET_ENABLED}, log=${LOG_ENABLED})"

  echo "Creating ${PROTOCOL_NAME} CloudLab payload tarball..."
  rm -f "${LOCAL_TAR}"
  COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 tar -czf "${LOCAL_TAR}" \
    --exclude='autobahn/.git' \
    --exclude='autobahn/target' \
    --exclude='autobahn/baselines/target' \
    -C "${REPO_ROOT}" autobahn config learning_agent proto scripts/apply_network_failures.py scripts/cloudlab/gen_committee.py

  echo "Copying payload to CloudLab nodes and building the Rust workspace..."
  pids=()
  nodes=()
  for i in "${!targets_nodes[@]}"; do
    node="${targets_nodes[$i]}"
    host="${targets_hosts[$i]}"
    user="${targets_users[$i]}"
    (
      remote_tar="/tmp/${PROTOCOL_NAME}_payload_$$.tar.gz"
      remote_failure_spec=""
      scp "${SSH_OPTS[@]}" "${LOCAL_TAR}" "${user}@${host}:${remote_tar}"
      scp "${SSH_OPTS[@]}" "${FAILURE_HOSTS_CONFIG_LOCAL}" "${user}@${host}:/tmp/${PROTOCOL_NAME}_hosts_failure_$$.config"
      scp "${SSH_OPTS[@]}" "${AGENT_HOSTS_CONFIG_LOCAL}" "${user}@${host}:/tmp/${PROTOCOL_NAME}_agent_hosts_$$.config"
      if (( FAILURE_SPEC_ENABLED == 1 )); then
        remote_failure_spec="/tmp/${PROTOCOL_NAME}_failure_spec_$$.xml"
        scp "${SSH_OPTS[@]}" "${FAILURE_SPEC_PATH}" "${user}@${host}:${remote_failure_spec}"
      fi
      ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- \
        "${remote_tar}" \
        "/tmp/${PROTOCOL_NAME}_hosts_failure_$$.config" \
        "${remote_failure_spec}" \
        "/tmp/${PROTOCOL_NAME}_agent_hosts_$$.config" \
        "${RUN_ROOT_REMOTE_NAME}" \
        "${WORKSPACE_REL}" \
        "${CLIENT_BIN_SRC}" <<'REMOTE'
set -euo pipefail

tar_path="$1"
failure_hosts_path="$2"
failure_spec_path="$3"
agent_hosts_path="$4"
run_root_name="$5"
workspace_rel="$6"
client_bin_src="$7"
run_root="$HOME/${run_root_name}"

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

# Keep the cargo target dir across runs to avoid full rebuilds.
for stale in "${run_root}/autobahn" "${run_root}/config" "${run_root}/learning_agent" "${run_root}/proto" "${run_root}/scripts" "${run_root}/logs" "${run_root}/db" "${run_root}/netcfg"; do
  remove_path_with_sudo_fallback "${stale}"
done
mkdir -p "${run_root}"
tar --warning=no-unknown-keyword -xzf "${tar_path}" -C "${run_root}"
mkdir -p "${run_root}/bin" "${run_root}/logs" "${run_root}/db" "${run_root}/netcfg"
cp "${failure_hosts_path}" "${run_root}/hosts.failure.config"
cp "${agent_hosts_path}" "${run_root}/agent_hosts.config"
if [[ -n "${failure_spec_path}" ]]; then
  cp "${failure_spec_path}" "${run_root}/failure_spec.xml"
fi
chmod +x "${run_root}/scripts/apply_network_failures.py" "${run_root}/scripts/cloudlab/gen_committee.py"
rm -f "${tar_path}" "${failure_hosts_path}" "${agent_hosts_path}"
if [[ -n "${failure_spec_path}" ]]; then
  rm -f "${failure_spec_path}"
fi

export PATH="$HOME/.cargo/bin:${PATH}"
if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found on $(hostname); run scripts/cloudlab/setup.sh first." >&2
  exit 1
fi
if ! command -v protoc >/dev/null 2>&1; then
  echo "protoc not found on $(hostname); run scripts/cloudlab/setup.sh first." >&2
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

echo "Using Rust toolchain: $(cargo --version)"
cd "${run_root}/${workspace_rel}"
CARGO_TARGET_DIR="${run_root}/target" cargo build --release --features node/benchmark --bin node --bin "${client_bin_src}"
cp "${run_root}/target/release/node" "${run_root}/bin/node"
cp "${run_root}/target/release/${client_bin_src}" "${run_root}/bin/client"
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

  echo "Generating keys and committee on node0..."
  # Only explicitly-given knobs are forwarded as overrides; everything else
  # comes from the checked-in template (shipped inside the payload tarball).
  parameter_override_args=()
  if [[ -n "${INITIAL_TIMEOUT_MS}" ]]; then parameter_override_args+=(--timeout-delay-ms "${INITIAL_TIMEOUT_MS}"); fi
  if [[ -n "${MAX_PAYLOAD_SIZE}" ]]; then parameter_override_args+=(--max-payload-size "${MAX_PAYLOAD_SIZE}"); fi
  if [[ -n "${MIN_BLOCK_DELAY_MS}" ]]; then parameter_override_args+=(--min-block-delay-ms "${MIN_BLOCK_DELAY_MS}"); fi
  if ! ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" "bash -s" -- \
    "${RUN_ROOT_REMOTE_NAME}" \
    "${replica_ips_csv}" \
    "${FRONT_PORT}" \
    "${MEMPOOL_PORT}" \
    "${CONSENSUS_PORT}" \
    "${LAUNCH_KIND}" \
    "${PARAMETERS_TEMPLATE_REL}" \
    ${parameter_override_args[@]+"${parameter_override_args[@]}"} <<'REMOTE'
set -euo pipefail
run_root="$HOME/$1"
ips="$2"
front_port="$3"
mempool_port="$4"
consensus_port="$5"
launch_kind="$6"
template_rel="$7"
shift 7
python3 "${run_root}/scripts/cloudlab/gen_committee.py" \
  --workspace "${launch_kind}" \
  --node-bin "${run_root}/bin/node" \
  --ips "${ips}" \
  --out-dir "${run_root}/netcfg" \
  --front-port "${front_port}" \
  --mempool-port "${mempool_port}" \
  --consensus-port "${consensus_port}" \
  --parameters-template "${run_root}/${template_rel}" \
  "$@"
REMOTE
  then
    echo "Committee generation failed on node0." >&2
    exit 1
  fi

  echo "Fetching committee artifacts from node0..."
  scp "${SSH_OPTS[@]}" "${node0_user}@${node0_host}:${RUN_ROOT_REMOTE_NAME}/netcfg/committee.json" "${ARTIFACT_DIR}/committee.json"
  scp "${SSH_OPTS[@]}" "${node0_user}@${node0_host}:${RUN_ROOT_REMOTE_NAME}/netcfg/parameters.json" "${ARTIFACT_DIR}/parameters.json"

  if (( machine_count > 1 )); then
    echo "Distributing committee, parameters, and keys to server machines..."
    remote_netcfg_tar="/tmp/${PROTOCOL_NAME}_netcfg_$$.tar.gz"
    local_netcfg_tar="${ARTIFACT_DIR}/netcfg.tar.gz"
    ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" \
      "tar -czf ${remote_netcfg_tar} -C \$HOME/${RUN_ROOT_REMOTE_NAME} netcfg"
    scp "${SSH_OPTS[@]}" "${node0_user}@${node0_host}:${remote_netcfg_tar}" "${local_netcfg_tar}"
    ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" "rm -f ${remote_netcfg_tar}"
    dist_failed=0
    for i in "${!server_nodes[@]}"; do
      node="${server_nodes[$i]}"
      host="${server_hosts[$i]}"
      user="${server_users[$i]}"
      [[ "${node}" == "node0" ]] && continue
      if scp "${SSH_OPTS[@]}" "${local_netcfg_tar}" "${user}@${host}:/tmp/${PROTOCOL_NAME}_netcfg_$$.tar.gz" \
        && ssh "${SSH_OPTS[@]}" "${user}@${host}" \
          "tar -xzf /tmp/${PROTOCOL_NAME}_netcfg_$$.tar.gz -C \$HOME/${RUN_ROOT_REMOTE_NAME} && rm -f /tmp/${PROTOCOL_NAME}_netcfg_$$.tar.gz"; then
        echo "  - ${node}: committee distributed"
      else
        echo "  - ${node}: committee distribution failed" >&2
        dist_failed=1
      fi
    done
    rm -f "${local_netcfg_tar}"
    if (( dist_failed != 0 )); then
      exit 1
    fi
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

  echo "Setting up ${PROTOCOL_NAME} per-replica network namespaces..."
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
      "${BRIDGE_NAME}" \
      "${OVERLAY_HOST_OFFSET}" \
      "${NS_PREFIX}" \
      "${HOST_IF_PREFIX}" \
      "${NS_IF_PREFIX}" \
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
ns_prefix="$9"
host_if_prefix="${10}"
ns_if_prefix="${11}"
shift 11
server_gateway_ips=("$@")

bridge_ip="${overlay_a}.${overlay_b}.$((machine_idx + 1)).1"
local_subnet="${overlay_a}.${overlay_b}.$((machine_idx + 1)).0/${overlay_mask}"

existing_ns="$(ip netns list | awk -v p="^${ns_prefix}[0-9]+" '$1 ~ p {print $1}')"
if [[ -n "${existing_ns}" ]]; then
  printf '%s\n' "${existing_ns}" | while read -r ns_name; do
    [[ -z "${ns_name}" ]] && continue
    sudo -n ip netns del "${ns_name}" >/dev/null 2>&1 || true
  done
fi
stale_veths="$(ip -o link show | awk -F': ' '{print $2}' | sed 's/@.*//' | awk -v p="^${host_if_prefix}[0-9]+$" '$0 ~ p')"
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
  ns_name="${ns_prefix}${replica_id}"
  host_if="${host_if_prefix}${replica_id}"
  ns_if="${ns_if_prefix}${replica_id}"
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
  # TCP tuning inside the replica namespace (namespace-scoped, host untouched).
  # With netem delays of 100-200 ms one-way, the default 4 MB send buffer caps
  # a single stream near 80 Mbit/s, below what batch dissemination needs;
  # 64 MB buffers and no slow start after idle lift that to several hundred
  # Mbit/s. Measured on amd121 (d6515) with iperf3 at 200 ms one-way:
  # 76 Mbit/s default vs 739 Mbit/s tuned.
  sudo -n ip netns exec "${ns_name}" sysctl -q -w \
    net.ipv4.tcp_rmem="4096 131072 67108864" \
    net.ipv4.tcp_wmem="4096 16384 67108864" \
    net.ipv4.tcp_slow_start_after_idle=0
done

for ((peer_idx = 0; peer_idx < server_machine_count; peer_idx++)); do
  if (( peer_idx == machine_idx )); then
    continue
  fi
  subnet="${overlay_a}.${overlay_b}.$((peer_idx + 1)).0/${overlay_mask}"
  gw_ip="${server_gateway_ips[$peer_idx]}"
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
    echo "Starting ${PROTOCOL_NAME} network failure appliers..."
    applier_start_failed=0
    for i in "${!server_nodes[@]}"; do
      node="${server_nodes[$i]}"
      host="${server_hosts[$i]}"
      user="${server_users[$i]}"
      machine_idx="${i}"
      if ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- \
        "${replicas_per_server_machine}" "${machine_idx}" "${START_UNIX_MS}" \
        "${RUN_ROOT_REMOTE_NAME}" "${NS_PREFIX}" "${NS_IF_PREFIX}" "${PROTOCOL_NAME}" <<'REMOTE'
set -euo pipefail

replicas_per_machine="$1"
machine_idx="$2"
start_unix_ms="$3"
run_root="$HOME/$4"
ns_prefix="$5"
ns_if_prefix="$6"
protocol="$7"

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  ns_name="${ns_prefix}${replica_id}"
  interface_name="${ns_if_prefix}${replica_id}"
  session_name="${protocol}_applier_${replica_id}"
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  tmux new-session -d -s "${session_name}" \
    "cd \"${run_root}\" && python3 \"${run_root}/scripts/apply_network_failures.py\" --failure-spec \"${run_root}/failure_spec.xml\" --hosts-config \"${run_root}/hosts.failure.config\" --replica-id ${replica_id} --netns ${ns_name} --interface ${interface_name} --start-unix-ms ${start_unix_ms} > \"${run_root}/logs/applier_${replica_id}.log\" 2>&1"
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
    echo "Starting ${PROTOCOL_NAME} learning agents..."
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
          "${AGENT_PORT}" \
          "${RUN_ROOT_REMOTE_NAME}" \
          "${NS_PREFIX}" \
          "${PROTOCOL_NAME}" \
          "${AGENT_PROTOCOL}" \
          "${SCHEDULE_FILE_REL}" \
          "${FAILURE_SPEC_ENABLED}" <<'REMOTE'
set -euo pipefail

replicas_per_machine="$1"
machine_idx="$2"
start_unix_ms="$3"
mode="$4"
dataset_enabled="$5"
model_type="$6"
agent_port="$7"
run_root="$HOME/$8"
ns_prefix="$9"
protocol="${10}"
agent_protocol="${11}"
schedule_file_rel="${12}"
failure_spec_enabled="${13}"
learning_agent_python="${HOME}/.venvs/adaptive-timer-learning-agent/bin/python"

if [[ ! -x "${learning_agent_python}" ]]; then
  echo "Learning-agent virtual environment not found on $(hostname): ${learning_agent_python}" >&2
  echo "Run scripts/cloudlab/setup.sh for this target file first." >&2
  exit 1
fi

agent_model_args=""
if [[ "${mode}" == "learning" ]]; then
  agent_model_args="--model-type ${model_type}"
elif [[ "${mode}" == "scheduled" ]]; then
  agent_model_args="--model-type scheduled --schedule-file ${schedule_file_rel}"
fi

failure_args=""
if [[ "${failure_spec_enabled}" == "1" ]]; then
  failure_args="--failure-spec ${run_root}/failure_spec.xml --injection-start-unix-ms ${start_unix_ms}"
fi

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  ns_name="${ns_prefix}${replica_id}"
  session_name="${protocol}_agent_${replica_id}"
  agent_log="${run_root}/logs/agent_${replica_id}.log"
  dataset_args=""
  if [[ "${dataset_enabled}" == "1" ]]; then
    rm -f "${run_root}/agent${replica_id}.csv"
    dataset_args="--dataset-csv ${run_root}/agent${replica_id}.csv"
  fi
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  tmux new-session -d -s "${session_name}" \
    "sudo -n ip netns exec ${ns_name} bash -lc 'cd \"${run_root}\" && exec \"${learning_agent_python}\" learning_agent/main.py --node-id ${replica_id} --protocol ${agent_protocol} --host 127.0.0.1 --port ${agent_port} --signing false ${agent_model_args} ${dataset_args} ${failure_args}' > \"${agent_log}\" 2>&1"
done

sleep 2
for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  session_name="${protocol}_agent_${replica_id}"
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
      echo "Learning-agent start failed on one or more nodes." >&2
      exit 1
    fi
  fi

  echo "Starting ${PROTOCOL_NAME} server replicas..."
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
        "${RUN_ROOT_REMOTE_NAME}" \
        "${NS_PREFIX}" \
        "${PROTOCOL_NAME}" \
        "${VERBOSE_ENABLED}" \
        "${START_UNIX_MS}" \
        "${FAILURE_SPEC_ENABLED}" \
        "${LEARNING_AGENT_ENABLED}" \
        "${AGENT_PORT}" \
        "${LEARNING_FEATURE_DURATION_MS}" \
        "${LEARNING_REPLY_WAIT_MS}" \
        "${LEARNING_WARMUP_DURATION_MS}" \
        "${LEARNING_REWARD_DURATION_MS}" \
        "${LAUNCH_KIND}" <<'REMOTE'
set -euo pipefail

replicas_per_machine="$1"
machine_idx="$2"
run_root="$HOME/$3"
ns_prefix="$4"
protocol="$5"
verbose_enabled="$6"
start_unix_ms="$7"
failure_enabled="$8"
learning_enabled="$9"
agent_port="${10}"
learning_feature_ms="${11}"
learning_reply_wait_ms="${12}"
learning_warmup_ms="${13}"
learning_reward_ms="${14}"
launch_kind="${15}"

verbosity="-vv"
if [[ "${verbose_enabled}" == "1" ]]; then
  verbosity="-vvv"
fi

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  ns_name="${ns_prefix}${replica_id}"
  session_name="${protocol}_server_${replica_id}"
  server_log="${run_root}/logs/server_${replica_id}.log"
  store_dir="${run_root}/db/replica_${replica_id}"
  rm -rf "${store_dir}" "${store_dir}_primary" "${store_dir}_worker"
  base_args=(
    "${run_root}/bin/node"
    "${verbosity}"
    "run"
    "--keys" "${run_root}/netcfg/node_${replica_id}.json"
    "--committee" "${run_root}/netcfg/committee.json"
    "--parameters" "${run_root}/netcfg/parameters.json"
  )
  adaptive_args=(
    "--replica-id" "${replica_id}"
    "--replica-map" "${run_root}/netcfg/replica_map.json"
  )
  if [[ "${failure_enabled}" == "1" ]]; then
    adaptive_args+=("--failure-spec" "${run_root}/failure_spec.xml" "--failure-start-unix-ms" "${start_unix_ms}")
  fi
  if [[ "${learning_enabled}" == "1" ]]; then
    adaptive_args+=(
      "--learning"
      "--agent-target" "127.0.0.1:${agent_port}"
      "--learning-feature-duration" "${learning_feature_ms}"
      "--learning-reply-wait" "${learning_reply_wait_ms}"
      "--learning-warmup-duration" "${learning_warmup_ms}"
      "--learning-reward-duration" "${learning_reward_ms}"
    )
  fi
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  : > "${server_log}"
  if [[ "${launch_kind}" == "autobahn" ]]; then
    # One primary and one worker process per replica, in the same namespace.
    args=("${base_args[@]}" "--store" "${store_dir}_primary" "${adaptive_args[@]}" "primary")
    printf -v server_cmd '%q ' "${args[@]}"
    tmux new-session -d -s "${session_name}" \
      "sudo -n ip netns exec ${ns_name} bash -lc 'exec ${server_cmd}' > \"${server_log}\" 2>&1"

    worker_session="${protocol}_worker_${replica_id}"
    worker_log="${run_root}/logs/worker_${replica_id}.log"
    args=("${base_args[@]}" "--store" "${store_dir}_worker" "worker" "--id" "0")
    printf -v worker_cmd '%q ' "${args[@]}"
    tmux kill-session -t "${worker_session}" >/dev/null 2>&1 || true
    : > "${worker_log}"
    tmux new-session -d -s "${worker_session}" \
      "sudo -n ip netns exec ${ns_name} bash -lc 'exec ${worker_cmd}' > \"${worker_log}\" 2>&1"
  else
    args=("${base_args[@]}" "--store" "${store_dir}" "--protocol" "${protocol}" "${adaptive_args[@]}")
    printf -v server_cmd '%q ' "${args[@]}"
    tmux new-session -d -s "${session_name}" \
      "sudo -n ip netns exec ${ns_name} bash -lc 'exec ${server_cmd}' > \"${server_log}\" 2>&1"
  fi
done
sleep 2
for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  sessions_to_check=("${protocol}_server_${replica_id}")
  if [[ "${launch_kind}" == "autobahn" ]]; then
    sessions_to_check+=("${protocol}_worker_${replica_id}")
  fi
  for session_name in "${sessions_to_check[@]}"; do
    if ! tmux has-session -t "${session_name}" 2>/dev/null; then
      echo "Session exited early: ${session_name}" >&2
      tail -n 100 "${run_root}/logs/server_${replica_id}.log" >&2 || true
      tail -n 100 "${run_root}/logs/worker_${replica_id}.log" >&2 || true
      exit 1
    fi
  done
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
    echo "Server start failed on one or more nodes." >&2
    exit 1
  fi

  client_delay_deadline="$(( $(date +%s) + CLIENT_START_DELAY_S ))"
  if ! wait_for_server_endpoints_from_node0_or_fail "${node0_host}" "${node0_user}" "${CLIENT_TARGETS_LOCAL}" "${client_delay_deadline}"; then
    exit 1
  fi
  remaining_client_delay="$((client_delay_deadline - $(date +%s)))"
  if (( remaining_client_delay > 0 )); then
    echo "Waiting ${remaining_client_delay}s before starting ${PROTOCOL_NAME} client..."
    sleep "${remaining_client_delay}"
  fi

  echo "Starting ${PROTOCOL_NAME} client on node0..."
  ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" "bash -s" -- \
    "${START_UNIX_MS}" \
    "${TX_RATE}" \
    "${TX_SIZE}" \
    "${REQUEST_TIMEOUT_MS}" \
    "${MONITOR_INTERVAL_MS}" \
    "${RUN_ROOT_REMOTE_NAME}" \
    "${PROTOCOL_NAME}" \
    "${CLIENT_TARGET_MODE}" \
    "${CLIENT_CONNECTIONS_PER_TARGET}" \
    "${CLIENT_RETRY_TIMEOUT_MS}" \
    ${client_targets_args} <<'REMOTE'
set -euo pipefail

start_unix_ms="$1"
tx_rate="$2"
tx_size="$3"
request_timeout_ms="$4"
monitor_interval_ms="$5"
run_root="$HOME/$6"
protocol="$7"
client_target_mode="$8"
client_connections_per_target="$9"
client_retry_timeout_ms="${10}"
shift 10
targets=("$@")

args=(
  "${run_root}/bin/client"
  "--targets" "${targets[@]}"
  "--rate" "${tx_rate}"
  "--size" "${tx_size}"
  "--request-timeout" "${request_timeout_ms}"
  "--monitor-interval" "${monitor_interval_ms}"
  "--start-unix-ms" "${start_unix_ms}"
  "--target-mode" "${client_target_mode}"
  "--connections-per-target" "${client_connections_per_target}"
  "--retry-timeout" "${client_retry_timeout_ms}"
)
printf -v client_cmd '%q ' "${args[@]}"
tmux kill-session -t "${protocol}_client" >/dev/null 2>&1 || true
: > "${run_root}/logs/client.log"
tmux new-session -d -s "${protocol}_client" \
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

  echo "Stopping ${PROTOCOL_NAME} tmux sessions..."
  stop_failed=0
  for i in "${!targets_nodes[@]}"; do
    node="${targets_nodes[$i]}"
    host="${targets_hosts[$i]}"
    user="${targets_users[$i]}"
    if ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- "${PROTOCOL_NAME}" <<'REMOTE'
set -euo pipefail
protocol="$1"
if ! command -v tmux >/dev/null 2>&1; then
  exit 0
fi
sessions="$(tmux ls -F '#S' 2>/dev/null || true)"
[[ -z "${sessions}" ]] && exit 0
printf '%s\n' "${sessions}" | while read -r session_name; do
  [[ -z "${session_name}" ]] && continue
  case "${session_name}" in
    "${protocol}_server_"*|"${protocol}_worker_"*|"${protocol}_applier_"*|"${protocol}_agent_"*|"${protocol}_client")
      tmux send-keys -t "${session_name}" C-c >/dev/null 2>&1 || true
      ;;
  esac
done
sleep 3
printf '%s\n' "${sessions}" | while read -r session_name; do
  [[ -z "${session_name}" ]] && continue
  case "${session_name}" in
    "${protocol}_server_"*|"${protocol}_worker_"*|"${protocol}_applier_"*|"${protocol}_agent_"*|"${protocol}_client")
      tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
      ;;
  esac
done
REMOTE
    then
      echo "  - ${node}: stopped"
    else
      echo "  - ${node}: failed to stop cleanly" >&2
      stop_failed=1
    fi
  done

  echo "Tearing down ${PROTOCOL_NAME} namespace setup..."
  teardown_failed=0
  for i in "${!server_nodes[@]}"; do
    node="${server_nodes[$i]}"
    host="${server_hosts[$i]}"
    user="${server_users[$i]}"
    machine_idx="${i}"
    if ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- \
      "${machine_idx}" "${server_machine_count}" "${OVERLAY_OCTET_A}" "${OVERLAY_OCTET_B}" "${OVERLAY_MASK_BITS}" "${BRIDGE_NAME}" "${NS_PREFIX}" <<'REMOTE'
set -euo pipefail
machine_idx="$1"
server_machine_count="$2"
overlay_a="$3"
overlay_b="$4"
overlay_mask="$5"
bridge_name="$6"
ns_prefix="$7"
for ((peer_idx = 0; peer_idx < server_machine_count; peer_idx++)); do
  (( peer_idx == machine_idx )) && continue
  subnet="${overlay_a}.${overlay_b}.$((peer_idx + 1)).0/${overlay_mask}"
  sudo -n ip route del "${subnet}" >/dev/null 2>&1 || true
done
existing_ns="$(ip netns list | awk -v p="^${ns_prefix}[0-9]+" '$1 ~ p {print $1}')"
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

  CLIENT_LOG_LOCAL="${ARTIFACT_DIR}/client.log"
  echo "Fetching ${PROTOCOL_NAME} client log..."
  queue_artifact_download "node0" "${node0_host}" "${node0_user}" \
    "${RUN_ROOT_REMOTE_NAME}/logs/client.log" \
    "${CLIENT_LOG_LOCAL}" \
    "${PROTOCOL_NAME} client.log" \
    "output"

  if (( LOG_ENABLED == 1 )); then
    LOGS_LOCAL_DIR="${ARTIFACT_DIR}/logs"
    rm -rf "${LOGS_LOCAL_DIR}"
    mkdir -p "${LOGS_LOCAL_DIR}/servers" "${LOGS_LOCAL_DIR}/appliers" "${LOGS_LOCAL_DIR}/agents"
    for i in "${!server_nodes[@]}"; do
      node="${server_nodes[$i]}"
      host="${server_hosts[$i]}"
      user="${server_users[$i]}"
      machine_idx="${i}"
      for ((n = 0; n < replicas_per_server_machine; n++)); do
        replica_id=$((machine_idx * replicas_per_server_machine + n))
        queue_artifact_download "${node}" "${host}" "${user}" \
          "${RUN_ROOT_REMOTE_NAME}/logs/server_${replica_id}.log" \
          "${LOGS_LOCAL_DIR}/servers/server_${replica_id}.log" \
          "server_${replica_id}.log" \
          "log"
        if [[ "${LAUNCH_KIND}" == "autobahn" ]]; then
          queue_artifact_download "${node}" "${host}" "${user}" \
            "${RUN_ROOT_REMOTE_NAME}/logs/worker_${replica_id}.log" \
            "${LOGS_LOCAL_DIR}/servers/worker_${replica_id}.log" \
            "worker_${replica_id}.log" \
            "log" \
            1
        fi
        if (( FAILURE_SPEC_ENABLED == 1 )); then
          queue_artifact_download "${node}" "${host}" "${user}" \
            "${RUN_ROOT_REMOTE_NAME}/logs/applier_${replica_id}.log" \
            "${LOGS_LOCAL_DIR}/appliers/applier_${replica_id}.log" \
            "applier_${replica_id}.log" \
            "log" \
            1
        fi
        if (( LEARNING_AGENT_ENABLED == 1 )); then
          queue_artifact_download "${node}" "${host}" "${user}" \
            "${RUN_ROOT_REMOTE_NAME}/logs/agent_${replica_id}.log" \
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
          "${RUN_ROOT_REMOTE_NAME}/agent${replica_id}.csv" \
          "${ARTIFACT_DIR}/agent${replica_id}.csv" \
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
    echo "Failed to fetch ${PROTOCOL_NAME} client log." >&2
    exit 1
  fi
  write_filtered_output "${CLIENT_LOG_LOCAL}" "${OUTPUT_LOCAL}"
  echo "Filtered ${PROTOCOL_NAME} output saved to ${OUTPUT_LOCAL}"

  if (( log_fetch_failed != 0 )); then
    echo "Some ${PROTOCOL_NAME} logs could not be fetched." >&2
  fi
  if (( dataset_fetch_failed != 0 )); then
    echo "Some ${PROTOCOL_NAME} dataset files could not be fetched." >&2
  fi
  if (( LEARNING_AGENT_ENABLED == 1 )) && (( DATASET_ENABLED == 1 )); then
    saved_agent_datasets=()
    for ((replica_id = 0; replica_id < NODES; replica_id++)); do
      agent_dataset_local="${ARTIFACT_DIR}/agent${replica_id}.csv"
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
  echo "Run artifacts saved to ${ARTIFACT_DIR}."
}

rustbft_main "$@"
