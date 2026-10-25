#!/usr/bin/env bash
# Shared CloudLab harness for the CppBedrock protocols (CppBedrock submodule).
# Not executable on its own: a thin per-protocol runner (run_pbft.sh,
# run_sbft.sh) sets the protocol variables below and then sources this file,
# which defines and finally invokes bedrock_main "$@".
#
# The phase structure, file formats, and conventions deliberately mirror
# scripts/cloudlab/lib_rustbft.sh and run_smartbft.sh so artifacts stay
# uniform across protocols: targets file (node host user; node0 = client
# machine), mkdir locks, timestamped artifact dir with run.env, payload
# tarball + remote cmake build, per-replica network namespaces on the
# 10.252.x.y overlay, one shared START_UNIX_MS, tmux sessions, scp
# collection, Monitor-line filtering.
#
# Required variables from the sourcing script:
#   PROTOCOL_NAME               e.g. pbft (artifact/session naming, agent protocol default)
#   BEDROCK_PROTOCOL_CONFIG_REL CppBedrock protocol YAML, relative to repo root
#   SUPPORTED_MODES             space-separated subset of: normal learning sharing scheduled
#   AGENT_PROTOCOL              --protocol value for learning_agent/main.py
#   SCHEDULE_FILE_REL           schedule CSV for scheduled mode, relative to repo root
#   FAILURE_SPEC_DEFAULT_REL    default failure spec, relative to repo root
#   NS_PREFIX                   netns name prefix, e.g. pb-r  -> pb-r<replica_id>
#   HOST_IF_PREFIX              host veth prefix (<=12 chars incl. id), e.g. pbh
#   NS_IF_PREFIX                namespace veth prefix, e.g. pbn
#   BRIDGE_NAME                 e.g. br-pbft

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INVOCATION_DIR="$(pwd -P)"
ORIGINAL_ARGS=("$@")

for required_var in PROTOCOL_NAME BEDROCK_PROTOCOL_CONFIG_REL SUPPORTED_MODES AGENT_PROTOCOL SCHEDULE_FILE_REL FAILURE_SPEC_DEFAULT_REL NS_PREFIX HOST_IF_PREFIX NS_IF_PREFIX BRIDGE_NAME; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "lib_bedrock.sh: required variable ${required_var} is not set." >&2
    exit 2
  fi
done

TARGETS_FILE_DEFAULT="${REPO_ROOT}/scripts/cloudlab/tmp/cloudlab_ssh_targets.txt"
BEDROCK_SRC="${REPO_ROOT}/CppBedrock"
LEARNING_AGENT_SRC_DIR="${REPO_ROOT}/learning_agent"
APPLIER_LOCAL_PATH="${REPO_ROOT}/scripts/apply_network_failures.py"
FAILURE_SPEC_DEFAULT="${REPO_ROOT}/${FAILURE_SPEC_DEFAULT_REL}"
PROTOCOL_CONFIG_LOCAL="${REPO_ROOT}/${BEDROCK_PROTOCOL_CONFIG_REL}"
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
LOG_ENABLED=0
VERBOSE_ENABLED=0
AGENT_MODEL_TYPE="discrete_random"
AGENT_MODEL_SET=0
REWARD_DESIGN="reward_tps_latency"
LATENCY_FLOOR_MULTIPLIER="0"
# Prior file the model loads, repo-relative after validation; "" (omitted) is untrained.
PRIOR_PATH=""
# PBFT discrete models: rungs replacing the 250,550,1200 ladder; "" keeps it.
TIMEOUT_LADDER=""
# continuous_heuristic: k replacing the protocol's own; "" keeps it.
HEURISTIC_K=""
# The latency floor's window in committing windows (median over them); "" keeps 1 (PBFT study E6).
LATENCY_FLOOR_WINDOW=""

TX_RATE="${TX_RATE:-40000}"
REQUEST_TIMEOUT_MS=10000
# Serialized bytes of every client request (0: the client's unpadded transfer request).
REQUEST_SIZE=512
# Parallel client streams (independent sender lanes) per replica.
CLIENT_CONNECTIONS_PER_TARGET=4
CLIENT_SIGN="false"
PROPOSAL_INTERVAL_MS=100
BATCH_MAX_REQUESTS=8192
# 4.5 MiB: with 512 B requests the 8192-request cap binds first (up to ~575 B per request).
BATCH_MAX_BYTES=4718592
# Batches the leader may keep unexecuted, and so the depth of the pipeline.
# A protocol's failure spec and oracle schedule are calibrated at one depth -
# 16 stalls execution for seconds at a time at 150 ms one way, which moves the
# usable timeout ladder - so each runner pins the depth its calibration assumes.
MAX_INFLIGHT_BATCHES="${MAX_INFLIGHT_BATCHES:-16}"
MAX_PENDING_REQUESTS=32768
# 32 MiB: with 512 B requests the 32,768-request cap binds first (up to 1 KiB per request).
MAX_PENDING_BYTES=33554432
MONITOR_INTERVAL_MS=1000
REPLICA_PORT=15000
AGENT_PORT=15501
# Sharing mode: each replica namespace also runs a SmartBFT report-chain node,
# which orders the agents' report batches so every agent decides from the
# same data (as in run_smartbft.sh and run_tendermint.sh).
SHARING_REPORT_PORT=15601
SHARING_CONSENSUS_PORT=15701
SHARING_COLLECT_WAIT_MS=100
INITIAL_TIMEOUT_MS=""
INITIAL_SLOW_PATH_TIMEOUT_MS=""
LEARNING_FEATURE_DURATION_MS=7000
LEARNING_REPLY_WAIT_MS=2000
LEARNING_WARMUP_DURATION_MS=4000
LEARNING_REWARD_DURATION_MS=7000
FAILURE_SPEC_PATH="${FAILURE_SPEC_DEFAULT}"
FAILURE_SPEC_ENABLED=1

OVERLAY_OCTET_A=10
OVERLAY_OCTET_B=252
OVERLAY_MASK_BITS=24
OVERLAY_HOST_OFFSET=10
ARTIFACT_FETCH_MAX_PARALLEL=4
# sshd throttles the burst of new connections at the end of a run (MaxStartups: "kex_exchange_identification:
# Connection reset"); three tries 1-2 s apart lost the last-queued file, the single-host agent3.csv, in most
# runs (2026-10-19, restart). Doubling waits give a throttled handshake about a minute to clear.
ARTIFACT_FETCH_MAX_ATTEMPTS=6
ARTIFACT_FETCH_RETRY_DELAY_S=1
LOCAL_RUN_LOCK_DIRS=("")

cleanup_local_run_locks() {
  local lock_dir
  for lock_dir in "${LOCAL_RUN_LOCK_DIRS[@]}"; do
    [[ -z "${lock_dir}" ]] && continue
    rm -f "${lock_dir}/pid"
    rmdir "${lock_dir}" 2>/dev/null || true
  done
}

# Run a bash script (read from stdin) on a remote host with positional
# arguments. ssh flattens its argument vector into one command string that the
# remote login shell re-parses, so every argument is shell-quoted here; without
# this an empty argument would vanish and shift every later positional.
remote_bash() {
  local user="$1"
  local host="$2"
  shift 2
  local quoted=""
  if (( $# > 0 )); then
    printf -v quoted ' %q' "$@"
  fi
  ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s --${quoted}"
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
    # The holder (pid, start time, script), so a lock left by a dead process can be told
    # from a live one; it is never taken over automatically (2026-10-19, restart).
    if [[ -f "${lock_dir}/pid" ]]; then
      local holder
      holder="$(cat "${lock_dir}/pid")"
      if kill -0 "${holder%% *}" 2>/dev/null; then
        echo "Held by: ${holder} (running)" >&2
      else
        echo "Held by: ${holder} (not running here; remove the lock if no other machine holds it)" >&2
      fi
    fi
    return 1
  fi
  printf '%s %s %s\n' "$$" "$(date +%FT%T)" "$0" > "${lock_dir}/pid"
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
Usage: $0 <private_key_path> [${SUPPORTED_MODES// /|}] [--log] [--output-dir <path>] [--model <name>] [-f <fault_tolerance>] [-t <experiment_seconds>] [targets_file]

Arguments:
  private_key_path  Local SSH private key for CloudLab nodes.
  mode              Optional; one of: ${SUPPORTED_MODES} (default: normal).
  targets_file      Optional path to cloudlab SSH targets file.

Flags:
  --log                             Download client, server, applier, and agent logs.
  --output-dir <path>               Store artifacts here instead of a timestamped run directory.
  --verbose                         Enable per-message debug logs on server replicas.
  --reward-design <reward_tps|reward_mixed|reward_tps_latency>
                                    Reward the agent learns from (default: reward_tps_latency).
  --model <discrete_random|discrete_pavlov|discrete_linear|discrete_tree_ts|discrete_tree_blocks|discrete_linucb|discrete_ppo|discrete_gp|continuous_gp|continuous_linucb|continuous_heuristic|continuous_ppo>
                                     Learning-agent model for learning mode (default: discrete_random).
  --latency-floor-multiplier <float> Never apply a timeout below this multiplier x the recent
                                     consensus latency; any model except scheduled (0 disables it;
                                     default: 0).
  --prior-path <path>                Prior file the model loads, for continuous_gp, discrete_gp, discrete_linucb,
                                     continuous_linucb, discrete_ppo or continuous_ppo in learning or
                                     sharing mode. It must be under learning_agent/, the only tree
                                     shipped to the nodes. Omitted (default): the model starts untrained.
  --timeout-ladder <ms,ms,...>       PBFT only, discrete models in learning or sharing mode: rungs replacing
                                     the 250,550,1200 ladder (a discrete_linucb/ppo prior must match it).
  --heuristic-k <float>              continuous_heuristic only: k replacing the protocol's own.
  --latency-floor-window <int>       With --latency-floor-multiplier: the floor is the median over this many
                                     committing windows (default: 1, the last one; PBFT study E6).
  -f <int>                          Fault tolerance f; nodes defaults to 3f+1 (default: 1).
  -t <int>                          Seconds to run after benchmark start (default: 300).

${PROTOCOL_NAME} options:
  --nodes <int>                     Override node count (default: 3f+1).
  --failure-spec <path>             failure_spec.xml: the <network> section drives the delay
                                     appliers, the <${PROTOCOL_NAME}> section the replicas'
                                     proposal-delay injection (default: ${FAILURE_SPEC_DEFAULT_REL}).
  --no-failure-spec                 Disable the network delay appliers and proposal-delay injection.
  --rate <int>                      Total client requests/s across all lanes (default: 1000).
  --proposal-interval <ms>          Fixed proposal cadence (default: 100).
  --batch-max-requests <int>        Requests per batch (default/max: 8192).
  --batch-max-bytes <int>           Bytes per batch (default: 4718592).
  --max-inflight-batches <int>      Unexecuted proposals (default and max: 16, the consensus window).
  --max-pending-requests <int>      Bound each ingress/pending queue (default: 32768). Raising it
                                     trades rejection for queueing delay; measured worse at 40k.
  --max-pending-bytes <int>         Bound each ingress/pending queue (default: 33554432).
  --request-size <bytes>            Serialized size of every client request, padded with a blank
                                     transaction; 0 sends unpadded transfer requests (default: 512).
  --request-timeout <ms>            Client per-request timeout in ms (default: 10000).
  --client-connections <int>        Client streams (independent sender lanes) per replica (default: 4).
  --client-sign <true|false>        RSA-sign every client request (default: false).
  --client-start-delay <seconds>    Wait after servers are ready before launching client (default: 10).
  --start-offset <seconds>          Delay from client launch to benchmark/failure start (default: 10).
  --replica-port <port>             gRPC port of every replica on its overlay IP (default: 15000).

Learning-agent options:
  --agent-port <port>               Learning-agent port inside each replica namespace (default: 15501).
  --initial-timeout <ms>            Initial election (view-change) timeout (default: from ${BEDROCK_PROTOCOL_CONFIG_REL}).
  --initial-slow-path-timeout <ms>  Initial SBFT slow-path timeout (default: from the protocol config).
  --learning-feature-duration <ms>  Feature window (default: 7000).
  --learning-reply-wait <ms>        Recommendation wait (default: 2000).
  --learning-warmup-duration <ms>   Post-apply warm-up (default: 4000).
  --learning-reward-duration <ms>   Reward window (default: 7000).
EOF
}

mode_enables_learning_agent() {
  case "$1" in
    learning|sharing|scheduled) return 0 ;;
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

# Model types retired on 2026-10-07 (RENAMED_MODEL_TYPES in
# learning_agent/models.py): prints the replacement, or fails for any other name.
renamed_model_type() {
  case "$1" in
    random|pavlov|linear|ppo) printf 'discrete_%s\n' "$1" ;;
    cmab|thompson) printf '%s\n' "discrete_tree_ts" ;;
    cmab_reset) printf '%s\n' "discrete_tree_blocks" ;;
    codex|codex_v1) printf '%s\n' "discrete_linucb" ;;
    continuous_pretrained_gp|continuous) printf '%s\n' "continuous_gp" ;;
    continuous_codex) printf '%s\n' "continuous_linucb" ;;
    latency_floor_heuristic|continuous_latency_floor) printf '%s\n' "continuous_heuristic" ;;
    *) return 1 ;;
  esac
}

normalize_agent_model_type() {
  local model_lower
  model_lower="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  case "${model_lower}" in
    discrete_random|discrete_pavlov|discrete_linear|discrete_tree_ts|discrete_tree_blocks|discrete_linucb|discrete_ppo|discrete_gp|continuous_gp|continuous_linucb|continuous_heuristic|continuous_ppo) printf '%s\n' "${model_lower}" ;;
    *)
      local renamed
      if renamed="$(renamed_model_type "${model_lower}")"; then
        echo "--model ${model_lower} was renamed to ${renamed}." >&2
      fi
      return 1
      ;;
  esac
}

# --prior-path: only learning_agent/ is shipped to the nodes and the agent runs
# from the extracted tree, so the prior must live under it. Prints the path
# relative to the repo root (learning_agent/...), or fails with the reason.
repo_relative_prior_path() {
  local path="$1"
  local repo
  local dir
  local rel
  if [[ ! -f "${path}" ]]; then
    echo "--prior-path file not found: ${path}" >&2
    return 1
  fi
  repo="$(cd "${REPO_ROOT}" && pwd -P)"
  dir="$(cd "$(dirname "${path}")" && pwd -P)"
  case "${dir}/" in
    "${repo}/learning_agent/"*) rel="${dir#"${repo}/"}/$(basename "${path}")" ;;
    *)
      echo "--prior-path ${path} is outside ${repo}/learning_agent/: only learning_agent/ is shipped to the nodes; put the prior under learning_agent/pretrained_models/." >&2
      return 1
      ;;
  esac
  # The remote agent command line carries it unquoted.
  if ! [[ "${rel}" =~ ^[A-Za-z0-9._/-]+$ ]]; then
    echo "--prior-path ${rel} has characters other than [A-Za-z0-9._/-], which the remote agent command cannot carry." >&2
    return 1
  fi
  printf '%s\n' "${rel}"
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

# Prints a failure spec's start-up warm-up in ms, read as the replicas and the
# network applier read it: <warmUpTimeMs>, else <warmUpTime> in seconds. Fails
# when the spec has neither.
failure_spec_warmup_ms() {
  python3 - "$1" <<'PY'
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
root = ET.parse(path).getroot()
ms = root.find("warmUpTimeMs")
secs = root.find("warmUpTime")
if ms is not None and ms.text and ms.text.strip():
    value = int(ms.text.strip())
elif secs is not None and secs.text and secs.text.strip():
    value = int(float(secs.text.strip()) * 1000.0 + 0.5)
else:
    sys.exit(f"{path}: no <warmUpTimeMs> or <warmUpTime>")
print(max(0, value))
PY
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
        sleep $((ARTIFACT_FETCH_RETRY_DELAY_S << attempt))
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

  echo "Timed out waiting for ${PROTOCOL_NAME} replica ports from ${node0_host}." >&2
  [[ -z "${last_unready}" ]] || echo "Still unreachable: ${last_unready}" >&2
  return 1
}

bedrock_main() {
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
      --reward-design) REWARD_DESIGN="${2:-}"; shift 2 ;;
      --latency-floor-multiplier) LATENCY_FLOOR_MULTIPLIER="${2:-}"; shift 2 ;;
      --prior-path)
        if [[ $# -lt 2 || -z "${2}" ]]; then
          echo "Missing value for --prior-path." >&2
          usage >&2
          exit 2
        fi
        PRIOR_PATH="${2}"
        shift 2
        ;;
      --timeout-ladder) TIMEOUT_LADDER="${2:-}"; [[ -n "${TIMEOUT_LADDER}" ]] || { echo "Missing value for --timeout-ladder." >&2; exit 2; }; shift 2 ;;
      --heuristic-k) HEURISTIC_K="${2:-}"; [[ -n "${HEURISTIC_K}" ]] || { echo "Missing value for --heuristic-k." >&2; exit 2; }; shift 2 ;;
      --latency-floor-window) LATENCY_FLOOR_WINDOW="${2:-}"; [[ -n "${LATENCY_FLOOR_WINDOW}" ]] || { echo "Missing value for --latency-floor-window." >&2; exit 2; }; shift 2 ;;
      --nodes) NODES="${2:-}"; NODES_SET=1; shift 2 ;;
      --failure-spec) FAILURE_SPEC_PATH="${2:-}"; FAILURE_SPEC_ENABLED=1; shift 2 ;;
      --no-failure-spec) FAILURE_SPEC_ENABLED=0; shift ;;
      --rate) TX_RATE="${2:-}"; shift 2 ;;
      --proposal-interval) PROPOSAL_INTERVAL_MS="${2:-}"; shift 2 ;;
      --batch-max-requests) BATCH_MAX_REQUESTS="${2:-}"; shift 2 ;;
      --batch-max-bytes) BATCH_MAX_BYTES="${2:-}"; shift 2 ;;
      --max-inflight-batches) MAX_INFLIGHT_BATCHES="${2:-}"; shift 2 ;;
      --max-pending-requests) MAX_PENDING_REQUESTS="${2:-}"; shift 2 ;;
      --max-pending-bytes) MAX_PENDING_BYTES="${2:-}"; shift 2 ;;
      --request-size) REQUEST_SIZE="${2:-}"; shift 2 ;;
      --request-timeout) REQUEST_TIMEOUT_MS="${2:-}"; shift 2 ;;
      --client-connections) CLIENT_CONNECTIONS_PER_TARGET="${2:-}"; shift 2 ;;
      --client-sign) CLIENT_SIGN="${2:-}"; shift 2 ;;
      --client-start-delay) CLIENT_START_DELAY_S="${2:-}"; shift 2 ;;
      --start-offset) START_OFFSET_AFTER_CLIENT_LAUNCH_S="${2:-}"; shift 2 ;;
      --replica-port) REPLICA_PORT="${2:-}"; shift 2 ;;
      --agent-port) AGENT_PORT="${2:-}"; shift 2 ;;
      --initial-timeout) INITIAL_TIMEOUT_MS="${2:-}"; shift 2 ;;
      --initial-slow-path-timeout) INITIAL_SLOW_PATH_TIMEOUT_MS="${2:-}"; shift 2 ;;
      --learning-feature-duration) LEARNING_FEATURE_DURATION_MS="${2:-}"; shift 2 ;;
      --learning-reply-wait) LEARNING_REPLY_WAIT_MS="${2:-}"; shift 2 ;;
      --learning-warmup-duration) LEARNING_WARMUP_DURATION_MS="${2:-}"; shift 2 ;;
      --learning-reward-duration) LEARNING_REWARD_DURATION_MS="${2:-}"; shift 2 ;;
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

  if ! mode_is_supported "${MODE}"; then
    echo "Mode '${MODE}' is not supported by run_${PROTOCOL_NAME}.sh (supported: ${SUPPORTED_MODES})." >&2
    exit 2
  fi
  if ! [[ "${F}" =~ ^[0-9]+$ ]] || (( F < 1 )); then
    echo "Invalid -f value: ${F}. It must be an integer >= 1." >&2
    exit 2
  fi
  if (( NODES_SET == 0 )); then
    NODES="$((3 * F + 1))"
  fi
  for value_name in NODES EXPERIMENT_TIME_S TX_RATE REQUEST_TIMEOUT_MS REPLICA_PORT AGENT_PORT PROPOSAL_INTERVAL_MS BATCH_MAX_REQUESTS BATCH_MAX_BYTES MAX_INFLIGHT_BATCHES MAX_PENDING_REQUESTS MAX_PENDING_BYTES; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
      echo "Invalid ${value_name}: ${value}" >&2
      exit 2
    fi
  done
  for value_name in INITIAL_TIMEOUT_MS INITIAL_SLOW_PATH_TIMEOUT_MS; do
    value="${!value_name}"
    [[ -z "${value}" ]] && continue
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
      echo "Invalid ${value_name}: ${value}" >&2
      exit 2
    fi
  done
  if ! [[ "${REQUEST_SIZE}" =~ ^[0-9]+$ ]]; then
    echo "Invalid --request-size value: ${REQUEST_SIZE}. Use an integer >= 0." >&2
    exit 2
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
  if (( NODES > 64 )); then
    echo "Invalid node count: ${NODES}. bedrock_client tracks replies for at most 64 replicas." >&2
    exit 2
  fi
  case "${CLIENT_SIGN}" in
    true|false) ;;
    *) echo "Invalid --client-sign value: ${CLIENT_SIGN}. Use true or false." >&2; exit 2 ;;
  esac
  if ! [[ "${CLIENT_CONNECTIONS_PER_TARGET}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid --client-connections value: ${CLIENT_CONNECTIONS_PER_TARGET}. Use a positive integer." >&2
    exit 2
  fi
  if ! AGENT_MODEL_TYPE="$(normalize_agent_model_type "${AGENT_MODEL_TYPE}")"; then
    echo "Invalid --model value. Use one of discrete_random|discrete_pavlov|discrete_linear|discrete_tree_ts|discrete_tree_blocks|discrete_linucb|discrete_ppo|discrete_gp|continuous_gp|continuous_linucb|continuous_heuristic|continuous_ppo." >&2
    exit 2
  fi
  if (( AGENT_MODEL_SET == 1 )) && [[ "${MODE}" != "learning" && "${MODE}" != "sharing" ]]; then
    echo "--model is only supported with learning or sharing mode." >&2
    exit 2
  fi
  if [[ -n "${PRIOR_PATH}" ]]; then
    if [[ "${MODE}" != "learning" && "${MODE}" != "sharing" ]]; then
      echo "--prior-path is only supported with learning or sharing mode." >&2
      exit 2
    fi
    case "${AGENT_MODEL_TYPE}" in
      continuous_gp|discrete_gp|discrete_linucb|continuous_linucb|discrete_ppo|continuous_ppo) ;;
      *) echo "--prior-path is not supported with --model ${AGENT_MODEL_TYPE}. Use one of continuous_gp|discrete_gp|discrete_linucb|continuous_linucb|discrete_ppo|continuous_ppo." >&2; exit 2 ;;
    esac
    if ! PRIOR_PATH="$(repo_relative_prior_path "${PRIOR_PATH}")"; then
      exit 2
    fi
  fi
  if [[ -n "${TIMEOUT_LADDER}" ]]; then
    if [[ "${PROTOCOL_NAME}" != "pbft" ]]; then
      echo "--timeout-ladder is supported only by run_pbft.sh." >&2; exit 2
    fi
    if [[ "${MODE}" != "learning" && "${MODE}" != "sharing" ]] || [[ "${AGENT_MODEL_TYPE}" != discrete_* ]]; then
      echo "--timeout-ladder applies to a discrete model in learning or sharing mode." >&2; exit 2
    fi
    if ! [[ "${TIMEOUT_LADDER}" =~ ^[0-9]+(,[0-9]+)+$ ]]; then
      echo "Invalid --timeout-ladder: ${TIMEOUT_LADDER}. Expected comma-separated integer ms, e.g. 250,450,1200." >&2; exit 2
    fi
  fi
  if [[ -n "${HEURISTIC_K}" ]]; then
    if [[ "${AGENT_MODEL_TYPE}" != "continuous_heuristic" ]]; then
      echo "--heuristic-k applies to --model continuous_heuristic only." >&2; exit 2
    fi
    if ! [[ "${HEURISTIC_K}" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "${HEURISTIC_K}" =~ ^0+([.]0+)?$ ]]; then
      echo "Invalid --heuristic-k: ${HEURISTIC_K}. Expected a positive number." >&2; exit 2
    fi
  fi
  if [[ -n "${LATENCY_FLOOR_WINDOW}" ]]; then
    if ! [[ "${LATENCY_FLOOR_WINDOW}" =~ ^[1-9][0-9]*$ ]]; then
      echo "Invalid --latency-floor-window: ${LATENCY_FLOOR_WINDOW}. Expected a positive integer." >&2; exit 2
    fi
    if ! [[ "${LATENCY_FLOOR_MULTIPLIER}" =~ [1-9] ]]; then
      echo "--latency-floor-window needs --latency-floor-multiplier above 0." >&2; exit 2
    fi
  fi
  case "${REWARD_DESIGN}" in
    reward_tps|reward_mixed|reward_tps_latency) ;;
    *) echo "Invalid --reward-design: ${REWARD_DESIGN}. Expected reward_tps|reward_mixed|reward_tps_latency." >&2; exit 2 ;;
  esac
  if ! [[ "${LATENCY_FLOOR_MULTIPLIER}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid --latency-floor-multiplier: ${LATENCY_FLOOR_MULTIPLIER}. Expected a non-negative number (0 disables the constraint)." >&2
    exit 2
  fi
  if [[ "${MODE}" == "scheduled" && "${LATENCY_FLOOR_MULTIPLIER}" =~ [1-9] ]]; then
    echo "--latency-floor-multiplier is only supported with learning or sharing mode; scheduled mode applies the schedule as written." >&2
    exit 2
  fi
  if ! mode_enables_learning_agent "${MODE}"; then
    LEARNING_AGENT_ENABLED=0
  else
    LEARNING_AGENT_ENABLED=1
  fi

  LOCAL_SSH_KEY_PATH="${LOCAL_SSH_KEY_PATH/#\~/$HOME}"
  FAILURE_SPEC_PATH="$(absolute_path "${FAILURE_SPEC_PATH}")"

  if [[ ! -f "${LOCAL_SSH_KEY_PATH}" ]]; then
    echo "Local SSH key not found: ${LOCAL_SSH_KEY_PATH}" >&2
    exit 1
  fi
  SSH_OPTS+=(-i "${LOCAL_SSH_KEY_PATH}" -o IdentitiesOnly=yes)
  if [[ ! -d "${BEDROCK_SRC}" || ! -f "${BEDROCK_SRC}/CMakeLists.txt" ]]; then
    echo "CppBedrock submodule not found: ${BEDROCK_SRC}" >&2
    exit 1
  fi
  if [[ ! -f "${PROTOCOL_CONFIG_LOCAL}" ]]; then
    echo "Protocol config not found: ${PROTOCOL_CONFIG_LOCAL}" >&2
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
  # Every replica anchors its episode grid on start + the spec's warm-up (also
  # with --no-failure-spec), and the agent maps episodes to schedule rows by the
  # episode period.
  if (( LEARNING_AGENT_ENABLED == 1 )); then
    for value_name in LEARNING_FEATURE_DURATION_MS LEARNING_REPLY_WAIT_MS LEARNING_WARMUP_DURATION_MS LEARNING_REWARD_DURATION_MS; do
      value="${!value_name}"
      if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "Invalid ${value_name}: ${value}" >&2
        exit 2
      fi
    done
    LEARNING_EPISODE_MS="$((LEARNING_FEATURE_DURATION_MS + LEARNING_REPLY_WAIT_MS + LEARNING_WARMUP_DURATION_MS + LEARNING_REWARD_DURATION_MS))"
    if (( LEARNING_EPISODE_MS <= 0 )); then
      echo "Invalid learning episode period: ${LEARNING_EPISODE_MS} ms" >&2
      exit 2
    fi
    if [[ ! -f "${FAILURE_SPEC_PATH}" ]]; then
      echo "Failure spec not found (the learning episode grid needs its warm-up): ${FAILURE_SPEC_PATH}" >&2
      exit 1
    fi
    if ! LEARNING_ANCHOR_WARMUP_MS="$(failure_spec_warmup_ms "${FAILURE_SPEC_PATH}")"; then
      echo "Cannot read the warm-up from failure spec ${FAILURE_SPEC_PATH}" >&2
      exit 1
    fi
  fi
  if [[ "${MODE}" == "scheduled" && ! -f "${REPO_ROOT}/${SCHEDULE_FILE_REL}" ]]; then
    echo "Schedule file not found: ${REPO_ROOT}/${SCHEDULE_FILE_REL}" >&2
    exit 1
  fi
  if [[ ! -f "${TARGETS_FILE}" ]]; then
    echo "Targets file not found: ${TARGETS_FILE}" >&2
    exit 1
  fi

  while read -r node host user _; do
    [[ -z "${node:-}" ]] && continue
    [[ "${node:0:1}" == "#" ]] && continue
    if [[ -z "${host:-}" || -z "${user:-}" ]]; then
      continue
    fi
    acquire_target_lock "${user}" "${host}"
    "${REPO_ROOT}/scripts/cloudlab/check_idle.sh" "${LOCAL_SSH_KEY_PATH}" "${user}@${host}" || exit 1
  done < "${TARGETS_FILE}"

  # The frozen code state (Kevin, 2026-10-19, restart): refuse to launch when the code a run
  # ships differs from FREEZE (scripts/cloudlab/code_state.py --check); ALLOW_UNFROZEN=1 marks
  # a run outside the frozen pipeline, and run.env records it. Before the artifact directory
  # is chosen, so runs with and without --output-dir are both checked.
  if [[ "${ALLOW_UNFROZEN:-0}" != "1" ]]; then
    python3 "${REPO_ROOT}/scripts/cloudlab/code_state.py" --check ${PRIOR_PATH:+--prior "${REPO_ROOT}/${PRIOR_PATH}"} > /dev/null \
      || { echo "The code or the prior differs from ${REPO_ROOT}/FREEZE (reason above); refreeze or retrain, or set ALLOW_UNFROZEN=1 for a run outside the frozen pipeline." >&2; exit 1; }
  fi
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
  cp "${PROTOCOL_CONFIG_LOCAL}" "${ARTIFACT_DIR}/$(basename "${PROTOCOL_CONFIG_LOCAL}")"
  if (( FAILURE_SPEC_ENABLED == 1 )); then
    cp "${FAILURE_SPEC_PATH}" "${ARTIFACT_DIR}/failure_spec.xml"
  fi
  if [[ "${MODE}" == "scheduled" ]]; then
    cp "${REPO_ROOT}/${SCHEDULE_FILE_REL}" "${ARTIFACT_DIR}/$(basename "${SCHEDULE_FILE_REL}")"
  fi
  {
    printf 'protocol=%q\n' "${PROTOCOL_NAME}"
    printf 'engine=%q\n' "cppbedrock"
    printf 'protocol_config=%q\n' "${BEDROCK_PROTOCOL_CONFIG_REL}"
    printf 'run_id=%q\n' "${RUN_ID}"
    printf 'artifact_dir=%q\n' "${ARTIFACT_DIR}"
    printf 'mode=%q\n' "${MODE}"
    printf 'fault_tolerance=%q\n' "${F}"
    printf 'nodes=%q\n' "${NODES}"
    printf 'experiment_time_s=%q\n' "${EXPERIMENT_TIME_S}"
    printf 'tx_rate=%q\n' "${TX_RATE}"
    printf 'request_size=%q\n' "${REQUEST_SIZE}"
    printf 'proposal_interval_ms=%q\n' "${PROPOSAL_INTERVAL_MS}"
    printf 'batch_max_requests=%q\n' "${BATCH_MAX_REQUESTS}"
    printf 'batch_max_bytes=%q\n' "${BATCH_MAX_BYTES}"
    printf 'max_inflight_batches=%q\n' "${MAX_INFLIGHT_BATCHES}"
    printf 'max_pending_requests=%q\n' "${MAX_PENDING_REQUESTS}"
    printf 'max_pending_bytes=%q\n' "${MAX_PENDING_BYTES}"
    printf 'request_timeout_ms=%q\n' "${REQUEST_TIMEOUT_MS}"
    printf 'client_connections_per_target=%q\n' "${CLIENT_CONNECTIONS_PER_TARGET}"
    printf 'client_sign=%q\n' "${CLIENT_SIGN}"
    printf 'proposal_signing=true\n'
    printf 'replica_port=%q\n' "${REPLICA_PORT}"
    printf 'initial_timeout_ms=%q\n' "${INITIAL_TIMEOUT_MS:-from-config}"
    printf 'initial_slow_path_timeout_ms=%q\n' "${INITIAL_SLOW_PATH_TIMEOUT_MS:-from-config}"
    printf 'agent_model=%q\n' "${AGENT_MODEL_TYPE}"
    printf 'reward_design=%q\n' "${REWARD_DESIGN}"
    printf 'latency_floor_multiplier=%q\n' "${LATENCY_FLOOR_MULTIPLIER}"
    printf 'prior_path=%q\n' "${PRIOR_PATH:-}"
    # Only when given, so other runs' run.env stays as it was.
    [[ -n "${TIMEOUT_LADDER}" ]] && printf 'timeout_ladder=%q\n' "${TIMEOUT_LADDER}"
    [[ -n "${HEURISTIC_K}" ]] && printf 'heuristic_k=%q\n' "${HEURISTIC_K}"
    [[ -n "${LATENCY_FLOOR_WINDOW}" ]] && printf 'latency_floor_window=%q\n' "${LATENCY_FLOOR_WINDOW}"
    printf 'learning_window_mode=wall-clock\n'
    printf 'learning_feature_duration_ms=%q\n' "${LEARNING_FEATURE_DURATION_MS}"
    printf 'learning_reply_wait_ms=%q\n' "${LEARNING_REPLY_WAIT_MS}"
    printf 'learning_warmup_duration_ms=%q\n' "${LEARNING_WARMUP_DURATION_MS}"
    printf 'learning_reward_duration_ms=%q\n' "${LEARNING_REWARD_DURATION_MS}"
    printf 'log_enabled=%q\n' "${LOG_ENABLED}"
    printf 'failure_spec_enabled=%q\n' "${FAILURE_SPEC_ENABLED}"
    printf 'targets_file=%q\n' "${TARGETS_FILE}"
    printf 'git_commit=%q\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'git_dirty=%q\n' "$([[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no 2>/dev/null)" ]] && echo 1 || echo 0)"
    printf 'cppbedrock_commit=%q\n' "$(git -C "${BEDROCK_SRC}" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'cppbedrock_dirty=%q\n' "$([[ -n "$(git -C "${BEDROCK_SRC}" status --porcelain --untracked-files=no 2>/dev/null)" ]] && echo 1 || echo 0)"
    printf 'freeze_name=%q\n' "$(sed -n 's/^freeze_name=//p' "${REPO_ROOT}/FREEZE" 2>/dev/null)"
    printf 'allow_unfrozen=%q\n' "${ALLOW_UNFROZEN:-0}"
    python3 "${REPO_ROOT}/scripts/cloudlab/code_state.py" ${PRIOR_PATH:+--prior "${REPO_ROOT}/${PRIOR_PATH}"}
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
  SHARING_HOSTS_CONFIG_LOCAL="${ARTIFACT_DIR}/${PROTOCOL_NAME}_sharing_hosts.config"
  CLIENT_TARGETS_LOCAL="${ARTIFACT_DIR}/client_targets.txt"
  : > "${FAILURE_HOSTS_CONFIG_LOCAL}"
  : > "${AGENT_HOSTS_CONFIG_LOCAL}"
  : > "${SHARING_HOSTS_CONFIG_LOCAL}"
  : > "${CLIENT_TARGETS_LOCAL}"
  sharing_consensus_port=0
  [[ "${MODE}" == "sharing" ]] && sharing_consensus_port="${SHARING_CONSENSUS_PORT}"
  replica_ips=()
  for ((replica_id = 0; replica_id < NODES; replica_id++)); do
    machine_idx=$((replica_id / replicas_per_server_machine))
    slot=$((replica_id % replicas_per_server_machine))
    ip="${OVERLAY_OCTET_A}.${OVERLAY_OCTET_B}.$((machine_idx + 1)).$((OVERLAY_HOST_OFFSET + slot))"
    replica_ips+=("${ip}")
    printf '%d %s %d\n' "${replica_id}" "${ip}" "${REPLICA_PORT}" >> "${FAILURE_HOSTS_CONFIG_LOCAL}"
    printf '%d %s 0 0 %d %d\n' "${replica_id}" "${ip}" "${AGENT_PORT}" "${sharing_consensus_port}" >> "${AGENT_HOSTS_CONFIG_LOCAL}"
    # Report-chain node ids are 1-based: node_id host report_port agent_port consensus_port.
    printf '%d %s %d %d %d\n' "$((replica_id + 1))" "${ip}" "${SHARING_REPORT_PORT}" "${AGENT_PORT}" "${SHARING_CONSENSUS_PORT}" >> "${SHARING_HOSTS_CONFIG_LOCAL}"
    printf '%s:%d\n' "${ip}" "${REPLICA_PORT}" >> "${CLIENT_TARGETS_LOCAL}"
  done
  replica_ips_csv="$(IFS=,; printf '%s' "${replica_ips[*]}")"

  echo "Selected mode: ${MODE}"
  echo "${PROTOCOL_NAME} nodes: ${NODES} (f=${F}, replicas_per_server_machine=${replicas_per_server_machine})"
  echo "Client rate: ${TX_RATE} req/s; request timeout: ${REQUEST_TIMEOUT_MS} ms; target mode: broadcast; lanes: ${CLIENT_CONNECTIONS_PER_TARGET}"
  echo "Learning-agent orchestration: ${LEARNING_AGENT_ENABLED} (log=${LOG_ENABLED})"

  echo "Creating ${PROTOCOL_NAME} CloudLab payload tarball..."
  rm -f "${LOCAL_TAR}"
  sharing_payload=""
  # The report chain is a SmartBFT binary built on the remote node.
  [[ "${MODE}" == "sharing" ]] && sharing_payload="SmartBFT"
  COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 tar -czf "${LOCAL_TAR}" \
    --exclude='SmartBFT/.git' \
    --exclude='SmartBFT/build' \
    --exclude='CppBedrock/.git' \
    --exclude='CppBedrock/build' \
    --exclude='CppBedrock/web' \
    --exclude='CppBedrock/keys/*.pem' \
    -C "${REPO_ROOT}" CppBedrock config learning_agent proto scripts/apply_network_failures.py scripts/cloudlab/gen_committee.py \
    ${sharing_payload}

  echo "Copying payload to CloudLab nodes and building CppBedrock..."
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
      scp "${SSH_OPTS[@]}" "${SHARING_HOSTS_CONFIG_LOCAL}" "${user}@${host}:/tmp/${PROTOCOL_NAME}_sharing_hosts_$$.config"
      if (( FAILURE_SPEC_ENABLED == 1 )); then
        remote_failure_spec="/tmp/${PROTOCOL_NAME}_failure_spec_$$.xml"
        scp "${SSH_OPTS[@]}" "${FAILURE_SPEC_PATH}" "${user}@${host}:${remote_failure_spec}"
      fi
      remote_bash "${user}" "${host}" \
        "${remote_tar}" \
        "/tmp/${PROTOCOL_NAME}_hosts_failure_$$.config" \
        "${remote_failure_spec}" \
        "/tmp/${PROTOCOL_NAME}_agent_hosts_$$.config" \
        "${RUN_ROOT_REMOTE_NAME}" \
        "/tmp/${PROTOCOL_NAME}_sharing_hosts_$$.config" \
        "${MODE}" <<'REMOTE'
set -euo pipefail

tar_path="$1"
failure_hosts_path="$2"
failure_spec_path="$3"
agent_hosts_path="$4"
run_root_name="$5"
sharing_hosts_path="$6"
mode="$7"
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

# Keep the cmake build dir across runs to avoid full rebuilds.
for stale in "${run_root}/CppBedrock" "${run_root}/SmartBFT" "${run_root}/config" "${run_root}/learning_agent" "${run_root}/proto" "${run_root}/scripts" "${run_root}/logs" "${run_root}/netcfg"; do
  remove_path_with_sudo_fallback "${stale}"
done
mkdir -p "${run_root}"
tar --warning=no-unknown-keyword -xzf "${tar_path}" -C "${run_root}"
mkdir -p "${run_root}/bin" "${run_root}/logs" "${run_root}/netcfg" "${run_root}/build"
cp "${failure_hosts_path}" "${run_root}/hosts.failure.config"
cp "${agent_hosts_path}" "${run_root}/agent_hosts.config"
cp "${sharing_hosts_path}" "${run_root}/sharing_hosts.config"
if [[ -n "${failure_spec_path}" ]]; then
  cp "${failure_spec_path}" "${run_root}/failure_spec.xml"
fi
chmod +x "${run_root}/scripts/apply_network_failures.py" "${run_root}/scripts/cloudlab/gen_committee.py"
rm -f "${tar_path}" "${failure_hosts_path}" "${agent_hosts_path}" "${sharing_hosts_path}"
if [[ -n "${failure_spec_path}" ]]; then
  rm -f "${failure_spec_path}"
fi

for tool in cmake protoc grpc_cpp_plugin tmux python3 openssl; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "${tool} not found on $(hostname); run scripts/cloudlab/setup.sh first." >&2
    exit 1
  fi
done

echo "Building CppBedrock with $(cmake --version | head -n1)"
cmake -S "${run_root}/CppBedrock" -B "${run_root}/build" -DCMAKE_BUILD_TYPE=Release \
  -DBEDROCK_AGENT_PROTO="${run_root}/proto/agent.proto" > "${run_root}/logs/cmake_configure.log" 2>&1 \
  || { tail -n 40 "${run_root}/logs/cmake_configure.log" >&2; exit 1; }
cmake --build "${run_root}/build" -j "$(nproc)" --target CppBedrock bedrock_client > "${run_root}/logs/cmake_build.log" 2>&1 \
  || { tail -n 60 "${run_root}/logs/cmake_build.log" >&2; exit 1; }
cp "${run_root}/build/CppBedrock" "${run_root}/bin/CppBedrock"
cp "${run_root}/build/bedrock_client" "${run_root}/bin/bedrock_client"

if [[ "${mode}" == "sharing" ]]; then
  export PATH="$HOME/go/bin:${PATH}"
  command -v go >/dev/null 2>&1 || { echo "go not found on $(hostname); run scripts/cloudlab/setup.sh first." >&2; exit 1; }
  echo "Building report-chain (sharing) binary with $(go version)"
  (cd "${run_root}/SmartBFT" && GOCACHE="${run_root}/gocache" go build -o "${run_root}/bin/smartbft-sharing" ./examples/sharing) \
    > "${run_root}/logs/sharing_build.log" 2>&1 || { tail -n 40 "${run_root}/logs/sharing_build.log" >&2; exit 1; }
fi
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
  if ! remote_bash "${node0_user}" "${node0_host}" \
    "${RUN_ROOT_REMOTE_NAME}" \
    "${replica_ips_csv}" \
    "${REPLICA_PORT}" <<'REMOTE'
set -euo pipefail
run_root="$HOME/$1"
ips="$2"
replica_port="$3"
python3 "${run_root}/scripts/cloudlab/gen_committee.py" \
  --workspace bedrock \
  --ips "${ips}" \
  --out-dir "${run_root}/netcfg" \
  --front-port "${replica_port}"
REMOTE
  then
    echo "Committee generation failed on node0." >&2
    exit 1
  fi

  echo "Fetching committee from node0..."
  scp "${SSH_OPTS[@]}" "${node0_user}@${node0_host}:${RUN_ROOT_REMOTE_NAME}/netcfg/committee.json" "${ARTIFACT_DIR}/committee.json"

  if (( machine_count > 1 )); then
    echo "Distributing committee and keys to server machines..."
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
    if remote_bash "${user}" "${host}" \
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
# Another protocol's harness, interrupted before its teardown, can leave its
# bridge holding this same overlay address; its route then shadows ours.
for foreign_if in $(ip -o -4 addr show | awk -v ip="${bridge_ip}/" -v own="${bridge_name}" 'index($4, ip) == 1 && $2 != own {print $2}' | sort -u); do
  sudo -n ip link del "${foreign_if}" >/dev/null 2>&1 || true
done
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
  # TCP tuning inside the replica namespace (namespace-scoped, host untouched),
  # matching scripts/cloudlab/lib_rustbft.sh. With netem delays of 100-200 ms
  # one-way, the default 4 MB send buffer caps a single stream well below what
  # batch dissemination needs, and slow start after idle re-throttles every
  # stream between proposal ticks.
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
    if ! remote_bash "${node0_user}" "${node0_host}" \
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
  LEARNING_ANCHOR_UNIX_MS=0
  if (( LEARNING_AGENT_ENABLED == 1 )); then
    LEARNING_ANCHOR_UNIX_MS="$((START_UNIX_MS + LEARNING_ANCHOR_WARMUP_MS))"
    echo "Learning episode grid: anchor_unix_ms=${LEARNING_ANCHOR_UNIX_MS} (start + warm-up ${LEARNING_ANCHOR_WARMUP_MS} ms) period_ms=${LEARNING_EPISODE_MS}"
  fi

  if (( FAILURE_SPEC_ENABLED == 1 )); then
    echo "Starting ${PROTOCOL_NAME} network failure appliers..."
    applier_start_failed=0
    for i in "${!server_nodes[@]}"; do
      node="${server_nodes[$i]}"
      host="${server_hosts[$i]}"
      user="${server_users[$i]}"
      machine_idx="${i}"
      if remote_bash "${user}" "${host}" \
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
    "cd \"${run_root}\" && python3 \"${run_root}/scripts/apply_network_failures.py\" --failure-spec \"${run_root}/failure_spec.xml\" --hosts-config \"${run_root}/hosts.failure.config\" --replica-id ${replica_id} --netns ${ns_name} --interface ${interface_name} --start-unix-ms ${start_unix_ms} --pre-apply-base > \"${run_root}/logs/applier_${replica_id}.log\" 2>&1"
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
    # The replicas must connect over the base delay (see --pre-apply-base), so wait for
    # every applier to have installed it before any replica starts.
    for i in "${!server_nodes[@]}"; do
      if ! remote_bash "${server_users[$i]}" "${server_hosts[$i]}" \
      "${replicas_per_server_machine}" "${i}" "${RUN_ROOT_REMOTE_NAME}" <<'REMOTE'
set -euo pipefail

replicas_per_machine="$1"
machine_idx="$2"
run_root="$HOME/$3"

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  applier_log="${run_root}/logs/applier_${replica_id}.log"
  ready=0
  for attempt in $(seq 1 60); do
    if grep -q "Pre-applied base delay" "${applier_log}" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 0.5
  done
  if (( ready == 0 )); then
    echo "Network applier ${replica_id} did not install the base delay within 30 s; see ${applier_log}:" >&2
    tail -n 5 "${applier_log}" >&2 || true
    exit 1
  fi
done
REMOTE
      then
        echo "  - ${server_nodes[$i]}: base delay not installed" >&2
        exit 1
      fi
    done
    echo "Base delay pre-applied on every replica."
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
        remote_bash "${user}" "${host}" \
          "${replicas_per_server_machine}" \
          "${machine_idx}" \
          "${START_UNIX_MS}" \
          "${MODE}" \
          "${AGENT_MODEL_TYPE}" \
          "${AGENT_PORT}" \
          "${RUN_ROOT_REMOTE_NAME}" \
          "${NS_PREFIX}" \
          "${PROTOCOL_NAME}" \
          "${AGENT_PROTOCOL}" \
          "${SCHEDULE_FILE_REL}" \
          "${FAILURE_SPEC_ENABLED}" \
          "${SHARING_COLLECT_WAIT_MS}" \
          "${REWARD_DESIGN}" \
          "${LATENCY_FLOOR_MULTIPLIER}" \
          "${LEARNING_FEATURE_DURATION_MS}" \
          "${LEARNING_REWARD_DURATION_MS}" \
          "${LEARNING_EPISODE_MS}" \
          "${PRIOR_PATH}" \
          "${TIMEOUT_LADDER}" \
          "${HEURISTIC_K}" \
          "${LATENCY_FLOOR_WINDOW}" <<'REMOTE'
set -euo pipefail

replicas_per_machine="$1"
machine_idx="$2"
start_unix_ms="$3"
mode="$4"
model_type="$5"
agent_port="$6"
run_root="$HOME/$7"
ns_prefix="$8"
protocol="$9"
agent_protocol="${10}"
schedule_file_rel="${11}"
failure_spec_enabled="${12}"
sharing_collect_wait_ms="${13}"
reward_design="${14}"
latency_floor_multiplier="${15}"
feature_window_ms="${16}"
reward_window_ms="${17}"
episode_ms="${18}"
prior_path="${19:-}"
timeout_ladder="${20:-}"
heuristic_k="${21:-}"
latency_floor_window="${22:-}"
learning_agent_python="${HOME}/.venvs/adaptive-timer-learning-agent/bin/python"

if [[ ! -x "${learning_agent_python}" ]]; then
  echo "Learning-agent virtual environment not found on $(hostname): ${learning_agent_python}" >&2
  echo "Run scripts/cloudlab/setup.sh for this target file first." >&2
  exit 1
fi

agent_model_args=""
# Sharing agents talk to each other and to their report-chain node, so they bind
# every interface and read their peers from the hosts config.
bind_args="--host 127.0.0.1 --port ${agent_port}"
if [[ "${mode}" == "learning" ]]; then
  agent_model_args="--model-type ${model_type}"
elif [[ "${mode}" == "sharing" ]]; then
  agent_model_args="--model-type ${model_type}"
  bind_args="--host 0.0.0.0 --sharing true --hosts-config ${run_root}/agent_hosts.config --collect-wait-ms ${sharing_collect_wait_ms}"
elif [[ "${mode}" == "scheduled" ]]; then
  agent_model_args="--model-type scheduled --schedule-file ${schedule_file_rel}"
fi

agent_model_args="${agent_model_args} --reward-design ${reward_design}"
# The floor is a constraint on the action space, so it only means anything where the
# agent is choosing the timeout at all (learning or sharing), as in run_smartbft.sh.
if [[ "${mode}" == "learning" || "${mode}" == "sharing" ]] && [[ "${latency_floor_multiplier}" != "0" ]]; then
  agent_model_args="${agent_model_args} --latency-floor-multiplier ${latency_floor_multiplier}"
fi

# The prior file the model loads, relative to run_root; "" leaves it untrained.
# The model and mode were checked locally.
if [[ -n "${prior_path}" ]]; then
  agent_model_args="${agent_model_args} --prior-path ${prior_path}"
fi
# Checked locally: PBFT discrete models (--timeout-ladder) and continuous_heuristic (--heuristic-k).
if [[ -n "${timeout_ladder}" ]]; then
  agent_model_args="${agent_model_args} --timeout-ladder ${timeout_ladder}"
fi
if [[ -n "${heuristic_k}" ]]; then
  agent_model_args="${agent_model_args} --heuristic-multiplier ${heuristic_k}"
fi
if [[ -n "${latency_floor_window}" ]]; then
  agent_model_args="${agent_model_args} --latency-floor-window ${latency_floor_window}"
fi

# The agent gives a window in which nothing committed its own length as its
# latency, so it needs the windows the replicas measure.
agent_model_args="${agent_model_args} --feature-window-ms ${feature_window_ms} --reward-window-ms ${reward_window_ms}"
# Episode e's schedule row and phase label are taken at its grid apply instant.
agent_model_args="${agent_model_args} --episode-ms ${episode_ms}"

failure_args=""
if [[ "${failure_spec_enabled}" == "1" ]]; then
  failure_args="--failure-spec ${run_root}/failure_spec.xml --injection-start-unix-ms ${start_unix_ms}"
fi

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  ns_name="${ns_prefix}${replica_id}"
  session_name="${protocol}_agent_${replica_id}"
  agent_log="${run_root}/logs/agent_${replica_id}.log"
  rm -f "${run_root}/agent${replica_id}.csv"
  dataset_args="--dataset-csv ${run_root}/agent${replica_id}.csv"
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  tmux new-session -d -s "${session_name}" \
    "sudo -n ip netns exec ${ns_name} bash -lc 'cd \"${run_root}\" && exec \"${learning_agent_python}\" learning_agent/main.py --node-id ${replica_id} --protocol ${agent_protocol} ${bind_args} --signing false ${agent_model_args} ${dataset_args} ${failure_args}' > \"${agent_log}\" 2>&1"
  if [[ "${mode}" == "sharing" ]]; then
    sharing_session="${protocol}_sharing_${replica_id}"
    # The node runs as the login user (not root) so its WAL stays removable here.
    rm -rf "${run_root}/sharing_wal_${replica_id}"
    tmux kill-session -t "${sharing_session}" >/dev/null 2>&1 || true
    tmux new-session -d -s "${sharing_session}" \
      "sudo -n ip netns exec ${ns_name} sudo -n -u $(id -un) env HOME=${HOME} bash -lc 'cd \"${run_root}\" && exec ./bin/smartbft-sharing --node-id $((replica_id + 1)) --protocol ${agent_protocol} --hosts-config \"${run_root}/sharing_hosts.config\" --data-dir \"${run_root}/sharing_wal_${replica_id}\"' > \"${run_root}/logs/sharing_${replica_id}.log\" 2>&1"
  fi
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
  if [[ "${mode}" == "sharing" ]] && ! tmux has-session -t "${protocol}_sharing_${replica_id}" 2>/dev/null; then
    echo "Report-chain session exited early: ${protocol}_sharing_${replica_id}" >&2
    tail -n 80 "${run_root}/logs/sharing_${replica_id}.log" >&2 || true
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
      remote_bash "${user}" "${host}" \
        "${replicas_per_server_machine}" \
        "${machine_idx}" \
        "${RUN_ROOT_REMOTE_NAME}" \
        "${NS_PREFIX}" \
        "${PROTOCOL_NAME}" \
        "${VERBOSE_ENABLED}" \
        "${LEARNING_AGENT_ENABLED}" \
        "${AGENT_PORT}" \
        "${LEARNING_FEATURE_DURATION_MS}" \
        "${LEARNING_REPLY_WAIT_MS}" \
        "${LEARNING_WARMUP_DURATION_MS}" \
        "${LEARNING_REWARD_DURATION_MS}" \
        "${BEDROCK_PROTOCOL_CONFIG_REL}" \
        "${INITIAL_TIMEOUT_MS}" \
        "${INITIAL_SLOW_PATH_TIMEOUT_MS}" \
        "${START_UNIX_MS}" \
        "${FAILURE_SPEC_ENABLED}" \
        "${PROPOSAL_INTERVAL_MS}" "${BATCH_MAX_REQUESTS}" "${BATCH_MAX_BYTES}" \
        "${MAX_INFLIGHT_BATCHES}" "${MAX_PENDING_REQUESTS}" "${MAX_PENDING_BYTES}" \
        "${LEARNING_ANCHOR_UNIX_MS}" <<'REMOTE'
set -euo pipefail

replicas_per_machine="$1"
machine_idx="$2"
run_root="$HOME/$3"
ns_prefix="$4"
protocol="$5"
verbose_enabled="$6"
learning_enabled="$7"
agent_port="$8"
learning_feature_ms="$9"
learning_reply_wait_ms="${10}"
learning_warmup_ms="${11}"
learning_reward_ms="${12}"
protocol_config_rel="${13}"
initial_timeout_ms="${14}"
initial_slow_path_timeout_ms="${15}"
start_unix_ms="${16}"
failure_spec_enabled="${17}"
proposal_interval="${18}"
batch_max_requests="${19}"
batch_max_bytes="${20}"
max_inflight_batches="${21}"
max_pending_requests="${22}"
max_pending_bytes="${23}"
learning_anchor_unix_ms="${24}"

for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  ns_name="${ns_prefix}${replica_id}"
  session_name="${protocol}_server_${replica_id}"
  server_log="${run_root}/logs/server_${replica_id}.log"
  args=(
    "${run_root}/bin/CppBedrock"
    "--node-id" "${replica_id}"
    "--committee" "${run_root}/netcfg/committee.json"
    "--protocol-config" "${run_root}/${protocol_config_rel}"
    "--keys-dir" "${run_root}/netcfg/keys"
    "--proposal-signing" "true"
    "--proposal-interval" "${proposal_interval}"
    "--batch-max-requests" "${batch_max_requests}"
    "--batch-max-bytes" "${batch_max_bytes}"
    "--max-inflight-batches" "${max_inflight_batches}"
    "--max-pending-requests" "${max_pending_requests}"
    "--max-pending-bytes" "${max_pending_bytes}"
  )
  if [[ "${verbose_enabled}" == "1" ]]; then
    args+=("--verbose")
  fi
  if [[ -n "${initial_timeout_ms}" ]]; then
    args+=("--initial-timeout" "${initial_timeout_ms}")
  fi
  if [[ -n "${initial_slow_path_timeout_ms}" ]]; then
    args+=("--initial-slow-path-timeout" "${initial_slow_path_timeout_ms}")
  fi
  if [[ "${failure_spec_enabled}" == "1" ]]; then
    args+=("--failure-spec" "${run_root}/failure_spec.xml" "--failure-start-unix-ms" "${start_unix_ms}")
  fi
  if [[ "${learning_enabled}" == "1" ]]; then
    args+=(
      "--agent-port" "${agent_port}"
      "--learning-anchor-unix-ms" "${learning_anchor_unix_ms}"
      "--learning-feature-duration" "${learning_feature_ms}"
      "--learning-reply-wait" "${learning_reply_wait_ms}"
      "--learning-warmup-duration" "${learning_warmup_ms}"
      "--learning-reward-duration" "${learning_reward_ms}"
    )
  fi
  printf -v server_cmd '%q ' "${args[@]}"
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  : > "${server_log}"
  tmux new-session -d -s "${session_name}" \
    "sudo -n ip netns exec ${ns_name} bash -lc 'exec ${server_cmd}' > \"${server_log}\" 2>&1"
done
sleep 2
for ((n = 0; n < replicas_per_machine; n++)); do
  replica_id=$((machine_idx * replicas_per_machine + n))
  session_name="${protocol}_server_${replica_id}"
  if ! tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "Session exited early: ${session_name}" >&2
    tail -n 100 "${run_root}/logs/server_${replica_id}.log" >&2 || true
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
  remote_bash "${node0_user}" "${node0_host}" \
    "${START_UNIX_MS}" \
    "${TX_RATE}" \
    "${REQUEST_TIMEOUT_MS}" \
    "${MONITOR_INTERVAL_MS}" \
    "${RUN_ROOT_REMOTE_NAME}" \
    "${PROTOCOL_NAME}" \
    "${CLIENT_CONNECTIONS_PER_TARGET}" \
    "${CLIENT_SIGN}" \
    "${REQUEST_SIZE}" <<'REMOTE'
set -euo pipefail

start_unix_ms="$1"
tx_rate="$2"
request_timeout_ms="$3"
monitor_interval_ms="$4"
run_root="$HOME/$5"
protocol="$6"
client_connections_per_target="$7"
client_sign="$8"
request_size="$9"

args=(
  "${run_root}/bin/bedrock_client"
  "--committee" "${run_root}/netcfg/committee.json"
  "--rate" "${tx_rate}"
  "--request-timeout" "${request_timeout_ms}"
  "--monitor-interval" "${monitor_interval_ms}"
  "--start-unix-ms" "${start_unix_ms}"
  "--target-mode" "broadcast"
  "--connections" "${client_connections_per_target}"
  "--sign" "${client_sign}"
  "--client-key" "${run_root}/netcfg/keys/client_private.pem"
  "--request-size" "${request_size}"
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
    if remote_bash "${user}" "${host}" "${PROTOCOL_NAME}" <<'REMOTE'
set -euo pipefail
protocol="$1"
if ! command -v tmux >/dev/null 2>&1; then
  exit 0
fi
sessions="$(tmux ls -F '#S' 2>/dev/null || true)"
[[ -z "${sessions}" ]] && exit 0
# The client first, so it prints its summary while replicas still answer.
printf '%s\n' "${sessions}" | while read -r session_name; do
  [[ -z "${session_name}" ]] && continue
  case "${session_name}" in
    "${protocol}_client")
      tmux send-keys -t "${session_name}" C-c >/dev/null 2>&1 || true
      ;;
  esac
done
sleep 3
printf '%s\n' "${sessions}" | while read -r session_name; do
  [[ -z "${session_name}" ]] && continue
  case "${session_name}" in
    "${protocol}_server_"*|"${protocol}_applier_"*|"${protocol}_agent_"*|"${protocol}_sharing_"*)
      tmux send-keys -t "${session_name}" C-c >/dev/null 2>&1 || true
      ;;
  esac
done
sleep 3
printf '%s\n' "${sessions}" | while read -r session_name; do
  [[ -z "${session_name}" ]] && continue
  case "${session_name}" in
    "${protocol}_server_"*|"${protocol}_applier_"*|"${protocol}_agent_"*|"${protocol}_sharing_"*|"${protocol}_client")
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
    if remote_bash "${user}" "${host}" \
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
    if remote_bash "${node0_user}" "${node0_host}" \
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
    [[ "${MODE}" == "sharing" ]] && mkdir -p "${LOGS_LOCAL_DIR}/sharing"
    # Replica logs are compressed on their host (the replicas are stopped by
    # now) and pulled as server_<id>.log.gz; a missing log is left to the
    # download below to report, as before.
    for i in "${!server_nodes[@]}"; do
      if ! remote_bash "${server_users[$i]}" "${server_hosts[$i]}" "${RUN_ROOT_REMOTE_NAME}" <<'REMOTE'
set -euo pipefail
cd "$HOME/$1/logs"
for f in server_*.log; do
  [[ -f "${f}" ]] || continue
  gzip -f "${f}"
done
REMOTE
      then
        echo "  - ${server_nodes[$i]}: failed to compress server logs" >&2
      fi
    done
    for i in "${!server_nodes[@]}"; do
      node="${server_nodes[$i]}"
      host="${server_hosts[$i]}"
      user="${server_users[$i]}"
      machine_idx="${i}"
      for ((n = 0; n < replicas_per_server_machine; n++)); do
        replica_id=$((machine_idx * replicas_per_server_machine + n))
        queue_artifact_download "${node}" "${host}" "${user}" \
          "${RUN_ROOT_REMOTE_NAME}/logs/server_${replica_id}.log.gz" \
          "${LOGS_LOCAL_DIR}/servers/server_${replica_id}.log.gz" \
          "server_${replica_id}.log.gz" \
          "log"
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
        if [[ "${MODE}" == "sharing" ]]; then
          queue_artifact_download "${node}" "${host}" "${user}" \
            "${RUN_ROOT_REMOTE_NAME}/logs/sharing_${replica_id}.log" \
            "${LOGS_LOCAL_DIR}/sharing/sharing_${replica_id}.log" \
            "sharing_${replica_id}.log" \
            "log" \
            1
        fi
      done
    done
  fi

  if (( LEARNING_AGENT_ENABLED == 1 )); then
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
  if (( LEARNING_AGENT_ENABLED == 1 )); then
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

bedrock_main "$@"
