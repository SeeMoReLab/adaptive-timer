#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INVOCATION_DIR="$(pwd -P)"
ORIGINAL_ARGS=("$@")

TARGETS_FILE_DEFAULT="${REPO_ROOT}/scripts/cloudlab/tmp/cloudlab_ssh_targets.txt"
TARGETS_FILE="${TARGETS_FILE_DEFAULT}"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
MODE="normal"
F=1
EXPERIMENT_TIME_S=60
LOG_ENABLED=0
DATASET_ENABLED=0
AGENT_MODEL_TYPE="discrete_random"
AGENT_MODEL_SET=0
REWARD_DESIGN="reward_tps"
# Domain constraint on the action space: never apply a timeout below k times
# the consensus latency the replicas are already achieving. 0 disables it.
LATENCY_FLOOR_K=0
# --model-type=continuous_pretrained_gp prior: "" the shipped one, "none" untrained, else a path.
PRIOR_PATH=""
SCHEDULE_FILE="learning_agent/config/schedule_tendermint.csv"
LOAD_CONNECTIONS=5
LOAD_RATE=20
LOAD_SEND_PERIOD_MS=100
LOG_FETCH_MAX_PARALLEL=4
LOG_FETCH_MAX_ATTEMPTS=3
LOG_FETCH_RETRY_DELAY_S=1

LOCAL_TMP_DIR="${REPO_ROOT}/scripts/cloudlab/tmp"
RUNS_ROOT="${LOCAL_TMP_DIR}/runs/tendermint"
OUTPUT_DIR=""
APPLIER_LOCAL_PATH="${REPO_ROOT}/scripts/apply_network_failures.py"
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
    echo "Cannot start Tendermint run: ${description} is already in use." >&2
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
Usage: $0 <private_key_path> [normal|learning|sharing|scheduled] [--dataset] [--log] [--output-dir <path>] [--model <discrete_random|discrete_pavlov|discrete_linear|discrete_tree_ts|discrete_tree_blocks|discrete_linucb|discrete_ppo|continuous_pretrained_gp|continuous_linucb|continuous_heuristic|continuous_ppo|continuous_latency_floor>] [-f <fault_tolerance>] [-t <experiment_seconds>] [targets_file]

Arguments:
  private_key_path  Local SSH private key for CloudLab nodes.
  mode              Optional; one of normal|learning|sharing|scheduled (default: normal).
  targets_file      Optional path to cloudlab SSH targets file.
                    One machine runs everything; with more, node0 runs the load client and
                    the others host an equal share of the replicas, as in the other harnesses.

Flags:
  --dataset         Collect per-replica learning-agent dataset CSVs as agentN.csv.
  --log             Fetch remote Tendermint, ABCI, and client logs into the run artifact directory.
  --output-dir <path>
                    Store artifacts here instead of creating a timestamped run directory.
  --model           Learning-agent model for learning/sharing modes: discrete_random|discrete_pavlov|discrete_linear|discrete_tree_ts|discrete_tree_blocks|discrete_linucb|discrete_ppo|continuous_pretrained_gp|continuous_linucb|continuous_heuristic|continuous_ppo|continuous_latency_floor (default: discrete_random).
  --reward-design   Reward the agent learns from: reward_tps|reward_mixed|reward_tps_latency (default: reward_tps).
                    reward_tps is throughput alone, concave past a 1000 tps knee. reward_mixed
                    also discounts it by the share of the window one fault would cost.
  --latency-floor-k <float>
                    Never apply a timeout below k * the consensus latency the replicas are
                    already achieving, estimated as a low quantile over recent committing
                    windows. 0 disables it (default: 0).
  --prior-path <path|none>           Which pretrained prior --model continuous_pretrained_gp loads.
                                     Empty (default) is the protocol's shipped
                                     return prior; a path is an ablation prior
                                     (<protocol>_sim.py train --no-proposal-faults);
                                     "none" runs it untrained, with a zero prior.
  --schedule-file <repo-rel-path>
                    scheduled-mode oracle CSV, relative to the repo root
                    (default: learning_agent/config/schedule_tendermint.csv). It must be inside
                    the deployed payload to exist on the remote.
  -f <int>          Fault tolerance f. Number of nodes is derived as 3f+1 (default: 1).
  -t <int>          Load-generation duration in seconds (default: 60).
EOF
}

mode_enables_learning_agent() {
  case "$1" in
    learning|sharing|scheduled)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

# Model types retired on 2026-10-07 (RENAMED_MODEL_TYPES in
# learning_agent/models.py): prints the replacement, or fails for any other name.
renamed_model_type() {
  case "$1" in
    random|pavlov|linear|ppo) printf 'discrete_%s\n' "$1" ;;
    cmab|thompson) printf '%s\n' "discrete_tree_ts" ;;
    cmab_reset) printf '%s\n' "discrete_tree_blocks" ;;
    codex|codex_v1) printf '%s\n' "discrete_linucb" ;;
    continuous) printf '%s\n' "continuous_pretrained_gp" ;;
    continuous_codex) printf '%s\n' "continuous_linucb" ;;
    latency_floor_heuristic) printf '%s\n' "continuous_latency_floor" ;;
    *) return 1 ;;
  esac
}

normalize_agent_model_type() {
  local model_lower
  model_lower="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  case "${model_lower}" in
    discrete_random|discrete_pavlov|discrete_linear|discrete_tree_ts|discrete_tree_blocks|discrete_linucb|discrete_ppo|continuous_pretrained_gp|continuous_linucb|continuous_heuristic|continuous_ppo|continuous_latency_floor)
      printf '%s\n' "${model_lower}"
      ;;
    *)
      local renamed
      if renamed="$(renamed_model_type "${model_lower}")"; then
        echo "--model ${model_lower} was renamed to ${renamed}." >&2
      fi
      return 1
      ;;
  esac
}

# Fetch the run's logs into the layout the other protocols' harnesses use:
# client.log in the run directory, replica processes under servers/, then
# appliers/, agents/ and sharing/. Each replica's logs come from the machine
# hosting it, client.log from the client machine.
fetch_tendermint_logs_parallel() {
  local local_logs_dir="$1"
  local i idx machine src failed=0
  local sources=("${client_user}@${client_host}")
  local remote_files=("client.log")
  local local_paths=("$(dirname "${local_logs_dir}")/client.log")
  local pids=()

  mkdir -p "${local_logs_dir}/servers" "${local_logs_dir}/appliers" "${local_logs_dir}/agents"
  if [[ "${MODE}" == "sharing" ]]; then
    mkdir -p "${local_logs_dir}/sharing"
  fi
  for ((i = 0; i < LOCAL_NUM_NODES; i++)); do
    machine=$((i / replicas_per_machine))
    src="${server_users[$machine]}@${server_hosts[$machine]}"
    sources+=("${src}" "${src}" "${src}")
    remote_files+=("node_${i}.log" "abci_${i}.log" "applier_${i}.log")
    local_paths+=(
      "${local_logs_dir}/servers/server_${i}.log"
      "${local_logs_dir}/servers/abci_${i}.log"
      "${local_logs_dir}/appliers/applier_${i}.log"
    )
    if (( LEARNING_AGENT_ENABLED == 1 )); then
      sources+=("${src}")
      remote_files+=("agent_${i}.log")
      local_paths+=("${local_logs_dir}/agents/agent_${i}.log")
    fi
    if [[ "${MODE}" == "sharing" ]]; then
      sources+=("${src}")
      remote_files+=("sharing_${i}.log")
      local_paths+=("${local_logs_dir}/sharing/sharing_${i}.log")
    fi
  done

  for idx in "${!remote_files[@]}"; do
    while (( $(jobs -pr | wc -l | tr -d '[:space:]') >= LOG_FETCH_MAX_PARALLEL )); do
      sleep 0.05
    done

    (
      local attempt
      for ((attempt = 1; attempt <= LOG_FETCH_MAX_ATTEMPTS; attempt++)); do
        rm -f "${local_paths[$idx]}"
        if scp -q "${SSH_OPTS[@]}" \
          "${sources[$idx]}:tendermint_cloudlab/logs/${remote_files[$idx]}" \
          "${local_paths[$idx]}"; then
          exit 0
        fi
        if (( attempt < LOG_FETCH_MAX_ATTEMPTS )); then
          sleep $((LOG_FETCH_RETRY_DELAY_S * attempt))
        fi
      done
      exit 1
    ) &
    pids+=("$!")
  done

  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "Missing or failed to fetch ${remote_files[$idx]} from ${sources[$idx]}." >&2
      failed=1
    fi
  done

  return "${failed}"
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
    --dataset)
      DATASET_ENABLED=1
      shift
      ;;
    --output-dir)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --output-dir." >&2
        usage >&2
        exit 2
      fi
      OUTPUT_DIR="${2}"
      shift 2
      ;;
    --reward-design)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --reward-design." >&2
        usage >&2
        exit 2
      fi
      REWARD_DESIGN="${2}"
      shift 2
      ;;
    --latency-floor-k)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --latency-floor-k." >&2
        usage >&2
        exit 2
      fi
      LATENCY_FLOOR_K="${2}"
      shift 2
      ;;
    --prior-path)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --prior-path." >&2
        exit 2
      fi
      PRIOR_PATH="${2}"
      shift 2
      ;;
    --schedule-file)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for --schedule-file." >&2
        usage >&2
        exit 2
      fi
      SCHEDULE_FILE="${2}"
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

if ! [[ "${F}" =~ ^[0-9]+$ ]] || (( F < 1 )); then
  echo "Invalid -f value: ${F}. It must be an integer >= 1." >&2
  exit 2
fi
if ! [[ "${EXPERIMENT_TIME_S}" =~ ^[0-9]+$ ]] || (( EXPERIMENT_TIME_S < 1 )); then
  echo "Invalid -t value: ${EXPERIMENT_TIME_S}. It must be an integer >= 1." >&2
  exit 2
fi
if [[ "${MODE}" != "normal" && "${MODE}" != "learning" && "${MODE}" != "sharing" && "${MODE}" != "scheduled" ]]; then
  echo "Mode '${MODE}' is not implemented. Supported modes: normal, learning, sharing, scheduled." >&2
  exit 2
fi
if ! mode_enables_learning_agent "${MODE}"; then
  LEARNING_AGENT_ENABLED=0
else
  LEARNING_AGENT_ENABLED=1
fi
case "${REWARD_DESIGN}" in
  reward_tps|reward_mixed|reward_tps_latency) ;;
  *) echo "Invalid --reward-design: ${REWARD_DESIGN}. Expected reward_tps|reward_mixed|reward_tps_latency." >&2; exit 2 ;;
esac
if ! [[ "${LATENCY_FLOOR_K}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Invalid --latency-floor-k: ${LATENCY_FLOOR_K}. Expected a non-negative number (0 disables the constraint)." >&2
  exit 2
fi
if ! AGENT_MODEL_TYPE="$(normalize_agent_model_type "${AGENT_MODEL_TYPE}")"; then
  echo "Invalid --model value: ${AGENT_MODEL_TYPE}. Use one of discrete_random|discrete_pavlov|discrete_linear|discrete_tree_ts|discrete_tree_blocks|discrete_linucb|discrete_ppo|continuous_pretrained_gp|continuous_linucb|continuous_heuristic|continuous_ppo|continuous_latency_floor." >&2
  exit 2
fi
if (( AGENT_MODEL_SET == 1 )) && [[ "${MODE}" != "learning" && "${MODE}" != "sharing" ]]; then
  echo "--model is only supported with learning or sharing mode." >&2
  exit 2
fi
if (( DATASET_ENABLED == 1 )) && (( LEARNING_AGENT_ENABLED == 0 )); then
  echo "--dataset requires a learning-agent mode: learning|sharing|scheduled." >&2
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

if [[ ! -d "${REPO_ROOT}/cometbft" ]]; then
  echo "Local cometbft directory not found: ${REPO_ROOT}/cometbft" >&2
  exit 1
fi
if [[ ! -f "${REPO_ROOT}/config/tendermint/config.toml" ]]; then
  echo "Master Tendermint config not found: ${REPO_ROOT}/config/tendermint/config.toml" >&2
  exit 1
fi
if [[ ! -f "${REPO_ROOT}/config/tendermint/failure_spec.xml" ]]; then
  echo "Tendermint failure spec not found: ${REPO_ROOT}/config/tendermint/failure_spec.xml" >&2
  exit 1
fi
if [[ ! -f "${APPLIER_LOCAL_PATH}" ]]; then
  echo "Network applier script not found: ${APPLIER_LOCAL_PATH}" >&2
  exit 1
fi
if (( LEARNING_AGENT_ENABLED == 1 )); then
  if [[ ! -d "${REPO_ROOT}/learning_agent" ]]; then
    echo "Local learning_agent directory not found: ${REPO_ROOT}/learning_agent" >&2
    exit 1
  fi
fi
if [[ "${MODE}" == "scheduled" ]]; then
  if [[ ! -f "${REPO_ROOT}/${SCHEDULE_FILE}" ]]; then
    echo "Missing Tendermint schedule file: ${REPO_ROOT}/${SCHEDULE_FILE}" >&2
    exit 1
  fi
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

# Machine roles, as in the other protocols' harnesses: a single machine runs
# everything; with more, node0 runs the load client and every other machine
# hosts an equal share of the replicas, replica i on server machine
# i / replicas_per_machine.
machine_count=${#targets_nodes[@]}
client_host=""
client_user=""
for i in "${!targets_nodes[@]}"; do
  if [[ "${targets_nodes[$i]}" == "node0" ]]; then
    client_host="${targets_hosts[$i]}"
    client_user="${targets_users[$i]}"
  fi
done
if [[ -z "${client_host}" || -z "${client_user}" ]]; then
  echo "node0 entry not found in ${TARGETS_FILE}" >&2
  exit 1
fi
server_names=()
server_hosts=()
server_users=()
for i in "${!targets_nodes[@]}"; do
  if (( machine_count == 1 )) || [[ "${targets_nodes[$i]}" != "node0" ]]; then
    server_names+=("${targets_nodes[$i]}")
    server_hosts+=("${targets_hosts[$i]}")
    server_users+=("${targets_users[$i]}")
  fi
done
server_machine_count=${#server_hosts[@]}
LOCAL_NUM_NODES=$((F * 3 + 1))
if (( LOCAL_NUM_NODES % server_machine_count != 0 )); then
  echo "Replica layout mismatch: nodes=${LOCAL_NUM_NODES} not divisible by server machine count=${server_machine_count}" >&2
  exit 1
fi
replicas_per_machine=$((LOCAL_NUM_NODES / server_machine_count))
for i in "${!targets_nodes[@]}"; do
  acquire_target_lock "${targets_users[$i]}" "${targets_hosts[$i]}"
done

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

OUTPUT_LOCAL="${ARTIFACT_DIR}/tendermint_output.txt"
LOGS_LOCAL_DIR="${ARTIFACT_DIR}/logs"
LOCAL_PAYLOAD="${ARTIFACT_DIR}/tendermint_cloudlab_payload.tar.gz"
rm -f "${LOCAL_PAYLOAD}" "${OUTPUT_LOCAL}"
: > "${OUTPUT_LOCAL}"
if (( LOG_ENABLED == 1 )); then
  rm -rf "${LOGS_LOCAL_DIR}"
  mkdir -p "${LOGS_LOCAL_DIR}"
fi

cp "${REPO_ROOT}/config/tendermint/config.toml" "${ARTIFACT_DIR}/config.toml"
cp "${REPO_ROOT}/config/tendermint/failure_spec.xml" "${ARTIFACT_DIR}/failure_spec.xml"
cp "${TARGETS_FILE}" "${ARTIFACT_DIR}/cloudlab_ssh_targets.txt"
if [[ "${MODE}" == "scheduled" ]]; then
  cp "${REPO_ROOT}/${SCHEDULE_FILE}" "${ARTIFACT_DIR}/schedule_tendermint.csv"
fi
{
  printf 'protocol=%q\n' "tendermint"
  printf 'run_id=%q\n' "${RUN_ID}"
  printf 'artifact_dir=%q\n' "${ARTIFACT_DIR}"
  printf 'mode=%q\n' "${MODE}"
  printf 'fault_tolerance=%q\n' "${F}"
  printf 'experiment_time_s=%q\n' "${EXPERIMENT_TIME_S}"
  printf 'agent_model=%q\n' "${AGENT_MODEL_TYPE}"
  printf 'reward_design=%q\n' "${REWARD_DESIGN}"
  printf 'latency_floor_k=%q\n' "${LATENCY_FLOOR_K}"
  printf 'prior_path=%q\n' "${PRIOR_PATH:-}"
  printf 'schedule_file=%q\n' "${SCHEDULE_FILE}"
  printf 'dataset_enabled=%q\n' "${DATASET_ENABLED}"
  printf 'log_enabled=%q\n' "${LOG_ENABLED}"
  printf 'targets_file=%q\n' "${TARGETS_FILE}"
  printf 'server_machines=%q\n' "${server_names[*]}"
  printf 'replicas_per_machine=%q\n' "${replicas_per_machine}"
  printf 'command='
  printf '%q ' "$0" "${ORIGINAL_ARGS[@]}"
  printf '\n'
} > "${ARTIFACT_DIR}/run.env"

echo "Run artifacts: ${ARTIFACT_DIR}"

SHARING_PAYLOAD_PATHS=""
if [[ "${MODE}" == "sharing" ]]; then
  # The report chain is a SmartBFT binary built on the remote node.
  SHARING_PAYLOAD_PATHS="SmartBFT"
fi

echo "Creating CometBFT payload tarball..."
if (( LEARNING_AGENT_ENABLED == 1 )); then
  COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 tar -C "${REPO_ROOT}" -czf "${LOCAL_PAYLOAD}" \
    --exclude='cometbft/mytestnet' \
    --exclude='cometbft/logs' \
    --exclude='cometbft/build' \
    --exclude='learning_agent/__pycache__' \
    --exclude='SmartBFT/.git' \
    --exclude='SmartBFT/build' \
    cometbft \
    config/tendermint \
    learning_agent \
    ${SHARING_PAYLOAD_PATHS} \
    scripts/tendermint_phase_report.py
else
  COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 tar -C "${REPO_ROOT}" -czf "${LOCAL_PAYLOAD}" \
    --exclude='cometbft/mytestnet' \
    --exclude='cometbft/logs' \
    --exclude='cometbft/build' \
    cometbft \
    config/tendermint \
    scripts/tendermint_phase_report.py
fi

REMOTE_PAYLOAD="/tmp/tendermint_cloudlab_payload_$$.tar.gz"
REMOTE_APPLIER="/tmp/tendermint_apply_network_failures_$$.py"
REMOTE_TESTNET="tendermint_testnet_$$.tar.gz"

is_ipv4_literal() {
  [[ "$1" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]
}

# Server machines reach each other's replica subnets through these, as in
# lib_bedrock.sh: the first address of `hostname -I` on each server machine.
GATEWAY_IPS_CSV="-"
if (( machine_count > 1 )); then
  echo "Resolving server gateway IPs..."
  gateway_ips=()
  for m in "${!server_hosts[@]}"; do
    if is_ipv4_literal "${server_hosts[$m]}"; then
      gateway_ip="${server_hosts[$m]}"
    else
      gateway_ip="$(ssh "${SSH_OPTS[@]}" "${server_users[$m]}@${server_hosts[$m]}" "hostname -I | awk '{print \$1}'" 2>/dev/null || true)"
    fi
    if ! is_ipv4_literal "${gateway_ip}"; then
      echo "Failed to resolve gateway IP for ${server_names[$m]} (${server_hosts[$m]})." >&2
      exit 1
    fi
    gateway_ips+=("${gateway_ip}")
    echo "  - ${server_names[$m]}: ${gateway_ip}"
  done
  GATEWAY_IPS_CSV="$(IFS=,; echo "${gateway_ips[*]}")"
fi
# The client machine resolves each server machine by its node name first, as
# lib_bedrock.sh's client routes do, then by its host name.
SERVER_ROUTE_NAMES_CSV="$(IFS=,; echo "${server_names[*]}")"
SERVER_ROUTE_HOSTS_CSV="$(IFS=,; echo "${server_hosts[*]}")"
START_UNIX_MS=0
# Seconds from choosing the shared start to the benchmark start: covers the
# replicas' launch, their RPC readiness and peering, and the client's launch.
START_DELAY_S=45

# The remote side, one phase per invocation:
#   prepare         every machine: unpack the payload, build
#   testnet         server machine 0: generate every replica's home, tar it
#   install_testnet other server machines: unpack that tar
#   start           every server machine: namespaces, routes, agents, nodes, appliers
#   client          client machine: routes, wait for peering, run the load
#   stop            every server machine: stop the replicas
#   report          server machine 0: latency, throughput and phase reports on stdout
#   teardown        every machine: namespaces, bridge, routes
IFS= read -r -d '' REMOTE_SCRIPT <<'REMOTE' || true
set -euo pipefail

phase="$1"
payload_path="$2"
applier_path="$3"
fault_tolerance="$4"
experiment_time_s="$5"
mode="$6"
log_enabled="$7"
dataset_enabled="$8"
load_connections="$9"
load_rate="${10}"
load_send_period_ms="${11}"
agent_model_type="${12}"
reward_design="${13}"
latency_floor_k="${14}"
schedule_file_rel="${15}"
# Index among the server machines, or -1 on a client-only machine.
machine_idx="${16}"
is_client="${17}"
server_machine_count="${18}"
replicas_per_machine="${19}"
start_unix_ms="${20}"
gateway_ips_csv="${21}"
server_route_names_csv="${22}"
server_route_hosts_csv="${23}"
testnet_tar="$HOME/${24}"
prior_path="${25:-}"

num_nodes=$((fault_tolerance * 3 + 1))
learning_agent_enabled=0
if [[ "${mode}" == "learning" || "${mode}" == "sharing" || "${mode}" == "scheduled" ]]; then
  learning_agent_enabled=1
fi
IFS=, read -r -a gateway_ips <<< "${gateway_ips_csv}"
IFS=, read -r -a server_route_names <<< "${server_route_names_csv}"
IFS=, read -r -a server_route_hosts <<< "${server_route_hosts_csv}"

run_root="$HOME/tendermint_cloudlab"
src_dir="${run_root}/cometbft"
learning_agent_dir="${run_root}/learning_agent"
master_config="${run_root}/config/tendermint/config.toml"
testnet_dir="${run_root}/mytestnet"
logs_dir="${run_root}/logs"
output_file="$HOME/tendermint_output.txt"
results_csv="$HOME/results.csv"
client_log="${logs_dir}/client.log"
agent_hosts_config="${run_root}/agent_hosts.config"
sharing_hosts_config="${run_root}/sharing_hosts.config"
sharing_src_dir="${run_root}/SmartBFT"
sharing_bin="${run_root}/bin/smartbft-sharing"
schedule_file="${run_root}/${schedule_file_rel}"
applier_hosts_config="${run_root}/applier_hosts.config"
applier_script="${run_root}/apply_network_failures.py"
cometbft_bin="${src_dir}/build/cometbft"

load_size=500
post_peer_connect_delay=5
buildvcs_flag="-buildvcs=false"
failure_spec="${run_root}/config/tendermint/failure_spec.xml"
report_state_log="${logs_dir}/node_0.log"
overlay_a=10
overlay_b=251
overlay_mask=24
overlay_host_offset=10
overlay_bridge_name="br-tendermint"
max_overlay_nodes=$((255 - overlay_host_offset))
run_as_user="$(id -un)"
run_as_home="${HOME}"
learning_agent_python="${run_as_home}/.venvs/adaptive-timer-learning-agent/bin/python"
export PATH="$HOME/go/bin:${PATH}"

p2p_port() { echo $((26656 + $1 * 3)); }
rpc_port() { echo $((26657 + $1 * 3)); }
abci_port() { echo $((26658 + $1 * 3)); }
agent_port() { echo $((50000 + $1)); }
sharing_report_port() { echo $((50200 + $1)); }
sharing_consensus_port() { echo $((50400 + $1)); }
# Replica i lives on server machine i / replicas_per_machine, whose replicas
# share the subnet a.b.(machine + 1).0/24; one machine gives a.b.1.(offset + i).
node_ip() { echo "${overlay_a}.${overlay_b}.$(($1 / replicas_per_machine + 1)).$((overlay_host_offset + $1 % replicas_per_machine))"; }
# What `cometbft testnet --starting-ip-address` assigns: consecutive addresses.
testnet_ip() { echo "${overlay_a}.${overlay_b}.1.$((overlay_host_offset + $1))"; }
machine_subnet() { echo "${overlay_a}.${overlay_b}.$(($1 + 1)).0/${overlay_mask}"; }
ns_name() { echo "tm-r$1"; }
ns_host_if() { echo "tm-vh$1"; }
ns_inner_if() { echo "tm-vn$1"; }
first_local_replica=$((machine_idx * replicas_per_machine))
last_local_replica=$((first_local_replica + replicas_per_machine - 1))
load_endpoints_csv() {
  local endpoints=()
  local i
  for ((i = 0; i < num_nodes; i++)); do
    endpoints+=("ws://$(node_ip "${i}"):$(rpc_port "${i}")/websocket")
  done
  local IFS=,
  echo "${endpoints[*]}"
}

write_applier_hosts_config() {
  : > "${applier_hosts_config}"
  local i
  for ((i = 0; i < num_nodes; i++)); do
    echo "${i} $(node_ip "${i}")" >> "${applier_hosts_config}"
  done
}

apply_master_config() {
  local master_cfg="$1"
  local node_cfg="$2"
  python3 - "$master_cfg" "$node_cfg" <<'PY'
import re
import sys

master_path = sys.argv[1]
node_path = sys.argv[2]

section_re = re.compile(r'^\s*\[([^\]]+)\]\s*$')
assign_re = re.compile(r'^([A-Za-z0-9_.-]+)\s*=\s*(.*)$')
line_assign_re = re.compile(r'^(\s*)([A-Za-z0-9_.-]+)(\s*=\s*)(.*?)(\s*(?:#.*)?)\n?$')

def collect_assignments(path):
    assignments = {}
    section = ""
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            sm = section_re.match(line)
            if sm:
                section = sm.group(1).strip()
                continue
            am = assign_re.match(line)
            if am:
                key = am.group(1)
                value = am.group(2)
                assignments[(section, key)] = value
    return assignments

master_values = collect_assignments(master_path)

out_lines = []
section = ""
with open(node_path, "r", encoding="utf-8") as fh:
    for raw in fh:
        stripped = raw.strip()
        sm = section_re.match(stripped)
        if sm:
            section = sm.group(1).strip()
            out_lines.append(raw)
            continue

        lm = line_assign_re.match(raw)
        if not lm:
            out_lines.append(raw)
            continue

        indent, key, equals, _, trailing = lm.groups()
        replacement = master_values.get((section, key))
        if replacement is None:
            out_lines.append(raw)
            continue
        out_lines.append(f"{indent}{key}{equals}{replacement}{trailing}\n")

with open(node_path, "w", encoding="utf-8") as fh:
    fh.writelines(out_lines)
PY
}

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
    if [[ "${session_name}" == tm_tendermint_node_* || "${session_name}" == tm_abci_* || "${session_name}" == tm_tendermint_agent_* || "${session_name}" == tm_applier_* || "${session_name}" == tm_sharing_* ]]; then
      tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
    fi
  done <<< "${sessions}"
}

collect_run_pids() {
  ps ax -o pid= -o command= | awk -v root="${testnet_dir}" '
    /cometbft node/ && index($0, root "/node") { print $1 }
  '
  for ((k = 0; k < num_nodes; k++)); do
    local port
    port="$(abci_port "${k}")"
    ps ax -o pid= -o command= | awk -v addr="tcp://127.0.0.1:""${port}" '
      /abci-cli kvstore/ && index($0, addr) { print $1 }
    '
  done
}

force_stop_run_processes() {
  local pids
  pids="$(collect_run_pids | awk '!seen[$1]++')"
  if [[ -z "${pids}" ]]; then
    return 0
  fi

  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    kill "${pid}" >/dev/null 2>&1 || true
  done <<< "${pids}"

  for _ in $(seq 1 6); do
    sleep 1
    pids="$(collect_run_pids | awk '!seen[$1]++')"
    [[ -z "${pids}" ]] && return 0
  done

  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    kill -9 "${pid}" >/dev/null 2>&1 || true
  done <<< "${pids}"
}

require_sudo() {
  if ! sudo -n true >/dev/null 2>&1; then
    echo "Namespace and route setup requires passwordless sudo on $(hostname)." >&2
    echo "Verify with: ssh <node> 'sudo -n true'" >&2
    return 1
  fi
}

teardown_network_namespaces() {
  local existing_ns
  existing_ns="$(ip netns list | awk '/^tm-r[0-9]+/ {print $1}')"
  if [[ -n "${existing_ns}" ]]; then
    printf '%s\n' "${existing_ns}" | while read -r name; do
      [[ -z "${name}" ]] && continue
      sudo -n ip netns del "${name}" >/dev/null 2>&1 || true
    done
  fi

  local stale_veths
  stale_veths="$(ip -o link show | awk -F': ' '{print $2}' | sed 's/@.*//' | awk '/^tm-vh[0-9]+$/')"
  if [[ -n "${stale_veths}" ]]; then
    printf '%s\n' "${stale_veths}" | while read -r if_name; do
      [[ -z "${if_name}" ]] && continue
      sudo -n ip link del "${if_name}" >/dev/null 2>&1 || true
    done
  fi

  sudo -n ip link set "${overlay_bridge_name}" down >/dev/null 2>&1 || true
  sudo -n ip link del "${overlay_bridge_name}" type bridge >/dev/null 2>&1 || true
}

# Routes to the other server machines' replica subnets (never to this one's).
delete_peer_routes() {
  local peer
  for ((peer = 0; peer < server_machine_count; peer++)); do
    (( peer == machine_idx )) && continue
    sudo -n ip route del "$(machine_subnet "${peer}")" >/dev/null 2>&1 || true
  done
}

setup_network_namespaces() {
  require_sudo
  teardown_network_namespaces

  local bridge_ip="${overlay_a}.${overlay_b}.$((machine_idx + 1)).1"
  sudo -n ip link add name "${overlay_bridge_name}" type bridge
  sudo -n ip addr replace "${bridge_ip}/${overlay_mask}" dev "${overlay_bridge_name}"
  sudo -n ip link set "${overlay_bridge_name}" up
  sudo -n sysctl -w net.ipv4.ip_forward=1 >/dev/null

  local i ns host_if inner_if ip
  for ((i = first_local_replica; i <= last_local_replica; i++)); do
    ns="$(ns_name "${i}")"
    host_if="$(ns_host_if "${i}")"
    inner_if="$(ns_inner_if "${i}")"
    ip="$(node_ip "${i}")"

    sudo -n ip netns add "${ns}"
    sudo -n ip link add "${host_if}" type veth peer name "${inner_if}"
    sudo -n ip link set "${inner_if}" netns "${ns}"
    sudo -n ip link set "${host_if}" master "${overlay_bridge_name}"
    sudo -n ip link set "${host_if}" up
    sudo -n ip netns exec "${ns}" ip link set lo up
    sudo -n ip netns exec "${ns}" ip addr replace "${ip}/${overlay_mask}" dev "${inner_if}"
    sudo -n ip netns exec "${ns}" ip link set "${inner_if}" up
    sudo -n ip netns exec "${ns}" ip route replace default via "${bridge_ip}"
  done

  local peer
  for ((peer = 0; peer < server_machine_count; peer++)); do
    (( peer == machine_idx )) && continue
    sudo -n ip route replace "$(machine_subnet "${peer}")" via "${gateway_ips[$peer]}"
  done
}

# A client-only machine reaches every server machine's replica subnet, resolving
# each server machine as lib_bedrock.sh's client routes do.
setup_client_routes() {
  require_sudo
  local peer gw_ip route_get route_dev
  for ((peer = 0; peer < server_machine_count; peer++)); do
    gw_ip="$(getent ahostsv4 "${server_route_names[$peer]}" 2>/dev/null | awk 'NR==1 {print $1}' || true)"
    if ! [[ "${gw_ip}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
      gw_ip="$(getent ahostsv4 "${server_route_hosts[$peer]}" 2>/dev/null | awk 'NR==1 {print $1}' || true)"
    fi
    if ! [[ "${gw_ip}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
      echo "Failed to resolve a gateway for ${server_route_names[$peer]}/${server_route_hosts[$peer]}" >&2
      return 1
    fi
    if ! sudo -n ip route replace "$(machine_subnet "${peer}")" via "${gw_ip}"; then
      route_get="$(ip -4 route get "${gw_ip}" 2>/dev/null | head -n1 || true)"
      route_dev="$(awk '{for (i = 1; i <= NF; i++) if ($i == "dev") { print $(i + 1); exit }}' <<< "${route_get}")"
      if [[ -z "${route_dev}" ]] || ! sudo -n ip route replace "$(machine_subnet "${peer}")" via "${gw_ip}" dev "${route_dev}" onlink; then
        echo "Failed route replace: subnet=$(machine_subnet "${peer}") gw=${gw_ip}" >&2
        return 1
      fi
    fi
  done
}

start_network_appliers() {
  local i ns iface session applier_log
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found; cannot start network appliers." >&2
    return 1
  fi
  if [[ ! -f "${applier_script}" ]]; then
    echo "Missing network applier script: ${applier_script}" >&2
    return 1
  fi
  if [[ ! -f "${applier_hosts_config}" ]]; then
    echo "Missing applier hosts config: ${applier_hosts_config}" >&2
    return 1
  fi

  for ((i = first_local_replica; i <= last_local_replica; i++)); do
    ns="$(ns_name "${i}")"
    iface="$(ns_inner_if "${i}")"
    session="tm_applier_${i}"
    applier_log="${logs_dir}/applier_${i}.log"
    tmux kill-session -t "${session}" >/dev/null 2>&1 || true
    tmux new-session -d -s "${session}" \
      "cd \"${run_root}\" && python3 \"${applier_script}\" --failure-spec \"${failure_spec}\" --hosts-config \"${applier_hosts_config}\" --replica-id ${i} --netns ${ns} --interface ${iface} --start-unix-ms ${start_unix_ms} > \"${applier_log}\" 2>&1"
  done
}

run_report_with_retry() {
  local max_attempts=30
  local attempt=1
  local tmp_out=""
  while true; do
    tmp_out="$(mktemp)"
    if report "$@" >"${tmp_out}" 2>&1; then
      cat "${tmp_out}"
      rm -f "${tmp_out}"
      return 0
    fi

    if grep -q "resource temporarily unavailable" "${tmp_out}"; then
      if (( attempt == 1 )); then
        echo "Report DB is still locked; waiting for node shutdown to complete..."
      fi
      rm -f "${tmp_out}"
      if (( attempt >= max_attempts )); then
        echo "Report failed after ${max_attempts} attempts waiting for DB lock release."
        return 1
      fi
      sleep 1
      attempt=$((attempt + 1))
      continue
    fi

    cat "${tmp_out}"
    rm -f "${tmp_out}"
    return 1
  done
}

stop_progress_reporter() {
  if [[ -z "${progress_reporter_pid:-}" ]]; then
    return
  fi
  kill "${progress_reporter_pid}" >/dev/null 2>&1 || true
  wait "${progress_reporter_pid}" >/dev/null 2>&1 || true
  progress_reporter_pid=""
}

report_time_remaining() {
  local end_epoch_s="$1"
  local now_epoch_s remaining_s remaining_minutes sleep_s

  while true; do
    now_epoch_s="$(date +%s)"
    remaining_s="$((end_epoch_s - now_epoch_s))"
    if (( remaining_s <= 0 )); then
      echo "Experiment deadline reached."
      return
    fi
    remaining_minutes="$(((remaining_s + 59) / 60))"
    echo "Time left until experiment deadline: ${remaining_s}s (~${remaining_minutes}m)"
    sleep_s=300
    (( remaining_s < sleep_s )) && sleep_s="${remaining_s}"
    sleep "${sleep_s}"
  done
}

phase_prepare() {
  for cmd in go tmux curl python3 tar awk sed ip sudo; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      echo "Required command not found on $(hostname): ${cmd}" >&2
      exit 1
    fi
  done

  rm -rf "${run_root}"
  mkdir -p "${run_root}" "${logs_dir}"
  if (( is_client == 1 )); then
    # Ensure the caller can always fetch a run log, even on early failures.
    {
      echo "==> remote run initialized"
      echo "==> mode=${mode}"
      echo "==> f=${fault_tolerance} n=${num_nodes}"
      echo "==> duration=${experiment_time_s}s"
    } > "${output_file}"
  fi
  tar --warning=no-unknown-keyword -xzf "${payload_path}" -C "${run_root}"
  cp -a "${applier_path}" "${applier_script}"
  chmod +x "${applier_script}"
  rm -f "${payload_path}" "${applier_path}"

  if [[ ! -d "${src_dir}" ]]; then
    echo "Extracted cometbft directory missing: ${src_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${master_config}" ]]; then
    echo "Extracted master config missing: ${master_config}" >&2
    exit 1
  fi
  if [[ ! -f "${failure_spec}" ]]; then
    echo "Missing failure spec: ${failure_spec}" >&2
    exit 1
  fi
  if (( machine_idx >= 0 && learning_agent_enabled == 1 )); then
    if [[ ! -f "${learning_agent_dir}/main.py" ]]; then
      echo "Extracted learning_agent missing: ${learning_agent_dir}/main.py" >&2
      exit 1
    fi
    if [[ ! -f "${schedule_file}" ]]; then
      echo "Missing Tendermint schedule file: ${schedule_file}" >&2
      exit 1
    fi
    if [[ ! -x "${learning_agent_python}" ]]; then
      echo "Learning-agent virtual environment not found: ${learning_agent_python}" >&2
      echo "Run scripts/cloudlab/setup.sh first." >&2
      exit 1
    fi
    if ! "${learning_agent_python}" -c "import cryptography, grpc, sklearn" >/dev/null 2>&1; then
      echo "Learning-agent dependencies are unavailable in ${learning_agent_python}." >&2
      echo "Run scripts/cloudlab/setup.sh first." >&2
      exit 1
    fi
  fi
  if (( replicas_per_machine > max_overlay_nodes )); then
    echo "Too many replicas per machine (${replicas_per_machine}) for overlay host offset ${overlay_host_offset} in a /24 subnet." >&2
    exit 1
  fi

  cd "${src_dir}"
  echo "Installing required CLIs (abci-cli/load/report)..."
  go install "${buildvcs_flag}" ./abci/cmd/abci-cli
  go install "${buildvcs_flag}" ./test/loadtime/cmd/load
  go install "${buildvcs_flag}" ./test/loadtime/cmd/report

  echo "Building cometbft binary..."
  GOFLAGS="${GOFLAGS:-} ${buildvcs_flag}" make build
  if [[ ! -x "${cometbft_bin}" ]]; then
    echo "Missing built cometbft binary: ${cometbft_bin}" >&2
    exit 1
  fi

  if [[ "${mode}" == "sharing" && "${machine_idx}" -ge 0 ]]; then
    echo "Building report-chain (sharing) binary..."
    if [[ ! -d "${sharing_src_dir}" ]]; then
      echo "Sharing mode requires the SmartBFT sources in the payload: ${sharing_src_dir}" >&2
      exit 1
    fi
    mkdir -p "${run_root}/bin"
    (cd "${sharing_src_dir}" && GOCACHE="${run_root}/gocache" go build "${buildvcs_flag}" -o "${sharing_bin}" ./examples/sharing)
  fi
}

phase_testnet() {
  echo "Generating testnet config (n=${num_nodes})..."
  "${cometbft_bin}" testnet \
    --v "${num_nodes}" \
    --o "${testnet_dir}" \
    --populate-persistent-peers \
    --starting-ip-address "$(testnet_ip 0)"

  echo "Applying master Tendermint config: ${master_config}"
  local i j config_file
  for ((i = 0; i < num_nodes; i++)); do
    config_file="${testnet_dir}/node${i}/config/config.toml"
    if [[ ! -f "${config_file}" ]]; then
      echo "Missing config file: ${config_file}" >&2
      exit 1
    fi
    apply_master_config "${master_config}" "${config_file}"
    # Peers get each replica's own address and port: testnet assigns consecutive
    # addresses on port 26656.
    for ((j = 0; j < num_nodes; j++)); do
      sed -i "s|@$(testnet_ip "${j}"):26656|@$(node_ip "${j}"):$(p2p_port "${j}")|g" "${config_file}"
    done
    if (( learning_agent_enabled == 1 )); then
      sed -i "s|adaptive_timer_addr = \".*\"|adaptive_timer_addr = \"$(node_ip "${i}"):$(agent_port "${i}")\"|" "${config_file}"
      sed -i "s|adaptive_timer_node_index = .*|adaptive_timer_node_index = ${i}|" "${config_file}"
    else
      sed -i "s|adaptive_timer_addr = \".*\"|adaptive_timer_addr = \"\"|" "${config_file}"
      sed -i "s|adaptive_timer_node_index = .*|adaptive_timer_node_index = 0|" "${config_file}"
    fi
  done
  tar -czf "${testnet_tar}" -C "${run_root}" mytestnet
}

phase_install_testnet() {
  rm -rf "${testnet_dir}"
  tar -xzf "${testnet_tar}" -C "${run_root}"
  rm -f "${testnet_tar}"
}

phase_start() {
  echo "Setting up namespaces for replicas ${first_local_replica}-${last_local_replica} on $(hostname)..."
  setup_network_namespaces

  local i
  if (( learning_agent_enabled == 1 )); then
    if [[ "${mode}" == "sharing" ]]; then
      : > "${sharing_hosts_config}"
      for ((i = 0; i < num_nodes; i++)); do
        # report-chain ids are 1-based: node_id host report_port agent_port consensus_port
        echo "$((i + 1)) $(node_ip "${i}") $(sharing_report_port "${i}") $(agent_port "${i}") $(sharing_consensus_port "${i}")" \
          >> "${sharing_hosts_config}"
      done
    fi
    : > "${agent_hosts_config}"
    for ((i = 0; i < num_nodes; i++)); do
      echo "${i} $(node_ip "${i}") 0 0 $(agent_port "${i}") $(sharing_consensus_port "${i}")" >> "${agent_hosts_config}"
    done
  fi
  echo "Starting replicas ${first_local_replica}-${last_local_replica} (kvstore and CometBFT) in tmux..."
  kill_run_sessions
  if (( learning_agent_enabled == 1 )); then
    local agent_mode_args=""
    if [[ "${mode}" == "sharing" ]]; then
      agent_mode_args="--sharing true --model-type ${agent_model_type} --collect-wait-ms 100"
    elif [[ "${mode}" == "learning" ]]; then
      agent_mode_args="--model-type ${agent_model_type}"
    elif [[ "${mode}" == "scheduled" ]]; then
      agent_mode_args="--model-type scheduled --schedule-file \"${schedule_file}\""
    fi
    # The reward shaping applies to every agent mode that trains, including the
    # scheduled replay, whose CSV records the reward each fixed timeout earned.
    if [[ "${agent_mode_args}" != "" ]]; then
      agent_mode_args="${agent_mode_args} --reward-design ${reward_design}"
    fi
    # The floor is a constraint on the action space, so it only means anything
    # where the agent is choosing the timeout at all.
    if [[ "${agent_mode_args}" != "" && "${latency_floor_k}" != "0" ]]; then
      agent_mode_args="${agent_mode_args} --latency-floor-k ${latency_floor_k}"
    fi
    # Which pretrained prior the continuous model loads: "" the shipped return
    # prior, "none" untrained, else a path to an ablation prior.
    if [[ "${agent_model_type}" == "continuous_pretrained_gp" && -n "${prior_path}" ]]; then
      agent_mode_args="${agent_mode_args} --prior-path ${prior_path}"
    fi
    local agent_session ns agent_log dataset_args
    for ((i = first_local_replica; i <= last_local_replica; i++)); do
      agent_session="tm_tendermint_agent_${i}"
      ns="$(ns_name "${i}")"
      agent_log="${logs_dir}/agent_${i}.log"
      dataset_args=""
      if (( dataset_enabled == 1 )); then
        rm -f "${run_root}/agent${i}.csv"
        dataset_args="--dataset-csv \"${run_root}/agent${i}.csv\""
      fi
      tmux new-session -d -s "${agent_session}" \
        "sudo -n ip netns exec ${ns} sudo -n -u ${run_as_user} env HOME=${run_as_home} PATH=${run_as_home}/go/bin:${PATH} PYTHONDONTWRITEBYTECODE=1 bash -lc 'cd \"${run_root}\" && exec \"${learning_agent_python}\" \"${learning_agent_dir}/main.py\" --node-id \"${i}\" --protocol tendermint --hosts-config \"${agent_hosts_config}\" --failure-spec \"${failure_spec}\" --injection-start-unix-ms \"${start_unix_ms}\" ${agent_mode_args} ${dataset_args}' > \"${agent_log}\" 2>&1"
    done
    sleep 1
    for ((i = first_local_replica; i <= last_local_replica; i++)); do
      agent_session="tm_tendermint_agent_${i}"
      if ! tmux has-session -t "${agent_session}" 2>/dev/null; then
        echo "Learning-agent session exited early: ${agent_session}" >&2
        tail -n 80 "${logs_dir}/agent_${i}.log" >&2 || true
        exit 1
      fi
    done

    if [[ "${mode}" == "sharing" ]]; then
      echo "Starting report-chain (sharing) nodes..."
      local sharing_session sharing_log
      for ((i = first_local_replica; i <= last_local_replica; i++)); do
        sharing_session="tm_sharing_${i}"
        ns="$(ns_name "${i}")"
        sharing_log="${logs_dir}/sharing_${i}.log"
        rm -rf "${run_root}/sharing_wal_${i}"
        # The report chain orders report batches with SmartBFT regardless of the
        # protocol being learned, so the same binary serves Tendermint runs.
        tmux new-session -d -s "${sharing_session}" \
          "sudo -n ip netns exec ${ns} sudo -n -u ${run_as_user} env HOME=${run_as_home} PATH=${run_as_home}/go/bin:${PATH} bash -lc 'cd \"${run_root}\" && exec \"${sharing_bin}\" --node-id $((i + 1)) --protocol tendermint --hosts-config \"${sharing_hosts_config}\" --data-dir \"${run_root}/sharing_wal_${i}\"' > \"${sharing_log}\" 2>&1"
      done
      sleep 2
      for ((i = first_local_replica; i <= last_local_replica; i++)); do
        if ! tmux has-session -t "tm_sharing_${i}" 2>/dev/null; then
          echo "Sharing session exited early: tm_sharing_${i}" >&2
          tail -n 80 "${logs_dir}/sharing_${i}.log" >&2 || true
          exit 1
        fi
      done
    fi
  fi
  local abci_session node_session abci_log node_log abci_p p2p_p rpc_p node_home
  for ((i = first_local_replica; i <= last_local_replica; i++)); do
    abci_session="tm_abci_${i}"
    node_session="tm_tendermint_node_${i}"
    ns="$(ns_name "${i}")"
    abci_log="${logs_dir}/abci_${i}.log"
    node_log="${logs_dir}/node_${i}.log"
    abci_p="$(abci_port "${i}")"
    p2p_p="$(p2p_port "${i}")"
    rpc_p="$(rpc_port "${i}")"
    node_home="${testnet_dir}/node${i}"

    tmux new-session -d -s "${abci_session}" \
      "sudo -n ip netns exec ${ns} sudo -n -u ${run_as_user} env HOME=${run_as_home} PATH=${run_as_home}/go/bin:${PATH} bash -lc 'cd \"${src_dir}\" && exec abci-cli kvstore --address tcp://127.0.0.1:${abci_p}' > \"${abci_log}\" 2>&1"
    tmux new-session -d -s "${node_session}" \
      "sudo -n ip netns exec ${ns} sudo -n -u ${run_as_user} env HOME=${run_as_home} PATH=${run_as_home}/go/bin:${PATH} COMETBFT_NODE_INDEX=${i} COMETBFT_PROPOSE_DELAY_FAILURE_SPEC=\"${failure_spec}\" COMETBFT_PROPOSE_DELAY_FAULTY_NODES=${fault_tolerance} COMETBFT_PROPOSE_DELAY_START_UNIX_MS=${start_unix_ms} bash -lc 'cd \"${src_dir}\" && exec \"${cometbft_bin}\" node --home \"${node_home}\" --proxy_app tcp://127.0.0.1:${abci_p} --p2p.laddr tcp://0.0.0.0:${p2p_p} --rpc.laddr tcp://0.0.0.0:${rpc_p}' > \"${node_log}\" 2>&1"
  done

  write_applier_hosts_config
  echo "Starting per-replica network failure appliers in tmux..."
  start_network_appliers

  local ready attempt
  for ((i = first_local_replica; i <= last_local_replica; i++)); do
    rpc_p="$(rpc_port "${i}")"
    ready=0
    for attempt in $(seq 1 60); do
      if curl -sf "http://$(node_ip "${i}"):${rpc_p}/status" >/dev/null 2>&1; then
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
}

phase_client() {
  if (( machine_idx < 0 )); then
    echo "Configuring client routes to the replica subnets..."
    setup_client_routes
  fi
  trap stop_progress_reporter EXIT

  local expected_peers=$((num_nodes - 1))
  local rpc0 rpc0_ip load_endpoints experiment_end_epoch_s peers attempt
  rpc0="$(rpc_port 0)"
  rpc0_ip="$(node_ip 0)"
  load_endpoints="$(load_endpoints_csv)"
  experiment_end_epoch_s="$(((start_unix_ms + experiment_time_s * 1000) / 1000))"
  for attempt in $(seq 1 30); do
    peers="$(curl -sf "http://${rpc0_ip}:${rpc0}/net_info" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["n_peers"])' 2>/dev/null || echo "0")"
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
    echo "==> f=${fault_tolerance} n=${num_nodes} server_machines=${server_machine_count} replicas_per_machine=${replicas_per_machine}"
    echo "==> post-peer-connect wait: ${post_peer_connect_delay}s"
    echo "==> synced start-unix-ms: ${start_unix_ms}"
    echo "==> load: duration=${experiment_time_s}s, connections=${load_connections}, rate=${load_rate}, send-period=${load_send_period_ms}ms, size=${load_size}B"
    echo "==> client log: ${client_log}"
  } > "${output_file}"

  echo "Experiment deadline unix_ms=$((start_unix_ms + experiment_time_s * 1000)) (start_unix_ms + ${experiment_time_s}s)"
  report_time_remaining "${experiment_end_epoch_s}" &
  progress_reporter_pid=$!

  load \
    --endpoints "${load_endpoints}" \
    --start-unix-ms "${start_unix_ms}" \
    --connections "${load_connections}" \
    --rate "${load_rate}" \
    --send-period "${load_send_period_ms}ms" \
    --time "${experiment_time_s}" \
    --size "${load_size}" 2>&1 | tee "${client_log}" | awk '
      # As in the other harnesses: everything up to the first Monitor line (which
      # carries the benchmark start line), then the Monitor lines and their errors.
      /Monitor (duration=|error:)/ { seen_monitor = 1; print; fflush(); next }
      seen_monitor == 0 { print; fflush() }
    ' >> "${output_file}"
  stop_progress_reporter
}

phase_stop() {
  kill_run_sessions
  force_stop_run_processes
}

phase_report() {
  echo ""
  echo "==> Latency report:"
  if [[ -f "${report_state_log}" ]]; then
    echo "Using state log for wall-clock latency: ${report_state_log}"
    run_report_with_retry --data-dir "${testnet_dir}/node0/data" --database-type goleveldb --state-log "${report_state_log}" --csv "${results_csv}"
    run_report_with_retry --data-dir "${testnet_dir}/node0/data" --database-type goleveldb --state-log "${report_state_log}"
  else
    echo "State log for node0 not found; falling back to chain block timestamp latency."
    run_report_with_retry --data-dir "${testnet_dir}/node0/data" --database-type goleveldb --csv "${results_csv}"
    run_report_with_retry --data-dir "${testnet_dir}/node0/data" --database-type goleveldb
  fi
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
  echo ""
  python3 "${run_root}/scripts/tendermint_phase_report.py" \
    --logs-dir "${logs_dir}" \
    --failure-spec "${failure_spec}" \
    --start-unix-ms "${start_unix_ms}"
  if (( log_enabled == 0 )); then
    rm -f "${results_csv}"
  fi
}

phase_teardown() {
  teardown_network_namespaces
  if (( machine_idx < 0 )); then
    local peer
    for ((peer = 0; peer < server_machine_count; peer++)); do
      sudo -n ip route del "$(machine_subnet "${peer}")" >/dev/null 2>&1 || true
    done
  else
    delete_peer_routes
  fi
  rm -f "${payload_path}" "${applier_path}" "${testnet_tar}"
}

case "${phase}" in
  prepare) phase_prepare ;;
  testnet) phase_testnet ;;
  install_testnet) phase_install_testnet ;;
  start) phase_start ;;
  client) phase_client ;;
  stop) phase_stop ;;
  report) phase_report ;;
  teardown) phase_teardown ;;
  *) echo "Unknown remote phase: ${phase}" >&2; exit 2 ;;
esac
REMOTE

# Run one remote phase: remote_phase <user> <host> <phase> <server machine index or -1> <is client>
remote_phase() {
  local user="$1" host="$2" phase="$3" machine_idx="$4" is_client="$5"
  ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- \
    "${phase}" "${REMOTE_PAYLOAD}" "${REMOTE_APPLIER}" "${F}" "${EXPERIMENT_TIME_S}" "${MODE}" \
    "${LOG_ENABLED}" "${DATASET_ENABLED}" "${LOAD_CONNECTIONS}" "${LOAD_RATE}" "${LOAD_SEND_PERIOD_MS}" \
    "${AGENT_MODEL_TYPE}" "${REWARD_DESIGN}" "${LATENCY_FLOOR_K}" "${SCHEDULE_FILE}" \
    "${machine_idx}" "${is_client}" "${server_machine_count}" "${replicas_per_machine}" \
    "${START_UNIX_MS}" "${GATEWAY_IPS_CSV}" "${SERVER_ROUTE_NAMES_CSV}" "${SERVER_ROUTE_HOSTS_CSV}" \
    "${REMOTE_TESTNET}" "${PRIOR_PATH}" <<< "${REMOTE_SCRIPT}"
}

# A machine's role arguments: its server machine index (or -1) and whether it runs the client.
client_is_server=0
(( machine_count == 1 )) && client_is_server=1
server_phase() {
  local m="$1" phase="$2"
  remote_phase "${server_users[$m]}" "${server_hosts[$m]}" "${phase}" "${m}" "${client_is_server}" 2>&1 \
    | while IFS= read -r line; do printf '  [%s] %s\n' "${server_names[$m]}" "${line}"; done
  return "${PIPESTATUS[0]}"
}

# Run a phase on every server machine (and, with with_client=1, a separate client
# machine) in parallel; fail if any of them fails.
parallel_phase() {
  local phase="$1" with_client="${2:-0}"
  local m failed=0
  local pids=()
  for m in "${!server_hosts[@]}"; do
    server_phase "${m}" "${phase}" &
    pids+=("$!")
  done
  if (( with_client == 1 && client_is_server == 0 )); then
    (
      remote_phase "${client_user}" "${client_host}" "${phase}" -1 1 2>&1 \
        | while IFS= read -r line; do printf '  [node0] %s\n' "${line}"; done
      exit "${PIPESTATUS[0]}"
    ) &
    pids+=("$!")
  fi
  for m in "${!pids[@]}"; do
    wait "${pids[$m]}" || failed=1
  done
  return "${failed}"
}

all_hosts=("${server_hosts[@]}")
all_users=("${server_users[@]}")
if (( client_is_server == 0 )); then
  all_hosts+=("${client_host}")
  all_users+=("${client_user}")
fi

echo "Copying payload to ${#all_hosts[@]} machine(s) (servers: ${server_names[*]}; client: node0)..."
copy_pids=()
for i in "${!all_hosts[@]}"; do
  (
    scp -q "${SSH_OPTS[@]}" "${LOCAL_PAYLOAD}" "${all_users[$i]}@${all_hosts[$i]}:${REMOTE_PAYLOAD}" &&
      scp -q "${SSH_OPTS[@]}" "${APPLIER_LOCAL_PATH}" "${all_users[$i]}@${all_hosts[$i]}:${REMOTE_APPLIER}"
  ) &
  copy_pids+=("$!")
done
for pid in "${copy_pids[@]}"; do
  if ! wait "${pid}"; then
    echo "Failed to copy the payload to every machine." >&2
    exit 1
  fi
done
rm -f "${LOCAL_PAYLOAD}"

teardown_started=0
teardown_all() {
  (( teardown_started == 1 )) && return 0
  teardown_started=1
  parallel_phase stop || true
  parallel_phase teardown 1 || true
}

echo "Running CometBFT experiment (mode=${MODE}, f=${F}, t=${EXPERIMENT_TIME_S}s, ${server_machine_count} server machine(s), ${replicas_per_machine} replica(s) each)..."
remote_failed=0
echo "Preparing machines (unpack and build)..."
if ! parallel_phase prepare 1; then
  remote_failed=1
fi
if (( remote_failed == 0 )); then
  echo "Generating the testnet on ${server_names[0]}..."
  if ! server_phase 0 testnet; then
    remote_failed=1
  elif (( server_machine_count > 1 )); then
    local_testnet="${ARTIFACT_DIR}/${REMOTE_TESTNET}"
    if ! scp -q "${SSH_OPTS[@]}" "${server_users[0]}@${server_hosts[0]}:${REMOTE_TESTNET}" "${local_testnet}"; then
      remote_failed=1
    else
      for ((m = 1; m < server_machine_count; m++)); do
        if ! scp -q "${SSH_OPTS[@]}" "${local_testnet}" "${server_users[$m]}@${server_hosts[$m]}:${REMOTE_TESTNET}" ||
          ! server_phase "${m}" install_testnet; then
          remote_failed=1
        fi
      done
    fi
    rm -f "${local_testnet}"
  fi
fi
if (( remote_failed == 0 )); then
  START_UNIX_MS="$(( ($(date +%s) + START_DELAY_S) * 1000 ))"
  echo "Using shared start_unix_ms=${START_UNIX_MS}"
  printf 'start_unix_ms=%q\n' "${START_UNIX_MS}" >> "${ARTIFACT_DIR}/run.env"
  trap 'teardown_all; cleanup_local_run_locks' EXIT
  echo "Starting replicas..."
  if ! parallel_phase start; then
    remote_failed=1
  fi
fi
if (( remote_failed == 0 )); then
  echo "Running the load client on node0..."
  if ! remote_phase "${client_user}" "${client_host}" client "$(( client_is_server == 1 ? 0 : -1 ))" 1; then
    remote_failed=1
  fi
fi
REPORT_LOCAL="${ARTIFACT_DIR}/.tendermint_report.txt"
rm -f "${REPORT_LOCAL}"
if (( START_UNIX_MS > 0 )); then
  echo "Stopping replicas..."
  parallel_phase stop || true
  if (( remote_failed == 0 )); then
    echo "Building the latency and phase reports on ${server_names[0]}..."
    if ! remote_phase "${server_users[0]}" "${server_hosts[0]}" report 0 "${client_is_server}" > "${REPORT_LOCAL}" 2>&1; then
      echo "Report phase failed on ${server_names[0]}; its output is appended to ${OUTPUT_LOCAL}." >&2
      remote_failed=1
    fi
  fi
fi

echo "Fetching tendermint output to ${OUTPUT_LOCAL}..."
if ! scp "${SSH_OPTS[@]}" "${client_user}@${client_host}:tendermint_output.txt" "${OUTPUT_LOCAL}" >/dev/null 2>&1; then
  {
    echo "Failed to fetch remote output file from ${client_host}."
    echo "remote_failed=${remote_failed}"
  } > "${OUTPUT_LOCAL}"
  if (( remote_failed == 0 )); then
    echo "Failed to fetch tendermint output from ${client_host}." >&2
    exit 1
  fi
fi
if [[ -f "${REPORT_LOCAL}" ]]; then
  cat "${REPORT_LOCAL}" >> "${OUTPUT_LOCAL}"
  rm -f "${REPORT_LOCAL}"
fi

if (( LOG_ENABLED == 1 )); then
  echo "Fetching logs in parallel to ${LOGS_LOCAL_DIR}..."
  if ! fetch_tendermint_logs_parallel "${LOGS_LOCAL_DIR}"; then
    echo "Failed to fetch some/all logs." >&2
  fi
fi
if (( DATASET_ENABLED == 1 )); then
  echo "Fetching learning-agent datasets to ${ARTIFACT_DIR}..."
  for ((i = 0; i < LOCAL_NUM_NODES; i++)); do
    m=$((i / replicas_per_machine))
    local_csv="${ARTIFACT_DIR}/agent${i}.csv"
    rm -f "${local_csv}"
    if ! scp "${SSH_OPTS[@]}" "${server_users[$m]}@${server_hosts[$m]}:tendermint_cloudlab/agent${i}.csv" "${local_csv}"; then
      echo "Missing or failed to fetch agent${i}.csv from ${server_hosts[$m]}." >&2
    fi
  done
fi

if (( START_UNIX_MS > 0 )); then
  echo "Tearing down namespaces and routes..."
  teardown_all
fi

if (( remote_failed == 1 )); then
  echo "Remote run failed. Retrieved output file may contain failure details: ${OUTPUT_LOCAL}" >&2
  exit 1
fi

echo "Done. Client output saved to ${OUTPUT_LOCAL}."
if (( LOG_ENABLED == 1 )); then
  echo "Logs saved to ${LOGS_LOCAL_DIR} (client.log in ${ARTIFACT_DIR})."
fi
if (( DATASET_ENABLED == 1 )); then
  echo "Datasets saved to ${ARTIFACT_DIR}/agentN.csv."
fi
echo "Run artifacts saved to ${ARTIFACT_DIR}."
