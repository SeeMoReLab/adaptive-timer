#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PREPARE_SCRIPT="${REPO_ROOT}/scripts/prepare_pbft.sh"
TARGETS_FILE_DEFAULT="${REPO_ROOT}/scripts/tmp/cloudlab_ssh_targets.txt"
LIBRARY_DIR="${REPO_ROOT}/scripts/tmp/library"
LOCAL_ZIP="${REPO_ROOT}/scripts/tmp/library.zip"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
MODE="normal"
F=1
TARGETS_FILE="${TARGETS_FILE_DEFAULT}"
EXPERIMENT_TIME_S=60
CLIENT_START_DELAY_S=10
OUTPUT_LOCAL="${REPO_ROOT}/scripts/tmp/output.txt"

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
    scp "${SSH_OPTS[@]}" "${LOCAL_ZIP}" "${user}@${host}:${remote_zip}"

    ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- "${remote_zip}" "${replica_count}" "${create_client}" <<'REMOTE'
set -euo pipefail

zip_path="$1"
replica_count="$2"
create_client="$3"
tmp_extract_dir="$HOME/.library_extract_$$"

cleanup() {
  rm -f "$zip_path"
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

echo "Starting PBFT server replicas in tmux..."
start_failed=0
for i in "${!server_nodes[@]}"; do
  node="${server_nodes[$i]}"
  host="${server_hosts[$i]}"
  user="${server_users[$i]}"
  machine_idx="${i}"

  if ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- "${replicas_per_server_machine}" "${machine_idx}" <<'REMOTE'
set -euo pipefail

replicas_per_machine="$1"
machine_idx="$2"

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
  session_name="pbft_server_${replica_id}"
  tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
  tmux new-session -d -s "${session_name}" \
    "cd \"$HOME/library\" && bash smartrun.sh bftsmart.demo.smallbank.SmallBankServer ${replica_id} smallbank${n}"
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
ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" "bash -s" <<'REMOTE'
set -euo pipefail

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
  "cd \"$HOME/library\" && bash smartrun.sh bftsmart.demo.smallbank.SmallBankClient client --create --execute > output.txt 2>&1"
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
  if [[ "${session_name}" == pbft_server_* || "${session_name}" == pbft_client ]]; then
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

mkdir -p "$(dirname "${OUTPUT_LOCAL}")"
rm -f "${OUTPUT_LOCAL}"
echo "Fetching client output to ${OUTPUT_LOCAL}..."
scp "${SSH_OPTS[@]}" "${node0_user}@${node0_host}:~/library/output.txt" "${OUTPUT_LOCAL}"

if (( stop_failed != 0 )); then
  echo "Some nodes reported tmux stop issues; output was still collected." >&2
fi

echo "Done. Client output saved to ${OUTPUT_LOCAL}."
