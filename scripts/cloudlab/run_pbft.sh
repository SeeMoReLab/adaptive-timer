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

usage() {
  cat <<EOF
Usage: $0 <private_key_path> [normal|heuristic|learning|sharing] [-f <fault_tolerance>] [targets_file]

Arguments:
  private_key_path  Local SSH private key for CloudLab nodes.
  mode              Optional; one of normal|heuristic|learning|sharing (default: normal).
  targets_file      Optional path to cloudlab SSH targets file.

Flags:
  -f <int>          Fault tolerance f passed to prepare_pbft.sh (default: 1).
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

mkdir -p "$(dirname "${LOCAL_ZIP}")"
rm -f "${LOCAL_ZIP}"
trap 'rm -f "${LOCAL_ZIP}"' EXIT

echo "Creating temporary library zip..."
(
  cd "$(dirname "${LIBRARY_DIR}")"
  zip -qr "${LOCAL_ZIP}" "$(basename "${LIBRARY_DIR}")"
)

echo "Copying library zip to all CloudLab nodes and extracting to ~/library (parallel)..."
pids=()
nodes=()

while read -r node host user _; do
  [[ -z "${node:-}" ]] && continue
  [[ "${node:0:1}" == "#" ]] && continue

  (
    remote_zip="/tmp/library_payload_$$.zip"
    scp "${SSH_OPTS[@]}" "${LOCAL_ZIP}" "${user}@${host}:${remote_zip}"

    ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- "${remote_zip}" <<'REMOTE'
set -euo pipefail

zip_path="$1"
trap 'rm -f "$zip_path"' EXIT

rm -rf "$HOME/library"

if command -v unzip >/dev/null 2>&1; then
  unzip -q "$zip_path" -d "$HOME"
elif command -v python3 >/dev/null 2>&1; then
  python3 - "$zip_path" "$HOME" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as zf:
    zf.extractall(sys.argv[2])
PY
else
  echo "Neither unzip nor python3 found on $(hostname)" >&2
  exit 1
fi

if [[ ! -d "$HOME/library" ]]; then
  echo "library directory not found after extraction on $(hostname)" >&2
  exit 1
fi
REMOTE
  ) &

  pids+=("$!")
  nodes+=("${node}")
done < "${TARGETS_FILE}"

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
echo "Done."
