#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_TARGETS_FILE="${SCRIPT_DIR}/tmp/cloudlab_ssh_targets.txt"
IP_OUTPUT_FILE="${SCRIPT_DIR}/tmp/cloudlab_ip.txt"
LEARNING_AGENT_REQUIREMENTS="${REPO_ROOT}/learning_agent/requirements.txt"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)

usage() {
  cat <<EOF
Usage: $0 <private_key_path> [targets_file]

Options:
  private_key_path  Local SSH private key used from this machine to reach CloudLab nodes.
  targets_file      Optional path to cloudlab_ssh_targets.txt.

Environment:
  LOCAL_SSH_KEY_PATH  Alternative to the positional private_key_path.
  IP_OUTPUT_FILE      Output file for node IPs from first hostname -I entry (default: scripts/cloudlab/tmp/cloudlab_ip.txt).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 2 ]]; then
  echo "Too many arguments." >&2
  usage >&2
  exit 2
fi

LOCAL_SSH_KEY_PATH="${LOCAL_SSH_KEY_PATH:-${1:-}}"
TARGETS_FILE="${2:-${DEFAULT_TARGETS_FILE}}"

if [[ ! -f "${TARGETS_FILE}" ]]; then
  echo "Targets file not found: ${TARGETS_FILE}" >&2
  echo "Generate it first with scripts/cloudlab/extract_ssh_targets.py" >&2
  exit 1
fi

if [[ -z "${LOCAL_SSH_KEY_PATH}" ]]; then
  echo "Missing required local SSH key path." >&2
  echo "Provide positional <private_key_path> or set LOCAL_SSH_KEY_PATH." >&2
  usage >&2
  exit 2
fi

LOCAL_SSH_KEY_PATH="${LOCAL_SSH_KEY_PATH/#\~/$HOME}"
if [[ ! -f "${LOCAL_SSH_KEY_PATH}" ]]; then
  echo "Local SSH key not found: ${LOCAL_SSH_KEY_PATH}" >&2
  exit 1
fi
SSH_OPTS+=(-i "${LOCAL_SSH_KEY_PATH}" -o IdentitiesOnly=yes)

tmp_peers="$(mktemp)"
tmp_all_nodes="$(mktemp)"
tmp_ips="$(mktemp)"
trap 'rm -f "${tmp_peers}" "${tmp_all_nodes}" "${tmp_ips}"' EXIT

node0_host=""
node0_user=""
has_peers=0

while read -r node host user _; do
  [[ -z "${node:-}" ]] && continue
  [[ "${node:0:1}" == "#" ]] && continue

  if [[ -z "${host:-}" || -z "${user:-}" ]]; then
    echo "Skipping malformed line in ${TARGETS_FILE}: ${node} ${host:-} ${user:-}" >&2
    continue
  fi

  if [[ "${node}" == "node0" ]]; then
    node0_host="${host}"
    node0_user="${user}"
  else
    printf '%s %s %s\n' "${node}" "${host}" "${user}" >> "${tmp_peers}"
  fi

  printf '%s %s %s\n' "${node}" "${host}" "${user}" >> "${tmp_all_nodes}"
done < "${TARGETS_FILE}"

if [[ -z "${node0_host}" || -z "${node0_user}" ]]; then
  echo "Could not find node0 entry in ${TARGETS_FILE}" >&2
  exit 1
fi
if [[ ! -f "${LEARNING_AGENT_REQUIREMENTS}" ]]; then
  echo "learning_agent requirements not found: ${LEARNING_AGENT_REQUIREMENTS}" >&2
  exit 1
fi

if [[ -s "${tmp_peers}" ]]; then
  has_peers=1
else
  echo "Single-node CloudLab targets detected (node0 only); skipping node0->peer SSH setup."
fi

if (( has_peers == 1 )); then
  echo "Multi-node targets detected; skipping node0->peer SSH bootstrap."
fi

echo "Installing PBFT + Tendermint dependencies on all nodes in parallel..."
pids=()
nodes=()
while read -r node host user; do
  (
    ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" <<'REMOTE'
set -euo pipefail

if command -v apt-get >/dev/null 2>&1; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    openjdk-17-jre-headless \
    python3-pip \
    python3-venv \
    tmux \
    golang-go \
    build-essential \
    curl \
    tar
else
  echo "apt-get not found on $(hostname); expected Ubuntu nodes." >&2
  exit 1
fi

go_version_at_least() {
  local min_version="$1"
  python3 - "$min_version" <<'PY'
import re
import subprocess
import sys

min_major, min_minor = [int(x) for x in sys.argv[1].split(".", 1)]
out = subprocess.check_output(["go", "version"], text=True)
m = re.search(r"go(\d+)\.(\d+)", out)
if not m:
    raise SystemExit(1)
major = int(m.group(1))
minor = int(m.group(2))
raise SystemExit(0 if (major, minor) >= (min_major, min_minor) else 1)
PY
}

ensure_modern_go_toolchain() {
  local min_go="1.22"
  local desired_go="${GO_BOOTSTRAP_VERSION:-1.25.7}"
  if go_version_at_least "${min_go}"; then
    return 0
  fi

  local arch
  case "$(uname -m)" in
    x86_64|amd64)
      arch="amd64"
      ;;
    aarch64|arm64)
      arch="arm64"
      ;;
    *)
      echo "Unsupported CPU architecture for Go bootstrap: $(uname -m)" >&2
      return 1
      ;;
  esac

  local install_root="${HOME}/.local/go-${desired_go}"
  local go_bin="${install_root}/go/bin/go"
  local archive="/tmp/go${desired_go}.linux-${arch}.tar.gz"
  local url="https://go.dev/dl/go${desired_go}.linux-${arch}.tar.gz"

  if [[ ! -x "${go_bin}" ]]; then
    rm -rf "${install_root}"
    mkdir -p "${install_root}"
    curl -fsSL "${url}" -o "${archive}"
    tar -xzf "${archive}" -C "${install_root}"
    rm -f "${archive}"
  fi

  export PATH="${install_root}/go/bin:${PATH}"
  hash -r
  if ! go_version_at_least "${min_go}"; then
    echo "Go bootstrap failed. Active go: $(go version 2>/dev/null || echo unavailable)" >&2
    return 1
  fi

  # CloudLab run scripts prepend $HOME/go/bin to PATH on remote execution.
  # Ensure that path resolves to the bootstrapped modern toolchain.
  mkdir -p "${HOME}/go/bin"
  ln -sf "${install_root}/go/bin/go" "${HOME}/go/bin/go"
  ln -sf "${install_root}/go/bin/gofmt" "${HOME}/go/bin/gofmt"
}

ensure_modern_go_toolchain

# Allow module-aware commands to auto-fix go.mod/go.sum during build/install.
# This avoids failures like:
#   "go: updates to go.mod needed; to update it: go mod tidy"
go env -w GOFLAGS=-mod=mod

java -version >/dev/null 2>&1
python3 -m pip --version >/dev/null 2>&1
tmux -V >/dev/null 2>&1
go version
make --version >/dev/null 2>&1
curl --version >/dev/null 2>&1
tar --version >/dev/null 2>&1
REMOTE
  ) &
  pids+=("$!")
  nodes+=("${node}")
done < "${tmp_all_nodes}"

install_failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "  - ${nodes[$i]}: installed"
  else
    echo "  - ${nodes[$i]}: install failed" >&2
    install_failed=1
  fi
done

if [[ "${install_failed}" -ne 0 ]]; then
  echo "Dependency installation failed on one or more nodes." >&2
  exit 1
fi

echo "Installing learning-agent Python requirements on all nodes in parallel..."
pids=()
nodes=()
while read -r node host user; do
  (
    remote_req_path="/tmp/learning_agent_requirements_$$_${node}.txt"
    ssh "${SSH_OPTS[@]}" "${user}@${host}" \
      "cat > \"${remote_req_path}\"" < "${LEARNING_AGENT_REQUIREMENTS}"
    ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- "${remote_req_path}" <<'REMOTE'
set -euo pipefail

requirements_path="$1"
venv_dir="${HOME}/.venvs/adaptive-timer-learning-agent"
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on $(hostname)" >&2
  exit 1
fi
if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "python3 venv support not found on $(hostname)" >&2
  exit 1
fi
if [[ ! -x "${venv_dir}/bin/python" ]]; then
  mkdir -p "$(dirname "${venv_dir}")"
  python3 -m venv "${venv_dir}"
fi
PIP_DISABLE_PIP_VERSION_CHECK=1 "${venv_dir}/bin/python" -m pip install -r "${requirements_path}"
"${venv_dir}/bin/python" -c "import cryptography, grpc, matplotlib, numpy, pandas, scipy, sklearn, tqdm, xgboost"
rm -f "${requirements_path}"
REMOTE
  ) &
  pids+=("$!")
  nodes+=("${node}")
done < "${tmp_all_nodes}"

pip_failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "  - ${nodes[$i]}: learning-agent virtual environment ready"
  else
    echo "  - ${nodes[$i]}: learning-agent virtual environment setup failed" >&2
    pip_failed=1
  fi
done

if [[ "${pip_failed}" -ne 0 ]]; then
  echo "Python requirements installation failed on one or more nodes." >&2
  exit 1
fi

if (( has_peers == 0 )); then
  echo "Single-node mode: writing node0 IP as 127.0.0.1."
  printf 'node0 127.0.0.1\n' > "${tmp_ips}"
else
  echo "Collecting node IPs directly from each node (first entry from hostname -I)..."
  while read -r node host user; do
    node_ip="$(
      ssh "${SSH_OPTS[@]}" "${user}@${host}" \
        "hostname -I | awk '{print \$1}'"
    )"

    if [[ -z "${node_ip}" ]]; then
      echo "Could not determine IP for ${node}" >&2
      exit 1
    fi

    printf '%s %s\n' "${node}" "${node_ip}" >> "${tmp_ips}"
  done < "${tmp_all_nodes}"
fi

mkdir -p "$(dirname "${IP_OUTPUT_FILE}")"
mv "${tmp_ips}" "${IP_OUTPUT_FILE}"
echo "Wrote node IPs to ${IP_OUTPUT_FILE}"

echo "Done."
