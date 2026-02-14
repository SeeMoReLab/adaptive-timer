#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TARGETS_FILE="${SCRIPT_DIR}/../tmp/cloudlab_ssh_targets.txt"
IP_OUTPUT_FILE="${SCRIPT_DIR}/../tmp/cloudlab_ip.txt"
NODE0_KEY_PATH="${NODE0_KEY_PATH:-~/.ssh/cloudlab_node0_ed25519}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)

usage() {
  cat <<EOF
Usage: $0 <private_key_path> [targets_file]

Options:
  private_key_path  Local SSH private key used from this machine to reach CloudLab nodes.
  targets_file      Optional path to cloudlab_ssh_targets.txt.

Environment:
  LOCAL_SSH_KEY_PATH  Alternative to the positional private_key_path.
  IP_OUTPUT_FILE      Output file for node IPs from first hostname -I entry (default: scripts/tmp/cloudlab_ip.txt).
  NODE0_KEY_PATH      Key path created on node0 for node0->peer SSH.
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
tmp_pubkey="$(mktemp)"
tmp_ips="$(mktemp)"
trap 'rm -f "${tmp_peers}" "${tmp_all_nodes}" "${tmp_pubkey}" "${tmp_ips}"' EXIT

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

if [[ -s "${tmp_peers}" ]]; then
  has_peers=1
else
  echo "Single-node CloudLab targets detected (node0 only); skipping node0->peer SSH setup."
fi

if (( has_peers == 1 )); then
  echo "Generating a fresh SSH key on node0 (${node0_user}@${node0_host})..."
  node0_pubkey_raw="$(
    ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" "bash -s" -- "${NODE0_KEY_PATH}" <<'REMOTE'
set -euo pipefail

key_path="$1"
key_path="${key_path/#\~/$HOME}"
pub_path="${key_path}.pub"

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

rm -f "$key_path" "$pub_path"
ssh-keygen -q -t ed25519 -N "" -f "$key_path" -C "cloudlab-node0-$(hostname)-$(date +%Y%m%d%H%M%S)"
chmod 600 "$key_path"
chmod 644 "$pub_path"

cat "$pub_path"
REMOTE
  )"
  node0_pubkey="$(
    printf '%s\n' "${node0_pubkey_raw}" | awk '
      /^ssh-(ed25519|rsa|ecdsa-sha2-nistp(256|384|521)|sk-ssh-ed25519|sk-ecdsa-sha2-nistp256)/ {
        print
        exit
      }
    '
  )"

  if [[ -z "${node0_pubkey}" ]]; then
    echo "Failed to parse a valid SSH public key from node0 output." >&2
    echo "Raw output from node0 was:" >&2
    printf '%s\n' "${node0_pubkey_raw}" >&2
    exit 1
  fi
  printf '%s\n' "${node0_pubkey}" > "${tmp_pubkey}"

  echo "Installing node0 public key on peer nodes..."
  while read -r node host user; do
    echo "  - ${node} (${user}@${host})"
    remote_pubkey_path="/tmp/cloudlab_node0_key_${node}.pub"
    scp "${SSH_OPTS[@]}" "${tmp_pubkey}" "${user}@${host}:${remote_pubkey_path}"

    ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" -- "${remote_pubkey_path}" <<'REMOTE'
set -euo pipefail

pubkey_path="${1:-}"
if [[ -z "${pubkey_path}" || ! -f "${pubkey_path}" ]]; then
  echo "Public key file missing on peer: ${pubkey_path:-<empty>}" >&2
  exit 1
fi
pubkey="$(cat "$pubkey_path")"

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys"
chmod 600 "$HOME/.ssh/authorized_keys"

if ! grep -qxF "$pubkey" "$HOME/.ssh/authorized_keys"; then
  printf '%s\n' "$pubkey" >> "$HOME/.ssh/authorized_keys"
fi

grep -qxF "$pubkey" "$HOME/.ssh/authorized_keys"
rm -f "$pubkey_path"
REMOTE
  done < "${tmp_peers}"

  echo "Updating node0 SSH config for peer hosts..."
  ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" "bash -s" -- "${NODE0_KEY_PATH}" <<REMOTE
set -euo pipefail

key_path="\$1"
key_path="\${key_path/#\\~/\$HOME}"
config_file="\$HOME/.ssh/config"
tmp_config="\$(mktemp)"
peer_list="\$(mktemp)"

trap 'rm -f "\$tmp_config" "\$peer_list"' EXIT

mkdir -p "\$HOME/.ssh"
chmod 700 "\$HOME/.ssh"

cat > "\$peer_list" <<'PEERS'
$(cat "${tmp_peers}")
PEERS

if [[ -f "\$config_file" ]]; then
  awk '
    BEGIN {skip=0}
    /^# BEGIN CLOUDLAB NODE0 PEERS\$/ {skip=1; next}
    /^# END CLOUDLAB NODE0 PEERS\$/ {skip=0; next}
    skip==0 {print}
  ' "\$config_file" > "\$tmp_config"
else
  : > "\$tmp_config"
fi

{
  echo "# BEGIN CLOUDLAB NODE0 PEERS"
  while read -r node host user; do
    [[ -z "\${node:-}" ]] && continue
    cat <<EOF
Host \$node \$host
    HostName \$host
    User \$user
    IdentityFile \$key_path
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new

EOF
  done < "\$peer_list"
  echo "# END CLOUDLAB NODE0 PEERS"
} >> "\$tmp_config"

chmod 600 "\$tmp_config"
mv "\$tmp_config" "\$config_file"
REMOTE

  echo "Verifying node0 can SSH to peer nodes..."
  while read -r node host user; do
    ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" "ssh -o BatchMode=yes ${node} true"
    echo "  - ${node}: OK"
  done < "${tmp_peers}"
fi

echo "Installing Java 17 and pip on all nodes in parallel..."
pids=()
nodes=()
while read -r node host user; do
  (
    ssh "${SSH_OPTS[@]}" "${user}@${host}" "bash -s" <<'REMOTE'
set -euo pipefail

if command -v apt-get >/dev/null 2>&1; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-17-jre-headless python3-pip tmux
else
  echo "apt-get not found on $(hostname); expected Ubuntu nodes." >&2
  exit 1
fi

java -version >/dev/null 2>&1
python3 -m pip --version >/dev/null 2>&1
tmux -V >/dev/null 2>&1
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

if (( has_peers == 1 )); then
  echo "Collecting node IPs (first entry from hostname -I)..."
  node0_ip="$(
    ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" \
      "hostname -I | awk '{print \$1}'"
  )"
  if [[ -z "${node0_ip}" ]]; then
    echo "Could not determine IP for node0" >&2
    exit 1
  fi
  printf 'node0 %s\n' "${node0_ip}" > "${tmp_ips}"

  while read -r node host user; do
    peer_ip="$(
      ssh "${SSH_OPTS[@]}" "${node0_user}@${node0_host}" "bash -s" -- "${node}" <<'REMOTE'
set -euo pipefail
peer="$1"
ssh -o BatchMode=yes "$peer" \
  "hostname -I | awk '{print \$1}'"
REMOTE
    )"

    if [[ -z "${peer_ip}" ]]; then
      echo "Could not determine IP for ${node}" >&2
      exit 1
    fi

    printf '%s %s\n' "${node}" "${peer_ip}" >> "${tmp_ips}"
  done < "${tmp_peers}"
else
  echo "Single-node mode: writing node0 IP as 127.0.0.1."
  printf 'node0 127.0.0.1\n' > "${tmp_ips}"
fi

mkdir -p "$(dirname "${IP_OUTPUT_FILE}")"
mv "${tmp_ips}" "${IP_OUTPUT_FILE}"
echo "Wrote node IPs to ${IP_OUTPUT_FILE}"

echo "Done."
if (( has_peers == 1 )); then
  echo "From node0 you can now SSH peers via: ssh node1 (or ssh <peer-hostname>)"
fi
