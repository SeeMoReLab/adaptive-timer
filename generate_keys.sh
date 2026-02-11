#!/usr/bin/env bash
set -euo pipefail

HOSTS_CONFIG="config/hosts.config"
KEY_DIR="config/keys"
FORCE=0

usage() {
  echo "Usage: $0 [--hosts-config PATH] [--key-dir PATH] [--force]" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hosts-config)
      [[ $# -lt 2 ]] && usage
      HOSTS_CONFIG="$2"
      shift 2
      ;;
    --key-dir)
      [[ $# -lt 2 ]] && usage
      KEY_DIR="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    *)
      usage
      ;;
  esac
done

if ! command -v openssl >/dev/null 2>&1; then
  echo "openssl is required to generate ECDSA keys." >&2
  exit 1
fi

if [[ ! -f "$HOSTS_CONFIG" ]]; then
  echo "Hosts config not found: $HOSTS_CONFIG" >&2
  exit 1
fi

mkdir -p "$KEY_DIR"

node_ids=$(awk 'NF > 0 && $1 !~ /^#/ {print $1}' "$HOSTS_CONFIG")
if [[ -z "$node_ids" ]]; then
  echo "No node ids found in $HOSTS_CONFIG" >&2
  exit 1
fi

created=()
skipped=()

for node_id in $node_ids; do
  priv_path="$KEY_DIR/node_${node_id}.ecdsa"
  pub_path="$KEY_DIR/node_${node_id}.ecdsa.pub"

  if [[ $FORCE -eq 0 && -f "$priv_path" && -f "$pub_path" ]]; then
    skipped+=("$node_id")
    continue
  fi

  openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:secp256k1 -out "$priv_path" >/dev/null 2>&1
  openssl pkey -in "$priv_path" -pubout -out "$pub_path" >/dev/null 2>&1
  chmod 600 "$priv_path"
  created+=("$node_id")
done

if [[ ${#created[@]} -gt 0 ]]; then
  echo "Generated keys for nodes: ${created[*]}"
fi
if [[ ${#skipped[@]} -gt 0 ]]; then
  echo "Skipped existing keys for nodes: ${skipped[*]}"
fi
