#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SRC_LIBRARY_DIR="${REPO_ROOT}/library"
TMP_DIR="${SCRIPT_DIR}/tmp"
DST_LIBRARY_DIR="${TMP_DIR}/library"

BASE_DIR="${DST_LIBRARY_DIR}/base"
SMALLBANK_DIR="${DST_LIBRARY_DIR}/smallbank"
DATAEXCHANGE_DIR="${DST_LIBRARY_DIR}/dataexchange"

if [[ ! -d "${SRC_LIBRARY_DIR}" ]]; then
  echo "Source library directory not found: ${SRC_LIBRARY_DIR}" >&2
  exit 1
fi

mkdir -p "${TMP_DIR}"
rm -rf "${DST_LIBRARY_DIR}"
cp -a "${SRC_LIBRARY_DIR}" "${DST_LIBRARY_DIR}"

if [[ ! -d "${BASE_DIR}" ]]; then
  echo "Base directory not found after copy: ${BASE_DIR}" >&2
  exit 1
fi

rm -rf "${SMALLBANK_DIR}" "${DATAEXCHANGE_DIR}"
cp -a "${BASE_DIR}" "${SMALLBANK_DIR}"
cp -a "${BASE_DIR}" "${DATAEXCHANGE_DIR}"

DATAEXCHANGE_SYSTEM_CONFIG="${DATAEXCHANGE_DIR}/system.config"
DATAEXCHANGE_HOSTS_CONFIG="${DATAEXCHANGE_DIR}/hosts.config"
if [[ ! -f "${DATAEXCHANGE_SYSTEM_CONFIG}" ]]; then
  echo "Dataexchange system.config not found: ${DATAEXCHANGE_SYSTEM_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${DATAEXCHANGE_HOSTS_CONFIG}" ]]; then
  echo "Dataexchange hosts.config not found: ${DATAEXCHANGE_HOSTS_CONFIG}" >&2
  exit 1
fi

sed -i \
  's/^\([[:space:]]*system\.optimizations\.episode_requests[[:space:]]*=[[:space:]]*\)false[[:space:]]*$/\1true/' \
  "${DATAEXCHANGE_SYSTEM_CONFIG}"

tmp_hosts="$(mktemp)"
awk '
  /^[[:space:]]*#/ || NF == 0 { print; next }
  {
    $3 = $3 + 10000
    $4 = $4 + 10000
    print
  }
' "${DATAEXCHANGE_HOSTS_CONFIG}" > "${tmp_hosts}"
mv "${tmp_hosts}" "${DATAEXCHANGE_HOSTS_CONFIG}"

rm -f "${SMALLBANK_DIR}/shards.config" "${DATAEXCHANGE_DIR}/shards.config"

echo "Generated configs under: ${DST_LIBRARY_DIR}"
