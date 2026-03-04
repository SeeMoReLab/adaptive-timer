#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BFT_SMART_DIR="${REPO_ROOT}/bft-smart"
GRADLEW="${BFT_SMART_DIR}/gradlew"
SRC_LIBRARY_DIR="${BFT_SMART_DIR}/build/install/library"
SRC_CONFIG_DIR="${REPO_ROOT}/config/bft-smart"
TMP_DIR="${SCRIPT_DIR}/tmp"
DST_LIBRARY_DIR="${TMP_DIR}/library"
CLOUDLAB_IP_FILE_DEFAULT="${SCRIPT_DIR}/cloudlab/tmp/cloudlab_ip.txt"

DST_CONFIG_DIR="${DST_LIBRARY_DIR}/config"
SMALLBANK_DIR="${DST_LIBRARY_DIR}/smallbank"
DATAEXCHANGE_DIR="${DST_LIBRARY_DIR}/dataexchange"

MODE="local"
F=1
N=1
SINGLE_NODE_CLOUDLAB=0
CLOUDLAB_IP_FILE="${CLOUDLAB_IP_FILE_DEFAULT}"
CLOUDLAB_IP_FILE_SET=0

usage() {
  cat <<EOF_USAGE
Usage: $0 [local|cloudlab] [-f <fault_tolerance>] [cloudlab_ip_file]

Arguments:
  local|cloudlab      Optional mode (default: local).
  cloudlab_ip_file    Optional path to cloudlab_ip.txt (cloudlab mode only).

Flags:
  -f <int>            Fault tolerance value f (default: 1). Total servers = 3f + 1.

Notes:
  - In cloudlab mode, n is derived from cloudlab_ip_file:
    n = (3f + 1) / (m - 1), where m is the number of non-empty, non-comment lines.
  - Server id i maps to node(i / n + 1).
  - If cloudlab_ip_file only includes node0, all replicas use 127.0.0.1 on node0.
EOF_USAGE
}

build_initial_view() {
  local num_servers="$1"
  local view=""
  local i

  for ((i = 0; i < num_servers; i++)); do
    if [[ -n "${view}" ]]; then
      view+=","
    fi
    view+="${i}"
  done

  printf '%s\n' "${view}"
}

update_system_config() {
  local config_file="$1"
  local num_servers="$2"
  local fault_tolerance="$3"
  local initial_view
  local tmp_config

  initial_view="$(build_initial_view "${num_servers}")"
  tmp_config="$(mktemp)"

  awk \
    -v num_servers="${num_servers}" \
    -v fault_tolerance="${fault_tolerance}" \
    -v initial_view="${initial_view}" '
    /^[[:space:]]*system\.servers\.num[[:space:]]*=/ {
      print "system.servers.num = " num_servers
      seen_num = 1
      next
    }
    /^[[:space:]]*system\.servers\.f[[:space:]]*=/ {
      print "system.servers.f = " fault_tolerance
      seen_f = 1
      next
    }
    /^[[:space:]]*system\.initial\.view[[:space:]]*=/ {
      print "system.initial.view = " initial_view
      seen_view = 1
      next
    }
    { print }
    END {
      if (!seen_num) {
        print "system.servers.num = " num_servers
      }
      if (!seen_f) {
        print "system.servers.f = " fault_tolerance
      }
      if (!seen_view) {
        print "system.initial.view = " initial_view
      }
    }
  ' "${config_file}" > "${tmp_config}"

  mv "${tmp_config}" "${config_file}"
}

generate_hosts_config() {
  local hosts_file="$1"
  local mode="$2"
  local num_servers="$3"
  local nodes_per_machine="$4"
  local cloudlab_ip_file="$5"
  local port3_base="$6"
  local port4_base="$7"
  local port5_base="$8"
  local port6_base="$9"
  local single_node_cloudlab="${10}"

  local tmp_hosts
  local server_id
  local slot
  local node_name
  local ip

  tmp_hosts="$(mktemp)"

  echo "#server id, address and two ports (the ids from 0 to n-1 are the service replicas), agent port, data exchange port" > "${tmp_hosts}"

  for ((server_id = 0; server_id < num_servers; server_id++)); do
    if [[ "${mode}" == "local" ]]; then
      ip="127.0.0.1"
      slot="${server_id}"
    elif [[ "${single_node_cloudlab}" == "1" ]]; then
      ip="127.0.0.1"
      slot="${server_id}"
    else
      slot=$((server_id % nodes_per_machine))
      node_name="node$((server_id / nodes_per_machine + 1))"
      ip="$(awk -v node_name="${node_name}" '$1 == node_name { print $2; exit }' "${cloudlab_ip_file}")"

      if [[ -z "${ip}" ]]; then
        rm -f "${tmp_hosts}"
        echo "Missing IP mapping for ${node_name} in ${cloudlab_ip_file}" >&2
        return 1
      fi
    fi

    printf '%d %s %d %d %d %d\n' \
      "${server_id}" \
      "${ip}" \
      "$((port3_base + slot))" \
      "$((port4_base + slot))" \
      "$((port5_base + slot))" \
      "$((port6_base + slot))" >> "${tmp_hosts}"
  done

  mv "${tmp_hosts}" "${hosts_file}"
}

configure_dataexchange_system_config() {
  local config_file="$1"
  local tmp_config

  tmp_config="$(mktemp)"
  awk '
    BEGIN {
      seen_episode_requests = 0
      seen_use_signatures = 0
    }
    /^[[:space:]]*system\.optimizations\.episode_requests[[:space:]]*=/ {
      print "system.optimizations.episode_requests = true"
      seen_episode_requests = 1
      next
    }
    /^[[:space:]]*system\.communication\.useSignatures[[:space:]]*=/ {
      # DataExchangeServer submits TOM messages directly and does not attach a
      # client signature payload; keep signature checks off for this config.
      print "system.communication.useSignatures = 0"
      seen_use_signatures = 1
      next
    }
    { print }
    END {
      if (!seen_episode_requests) {
        print "system.optimizations.episode_requests = true"
      }
      if (!seen_use_signatures) {
        print "system.communication.useSignatures = 0"
      }
    }
  ' "${config_file}" > "${tmp_config}"
  mv "${tmp_config}" "${config_file}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -f)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing value for -f" >&2
        usage >&2
        exit 2
      fi
      F="${2}"
      shift 2
      ;;
    local|cloudlab)
      MODE="$1"
      shift
      ;;
    -* )
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ "${CLOUDLAB_IP_FILE_SET}" -eq 1 ]]; then
        echo "Unexpected extra positional argument: $1" >&2
        usage >&2
        exit 2
      fi
      CLOUDLAB_IP_FILE="$1"
      CLOUDLAB_IP_FILE_SET=1
      shift
      ;;
  esac
done

if ! [[ "${F}" =~ ^[0-9]+$ ]] || (( F < 1 )); then
  echo "Invalid -f value: ${F}. It must be an integer >= 1." >&2
  exit 2
fi

if [[ "${MODE}" != "local" && "${MODE}" != "cloudlab" ]]; then
  echo "Invalid mode: ${MODE}" >&2
  usage >&2
  exit 2
fi

if [[ "${MODE}" == "local" && "${CLOUDLAB_IP_FILE_SET}" -eq 1 ]]; then
  echo "cloudlab_ip_file can only be provided in cloudlab mode." >&2
  exit 2
fi

NUM_SERVERS=$((3 * F + 1))

if [[ ! -d "${BFT_SMART_DIR}" ]]; then
  echo "bft-smart directory not found: ${BFT_SMART_DIR}" >&2
  exit 1
fi

if [[ ! -d "${SRC_CONFIG_DIR}" ]]; then
  echo "Source config directory not found: ${SRC_CONFIG_DIR}" >&2
  exit 1
fi

if [[ ! -x "${GRADLEW}" ]]; then
  echo "gradlew not found or not executable: ${GRADLEW}" >&2
  exit 1
fi

if [[ "${MODE}" == "cloudlab" ]]; then
  if [[ ! -f "${CLOUDLAB_IP_FILE}" ]]; then
    echo "CloudLab IP file not found: ${CLOUDLAB_IP_FILE}" >&2
    exit 1
  fi

  MACHINE_COUNT="$(awk '/^[[:space:]]*#/ || NF == 0 { next } { count++ } END { print count + 0 }' "${CLOUDLAB_IP_FILE}")"
  if (( MACHINE_COUNT < 1 )); then
    echo "CloudLab IP file is empty: ${CLOUDLAB_IP_FILE}" >&2
    exit 1
  fi

  if (( MACHINE_COUNT == 1 )); then
    SINGLE_NODE_CLOUDLAB=1
    N="${NUM_SERVERS}"
    echo "Single-node CloudLab mode detected: using 127.0.0.1 for all replicas."
  else
    SERVER_MACHINES=$((MACHINE_COUNT - 1))
    if (( NUM_SERVERS % SERVER_MACHINES != 0 )); then
      echo "CloudLab size check failed: (3f + 1) must be divisible by (m - 1)." >&2
      echo "  m=${MACHINE_COUNT}, f=${F}" >&2
      echo "  3f + 1 = ${NUM_SERVERS}, m - 1 = ${SERVER_MACHINES}" >&2
      exit 1
    fi

    N=$((NUM_SERVERS / SERVER_MACHINES))
  fi
fi

echo "Building library distribution via gradlew installDist..."
(
  cd "${BFT_SMART_DIR}"
  ./gradlew installDist
)

if [[ ! -d "${SRC_LIBRARY_DIR}" ]]; then
  echo "Installed library directory not found after build: ${SRC_LIBRARY_DIR}" >&2
  exit 1
fi

mkdir -p "${TMP_DIR}"
rm -rf "${DST_LIBRARY_DIR}"
mkdir -p "${DST_LIBRARY_DIR}"
cp -a "${SRC_LIBRARY_DIR}/." "${DST_LIBRARY_DIR}"
rm -rf "${DST_CONFIG_DIR}" "${SMALLBANK_DIR}" "${DATAEXCHANGE_DIR}"

mkdir -p "${DST_CONFIG_DIR}"
cp -a "${SRC_CONFIG_DIR}/." "${DST_CONFIG_DIR}"
rm -f \
  "${DST_CONFIG_DIR}/smallbank.xml" \
  "${DST_CONFIG_DIR}/hosts.config" \
  "${DST_CONFIG_DIR}/shards.config" \
  "${DST_CONFIG_DIR}/system.config"

HOSTS_CONFIG_SRC="${SRC_CONFIG_DIR}/hosts.config"
SYSTEM_CONFIG_SRC="${SRC_CONFIG_DIR}/system.config"
SMALLBANK_CONFIG_SRC="${SRC_CONFIG_DIR}/smallbank.xml"
FAILURE_SPEC_SRC="${SRC_CONFIG_DIR}/failure_spec.xml"

for required_file in \
  "${HOSTS_CONFIG_SRC}" \
  "${SYSTEM_CONFIG_SRC}" \
  "${SMALLBANK_CONFIG_SRC}" \
  "${FAILURE_SPEC_SRC}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required config file not found: ${required_file}" >&2
    exit 1
  fi
done

mkdir -p "${SMALLBANK_DIR}" "${DATAEXCHANGE_DIR}"

cp -a "${HOSTS_CONFIG_SRC}" "${SMALLBANK_DIR}/hosts.config"
cp -a "${SYSTEM_CONFIG_SRC}" "${SMALLBANK_DIR}/system.config"
cp -a "${SMALLBANK_CONFIG_SRC}" "${SMALLBANK_DIR}/smallbank.xml"

cp -a "${HOSTS_CONFIG_SRC}" "${DATAEXCHANGE_DIR}/hosts.config"
cp -a "${SYSTEM_CONFIG_SRC}" "${DATAEXCHANGE_DIR}/system.config"

update_system_config "${SMALLBANK_DIR}/system.config" "${NUM_SERVERS}" "${F}"
update_system_config "${DATAEXCHANGE_DIR}/system.config" "${NUM_SERVERS}" "${F}"
configure_dataexchange_system_config "${DATAEXCHANGE_DIR}/system.config"

generate_hosts_config \
  "${SMALLBANK_DIR}/hosts.config" \
  "${MODE}" \
  "${NUM_SERVERS}" \
  "${N}" \
  "${CLOUDLAB_IP_FILE}" \
  11000 \
  10000 \
  50000 \
  55000 \
  "${SINGLE_NODE_CLOUDLAB}"

generate_hosts_config \
  "${DATAEXCHANGE_DIR}/hosts.config" \
  "${MODE}" \
  "${NUM_SERVERS}" \
  "${N}" \
  "${CLOUDLAB_IP_FILE}" \
  21000 \
  20000 \
  50000 \
  55000 \
  "${SINGLE_NODE_CLOUDLAB}"

echo "Generated configs under: ${DST_LIBRARY_DIR}"
echo "Mode=${MODE}, f=${F}, n=${N}, servers=${NUM_SERVERS}"
