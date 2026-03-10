#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_SCRIPT="${REPO_ROOT}/scripts/cloudlab/run_pbft.sh"
FAILURE_SPEC="${REPO_ROOT}/config/bft-smart/failure_spec.xml"
SYSTEM_CONFIG="${REPO_ROOT}/config/bft-smart/system.config"
RUN_OUTPUT="${REPO_ROOT}/scripts/cloudlab/tmp/output.txt"
SERVER_LOG_DIR="${REPO_ROOT}/scripts/cloudlab/tmp/server_logs"

SSH_KEY="${HOME}/.ssh/cloudlab"
MODE="normal"
F=1
DURATION_SECONDS=310
RESULTS_DIR="${REPO_ROOT}/experiment"
DRY_RUN=0
KEEP_CONFIG=0
CONTINUE_ON_ERROR=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/cloudlab/run_pbft_sweep.sh [options] <delay:timeout:backoff> [<delay:timeout:backoff> ...]

Options:
  --key <path>            SSH private key path (default: ~/.ssh/cloudlab)
  --mode <mode>           run_pbft mode (default: normal)
  -f, --fault-tolerance   Fault tolerance f passed to run_pbft (default: 1)
  --duration <seconds>    Passed to run_pbft as -t (default: 310)
  --results-dir <dir>     Directory to store per-run output snapshots (default: experiment)
  --dry-run               Print commands without executing run_pbft
  --keep-config           Keep last tuple config values instead of restoring originals
  --continue-on-error     Continue remaining tuples if one run fails
  -h, --help              Show this help

Notes:
  This script always runs run_pbft with --log so server regency can be appended to output.

Tuple format:
  delay:timeout:backoff
  - delay: non-negative integer (ms), written to first phase network global delayMs
  - timeout: positive integer (ms), written to system.totalordermulticast.timeout
  - backoff: true|false, written to system.totalordermulticast.timeout.backoff

Example:
  bash scripts/cloudlab/run_pbft_sweep.sh \
    100:200:false 120:300:true
USAGE
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

normalize_bool() {
  local value
  value="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    true|false)
      printf '%s\n' "${value}"
      ;;
    *)
      return 1
      ;;
  esac
}

extract_regency_from_log() {
  local log_file="$1"
  local last_line

  if [[ ! -f "${log_file}" ]]; then
    echo "unknown"
    return 0
  fi

  last_line="$(grep -Ei 'Last regency:|regency[[:space:]]+[0-9]+' "${log_file}" | tail -n 1 || true)"
  if [[ -z "${last_line}" ]]; then
    echo "unknown"
    return 0
  fi

  if [[ "${last_line}" =~ [Ll]ast[[:space:]]+regency:[[:space:]]*([0-9]+),[[:space:]]*next[[:space:]]+regency:[[:space:]]*([0-9]+) ]]; then
    local current_regency="${BASH_REMATCH[1]}"
    local next_regency="${BASH_REMATCH[2]}"
    if (( next_regency > current_regency )); then
      echo "${next_regency}"
    else
      echo "${current_regency}"
    fi
    return 0
  fi

  if [[ "${last_line}" =~ [Rr]egency[[:space:]]+([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi

  echo "unknown"
}

prepend_regency_snapshot_to_output() {
  local output_file="$1"
  local tmp_header_file
  local tmp_output_file
  local found_logs=0

  if [[ ! -f "${output_file}" ]]; then
    return 1
  fi

  if [[ ! -d "${SERVER_LOG_DIR}" ]]; then
    return 1
  fi

  tmp_header_file="$(mktemp)"
  tmp_output_file="$(mktemp)"

  {
    echo "Regency snapshot (generated at $(date -u '+%Y-%m-%dT%H:%M:%SZ'))"
    while IFS= read -r log_file; do
      [[ -z "${log_file}" ]] && continue
      found_logs=1
      echo "  - $(basename "${log_file}"): regency=$(extract_regency_from_log "${log_file}")"
    done < <(find "${SERVER_LOG_DIR}" -maxdepth 1 -type f -name 'server_*.log' | sort)
    echo
  } > "${tmp_header_file}"

  if (( found_logs == 0 )); then
    rm -f "${tmp_header_file}" "${tmp_output_file}"
    return 1
  fi

  {
    cat "${tmp_header_file}"
    cat "${output_file}"
  } > "${tmp_output_file}"
  mv "${tmp_output_file}" "${output_file}"
  rm -f "${tmp_header_file}"
  return 0
}

update_failure_spec_first_phase_delay() {
  local delay_ms="$1"
  local tmp_file
  tmp_file="$(mktemp)"

  if ! awk -v delay_ms="${delay_ms}" '
    BEGIN {
      in_phase = 0
      first_phase_done = 0
      in_global = 0
      changed = 0
    }
    {
      if (!first_phase_done && $0 ~ /^[[:space:]]*<phase>[[:space:]]*$/) {
        in_phase = 1
      }

      if (in_phase && $0 ~ /^[[:space:]]*<global>[[:space:]]*$/) {
        in_global = 1
      }

      if (in_global && changed == 0 && $0 ~ /<delayMs>[0-9]+<\/delayMs>/) {
        sub(/<delayMs>[0-9]+<\/delayMs>/, "<delayMs>" delay_ms "</delayMs>")
        changed = 1
      }

      print

      if (in_global && $0 ~ /^[[:space:]]*<\/global>[[:space:]]*$/) {
        in_global = 0
      }

      if (in_phase && $0 ~ /^[[:space:]]*<\/phase>[[:space:]]*$/) {
        in_phase = 0
        first_phase_done = 1
      }
    }
    END {
      if (changed == 0) {
        exit 2
      }
    }
  ' "${FAILURE_SPEC}" > "${tmp_file}"; then
    rm -f "${tmp_file}"
    echo "Failed to update first phase global delayMs in ${FAILURE_SPEC}" >&2
    exit 1
  fi

  mv "${tmp_file}" "${FAILURE_SPEC}"
}

update_system_config_timeout() {
  local timeout_ms="$1"
  local backoff="$2"
  local tmp_file
  tmp_file="$(mktemp)"

  if ! awk -v timeout_ms="${timeout_ms}" -v backoff="${backoff}" '
    BEGIN {
      timeout_changed = 0
      backoff_changed = 0
    }
    {
      if ($0 ~ /^[[:space:]]*system\.totalordermulticast\.timeout[[:space:]]*=/) {
        print "system.totalordermulticast.timeout = " timeout_ms
        timeout_changed = 1
        next
      }
      if ($0 ~ /^[[:space:]]*system\.totalordermulticast\.timeout\.backoff[[:space:]]*=/) {
        print "system.totalordermulticast.timeout.backoff = " backoff
        backoff_changed = 1
        next
      }
      print
    }
    END {
      if (timeout_changed == 0 || backoff_changed == 0) {
        exit 3
      }
    }
  ' "${SYSTEM_CONFIG}" > "${tmp_file}"; then
    rm -f "${tmp_file}"
    echo "Failed to update timeout/backoff in ${SYSTEM_CONFIG}" >&2
    exit 1
  fi

  mv "${tmp_file}" "${SYSTEM_CONFIG}"
}

TUPLES=()
while (( $# > 0 )); do
  case "$1" in
    --key)
      shift
      if (( $# == 0 )); then
        echo "Missing value for --key" >&2
        exit 1
      fi
      SSH_KEY="$1"
      ;;
    --mode)
      shift
      if (( $# == 0 )); then
        echo "Missing value for --mode" >&2
        exit 1
      fi
      MODE="$1"
      ;;
    -f|--fault-tolerance)
      option_name="$1"
      shift
      if (( $# == 0 )); then
        echo "Missing value for ${option_name}" >&2
        exit 1
      fi
      F="$1"
      ;;
    --duration)
      shift
      if (( $# == 0 )); then
        echo "Missing value for --duration" >&2
        exit 1
      fi
      DURATION_SECONDS="$1"
      ;;
    --results-dir)
      shift
      if (( $# == 0 )); then
        echo "Missing value for --results-dir" >&2
        exit 1
      fi
      RESULTS_DIR="$1"
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --keep-config)
      KEEP_CONFIG=1
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      TUPLES+=("$1")
      ;;
  esac
  shift
done

if (( ${#TUPLES[@]} == 0 )); then
  usage >&2
  exit 1
fi

if [[ ! "${DURATION_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --duration value: ${DURATION_SECONDS}" >&2
  exit 1
fi

if [[ ! "${F}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid -f/--fault-tolerance value: ${F}" >&2
  exit 1
fi

require_file "${RUN_SCRIPT}"
require_file "${FAILURE_SPEC}"
require_file "${SYSTEM_CONFIG}"

mkdir -p "${RESULTS_DIR}"

backup_dir="$(mktemp -d)"
cp "${FAILURE_SPEC}" "${backup_dir}/failure_spec.xml.orig"
cp "${SYSTEM_CONFIG}" "${backup_dir}/system.config.orig"

restore_configs() {
  if (( KEEP_CONFIG == 0 )); then
    cp "${backup_dir}/failure_spec.xml.orig" "${FAILURE_SPEC}"
    cp "${backup_dir}/system.config.orig" "${SYSTEM_CONFIG}"
  fi
  rm -rf "${backup_dir}"
}
trap restore_configs EXIT

run_index=0
failed_runs=0

for tuple in "${TUPLES[@]}"; do
  run_index=$((run_index + 1))

  if [[ ! "${tuple}" =~ ^([0-9]+):([1-9][0-9]*):([^:]+)$ ]]; then
    echo "Invalid tuple '${tuple}'. Expected delay:timeout:backoff" >&2
    exit 1
  fi

  delay_ms="${BASH_REMATCH[1]}"
  timeout_ms="${BASH_REMATCH[2]}"
  if ! backoff="$(normalize_bool "${BASH_REMATCH[3]}")"; then
    echo "Invalid backoff value in tuple '${tuple}'. Use true|false." >&2
    exit 1
  fi

  echo "== Run ${run_index}/${#TUPLES[@]}: delay=${delay_ms} timeout=${timeout_ms} backoff=${backoff} =="
  update_failure_spec_first_phase_delay "${delay_ms}"
  update_system_config_timeout "${timeout_ms}" "${backoff}"

  cmd=(
    bash "${RUN_SCRIPT}" "${SSH_KEY}" "${MODE}" --trim-output --log -f "${F}" -t "${DURATION_SECONDS}"
  )
  echo "Command: ${cmd[*]}"

  if (( DRY_RUN == 0 )); then
    if ! "${cmd[@]}"; then
      failed_runs=$((failed_runs + 1))
      echo "Run failed for tuple ${tuple}" >&2
      if (( CONTINUE_ON_ERROR == 0 )); then
        exit 1
      fi
      continue
    fi

    output_basename="$(printf "%03d_d%s_t%s_b%s.txt" "${run_index}" "${delay_ms}" "${timeout_ms}" "${backoff}")"
    output_snapshot="${RESULTS_DIR}/${output_basename}"

    if [[ -f "${RUN_OUTPUT}" ]]; then
      if ! prepend_regency_snapshot_to_output "${RUN_OUTPUT}"; then
        echo "Warning: unable to prepend regency snapshot (missing or unparsable server logs)." >&2
      fi
      cp "${RUN_OUTPUT}" "${output_snapshot}"
      echo "Saved output snapshot: ${output_snapshot}"
    else
      echo "Warning: run output not found at ${RUN_OUTPUT}" >&2
    fi
  fi
done

if (( failed_runs > 0 )); then
  echo "Completed with ${failed_runs} failed run(s)." >&2
  exit 1
fi

echo "All runs completed."
