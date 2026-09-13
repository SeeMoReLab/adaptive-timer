#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUNNER="${SCRIPT_DIR}/run_smartbft.sh"
FAILURE_SPEC="${REPO_ROOT}/config/smartbft/failure_spec.xml"
SCHEDULE="${REPO_ROOT}/learning_agent/config/schedule_smartbft.csv"
TMP_ROOT="${SCRIPT_DIR}/tmp"

MAX_TIMEOUT_MS=4294967295

duration_s=""
timeouts_spec=""
targets_file=""
learning_window_mode=""

usage() {
  cat <<EOF
Usage: $0 <private_key_path> -d <experiment_seconds> -t <timeouts>
          [--learning-window-mode <consensus|wall-clock>] [targets_file]

Runs each proposed timeout sequentially through CloudLab in scheduled mode with
--dataset and --log. The load settings come from run_smartbft.sh.

Arguments:
  private_key_path  Local SSH private key used by run_smartbft.sh.
  targets_file      Optional CloudLab SSH targets file.

Flags:
  -d <seconds>      Experiment duration for each run.
  -t <timeouts>     Semicolon-separated PBFT election timeout values in ms,
                    e.g. '300;437;1800'. Each value must be between 1 and
                    ${MAX_TIMEOUT_MS}.
  --learning-window-mode <consensus|wall-clock>
                    Episode boundary mode forwarded to run_smartbft.sh.
                    Omitted by default, so run_smartbft.sh picks its own.
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

private_key_path="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d)
      [[ $# -ge 2 ]] || { echo "Missing value for -d." >&2; exit 2; }
      duration_s="$2"
      shift 2
      ;;
    -t)
      [[ $# -ge 2 ]] || { echo "Missing value for -t." >&2; exit 2; }
      timeouts_spec="$2"
      shift 2
      ;;
    --learning-window-mode)
      [[ $# -ge 2 ]] || { echo "Missing value for --learning-window-mode." >&2; exit 2; }
      learning_window_mode="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "${targets_file}" ]]; then
        echo "Unexpected extra positional argument: $1" >&2
        usage >&2
        exit 2
      fi
      targets_file="$1"
      shift
      ;;
  esac
done

if ! [[ "${duration_s}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid or missing -d value: '${duration_s}'. Use an integer >= 1." >&2
  exit 2
fi
if ! [[ "${timeouts_spec}" =~ ^[1-9][0-9]*(\;[1-9][0-9]*)*$ ]]; then
  echo "Invalid or missing -t value: '${timeouts_spec}'." >&2
  exit 2
fi
case "${learning_window_mode}" in
  ""|consensus|wall-clock) ;;
  *)
    echo "Invalid --learning-window-mode value: '${learning_window_mode}'. Expected consensus or wall-clock." >&2
    exit 2
    ;;
esac
if [[ ! -x "${RUNNER}" ]]; then
  echo "CloudLab runner is missing or not executable: ${RUNNER}" >&2
  exit 1
fi
if [[ ! -f "${FAILURE_SPEC}" ]]; then
  echo "SmartBFT failure spec not found: ${FAILURE_SPEC}" >&2
  exit 1
fi
if [[ ! -f "${SCHEDULE}" ]]; then
  echo "SmartBFT schedule not found: ${SCHEDULE}" >&2
  exit 1
fi

private_key_path="${private_key_path/#\~/$HOME}"
if [[ ! -f "${private_key_path}" ]]; then
  echo "SSH private key not found: ${private_key_path}" >&2
  exit 1
fi
if [[ -n "${targets_file}" && ! -f "${targets_file}" ]]; then
  echo "Targets file not found: ${targets_file}" >&2
  exit 1
fi

IFS=';' read -r -a timeouts <<< "${timeouts_spec}"

for timeout_ms in "${timeouts[@]}"; do
  if (( ${#timeout_ms} > ${#MAX_TIMEOUT_MS} )); then
    echo "PBFT timeout is too large: ${timeout_ms}ms. Maximum: ${MAX_TIMEOUT_MS}ms." >&2
    exit 2
  fi
  if (( 10#${timeout_ms} > MAX_TIMEOUT_MS )); then
    echo "PBFT timeout is too large: ${timeout_ms}ms. Maximum: ${MAX_TIMEOUT_MS}ms." >&2
    exit 2
  fi
done

schedule_backup="$(mktemp "${TMPDIR:-/tmp}/pbft-schedule.XXXXXX")"
cp -p "${SCHEDULE}" "${schedule_backup}"

restore_files() {
  cp -p "${schedule_backup}" "${SCHEDULE}"
  rm -f "${schedule_backup}"
}
trap restore_files EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

set_schedule_timeout() {
  local timeout_ms="$1"
  printf '%s\n' \
    "# time_s,timeout_ms" \
    "-10,${timeout_ms}" \
    "0,${timeout_ms}" > "${SCHEDULE}"
}

combine_agent_datasets() {
  local agent_name combined_csv source_csv timeout_ms
  local expected_header source_header first_dataset

  while IFS= read -r agent_name; do
    [[ -n "${agent_name}" ]] || continue
    combined_csv="${results_root}/${agent_name}"
    rm -f "${combined_csv}"
    first_dataset=1

    for timeout_ms in "${timeouts[@]}"; do
      source_csv="${results_root}/timeout_${timeout_ms}/${agent_name}"
      [[ -s "${source_csv}" ]] || continue

      if (( first_dataset == 1 )); then
        cp "${source_csv}" "${combined_csv}"
        expected_header="$(head -n 1 "${source_csv}")"
        first_dataset=0
      else
        source_header="$(head -n 1 "${source_csv}")"
        if [[ "${source_header}" != "${expected_header}" ]]; then
          echo "Dataset header mismatch: ${source_csv}" >&2
          return 1
        fi
        printf '\n' >> "${combined_csv}"
        tail -n +2 "${source_csv}" >> "${combined_csv}"
      fi
    done
  done < <(
    find "${results_root}" -mindepth 2 -maxdepth 2 -type f \
      -name 'agent[0-9]*.csv' -exec basename {} \; | sort -u
  )
}

run_stamp="$(date +%Y%m%d_%H%M%S)"
results_root="${TMP_ROOT}/smartbft_sweep_${run_stamp}"
mkdir -p "${results_root}"
cp "${FAILURE_SPEC}" "${results_root}/failure_spec.xml"

total_runs=${#timeouts[@]}
run_number=0
failed_runs=0

printf 'duration_s=%s\ntimeouts=%s\nmode=scheduled\ndataset=true\nlearning_window_mode=%s\n' \
  "${duration_s}" "${timeouts_spec}" "${learning_window_mode:-default}" \
  > "${results_root}/sweep.conf"

for timeout_ms in "${timeouts[@]}"; do
  run_number=$((run_number + 1))
  set_schedule_timeout "${timeout_ms}"

  run_name="timeout_${timeout_ms}"
  run_dir="${results_root}/${run_name}"
  mkdir -p "${run_dir}"
  cp "${SCHEDULE}" "${run_dir}/schedule_smartbft.csv"

  echo
  echo "==> Sweep run ${run_number}/${total_runs}: timeout=${timeout_ms}ms"

  runner_args=(
    "${private_key_path}"
    scheduled
    --dataset
    --log
    --output-dir "${run_dir}"
    -t "${duration_s}"
  )
  if [[ -n "${learning_window_mode}" ]]; then
    runner_args+=(--learning-window-mode "${learning_window_mode}")
  fi
  if [[ -n "${targets_file}" ]]; then
    runner_args+=("${targets_file}")
  fi

  run_status="success"
  if ! "${RUNNER}" "${runner_args[@]}"; then
    run_status="failed"
    failed_runs=$((failed_runs + 1))
  fi

  {
    printf 'status=%s\n' "${run_status}"
    printf 'timeout_ms=%s\n' "${timeout_ms}"
    printf 'duration_s=%s\n' "${duration_s}"
    printf 'mode=scheduled\n'
    printf 'dataset=true\n'
    printf 'learning_window_mode=%s\n' "${learning_window_mode:-default}"
  } > "${run_dir}/run.conf"

  echo "==> Archived ${run_status} run: ${run_dir}"
done

combine_agent_datasets

echo
echo "Sweep complete: $((total_runs - failed_runs))/${total_runs} runs succeeded."
echo "Results: ${results_root}"
echo "Combined datasets: ${results_root}/agentN.csv"

if (( failed_runs > 0 )); then
  exit 1
fi
