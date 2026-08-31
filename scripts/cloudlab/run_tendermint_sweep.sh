#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUNNER="${SCRIPT_DIR}/run_tendermint.sh"
FAILURE_SPEC="${REPO_ROOT}/config/tendermint/failure_spec.xml"
SCHEDULE="${REPO_ROOT}/learning_agent/config/schedule_tendermint.csv"
TMP_ROOT="${SCRIPT_DIR}/tmp"

duration_s=""
timeouts_spec=""
targets_file=""

usage() {
  cat <<EOF
Usage: $0 <private_key_path> -d <experiment_seconds> -t <timeouts> [targets_file]

Runs each proposed timeout sequentially through CloudLab in scheduled mode with
--dataset and --log. The load settings come from run_tendermint.sh.

Arguments:
  private_key_path  Local SSH private key used by run_tendermint.sh.
  targets_file      Optional CloudLab SSH targets file.

Flags:
  -d <seconds>      Experiment duration for each run.
  -t <timeouts>     Semicolon-separated propose timeout values in ms,
                    e.g. '150;450;750'.
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
if [[ ! -x "${RUNNER}" ]]; then
  echo "CloudLab runner is missing or not executable: ${RUNNER}" >&2
  exit 1
fi
if [[ ! -f "${FAILURE_SPEC}" ]]; then
  echo "Tendermint failure spec not found: ${FAILURE_SPEC}" >&2
  exit 1
fi
if [[ ! -f "${SCHEDULE}" ]]; then
  echo "Tendermint schedule not found: ${SCHEDULE}" >&2
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

schedule_backup="$(mktemp "${TMPDIR:-/tmp}/tendermint-schedule.XXXXXX")"
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
  local propose_ms="$1"
  printf '%s\n' \
    "# time_s,propose_timeout_ms" \
    "-10,${propose_ms}" \
    "0,${propose_ms}" > "${SCHEDULE}"
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

IFS=';' read -r -a timeouts <<< "${timeouts_spec}"

run_stamp="$(date +%Y%m%d_%H%M%S)"
results_root="${TMP_ROOT}/tendermint_sweep_${run_stamp}"
mkdir -p "${results_root}"
cp "${FAILURE_SPEC}" "${results_root}/failure_spec.xml"

total_runs=${#timeouts[@]}
run_number=0
failed_runs=0

printf 'duration_s=%s\ntimeouts=%s\nmode=scheduled\ndataset=true\n' \
  "${duration_s}" "${timeouts_spec}" > "${results_root}/sweep.conf"

for propose_ms in "${timeouts[@]}"; do
  run_number=$((run_number + 1))
  set_schedule_timeout "${propose_ms}"

  run_name="timeout_${propose_ms}"
  run_dir="${results_root}/${run_name}"
  mkdir -p "${run_dir}"
  cp "${SCHEDULE}" "${run_dir}/schedule_tendermint.csv"

  echo
  echo "==> Sweep run ${run_number}/${total_runs}: propose_timeout=${propose_ms}ms"

  runner_args=(
    "${private_key_path}"
    scheduled
    --dataset
    --log
    --output-dir "${run_dir}"
    -t "${duration_s}"
  )
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
    printf 'timeout_propose_ms=%s\n' "${propose_ms}"
    printf 'duration_s=%s\n' "${duration_s}"
    printf 'mode=scheduled\n'
    printf 'dataset=true\n'
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
