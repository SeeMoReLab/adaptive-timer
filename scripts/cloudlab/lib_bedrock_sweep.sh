#!/usr/bin/env bash
# Shared timeout-sweep driver for the CppBedrock protocols (PBFT, SBFT).
# A wrapper (run_pbft_sweep.sh, run_sbft_sweep.sh) sets:
#   PROTOCOL_NAME        pbft | sbft
#   RUNNER_SCRIPT        the CloudLab runner (run_<protocol>.sh)
#   FAILURE_SPEC_REL     failure spec, relative to repo root
#   SCHEDULE_REL         schedule CSV, relative to repo root
#   SWEEP_SLOW_PATH      1 when the schedule carries a slow-path column
# and then sources this file. Each timeout value (each election/slow-path
# pair with SWEEP_SLOW_PATH=1) becomes one scheduled run with --dataset
# --log; the per-run agent datasets are combined at the end, mirroring
# run_smartbft_sweep.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

for required in PROTOCOL_NAME RUNNER_SCRIPT FAILURE_SPEC_REL SCHEDULE_REL SWEEP_SLOW_PATH; do
  if [[ -z "${!required:-}" ]]; then
    echo "lib_bedrock_sweep.sh: ${required} must be set by the wrapper script." >&2
    exit 2
  fi
done

RUNNER="${SCRIPT_DIR}/${RUNNER_SCRIPT}"
FAILURE_SPEC="${REPO_ROOT}/${FAILURE_SPEC_REL}"
SCHEDULE="${REPO_ROOT}/${SCHEDULE_REL}"
TMP_ROOT="${SCRIPT_DIR}/tmp"

MAX_TIMEOUT_MS=4294967295

duration_s=""
timeouts_spec=""
slow_path_spec=""
targets_file=""
extra_runner_args=()

usage() {
  cat <<EOF
Usage: $0 <private_key_path> -d <experiment_seconds> -t <election_timeouts>$( (( SWEEP_SLOW_PATH )) && printf ' -s <slow_path_timeouts>' )
          [--runner-arg <arg>]... [targets_file]

Runs each proposed timeout sequentially through CloudLab in scheduled mode with
--dataset and --log. The load settings come from ${RUNNER_SCRIPT}.

Arguments:
  private_key_path  Local SSH private key used by ${RUNNER_SCRIPT}.
  targets_file      Optional CloudLab SSH targets file.

Flags:
  -d <seconds>      Experiment duration for each run.
  -t <timeouts>     Semicolon-separated election timeout values in ms,
                    e.g. '300;437;1800'. Each value must be between 1 and
                    ${MAX_TIMEOUT_MS}.
EOF
  if (( SWEEP_SLOW_PATH )); then
    cat <<EOF
  -s <timeouts>     Semicolon-separated SBFT slow-path timeout values in ms,
                    e.g. '20;100'. Every election/slow-path pair is one run.
EOF
  fi
  cat <<EOF
  --runner-arg <a>  Extra argument forwarded verbatim to ${RUNNER_SCRIPT}
                    (repeatable, e.g. --runner-arg --rate --runner-arg 500).
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
    -s)
      [[ $# -ge 2 ]] || { echo "Missing value for -s." >&2; exit 2; }
      if ! (( SWEEP_SLOW_PATH )); then
        echo "-s is only valid for protocols with a slow-path timeout." >&2
        exit 2
      fi
      slow_path_spec="$2"
      shift 2
      ;;
    --runner-arg)
      [[ $# -ge 2 ]] || { echo "Missing value for --runner-arg." >&2; exit 2; }
      extra_runner_args+=("$2")
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
if (( SWEEP_SLOW_PATH )); then
  if ! [[ "${slow_path_spec}" =~ ^[1-9][0-9]*(\;[1-9][0-9]*)*$ ]]; then
    echo "Invalid or missing -s value: '${slow_path_spec}'." >&2
    exit 2
  fi
fi
if [[ ! -x "${RUNNER}" ]]; then
  echo "CloudLab runner is missing or not executable: ${RUNNER}" >&2
  exit 1
fi
if [[ ! -f "${FAILURE_SPEC}" ]]; then
  echo "${PROTOCOL_NAME} failure spec not found: ${FAILURE_SPEC}" >&2
  exit 1
fi
if [[ ! -f "${SCHEDULE}" ]]; then
  echo "${PROTOCOL_NAME} schedule not found: ${SCHEDULE}" >&2
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
slow_paths=("")
if (( SWEEP_SLOW_PATH )); then
  IFS=';' read -r -a slow_paths <<< "${slow_path_spec}"
fi

check_timeout_bound() {
  local value="$1" label="$2"
  if (( ${#value} > ${#MAX_TIMEOUT_MS} )) || (( 10#${value} > MAX_TIMEOUT_MS )); then
    echo "${label} is too large: ${value}ms. Maximum: ${MAX_TIMEOUT_MS}ms." >&2
    exit 2
  fi
}
for timeout_ms in "${timeouts[@]}"; do
  check_timeout_bound "${timeout_ms}" "${PROTOCOL_NAME} election timeout"
done
if (( SWEEP_SLOW_PATH )); then
  for slow_ms in "${slow_paths[@]}"; do
    check_timeout_bound "${slow_ms}" "${PROTOCOL_NAME} slow-path timeout"
  done
fi

schedule_backup="$(mktemp "${TMPDIR:-/tmp}/${PROTOCOL_NAME}-schedule.XXXXXX")"
cp -p "${SCHEDULE}" "${schedule_backup}"

restore_files() {
  cp -p "${schedule_backup}" "${SCHEDULE}"
  rm -f "${schedule_backup}"
}
trap restore_files EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# Writes the schedule for one run: the same value before and at the start,
# in the column layout the protocol's agent expects.
set_schedule_timeout() {
  local timeout_ms="$1" slow_ms="$2"
  if (( SWEEP_SLOW_PATH )); then
    printf '%s\n' \
      "# time_s,election_timeout_ms,slow_path_timeout_ms" \
      "-10,${timeout_ms},${slow_ms}" \
      "0,${timeout_ms},${slow_ms}" > "${SCHEDULE}"
  else
    printf '%s\n' \
      "# time_s,timeout_ms" \
      "-10,${timeout_ms}" \
      "0,${timeout_ms}" > "${SCHEDULE}"
  fi
}

run_name_for() {
  local timeout_ms="$1" slow_ms="$2"
  if (( SWEEP_SLOW_PATH )); then
    printf 'timeout_%s_slow_%s' "${timeout_ms}" "${slow_ms}"
  else
    printf 'timeout_%s' "${timeout_ms}"
  fi
}

combine_agent_datasets() {
  local agent_name combined_csv source_csv run_name
  local expected_header source_header first_dataset

  while IFS= read -r agent_name; do
    [[ -n "${agent_name}" ]] || continue
    combined_csv="${results_root}/${agent_name}"
    rm -f "${combined_csv}"
    first_dataset=1

    for run_name in "${run_names[@]}"; do
      source_csv="${results_root}/${run_name}/${agent_name}"
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
results_root="${TMP_ROOT}/${PROTOCOL_NAME}_sweep_${run_stamp}"
mkdir -p "${results_root}"
cp "${FAILURE_SPEC}" "${results_root}/failure_spec.xml"

run_names=()
for timeout_ms in "${timeouts[@]}"; do
  for slow_ms in "${slow_paths[@]}"; do
    run_names+=("$(run_name_for "${timeout_ms}" "${slow_ms}")")
  done
done
total_runs=${#run_names[@]}
run_number=0
failed_runs=0

printf 'protocol=%s\nduration_s=%s\ntimeouts=%s\nslow_path_timeouts=%s\nmode=scheduled\ndataset=true\nrunner_args=%s\n' \
  "${PROTOCOL_NAME}" "${duration_s}" "${timeouts_spec}" "${slow_path_spec:-n/a}" "${extra_runner_args[*]:-}" \
  > "${results_root}/sweep.conf"

for timeout_ms in "${timeouts[@]}"; do
  for slow_ms in "${slow_paths[@]}"; do
    run_number=$((run_number + 1))
    set_schedule_timeout "${timeout_ms}" "${slow_ms}"

    run_name="$(run_name_for "${timeout_ms}" "${slow_ms}")"
    run_dir="${results_root}/${run_name}"
    mkdir -p "${run_dir}"
    cp "${SCHEDULE}" "${run_dir}/$(basename "${SCHEDULE}")"

    echo
    if (( SWEEP_SLOW_PATH )); then
      echo "==> Sweep run ${run_number}/${total_runs}: election=${timeout_ms}ms slow_path=${slow_ms}ms"
    else
      echo "==> Sweep run ${run_number}/${total_runs}: timeout=${timeout_ms}ms"
    fi

    runner_args=(
      "${private_key_path}"
      scheduled
      --dataset
      --log
      --output-dir "${run_dir}"
      -t "${duration_s}"
      --initial-timeout "${timeout_ms}"
    )
    if (( SWEEP_SLOW_PATH )); then
      runner_args+=(--initial-slow-path-timeout "${slow_ms}")
    fi
    if (( ${#extra_runner_args[@]} > 0 )); then
      runner_args+=("${extra_runner_args[@]}")
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
      if (( SWEEP_SLOW_PATH )); then
        printf 'slow_path_timeout_ms=%s\n' "${slow_ms}"
      fi
      printf 'duration_s=%s\n' "${duration_s}"
      printf 'mode=scheduled\n'
      printf 'dataset=true\n'
    } > "${run_dir}/run.conf"

    echo "==> Archived ${run_status} run: ${run_dir}"
  done
done

combine_agent_datasets

echo
echo "Sweep complete: $((total_runs - failed_runs))/${total_runs} runs succeeded."
echo "Results: ${results_root}"
echo "Combined datasets: ${results_root}/agentN.csv"

if (( failed_runs > 0 )); then
  exit 1
fi
