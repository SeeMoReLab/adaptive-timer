#!/usr/bin/env bash
# Election-timeout sweep for CppBedrock PBFT: one scheduled CloudLab run per
# value via run_pbft.sh, combined agent datasets at the end.
PROTOCOL_NAME="pbft"
RUNNER_SCRIPT="run_pbft.sh"
FAILURE_SPEC_REL="config/pbft/failure_spec.xml"
SCHEDULE_REL="learning_agent/config/schedule_pbft.csv"
SWEEP_SLOW_PATH=0

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_bedrock_sweep.sh"
