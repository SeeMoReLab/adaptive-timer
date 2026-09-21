#!/usr/bin/env bash
# Timeout sweep for CppBedrock SBFT: one scheduled CloudLab run per
# election/slow-path timeout pair via run_sbft.sh, combined agent datasets
# at the end.
PROTOCOL_NAME="sbft"
RUNNER_SCRIPT="run_sbft.sh"
FAILURE_SPEC_REL="config/sbft/failure_spec.xml"
SCHEDULE_REL="learning_agent/config/schedule_sbft.csv"
SWEEP_SLOW_PATH=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_bedrock_sweep.sh"
