#!/usr/bin/env bash
# CloudLab runner for CppBedrock PBFT.
# All phases live in lib_bedrock.sh; this file only pins the protocol knobs.

PROTOCOL_NAME="pbft"
BEDROCK_PROTOCOL_CONFIG_REL="CppBedrock/config/config.pbft.yaml"
SUPPORTED_MODES="normal learning sharing scheduled"
AGENT_PROTOCOL="pbft"
SCHEDULE_FILE_REL="${SCHEDULE_FILE_REL:-learning_agent/config/schedule_pbft.csv}"
FAILURE_SPEC_DEFAULT_REL="config/pbft/failure_spec.xml"
NS_PREFIX="pb-r"
HOST_IF_PREFIX="pbh"
NS_IF_PREFIX="pbn"
BRIDGE_NAME="br-pbft"

# The PBFT ladder in CppBedrock/adaptive_timers.MD was calibrated at this
# depth. SBFT runs at 64; re-calibrate before raising this one to match.
MAX_INFLIGHT_BATCHES="${MAX_INFLIGHT_BATCHES:-16}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_bedrock.sh"
