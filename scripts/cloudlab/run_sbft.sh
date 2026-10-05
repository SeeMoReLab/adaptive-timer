#!/usr/bin/env bash
# CloudLab runner for CppBedrock SBFT.
# All phases live in lib_bedrock.sh; this file only pins the protocol knobs.

PROTOCOL_NAME="sbft"
BEDROCK_PROTOCOL_CONFIG_REL="CppBedrock/config/config.sbft.yaml"
SUPPORTED_MODES="normal learning sharing scheduled"
AGENT_PROTOCOL="sbft"
SCHEDULE_FILE_REL="${SCHEDULE_FILE_REL:-learning_agent/config/schedule_sbft.csv}"
FAILURE_SPEC_DEFAULT_REL="config/sbft/failure_spec.xml"
NS_PREFIX="sb-r"
HOST_IF_PREFIX="sbh"
NS_IF_PREFIX="sbn"
BRIDGE_NAME="br-sbft"
# config/sbft/failure_spec.xml and schedule_sbft.csv are calibrated at this
# depth; at 16 the pipeline stalls at 150 ms one way and the ladder doubles.
MAX_INFLIGHT_BATCHES="${MAX_INFLIGHT_BATCHES:-64}"
# The calibration depends on this rate, not only on the timeouts. Capacity is
# about 40,000 requests/s; at half of it the backlog a view change leaves takes
# a full timeout to drain and the phases become metastable, so the same
# configuration can give 6 view changes in one run and 296 in the next.
TX_RATE="${TX_RATE:-15000}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_bedrock.sh"
