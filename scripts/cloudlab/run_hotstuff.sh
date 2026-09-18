#!/usr/bin/env bash
# CloudLab runner for batched HotStuff (autobahn/baselines workspace).
# All phases live in lib_rustbft.sh; this file only pins the protocol knobs.

PROTOCOL_NAME="hotstuff"
SUPPORTED_MODES="normal learning scheduled"
AGENT_PROTOCOL="hotstuff"
SCHEDULE_FILE_REL="learning_agent/config/schedule_hotstuff.csv"
FAILURE_SPEC_DEFAULT_REL="config/hotstuff/failure_spec.xml"
NS_PREFIX="hs-r"
HOST_IF_PREFIX="hsh"
NS_IF_PREFIX="hsn"
BRIDGE_NAME="br-hotstuff"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_rustbft.sh"
