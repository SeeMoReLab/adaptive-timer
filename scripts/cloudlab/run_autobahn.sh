#!/usr/bin/env bash
# CloudLab runner for Autobahn (autobahn root workspace: one primary and one
# worker process per replica). All phases live in lib_rustbft.sh; this file
# only pins the protocol knobs.

PROTOCOL_NAME="autobahn"
SUPPORTED_MODES="normal learning sharing scheduled"
AGENT_PROTOCOL="autobahn"
SCHEDULE_FILE_REL="learning_agent/config/schedule_autobahn.csv"
FAILURE_SPEC_DEFAULT_REL="config/autobahn/failure_spec.xml"
NS_PREFIX="abn-r"
HOST_IF_PREFIX="abh"
NS_IF_PREFIX="abv"
BRIDGE_NAME="br-autobahn"
LAUNCH_KIND="autobahn"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_rustbft.sh"
