#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

for node_id in 0 1 2 3; do
  osascript -e "
    tell application \"Terminal\"
      do script \"cd '${REPO_ROOT}' && python3 learning_agent/main.py --node-id ${node_id} --sharing false --model-type scheduled --schedule-file learning_agent/config/schedule_tendermint.csv --protocol tendermint\"
      activate
    end tell
  "
done
