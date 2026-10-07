#!/usr/bin/env bash
# Regenerates every checked-in binding of proto/agent.proto:
#   - learning_agent/agent_pb2.py and agent_pb2_grpc.py (Python)
#   - SmartBFT/proto/adaptive_timers/agent.pb.go (gogo Go, used by the
#     SmartBFT smallbank and sharing examples)
#   - cometbft/proto/adaptive_timers/agent.proto and agent.pb.go (a copy of
#     proto/agent.proto with cometbft's go_package, then gogo Go)
# The Rust binding under autobahn/ is compiled from the proto at cargo build
# time and needs no step here.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

generate_python() {
  if python - <<'PY'
try:
    import grpc_tools.protoc  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
  then
    python -m grpc_tools.protoc \
        -I proto \
        --python_out=learning_agent \
        --grpc_python_out=learning_agent \
        proto/agent.proto
    return 0
  fi

  protoc -I proto --python_out=learning_agent proto/agent.proto

  if command -v grpc_python_plugin >/dev/null 2>&1; then
    protoc -I proto \
        --grpc_python_out=learning_agent \
        --plugin=protoc-gen-grpc_python="$(command -v grpc_python_plugin)" \
        proto/agent.proto
  else
    echo "grpc_python_plugin not found; agent_pb2_grpc.py not regenerated." >&2
    return 1
  fi
}

generate_smartbft_go() {
  local out_dir="SmartBFT/proto/adaptive_timers"
  if ! command -v protoc-gen-gogofaster >/dev/null 2>&1; then
    echo "protoc-gen-gogofaster not found; ${out_dir}/agent.pb.go not regenerated." >&2
    echo "Install it with: go install github.com/gogo/protobuf/protoc-gen-gogofaster@v1.3.2" >&2
    return 1
  fi
  # google.protobuf.Empty must map to the gogo types package that SmartBFT vendors.
  protoc -I proto \
      --gogofaster_out="plugins=grpc,paths=source_relative,Mgoogle/protobuf/empty.proto=github.com/gogo/protobuf/types:${out_dir}" \
      proto/agent.proto
}

generate_cometbft_go() {
  local out_dir="cometbft/proto/adaptive_timers"
  if ! command -v protoc-gen-gogofaster >/dev/null 2>&1; then
    echo "protoc-gen-gogofaster not found; ${out_dir}/agent.pb.go not regenerated." >&2
    return 1
  fi
  local source_package='option go_package = "github.com/tendermint/tendermint/proto/adaptive_timers;adaptivetimers";'
  local cometbft_package='option go_package = "github.com/cometbft/cometbft/proto/adaptive_timers;adaptivetimers";'
  if ! grep -qF "${source_package}" proto/agent.proto; then
    echo "proto/agent.proto no longer declares the expected go_package; update generate_cometbft_go." >&2
    return 1
  fi
  sed "s|${source_package}|${cometbft_package}|" proto/agent.proto > "${out_dir}/agent.proto"
  # Same options as cometbft/buf.gen.yaml.
  protoc -I cometbft/proto \
      --gogofaster_out="plugins=grpc,paths=source_relative,Mgoogle/protobuf/empty.proto=github.com/gogo/protobuf/types:cometbft/proto" \
      cometbft/proto/adaptive_timers/agent.proto
}

generate_python
generate_smartbft_go
generate_cometbft_go
