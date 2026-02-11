#!/usr/bin/env bash
set -euo pipefail

if python - <<'PY'
try:
    import grpc_tools.protoc  # noqa: F401
    print("grpc_tools available")
except Exception:
    raise SystemExit(1)
PY
then
  python -m grpc_tools.protoc \
      -I proto \
      --python_out=learning_agent \
      --grpc_python_out=learning_agent \
      proto/agent.proto
  exit 0
fi

protoc -I proto --python_out=learning_agent proto/agent.proto

if command -v grpc_python_plugin >/dev/null 2>&1; then
  protoc -I proto \
      --grpc_python_out=learning_agent \
      --plugin=protoc-gen-grpc_python="$(command -v grpc_python_plugin)" \
      proto/agent.proto
else
  echo "grpc_python_plugin not found; agent_pb2_grpc.py not regenerated." >&2
fi
