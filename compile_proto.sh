python -m grpc_tools.protoc \
    -I proto \
    --python_out=learning_agent \
    --grpc_python_out=learning_agent \
    proto/agent.proto

python -m grpc_tools.protoc \
    -I proto \
    --python_out=data_exchange \
    --grpc_python_out=data_exchange \
    proto/agent.proto
