#!/usr/bin/env bash

protoDir="."
outDir="../rl_algorithm/environment/gym_env/proto_out"

protoc -I ${protoDir} ${protoDir}/*.proto --cpp_out=${outDir}
protoc -I ${protoDir} ${protoDir}/*.proto --grpc_out=${outDir} --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` 