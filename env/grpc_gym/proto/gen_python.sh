#!/usr/bin/env bash

protoDir="./"
outDir="../gymServer/"

python3 -m grpc_tools.protoc -I ${protoDir} --python_out=${outDir} --grpc_python_out=${outDir} ${protoDir}/*.proto
