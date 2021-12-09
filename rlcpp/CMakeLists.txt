project(rlcpp)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -fPIC ${CMAKE_CXX_FLAGS}")
set(TargetName "ipcm")

find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

####### change this dir to proto src dir #######
set(PROTO_SRC_DIR ${PROJECT_SOURCE_DIR}/env/gym_env/proto_out)
################################################

include_directories(${PROTO_SRC_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB PROTO_SRC_FILE ${PROTO_SRC_DIR}/*.cc)
message("proto src file: ${PROTO_SRC_FILE}")

file(GLOB TRAIN_SRC_FILES ${PROJECT_SOURCE_DIR}/train/*.cpp)

# Building it
foreach(SRC_PATH ${TRAIN_SRC_FILES})
    message("src path: ${SRC_PATH}")
    get_filename_component(_target ${SRC_PATH} NAME_WE)
    message("_target is: ${_target}")
    add_executable(${_target} ${PROTO_SRC_FILE} ${SRC_PATH})
    target_link_libraries(${_target} gRPC::grpc++_reflection protobuf::libprotobuf)
endforeach()

