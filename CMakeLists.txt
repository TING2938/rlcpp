project(rlcpp)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 14)

find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

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

## Build dynet test
include_directories(/home/yeting/study/ML/dynet/examples/cpp-utils)
foreach(dynet_src train_xor train_linefit train_mnist)
    add_executable(${dynet_src} ${PROJECT_SOURCE_DIR}/test/dynet/${dynet_src}.cc)
    target_link_libraries(${dynet_src} dynet)
endforeach()

## Build test
add_executable(demo ${PROJECT_SOURCE_DIR}/test/demo.cpp)
