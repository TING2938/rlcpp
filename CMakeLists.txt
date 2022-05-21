cmake_minimum_required(VERSION 3.2)

project(rlcpp)

set(CMAKE_CXX_STANDARD 14)

## for dynet
set(DYNET_ROOT "/usr/local" CACHE PATH "the dynet installed prefix")
include_directories(${DYNET_ROOT}/include)
link_directories(${DYNET_ROOT}/lib)

# for third_party
add_subdirectory(third_party/spdlog)
add_subdirectory(third_party/pybind11)

include_directories(${PROJECT_SOURCE_DIR})
set(PUBLIC_LINK_LIB dynet spdlog::spdlog_header_only pybind11::embed)

# whether to use grpc
option(USE_GRPC "use grpc " OFF)
if(USE_GRPC)
    find_package(Protobuf REQUIRED)
    find_package(gRPC REQUIRED)
    include_directories(${GRPC_INCLUDE_DIRS})
    include_directories(${PROTOBUF_INCLUDE_DIRS})
    
    ####### change this dir to proto src dir #######
    set(PROTO_SRC_DIR ${PROJECT_SOURCE_DIR}/env/grpc_gym/proto_out)
    include_directories(${PROTO_SRC_DIR})
    ################################################

    file(GLOB PROTO_SRC_FILE ${PROTO_SRC_DIR}/*.cc)
    message("proto src file 3: ${PROTO_SRC_FILE}")

    add_library(protolib ${PROTO_SRC_FILE})
    set(GRPC_LINK_LIB gRPC::grpc++_reflection protobuf::libprotobuf protolib)
    set(PUBLIC_LINK_LIB ${PUBLIC_LINK_LIB} ${GRPC_LINK_LIB})
endif()

message("public link lib: ${PUBLIC_LINK_LIB}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_subdirectory(train)

option(BUILD_TEST "whether to build the test progrom" ON)
if(BUILD_TEST)
    add_subdirectory(test)
endif()
