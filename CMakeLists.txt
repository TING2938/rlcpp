project(rlcpp)
cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 14)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

## for dynet
set(DYNET_ROOT "/usr/local" CACHE PATH "the dynet installed prefix")
include_directories(${DYNET_ROOT}/include)
link_directories(${DYNET_ROOT}/lib)

set(PUBLIC_LINK_LIB dynet Threads::Threads ${Python3_LIBRARIES})

# whether to use grpc
option(USE_GRPC "use grpc " OFF)
if(USE_GRPC)
    find_package(Protobuf REQUIRED)
    find_package(gRPC REQUIRED)

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
