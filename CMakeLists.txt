cmake_minimum_required(VERSION 3.2)

project(rlcpp)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
    # set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# check if rlcpp is being used directly or via add_subdirectory
if(NOT DEFINED RLCPP_MASTER_PROJECT)
    if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(RLCPP_MASTER_PROJECT ON)
    else()
        set(RLCPP_MASTER_PROJECT OFF)
    endif()
endif()

add_library(rlcpp INTERFACE)
add_library(rlcpp::rlcpp ALIAS rlcpp)
target_include_directories(rlcpp INTERFACE ${CMAKE_CURRENT_LIST_DIR})

## for dynet
set(DYNET_ROOT "/usr/local" CACHE PATH "the dynet installed prefix")
target_include_directories(rlcpp INTERFACE ${DYNET_ROOT}/include)   
target_link_directories(rlcpp INTERFACE ${DYNET_ROOT}/lib)
target_link_libraries(rlcpp INTERFACE dynet)

if (RLCPP_MASTER_PROJECT)
    # for third_party
    add_subdirectory(third_party/spdlog)
    add_subdirectory(third_party/pybind11)

    set(PUBLIC_LINK_LIB rlcpp::rlcpp spdlog::spdlog_header_only pybind11::embed)

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
    add_subdirectory(train)

    option(BUILD_TEST "whether to build the test progrom" ON)
    if(BUILD_TEST)
        add_subdirectory(test)
    endif()
else()
    message("-- load rlcpp::rlcpp done")
endif()
