cmake_minimum_required(VERSION 3.11)

project(rlcpp)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
    # set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

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
if (NOT EXISTS ${DYNET_ROOT}/include/dynet/dynet.h)
    message(FATAL_ERROR "error of DYNET_ROOT, please set with -DDYNET_ROOT=/path/to/dynet: ${DYNET_ROOT}")
endif()
message(STATUS "DYNET_ROOT: ${DYNET_ROOT}")
target_include_directories(rlcpp INTERFACE ${DYNET_ROOT}/include)   
target_link_directories(rlcpp INTERFACE ${DYNET_ROOT}/lib)
target_link_libraries(rlcpp INTERFACE dynet)

include(FetchContent)

## find cpptools library
message(STATUS "rlcpp find cpptools library")
include(FetchContent)
FetchContent_Declare(
    cpptools
    GIT_REPOSITORY https://github.com/ting2938/cpptools.git
    GIT_TAG master
)
FetchContent_MakeAvailable(cpptools)
target_link_libraries(rlcpp INTERFACE cpptools::ct)

if (RLCPP_MASTER_PROJECT)
    include(fetch_lib)
    add_subdirectory(train)
    add_subdirectory(tools)

    option(RLCPP_BUILD_TEST "whether to build the test progrom" ON)
    if(RLCPP_BUILD_TEST)
        add_subdirectory(test)
    endif()
else()
    message(STATUS "load rlcpp::rlcpp done")
endif()
