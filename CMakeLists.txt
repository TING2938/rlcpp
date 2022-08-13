cmake_minimum_required(VERSION 3.11)

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
message(STATUS "DYNET_ROOT: ${DYNET_ROOT}")
target_include_directories(rlcpp INTERFACE ${DYNET_ROOT}/include)   
target_link_directories(rlcpp INTERFACE ${DYNET_ROOT}/lib)
target_link_libraries(rlcpp INTERFACE dynet)

## find cpptools library
message(STATUS "rlcpp find cpptools library")
if (NOT TARGET cpptools::ct)
    if (NOT DEFINED CPPTOOLS_ROOT)
        message(STATUS "fetch cpptools from github")
        include(FetchContent)
        FetchContent_Declare(
            cpptools
            GIT_REPOSITORY https://github.com/ting2938/cpptools.git
        )
        FetchContent_MakeAvailable(cpptools)
    else()
        message(STATUS "use cpptools: ${CPPTOOLS_ROOT}")
        add_subdirectory(${CPPTOOLS_ROOT} cpptools_dir)
    endif()
else()
    message(STATUS "rlcpp use external target cpptools::ct")
endif()
target_link_libraries(rlcpp INTERFACE cpptools::ct)

if (RLCPP_MASTER_PROJECT)
    add_subdirectory(train)

    option(RLCPP_BUILD_TEST "whether to build the test progrom" ON)
    if(RLCPP_BUILD_TEST)
        add_subdirectory(test)
    endif()
else()
    message(STATUS "load rlcpp::rlcpp done")
endif()
