project(rlcpp)
cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 14)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

set(PUBLIC_LINK_LIB dynet Threads::Threads ${Python3_LIBRARIES})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

add_subdirectory(train)
add_subdirectory(test)

