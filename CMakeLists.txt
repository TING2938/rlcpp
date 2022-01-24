project(rlcpp)
cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3 -mavx -msse3 -mavx -g -fno-inline")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)

include_directories(${Python3_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/third_party/tiny-dnn)
include_directories(${PROJECT_SOURCE_DIR}/third_party/eigen)

set(PUBLIC_LINK_LIB dynet Threads::Threads ${Python3_LIBRARIES})

add_subdirectory(train)

add_subdirectory(test)

