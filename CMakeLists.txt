project(rlcpp)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "-msse3 -mavx -msse3 -mavx")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)

include_directories(${Python3_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/third_party/tiny-dnn)
include_directories(${PROJECT_SOURCE_DIR}/third_party/eigen)

set(PUBLIC_LINK_LIB dynet Threads::Threads ${Python3_LIBRARIES})


file(GLOB TRAIN_SRC_FILES ${PROJECT_SOURCE_DIR}/train/*.cpp)

# Building it
foreach(SRC_PATH ${TRAIN_SRC_FILES})
    message("src path: ${SRC_PATH}")
    get_filename_component(_target ${SRC_PATH} NAME_WE)
    message("_target is: ${_target}")
    add_executable(${_target} ${PROTO_SRC_FILE} ${SRC_PATH})
    target_link_libraries(${_target} ${PUBLIC_LINK_LIB})
endforeach()

## Build dynet test
foreach(dynet_src train_xor train_linefit train_mnist my_mnist_test)
    add_executable(${dynet_src} ${PROJECT_SOURCE_DIR}/test/dynet/${dynet_src}.cc)
    target_link_libraries(${dynet_src} ${PUBLIC_LINK_LIB})
endforeach()

add_executable(dynet_network_test network/dynet_network/dynet_network_test.cpp)
target_link_libraries(dynet_network_test ${PUBLIC_LINK_LIB})

## Build test
add_executable(demo ${PROJECT_SOURCE_DIR}/test/demo.cpp)

add_executable(tiny_dnn_network_test ${PROJECT_SOURCE_DIR}/network/tiny_dnn/tiny_dnn_network_test.cpp)
target_link_libraries(tiny_dnn_network_test ${PUBLIC_LINK_LIB})

## py call from C++ test

add_executable(pycall test/pycall/demo.cpp)
target_link_libraries(pycall ${PUBLIC_LINK_LIB})

add_executable(call_gym test/pycall/call_gym.cpp)
target_link_libraries(call_gym ${PUBLIC_LINK_LIB})
