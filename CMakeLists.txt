project(rlcpp)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 14)

# set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG  -O3 -pthread  -msse3 -mavx -msse3 -mavx")

find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Development NumPy)

include_directories(${Python3_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/tiny-dnn)
include_directories(/home/yeting/study/ML/dynet/examples/cpp-utils)

set(GRPC_LINK_LIB gRPC::grpc++_reflection protobuf::libprotobuf)
set(PUBLIC_LINK_LIB dynet Threads::Threads ${Python3_LIBRARIES} Python3::NumPy)



####### change this dir to proto src dir #######
set(PROTO_SRC_DIR ${PROJECT_SOURCE_DIR}/env/gym_env/proto_out)
include_directories(${PROTO_SRC_DIR})
################################################

file(GLOB PROTO_SRC_FILE ${PROTO_SRC_DIR}/*.cc)
message("proto src file: ${PROTO_SRC_FILE}")

file(GLOB TRAIN_SRC_FILES ${PROJECT_SOURCE_DIR}/train/*.cpp)

# Building it
foreach(SRC_PATH ${TRAIN_SRC_FILES})
    message("src path: ${SRC_PATH}")
    get_filename_component(_target ${SRC_PATH} NAME_WE)
    message("_target is: ${_target}")
    add_executable(${_target} ${PROTO_SRC_FILE} ${SRC_PATH})
    target_link_libraries(${_target} ${GRPC_LINK_LIB} ${PUBLIC_LINK_LIB})
endforeach()

## Build dynet test
foreach(dynet_src train_xor train_linefit train_mnist)
    add_executable(${dynet_src} ${PROJECT_SOURCE_DIR}/test/dynet/${dynet_src}.cc)
    target_link_libraries(${dynet_src} ${GRPC_LINK_LIB} ${PUBLIC_LINK_LIB})
endforeach()

add_executable(dynet_network_test network/dynet_network/dynet_network_test.cpp)
target_link_libraries(dynet_network_test ${GRPC_LINK_LIB} ${PUBLIC_LINK_LIB})

## Build test
add_executable(demo ${PROJECT_SOURCE_DIR}/test/demo.cpp)

add_executable(tiny_dnn_network_test ${PROJECT_SOURCE_DIR}/network/tiny_dnn/tiny_dnn_network_test.cpp)
target_link_libraries(tiny_dnn_network_test ${GRPC_LINK_LIB} ${PUBLIC_LINK_LIB})

message(${CMAKE_CXX_FLAGS_RELEASE})

## py call from C++ test

add_executable(pycall test/pycall/demo.cpp)
target_link_libraries(pycall ${GRPC_LINK_LIB} ${PUBLIC_LINK_LIB})

add_executable(call_gym test/pycall/call_gym.cpp)
target_link_libraries(call_gym ${GRPC_LINK_LIB} ${PUBLIC_LINK_LIB})
