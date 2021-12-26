
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

####### change this dir to proto src dir #######
set(PROTO_SRC_DIR ${PROJECT_SOURCE_DIR}/env/grpc_gym/proto_out)
include_directories(${PROTO_SRC_DIR})
################################################

file(GLOB PROTO_SRC_FILE ${PROTO_SRC_DIR}/*.cc)
message("proto src file 3: ${PROTO_SRC_FILE}")

set(GRPC_LINK_LIB gRPC::grpc++_reflection protobuf::libprotobuf)

# Building it
file(GLOB TRAIN_SRC_FILES *.cpp)
foreach(SRC_PATH ${TRAIN_SRC_FILES})
    get_filename_component(_target ${SRC_PATH} NAME_WE)
    add_executable(${_target} ${SRC_PATH} ${PROTO_SRC_FILE})
    target_link_libraries(${_target} ${PUBLIC_LINK_LIB} ${GRPC_LINK_LIB})
endforeach()

