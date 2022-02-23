
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})


add_executable(dynet2torch_xor xor_model.cpp)
target_link_libraries(dynet2torch_xor "${TORCH_LIBRARIES}")
set_property(TARGET dynet2torch_xor PROPERTY CXX_STANDARD 14)

add_executable(dynet2torch_mnist mnist_model.cpp)
target_link_libraries(dynet2torch_mnist "${TORCH_LIBRARIES}")
set_property(TARGET dynet2torch_mnist PROPERTY CXX_STANDARD 14)
