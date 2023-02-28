
find_package(Torch QUIET)
if(Torch_FOUND)
    include_directories(${TORCH_INCLUDE_DIRS})

    add_executable(dynet2torch_xor xor_model.cpp)
    target_link_libraries(dynet2torch_xor "${TORCH_LIBRARIES}")
    set_property(TARGET dynet2torch_xor PROPERTY CXX_STANDARD 14)

    add_executable(dynet2torch_mnist mnist_model.cpp)
    target_link_libraries(dynet2torch_mnist "${TORCH_LIBRARIES}")
    set_property(TARGET dynet2torch_mnist PROPERTY CXX_STANDARD 14)
else()
    message("Torch Not Found, please set CMAKE_PREFIX_PATH or Torch_DIR for dynet2torch code")
endif()