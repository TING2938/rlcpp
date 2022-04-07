
## Build demo
add_executable(draft draft.cpp)
target_link_libraries(draft ${PUBLIC_LINK_LIB})

add_subdirectory(dynet)
add_subdirectory(pycall)
add_subdirectory(dynet2pytorch/cpp)