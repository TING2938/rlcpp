

# /usr/local/share/cmake/Torch
find_package(Torch REQUIRED)
MESSAGE( [Main] " TORCH_INCLUDE_DIRS = ${TORCH_INCLUDE_DIRS}") 
MESSAGE( [Main] " TORCH_LIBRARIES = ${TORCH_LIBRARIES}")  
include_directories(${TORCH_INCLUDE_DIRS})

add_executable(dynet2torch main.cpp)
target_link_libraries(dynet2torch "${TORCH_LIBRARIES}")
set_property(TARGET dynet2torch PROPERTY CXX_STANDARD 14)
