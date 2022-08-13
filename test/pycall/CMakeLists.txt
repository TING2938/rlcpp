
file(GLOB PYCALL_SRC_FILES *.cpp *.cc)

foreach(SRC_PATH ${PYCALL_SRC_FILES})
    get_filename_component(_target ${SRC_PATH} NAME_WE)
    add_executable(${_target} ${SRC_PATH})
    target_link_libraries(${_target} cpptools::pybind11)
endforeach()
