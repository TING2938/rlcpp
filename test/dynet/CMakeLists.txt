
file(GLOB DYNET_SRC_FILES *.cpp *.cc)

foreach(SRC_PATH ${DYNET_SRC_FILES})
    get_filename_component(_target ${SRC_PATH} NAME_WE)
    add_executable(${_target} ${SRC_PATH})
    target_link_libraries(${_target} cpptools::asan rlcpp::rlcpp cpptools::ct)
endforeach()
