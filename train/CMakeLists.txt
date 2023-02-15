CT_USE_FMT()
CT_USE_PYBIND11()

# Building it
file(GLOB TRAIN_SRC_FILES *.cpp)
foreach(SRC_PATH ${TRAIN_SRC_FILES})
    get_filename_component(_target ${SRC_PATH} NAME_WE)
    add_executable(${_target} ${SRC_PATH})
    target_link_libraries(${_target} rlcpp::rlcpp cpptools::ct cpptools::fmt cpptools::pybind11)
endforeach()

