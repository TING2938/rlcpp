CT_USE_SPDLOG()
CT_USE_JSON()

## Build demo
add_executable(draft draft.cpp)
target_link_libraries(draft rlcpp::rlcpp cpptools::ct cpptools::spdlog cpptools::json)

add_subdirectory(dynet)
add_subdirectory(pycall)
add_subdirectory(dynet2pytorch/cpp)