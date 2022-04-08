#define SPDLOG_HEADER_ONLY
#define _USE_MATH_DEFINES
#include "matplotlib.hpp"
#include "spdlog/fmt/ostr.h"  // must be included
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"
#include "tools/ring_vector.h"
#include "tools/vector_tools.h"

using namespace rlcpp::opt;

void custom_class_example()
{
    std::vector<int> vec = {1, 3, 5, 67};

    auto logger = spdlog::basic_logger_mt("ln", "logs/ln.dat");
    logger->info("vector print: {}", vec);
}

void ring_vector_example()
{
    rlcpp::RingVector<int> vec;
    vec.init(4);
    vec.store(1);
    fmt::print("[{}] the vec: {}\n", __LINE__, vec.lined_vector());

    vec.store(2);
    fmt::print("[{}] the vec: {}\n", __LINE__, vec.lined_vector());

    vec.store(3);
    fmt::print("[{}] the vec: {}\n", __LINE__, vec.lined_vector());

    vec.store(4);
    fmt::print("[{}] the vec: {}\n", __LINE__, vec.lined_vector());

    vec.store(5);
    fmt::print("[{}] the vec: {}\n", __LINE__, vec.lined_vector());

    vec.store(6);
    fmt::print("[{}] the vec: {}\n", __LINE__, vec.lined_vector());

    vec.store(7);
    fmt::print("[{}] the vec: {}\n", __LINE__, vec.lined_vector());

    vec.store(8);
    fmt::print("[{}] the vec: {}\n", __LINE__, vec.lined_vector());

    vec.store(9);
    fmt::print("[{}] the vec: {}\n", __LINE__, vec.lined_vector());
}

int main()
{
    // custom_class_example();
    // plt_example();
    // plt_subplot();
    // ring_vector_example();
}