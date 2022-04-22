#define SPDLOG_HEADER_ONLY
#define _USE_MATH_DEFINES
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
    std::vector<int> vec(10);
    std::iota(vec.begin(), vec.end(), 1);

    std::vector<int> vec2 = vec;
    std::random_shuffle(vec.begin(), vec.end());
    fmt::print("vec: {}\n", vec);

    fmt::print("vec2: {}\n", vec2);
    std::random_shuffle(vec2.begin(), vec2.end());
    fmt::print("vec22: {}\n", vec2);

    std::vector<int> ind = {2, 1, 5, 4, 3};

    std::vector<int> out;

    rlcpp::gather(vec, &out, ind.begin(), ind.end());

    fmt::print("out: {}\n", rlcpp::gather(vec, {4, 1, 3}));

    // custom_class_example();
    // plt_example();
    // plt_subplot();
    // ring_vector_example();
}