#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"
// add support for logging of std::vector
#include "spdlog/fmt/ostr.h"  // must be included

#include "tools/ring_vector.h"
#include "tools/utility.hpp"
#include "tools/vector_tools.h"

using namespace rlcpp::opt;

void spdlog_example()
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

void fnm_match_test()
{
    std::string fnm = "/home/yeting/work/project/rlcpp/train/*.cpp";
    auto ret        = rlcpp::file_match(fnm);
    fmt::print("ret: {}\n", ret);
}

void reward_name_test()
{
    std::string fnm = "/home/yeting/work/project/rlcpp/build/reward_fnm_test/base";
    auto ret        = rlcpp::load_best_reward_model_name(fnm);
    fmt::print("ret: {} {}\n", ret.first, ret.second);
}

int main()
{
    // reward_name_test();
    // fnm_match_test();
    spdlog_example();
    // plt_example();
    // plt_subplot();
    // ring_vector_example();
}