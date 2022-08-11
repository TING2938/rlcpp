#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"
// add support for logging of std::vector
#include "spdlog/fmt/ostr.h"  // must be included

#include <cpptools/ct_bits/ring_vector.h>
#include <cpptools/ct_bits/vector_tools.h>
#include <fstream>
#include <iostream>
#include "nlohmann/json.hpp"
#include "tools/utility.hpp"

using namespace ct::opt;

using json = nlohmann::json;

void spdlog_example()
{
    std::vector<int> vec = {1, 3, 5, 67};

    auto logger = spdlog::basic_logger_mt("ln", "logs/ln.dat");
    logger->info("vector print: {}", vec);
}

void ring_vector_example()
{
    ct::RingVector<int> vec;
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
    auto ret        = ct::file_match(fnm);
    fmt::print("ret: {}\n", ret);
}

void reward_name_test()
{
    std::string fnm = "/home/yeting/work/project/rlcpp/build/reward_fnm_test/base";
    auto ret        = rlcpp::load_best_reward_model_name(fnm);
    fmt::print("ret: {} {}\n", ret.first, ret.second);
}

void test_json()
{
    json j = {
        {"pi", 3.14},
        {"happy", true},
        {"list", {1, 3, 6}},
    };

    std::cout << "j: " << j << std::endl;

    auto j3 = json::parse(R"(
    {
        "happy": true,
        "list": [
            1,
            3,
            6
        ],
        "pi": 3.14
    }
    )");

    auto list = j3["list"].get<std::vector<int>>();
    for (auto&& l : list) {
        fmt::print("{}\n", l);
    }

    std::ifstream fid("/home/yeting/work/project/SA_gRPC/configData/config.json");
    json j4;
    fid >> j4;

    std::cout << "j4: " << j4.dump(4) << std::endl;
    std::cout << j4["AI_application_list"] << std::endl;
}

using Vecf = std::vector<float>;
void copy(const Vecf& src, Vecf::iterator it)
{
    std::copy(src.begin(), src.end(), it);
}
void copy(int src, Vecf::iterator it)
{
    *it = src;
}
void test_copy()
{
    Vecf batch_vec(6, 0);
    Vecf vec     = {2, 3};
    float scalar = 5;
    copy(vec, batch_vec.begin() + 4);
    copy(scalar, batch_vec.begin() + 2);
    fmt::print("{}\n", batch_vec);
}

int main()
{
    test_copy();
    // test_json();
    // reward_name_test();
    // fnm_match_test();
    // spdlog_example();
    // plt_example();
    // plt_subplot();
    // ring_vector_example();
}