#define SPDLOG_HEADER_ONLY
#include "spdlog/fmt/ostr.h"  // must be included
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"
#include "tools/vector_tools.h"

using namespace rlcpp::opt;

void custom_class_example()
{
    std::vector<int> vec = {1, 3, 5, 67};

    auto logger = spdlog::basic_logger_mt("ln", "logs/ln.dat");
    logger->info("vector print: {}", vec);
}

int main()
{
    custom_class_example();
}