#define SPDLOG_HEADER_ONLY
#include "matplotlibcpp.h"
#include "spdlog/fmt/ostr.h"  // must be included
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"
#include "tools/ring_vector.h"
#include "tools/vector_tools.h"

using namespace rlcpp::opt;
namespace plt = matplotlibcpp;

void custom_class_example()
{
    std::vector<int> vec = {1, 3, 5, 67};

    auto logger = spdlog::basic_logger_mt("ln", "logs/ln.dat");
    logger->info("vector print: {}", vec);
}

void plt_example()
{
    int n = 1000;
    std::vector<double> x, y, z;

    for (int i = 0; i < n; i++) {
        x.push_back(i * i);
        y.push_back(sin(2 * M_PI * i / 360.0));
        z.push_back(log(i));

        if (i % 10 == 0) {
            // Clear previous plot
            plt::clf();
            // Plot line from given x and y data. Color is selected automatically.
            plt::plot(x, y);
            // Plot a line whose name will show up as "log(x)" in the legend.
            plt::named_plot("log(x)", x, z);

            // Set x-axis to interval [0,1000000]
            plt::xlim(0, n * n);

            // Add graph title
            plt::title("Sample figure");
            // Enable legend.
            plt::legend();
            // Display plot continuously
            plt::pause(0.01);
        }
    }
    plt::detail::_interpreter::kill();
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
    ring_vector_example();
}