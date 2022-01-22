#include "common/rl_config.h"
#include <iostream>
#include "tools/random_tools.h"
#include "tools/vector_tools.h"


int main()
{
    rlcpp::Vecf vec = {1, 3, 5};
    rlcpp::Vecf prob = {0.7, 0.2, 0.1};
    vec *= 2.f;
    std::cout << vec << std::endl;

}
