#include "common/rl_config.h"
#include <iostream>
#include "tools/rand.h"
#include "tools/vector_tools.h"


int main()
{
    rlcpp::set_rand_seed();
    
    rlcpp::Veci vec = {1, 3, 5};
    rlcpp::Vecf prob = {0.7, 0.2, 0.1};
    int N = 1000000;

    auto ret = rlcpp::random_choise(3, N, prob);

    std::cout << "0: " << std::count(ret.begin(), ret.end(), 0) * 1.0 / N << "\n"
              << "1: " << std::count(ret.begin(), ret.end(), 1) * 1.0 / N << "\n" 
              << "2: " << std::count(ret.begin(), ret.end(), 2) * 1.0 / N << "\n";
    
}
