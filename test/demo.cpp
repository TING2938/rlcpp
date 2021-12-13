#include "common/rl_config.h"
#include <iostream>
#include "tools/rand.h"

int main()
{
    rlcpp::Veci vec = {1, 3, 5, 6, 43, 54, 75};

    for (int i = 0; i < 10; i++)
       std::cout << rlcpp::random_choise(vec) << ' ';
    
    bool bb = false;
    std::cout << '\n' << 1 - bb << std::endl;
    return 0;
}
