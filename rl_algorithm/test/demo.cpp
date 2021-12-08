#include "../rl_config.h"
#include <iostream>

int main()
{
    /*
    Veci vec = {1, 3, 5, 6, 43, 54, 75};

    Veci vec2;
    vec2 = {vec.begin(), vec.end()};
    std::cout << vec2.size() << " " << vec2.front() << " " << vec.back() << std::endl;
    */

    srand((unsigned)time(NULL));
    for (int i = 0; i < 10; i++)
        std::cout << rand() % 4 << '\t';
    std::cout << std::endl;
    return 0;
}