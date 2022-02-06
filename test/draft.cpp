#include <iostream>
#include "common/rl_config.h"
#include "tools/random_tools.h"
#include "tools/vector_tools.h"

int main()
{
    rlcpp::Veci aa = {1, 2, 3};
    std::cout << rlcpp::stddev(aa) << std::endl;

    std::string str = "aaac";
    std::cout << str << std::endl;

    size_t pos = 0;
    while (true) {
        pos = str.find('\n', pos);
        if (pos == std::string::npos)
            break;
        str.insert(pos + 1, 5, 'P');
        pos += 6;
    }
    std::cout << "\nafter insert: \n" << str << std::endl;
}
