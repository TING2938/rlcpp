#include <iostream>
#include <regex>
#include "common/rl_config.h"
#include "tools/random_tools.h"
#include "tools/vector_tools.h"

int main()
{
    std::regex reg(R"(<(.*)>(.*)</\1>)");
    std::string text = "<html>value</html>";
    std::smatch m;
    bool ret = std::regex_match(text, m, reg);

    std::cout << "matched result: " << m[0].str() << std::endl;
    std::cout << "matched result: " << m[1].str() << std::endl;
    std::cout << "matched result: " << m[2].str() << std::endl;

    for (auto&& i : m) {
        std::cout << i << std::endl;
    }

    std::cout << std::boolalpha << ret << std::endl;
}
