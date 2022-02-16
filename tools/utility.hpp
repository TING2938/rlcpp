#ifndef __RLCPP_UTILITY_HPP__
#define __RLCPP_UTILITY_HPP__

#include <ctime>
#include <string>

namespace rlcpp
{

/**
 * @brief 获取本地时间日期字符串
 * @param fmt 时间日期格式，参见 https://en.cppreference.com/w/cpp/io/manip/put_time
 * @return 时间日期字符串
 */
inline std::string localTime(const char* fmt = "%Y-%m-%d %H:%M:%S %A")
{
    time_t t;
    char buf[500];
    t = time(NULL);
    strftime(buf, 500, fmt, localtime(&t));
    return buf;
}


}  // namespace rlcpp
#endif  // !__RLCPP_UTILITY_HPP__