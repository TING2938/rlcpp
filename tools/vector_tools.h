/**
 * @file cout_operator.h
 * @author Ting Ye (yeting2938@163.com)
 * @brief some operator for ostream
 * @version 0.1
 * @date 2022-01-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <algorithm>
#include <numeric>
#include <ostream>
#include <vector>

namespace rlcpp
{
    template <typename T>
    T sum(const std::vector<T>& vec)
    {
        return std::accumulate(vec.begin(), vec.end(), T(0));
    }

    template <typename T>
    double mean(const std::vector<T>& vec)
    {
        if (vec.empty()) 
            return 0.0;
        return double(sum(vec)) / vec.size();
    }

    template <typename T>
    T max(const std::vector<T>& vec)
    {
        return *std::max_element(vec.begin(), vec.end());
    }

    template <typename T>
    T argmax(const std::vector<T>& vec)
    {
        return std::max_element(vec.begin(), vec.end()) - vec.begin();
    }

    template <typename T>
    T min(const std::vector<T>& vec)
    {
        return *std::min_element(vec.begin(), vec.end());
    }

    template <typename T>
    T argmin(const std::vector<T>& vec)
    {
        return std::min_element(vec.begin(), vec.end()) - vec.begin();
    }
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec)
{
    os << '{';
    if (vec.size() > 0)
    {
        os << vec[0];
    }
    for (int i = 1; i < vec.size(); i++)
    {
        os << ", " << vec[i];
    }
    os << '}';
    return os;
}

