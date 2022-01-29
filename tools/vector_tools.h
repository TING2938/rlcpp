/**
 * @file cout_operator.h
 * @author Ting Ye (yeting2938@163.com)
 * @brief some operator for vector
 * @version 0.1
 * @date 2022-01-21
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __RLCPP_VECTOR_TOOLS_H__
#define __RLCPP_VECTOR_TOOLS_H__

#include <algorithm>
#include <numeric>
#include <ostream>
#include <vector>

#include "common/rl_config.h"

namespace rlcpp
{
#define ALL(vec) vec.begin(), vec.end()

template <typename T>
inline T sum(const std::vector<T>& vec)
{
    return std::accumulate(ALL(vec), T(0));
}

template <typename T>
inline Real mean(const std::vector<T>& vec)
{
    if (vec.empty())
        return 0.0;
    return Real(sum(vec)) / Real(vec.size());
}

template <typename T>
inline T max(const std::vector<T>& vec)
{
    return *std::max_element(ALL(vec));
}

template <typename T>
inline T argmax(const std::vector<T>& vec)
{
    return std::max_element(ALL(vec)) - vec.begin();
}

template <typename T>
inline T min(const std::vector<T>& vec)
{
    return *std::min_element(ALL(vec));
}

template <typename T>
inline T argmin(const std::vector<T>& vec)
{
    return std::min_element(ALL(vec)) - vec.begin();
}

template <typename T>
inline T prod(const std::vector<T>& vec)
{
    return std::accumulate(ALL(vec), T(1), std::multiplies<T>());
}
}  // namespace rlcpp

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    os << '{';
    if (vec.size() > 0) {
        os << vec[0];
    }
    for (int i = 1; i < vec.size(); i++) {
        os << ", " << vec[i];
    }
    os << '}';
    return os;
}

#define RLCPP_VEC_OP_VEC_RET_VEC(op)                                                          \
    template <typename T>                                                                     \
    inline std::vector<T> operator op(const std::vector<T>& vec1, const std::vector<T>& vec2) \
    {                                                                                         \
        assert(vec1.size() == vec2.size());                                                   \
        std::vector<T> ret(vec1.size());                                                      \
        for (int i = 0; i < vec1.size(); i++) {                                               \
            ret[i] = vec1[i] op vec2[i];                                                      \
        }                                                                                     \
        return ret;                                                                           \
    }

RLCPP_VEC_OP_VEC_RET_VEC(+)
RLCPP_VEC_OP_VEC_RET_VEC(-)
RLCPP_VEC_OP_VEC_RET_VEC(*)
RLCPP_VEC_OP_VEC_RET_VEC(/)
RLCPP_VEC_OP_VEC_RET_VEC(%)

#define RLCPP_VEC_OP_VEC_INPLACE(op)                                                     \
    template <typename T>                                                                \
    inline std::vector<T>& operator op(std::vector<T>& vec1, const std::vector<T>& vec2) \
    {                                                                                    \
        assert(vec1.size() == vec2.size());                                              \
        for (int i = 0; i < vec1.size(); i++) {                                          \
            vec1[i] op vec2[i];                                                          \
        }                                                                                \
        return vec1;                                                                     \
    }

RLCPP_VEC_OP_VEC_INPLACE(+=)
RLCPP_VEC_OP_VEC_INPLACE(-=)
RLCPP_VEC_OP_VEC_INPLACE(*=)
RLCPP_VEC_OP_VEC_INPLACE(/=)
RLCPP_VEC_OP_VEC_INPLACE(%=)

#define RLCPP_VEC_OP_NUM_RET_VEC(op)                                           \
    template <typename T>                                                      \
    inline std::vector<T> operator op(const std::vector<T>& vec, const T& num) \
    {                                                                          \
        std::vector<T> ret(vec.size());                                        \
        for (int i = 0; i < vec.size(); i++) {                                 \
            ret[i] = vec[i] op num;                                            \
        }                                                                      \
        return ret;                                                            \
    }

RLCPP_VEC_OP_NUM_RET_VEC(+)
RLCPP_VEC_OP_NUM_RET_VEC(-)
RLCPP_VEC_OP_NUM_RET_VEC(*)
RLCPP_VEC_OP_NUM_RET_VEC(/)
RLCPP_VEC_OP_NUM_RET_VEC(%)

#define RLCPP_VEC_OP_NUM_INPLACE(op)                                      \
    template <typename T>                                                 \
    inline std::vector<T>& operator op(std::vector<T>& vec, const T& num) \
    {                                                                     \
        for (int i = 0; i < vec.size(); i++) {                            \
            vec[i] op num;                                                \
        }                                                                 \
        return vec;                                                       \
    }

RLCPP_VEC_OP_NUM_INPLACE(+=)
RLCPP_VEC_OP_NUM_INPLACE(-=)
RLCPP_VEC_OP_NUM_INPLACE(*=)
RLCPP_VEC_OP_NUM_INPLACE(/=)
RLCPP_VEC_OP_NUM_INPLACE(%=)

#endif  // !__RLCPP_VECTOR_TOOLS_H__