/**
 * @file random_tools.h
 * @author Ting Ye (yeting2938@163.com)
 * @brief 随机数实现
 * @version 0.1
 * @date 2022-01-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __RL_RANDOM_TOOLS_H__
#define __RL_RANDOM_TOOLS_H__

#include <cassert>
#include "tools/vector_tools.h"

namespace rlcpp
{
/**
 * @brief Set the rand seed
 */
void set_rand_seed(unsigned int __seed = time(nullptr))
{
    srand(__seed);
}

/**
 * @brief [low, up) Float type
 *
 * @param low
 * @param up
 * @return Float
 */
Real randf(Real low = 0.0, Real up = 1.0)
{
    return (float)rand() / ((float)RAND_MAX + 1) * (up - low) + low;
}

/**
 * @brief [low, up) Int type
 *
 * @param low
 * @param up
 * @return Int
 */
Int randd(Int low = 0, Int up = 10)
{
    return rand() % (up - low) + low;
}

/**
 * @brief choise one element from [0, a).
 *
 * @param a vector or int, for choose from vector or [0, a)
 * @param prob The probabilities associated with each entry in a.
 *             If empty, the sample assumes a uniform distribution over all entries in a
 * @return Int
 */
Int random_choise(Int a, const Vecf& prob = {})
{
    if (prob.empty()) {
        return randd(0, a);
    } else {
        assert(std::abs(sum(prob) - Real(1.0f)) < 1e-6);
        auto r = randf(0.0f, 1.0f);
        Real s = 0.0f;
        for (Int i = 0; i < prob.size(); i++) {
            s += prob[i];
            if (s >= r)
                return i;
        }
        return -1;  // else return error
    }
}

/**
 * @brief choise one element from vector a.
 *
 * @tparam T
 * @param a vector or int, for choose from vector or [0, a)
 * @param prob The probabilities associated with each entry in a.
 *             If empty, the sample assumes a uniform distribution over all entries in a
 * @return T
 */
template <typename T>
T random_choise(const std::vector<T>& a, const Vecf& prob = {})
{
    return a[random_choise(a.size(), prob)];
}

/**
 * @brief choise `size` elements from vector a.
 *
 * @tparam T
 * @param a vector or int, for choose from vector or [0, a)
 * @param size
 * @param prob The probabilities associated with each entry in a.
 *             If empty, the sample assumes a uniform distribution over all entries in a
 * @return std::vector<T>
 */
template <typename T>
std::vector<T> random_choise(const std::vector<T>& a, Int size, const Vecf& prob = {})
{
    std::vector<T> ret(size);
    for (Int i = 0; i < size; i++) {
        ret[i] = random_choise(a, prob);
    }
    return ret;
}

/**
 * @brief choise `size` elements from [0, a).
 *
 * @param a vector or int, for choose from vector or [0, a)
 * @param size number of samples to choise
 * @param prob The probabilities associated with each entry in a.
 *             If empty, the sample assumes a uniform distribution over all entries in a
 * @return Veci
 */
Veci random_choise(Int a, Int size, const Vecf& prob = {})
{
    Veci ret(size);
    for (Int i = 0; i < size; i++) {
        ret[i] = random_choise(a, prob);
    }
    return ret;
}
}  // namespace rlcpp

#endif  // !__RL_RANDOM_TOOLS_H__