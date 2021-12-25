#ifndef __RL_RAND_H__
#define __RL_RAND_H__

#include <random>
#include "common/rl_config.h"

namespace rlcpp
{
    // [low, up) Float type
    Float randf(Float low=0.0, Float up=1.0)
    {
        return (float)rand() / ((float)RAND_MAX + 1) * (up - low) + low;
    }

    // [low, up) Int type
    Int randd(Int low=0, Int up=10)
    {
        return rand() % (up - low) + low;
    }

    template<typename T>
    T random_choise(const std::vector<T>& a)
    {
        return a[randd(0, a.size())];
    }

    Int random_choise(Int a)
    {
        return randd(0, a);
    }

    template<typename T>
    std::vector<T> random_choise(const std::vector<T>& a, Int size)
    {
        std::vector<T> ret;
        ret.reserve(size);
        for (Int i = 0; i < size; i++)
        {
            ret.push_back(a[randd(0, a.size())]);
        }
    }

    Veci random_choise(Int a, Int size)
    {
        Veci ret(size);
        for (Int i = 0; i < size; i++)
        {
            ret[i] = randd(0, a);
        }
        return ret;
    }



}
#endif // !__RL_RAND_H__