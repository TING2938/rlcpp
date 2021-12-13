#ifndef __RL_RAND_H__
#define __RL_RAND_H__

#include <random>
#include "common/rl_config.h"

namespace rlcpp
{
    // for random engine
    namespace inner {
        std::random_device rd;
    }
    static std::default_random_engine engine(inner::rd());

    // [low, up) Float type
    Float randf(Float low=0.0, Float up=1.0)
    {
        std::uniform_real_distribution<Float> u(low, up);
        return u(engine);
    }

    // [low, up) Int type
    Int randd(Int low=0, Int up=10)
    {
        std::uniform_int_distribution<Int> u(low, up-1);
        return u(engine);
    }

    template<typename T>
    T random_choise(const std::vector<T>& vec)
    {
        return vec[randd(0, vec.size())];
    }

}
#endif // !__RL_RAND_H__