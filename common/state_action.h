#pragma once

#include "common/rl_config.h"

namespace rlcpp
{
struct Space
{
    Int n;           // num. of action if type is `Discrete`
    Veci shape;      // shape of Box if type is `Box`
    Vecf high;       // high boundary if type is `Box`
    Vecf low;        // low boundary if type is `Box`
    bool bDiscrete;  // type is discrete if true else Box
};

template <typename T>
constexpr bool is_scalar_type()
{
    return std::is_scalar<T>::value;
}

template <typename T>
int type_size(const T&)
{
    return 0;
}

template <typename T>
int type_size(const std::vector<T>& vec)
{
    return vec.size();
}

}  // namespace rlcpp
