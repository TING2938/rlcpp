#pragma once

#include "common/rl_config.h"
#include "tools/vector_tools.h"

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
}  // namespace rlcpp
