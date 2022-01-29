#ifndef __RL_STATE_ACTION_H__
#define __RL_STATE_ACTION_H__

#include "tools/vector_tools.h"

// discrete: 0, box: 1
#ifndef RLCPP_STATE_TYPE
#define RLCPP_STATE_TYPE 1
#endif  // !RLCPP_STATE_TYPE

#ifndef RLCPP_ACTION_TYPE
#define RLCPP_ACTION_TYPE 0
#endif  // !RLCPP_ACTION_TYPE

namespace rlcpp
{
#if RLCPP_STATE_TYPE == 0
using State = Int;
#elif RLCPP_STATE_TYPE == 1
using State  = Vecf;
#endif

#if RLCPP_ACTION_TYPE == 0
using Action = Int;
#elif RLCPP_ACTION_TYPE == 1
using Action = Vecf;
#endif

struct Space
{
    Int n;           // num. of action if type is `Discrete`
    Veci shape;      // shape of Box if type is `Box`
    Vecf high;       // high boundary if type is `Box`
    Vecf low;        // low boundary if type is `Box`
    bool bDiscrete;  // type is discrete if true else Box

    State getEmptyObs()
    {
#if RLCPP_STATE_TYPE == 0
        return 0;
#elif RLCPP_STATE_TYPE == 1
        return State(prod(this->shape), 0);
#endif
    }

    Action getEmptyAction()
    {
#if RLCPP_ACTION_TYPE == 0
        return 0;
#elif RLCPP_ACTION_TYPE == 1
        return Action(prod(this->shape), 0);
#endif
    }
};
}  // namespace rlcpp

#endif  // !__RL_STATE_ACTION_H__
