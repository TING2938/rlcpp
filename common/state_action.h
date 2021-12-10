#ifndef __RL_STATE_ACTION_H__
#define __RL_STATE_ACTION_H__

#include "common/rl_config.h"

namespace rlcpp
{
    using State = BoxType;

    using Action = DiscreteType;

    struct Space
    {   
        Int n; // num. of action if type is `Discrete` 
        Veci shape; // shape of Box if type is `Box`  
        Vecd high; // high boundary if type is `Box` 
        Vecd low;  // low boundary if type is `Box`
        bool bDiscrete; // type is discrete if true else Box 

        State getEmptyObs()
        {
            if (this->bDiscrete)
            {
                return State(1, 0.0);
            } else 
            {
                auto n = std::accumulate(this->shape.begin(), this->shape.end(), Int(1), std::multiplies<Int>());
                return State(n, 0);
            }
        }

        Action getEmptyAction()
        {
            return Action(0);
        }
    };
}

#endif // !__RL_STATE_ACTION_H__
