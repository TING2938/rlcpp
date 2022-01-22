#ifndef __RL_STATE_ACTION_H__
#define __RL_STATE_ACTION_H__

#include "tools/vector_tools.h"

namespace rlcpp
{
    using State = Vecf;
    using Action = Veci;

    struct Space
    {   
        Int n; // num. of action if type is `Discrete` 
        Veci shape; // shape of Box if type is `Box`  
        Vecf high; // high boundary if type is `Box` 
        Vecf low;  // low boundary if type is `Box`
        bool bDiscrete; // type is discrete if true else Box 

        State getEmptyObs()
        {
            if (this->bDiscrete)
            {
                return { 0.0 };
            } else 
            {
                return State(prod(this->shape), 0);
            }
        }

        Action getEmptyAction()
        {
            if (this->bDiscrete)
            {
                return { 0 };
            } else 
            {
                return Action(prod(this->shape), 0);
            }
        }
    };
}

#endif // !__RL_STATE_ACTION_H__
