#ifndef __RL_NETWORK__
#define __RL_NETWORK__

#include <memory>
#include "common/state_action.h"

namespace rlcpp
{
    class Network
    {
    public:
        virtual void predict(const std::vector<State>& batch_state, std::vector<Vecf>* batch_out) = 0;

        virtual void predict(const State& state, Vecf* out) = 0;

        virtual Float learn(const std::vector<State>& batch_state, const std::vector<Vecf>& batch_target_value) = 0;

        virtual void update_weights_from(const Network* other) = 0;
    };
} // ! namespace rlcpp

#endif // !__RL_NETWORK__
