#ifndef __RL_NETWORK__
#define __RL_NETWORK__

#include <memory>
#include "common/state_action.h"

namespace rlcpp
{
    class Network
    {
    public:
        virtual std::vector<Float> predict(const std::vector<State>& batch_state);

        virtual void learn(const Vecf& train_x, const Vecf& train_y);

        virtual void update_weights(const Network* other);

        virtual std::shared_ptr<Network> deepCopy();
    };
} // ! namespace rlcpp

#endif // !__RL_NETWORK__
