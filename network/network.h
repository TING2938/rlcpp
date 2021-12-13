#ifndef __RL_NETWORK__
#define __RL_NETWORK__

#include <memory>
#include "common/state_action.h"

namespace rlcpp
{
    class Network
    {
    public:
        virtual void predict_batch(const std::vector<State>& batch_state, std::vector<Vecf>* batch_out);

        virtual void predict_one(const State& state, Vecf* out);

        virtual void learn(const std::vector<State>& batch_state, const std::vector<Vecf>& batch_target_value);

        virtual void update_weights(const Network* other);

        virtual std::shared_ptr<Network> deepCopy();
    };
} // ! namespace rlcpp

#endif // !__RL_NETWORK__
