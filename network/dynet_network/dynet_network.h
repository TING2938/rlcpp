#ifndef __DYNET_NETWORK_H__
#define __DYNET_NETWORK_H__

#include "network/network.h"
#include "network/dynet_network/mlp.h"

namespace rlcpp
{
    class Dynet_Network : Network
    {
    public:
        void predict_batch(const std::vector<State>& batch_state, std::vector<Vecf>* batch_out) override
        {
        }

        void predict_one(const State& state, Vecf* out) override
        {
        }

        void learn(const std::vector<State>& batch_state, const std::vector<Vecf>& batch_target_value) override
        {
        }

        void update_weights(const Network* other) override
        {   
        }

        std::shared_ptr<Network> deepCopy() override
        {
        }
    };
} // namespace rlcpp

#endif // !__DYNET_NETWORK_H__