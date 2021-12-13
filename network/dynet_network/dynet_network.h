#ifndef __DYNET_NETWORK_H__
#define __DYNET_NETWORK_H__

#include "network/network.h"
#include "network/dynet_network/mlp.h"

namespace rlcpp
{
    class Dynet_Network : Network
    {
    public:
        Vecf predict(const std::vector<State>& batch_state) override
        {
        }

        void learn(const Vecf& train_x, const Vecf& train_y) override
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