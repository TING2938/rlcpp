#ifndef __DYNET_NETWORK_H__
#define __DYNET_NETWORK_H__

#include "network/network.h"
#include "network/dynet_network/mlp.h"

namespace rlcpp
{
    struct Dynet_Network : public Network
    {
    public:

        Dynet_Network(int in_dim, int out_dim) : nn(model), input_dim(in_dim), output_dim(out_dim) {}

        void build_model(const std::vector<dynet::Layer>& layers)
        {
            for (dynet::Layer layer : layers)
            {
                nn.append(model, layer);
            }
        }

        void predict(const std::vector<State>& batch_state, std::vector<Vecf>* batch_out)
        {

        }

        void predict(const Vecf& in, Vecf* out) override
        {
            dynet::ComputationGraph cg;
            dynet::Dim dim({input_dim}, in.size() / input_dim);
            dynet::Expression x = dynet::input(cg, dim, in);
            auto y = nn.run(x, cg);
            *out = as_vector(cg.forward(y));
        }

        void learn(const std::vector<State>& batch_state, const std::vector<Vecf>& batch_target_value)
        {

        }
        
        void update_weights_from(const Network* other) 
        {

        }

    public:
        dynet::ParameterCollection model;
        dynet::MLP nn;
        unsigned int input_dim;
        unsigned int output_dim;
    };
} // namespace rlcpp

#endif // !__DYNET_NETWORK_H__