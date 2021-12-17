#ifndef __DYNET_NETWORK_H__
#define __DYNET_NETWORK_H__

#include "network/network.h"
#include "network/dynet_network/mlp.h"

namespace rlcpp
{
    struct Dynet_Network : public Network
    {
    public:
        Dynet_Network(int in_dim, int out_dim) : input_dim(in_dim), output_dim(out_dim), nn(model) {}

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

        Float learn(const std::vector<State>& batch_state, const std::vector<Vecf>& batch_target_value) override
        {
            return 0.0;
        }
        
        void update_weights_from(const Network* other) 
        {
            auto other_network = (Dynet_Network*)other;
            auto params_self = this->model.parameters_list();
            auto params_other = other_network->model.parameters_list();
            assert(params_self.size() == params_other.size());
            for (int i = 0; i < params_self.size(); i++)
            {
                dynet::TensorTools::copy_elements(params_self[i]->values, params_other[i]->values);
            }
        }

    public:
        dynet::ParameterCollection model;
        dynet::MLP nn;
        unsigned int input_dim;
        unsigned int output_dim;
    };
} // namespace rlcpp

#endif // !__DYNET_NETWORK_H__