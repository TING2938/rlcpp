#ifndef __DYNET_NETWORK_H__
#define __DYNET_NETWORK_H__

#include "dynet/io.h"
#include "tools/dynet_network/mlp.h"

namespace rlcpp
{
struct Dynet_Network
{
public:
    Dynet_Network() = default;

    void build_model(const std::vector<dynet::Layer>& layers)
    {
        this->input_dim = layers.front().input_dim;
        for (dynet::Layer layer : layers) {
            this->nn.append(model, layer);
        }
    }

    std::vector<float> predict(const std::vector<float>& in)
    {
        dynet::ComputationGraph cg;
        dynet::Dim dim({input_dim}, in.size() / input_dim);
        dynet::Expression x = dynet::input(cg, dim, in);
        auto y              = nn.run(x, cg);
        return as_vector(cg.forward(y));
    }

    /**
     * @brief update parameters from other network.
     *
     * [this network] = [this network] * (1 - tau) + [other network] * tau
     *
     * @param other other network
     * @param tau update rate when use soft update
     */
    void update_weights_from(const Dynet_Network& other, float tau = 1.0f)
    {
        auto params_self  = this->model.parameters_list();
        auto params_other = other.model.parameters_list();
        for (unsigned int i = 0; i < params_self.size(); i++) {
            auto x = params_self[i]->values;
            auto y = params_other[i]->values;
            for (unsigned int j = 0; j < x.d.size(); j++) {
                x.v[j] = x.v[j] * (1 - tau) + y.v[j] * tau;
            }
        }
    }

    void save(const std::string& model_name, const std::string& key = "", bool append = false)
    {
        dynet::TextFileSaver saver(model_name, append);
        saver.save(this->model, key);
    }

    void load(const std::string& model_name, const std::string& key = "")
    {
        dynet::TextFileLoader loader(model_name);
        loader.populate(this->model, key);
    }

public:
    dynet::ParameterCollection model;
    dynet::MLP nn;
    unsigned int input_dim = 0;
};

std::ostream& operator<<(std::ostream& os, const Dynet_Network& network)
{
    return os << network.nn;
}

}  // namespace rlcpp

namespace dynet
{
inline Expression clip(const Expression& expr,
                       float low,
                       float up,
                       ComputationGraph& g,
                       Device* device = dynet::default_device)
{
    auto dim = expr.dim();
    return dynet::max(dynet::min(expr, dynet::constant(g, dim, up, device)), dynet::constant(g, dim, low, device));
}
}  // namespace dynet

#endif  // !__DYNET_NETWORK_H__