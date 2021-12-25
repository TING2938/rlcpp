#ifndef __DYNET_NETWORK_H__
#define __DYNET_NETWORK_H__

#define NN_TINYDNN

#include "network/network.h"
#include <tiny_dnn/tiny_dnn.h>
#include <sstream>

namespace rlcpp
{
    template <typename E>
    class TinyDNN_Network : public Network
    {
    public:
        void predict(const std::vector<State> &batch_state, std::vector<Vecf> *batch_out) override
        {
            for (int i = 0; i < batch_state.size(); i++)
            {
                this->predict(batch_state[i], &(*batch_out)[i]);
            }
        }

        void predict(const State &state, Vecf *out) override
        {
            auto ret = this->nn.predict(state);
            std::copy(ret.begin(), ret.end(), out->begin());
        }

        Float learn(const std::vector<State> &batch_state, const std::vector<Vecf> &batch_target_value) override
        {
            auto on_enumerate_epoch = [&]()
            {
                loss = nn.get_loss<E>(batch_state, batch_target_value);
            };
            this->nn.fit<E>(this->opt, batch_state, batch_target_value, this->minibatch_size, this->nepochs, []() {}, on_enumerate_epoch);
            return this->loss;
        }

        void update_weights_from(const Network *other) override
        {
            auto other_network = (TinyDNN_Network<E>*)other;
            for (int i = 0; i < this->nn.layer_size(); i++)
            {
                std::vector<tiny_dnn::vec_t *> wd = this->nn[i]->weights();
                auto ws = other_network->nn[i]->weights();
                assert(ws.size() == wd.size());
                for (int j = 0; j < ws.size(); j++)
                {
                    assert(ws[j]->size() == wd[j]->size());
                    *(wd[j]) = *(ws[j]);
                }
            }
        }

        tiny_dnn::network<tiny_dnn::sequential> nn;
        tiny_dnn::adamax opt;

        size_t minibatch_size = 1; // 16 samples for each network weight update
        int nepochs = 1;      // 2000 presentation of all samples
    
    private:
        Float loss;
    };
} // namespace rlcpp

#endif // !__DYNET_NETWORK_H__
