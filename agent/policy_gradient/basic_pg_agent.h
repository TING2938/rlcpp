// basic policy gradient algorithm

#ifndef __RLCPP_BASIC_PG_AGENT_H__
#define __RLCPP_BASIC_PG_AGENT_H__

#include <algorithm>
#include "agent/agent.h"
#include "tools/memory_reply.h"
#include "tools/rand.h"
#include "network/dynet_network/dynet_network.h"

namespace rlcpp
{
    // observation space: continuous
    // action space: discrete
    class Basic_PG_Agent : public Agent
    {   
        using Expression = dynet::Expression;

    public:
        Basic_PG_Agent(const std::vector<dynet::Layer>& layers, 
                  Int max_memory_size, Int batch_size,
                  Int update_target_steps = 500, Float gamma = 0.99)
        : network(), trainer(network.model)
        {
            this->network.build_model(layers);

            this->trainer.clip_threshold = 1.0;
            this->trainer.learning_rate = 5e-4;
            
            this->memory.init(max_memory_size);
            this->obs_dim = layers.front().input_dim;
            this->act_n = layers.back().output_dim;
            this->gamma = gamma;

            this->learn_step = 0;

            this->batch_state.resize(batch_size * this->obs_dim, 0);
            this->batch_action.resize(batch_size, 0);
            this->batch_reward.resize(batch_size, 0);
            this->batch_next_state.resize(batch_size * this->obs_dim, 0);
            this->batch_done.resize(batch_size, 0);
            this->batch_target_Q.resize(batch_size * this->act_n, 0);
        }

        // 根据观测值，采样输出动作，带探索过程
        void sample(const State &obs, Action *action) override
        {
            Vecf act_prob(this->act_n, 0);
            this->network.predict(obs, &act_prob);
            
        }

        // 根据输入观测值，预测下一步动作
        void predict(const State &obs, Action *action) override
        {
            Vecf act_prob(this->act_n, 0);
            this->network.predict(obs, &act_prob);
            action->front() = std::max_element(act_prob.begin(), act_prob.end()) - act_prob.begin();
        }

        void store(const State &state, const Action &action, Float reward, const State &next_state, bool done) override 
        {
            this->memory.store(state, action, reward, next_state, done);
        }

        Float learn() override 
        {
            this->memory.sample_onedim(this->batch_state, this->batch_action, this->batch_reward, this->batch_next_state, this->batch_done);
            unsigned batch_size = this->batch_reward.size();
            
            this->network.predict(this->batch_next_state, &this->batch_target_Q);

            Vecf target_values(batch_size);            
            for (int i = 0; i < batch_size; i++)
            {
                Float maxQ = *std::max_element(this->batch_target_Q.begin() + i * this->act_n, this->batch_target_Q.begin() + (i+1) * this->act_n);
                target_values[i] = this->batch_reward[i] + this->gamma * maxQ * (1 - this->batch_done[i]);
            }
            
            dynet::ComputationGraph cg;
            Expression batch_state_expr = dynet::input(cg, dynet::Dim({unsigned(this->obs_dim)}, batch_size), this->batch_state);
            Expression batch_Q_expr = this->network.nn.run(batch_state_expr, cg);
            Expression picked_values_expr = dynet::pick(batch_Q_expr, {this->batch_action.begin(), this->batch_action.end()});
            Expression target_values_expr = dynet::input(cg, dynet::Dim({1}, batch_size), target_values);
            Expression loss = dynet::sum_batches(dynet::squared_distance(picked_values_expr, target_values_expr));
            Float loss_value = dynet::as_scalar(cg.forward(loss)); 
            cg.backward(loss);
            this->trainer.update();
            
            this->learn_step += 1;
            return loss_value;
        }

    private:
        Int obs_dim; // dimension of observation space
        Int act_n;   // num. of action
        Float gamma;

        size_t learn_step;
        size_t update_target_steps;

        // input: state, output: probability of each action
        Dynet_Network network;
        dynet::AdamTrainer trainer;

        RandomReply memory;
        Vecf batch_state;
        Vecf batch_action;
        Vecf batch_reward;
        Vecf batch_next_state;
        std::vector<bool> batch_done;
        Vecf batch_target_Q;
    }; // !class

} // !namespace rlcpp

#endif // !__RLCPP_BASIC_PG_AGENT_H__
