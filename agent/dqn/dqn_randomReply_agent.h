#ifndef __RLCPP_DQN_RANDOMREPLY_AGENT_H__
#define __RLCPP_DQN_RANDOMREPLY_AGENT_H__

#include <algorithm>
#include "agent/agent.h"
#include "tools/memory_reply.h"
#include "tools/random_tools.h"
#include "network/dynet_network/dynet_network.h"

namespace rlcpp
{
    // observation space: continuous
    // action space: discrete
    class DQN_RandomReply_Agent : public Agent
    {   
        using Expression = dynet::Expression;

    public:
        DQN_RandomReply_Agent(const std::vector<dynet::Layer>& layers, 
                  Int max_memory_size, bool use_double_dqn, Int batch_size,
                  Int update_target_steps = 500, Real gamma = 0.99, 
                  Real epsilon = 1.0, Real epsilon_decrease = 1e-4)
        : network(), trainer(network.model)
        {
            this->network.build_model(layers);
            this->use_double_dqn = use_double_dqn;
            if (this->use_double_dqn)
            {
                this->target_network.build_model(layers);
                this->target_network.update_weights_from(&this->network);
                this->update_target_steps = update_target_steps;
            }

            this->trainer.clip_threshold = 1.0;
            this->trainer.learning_rate = 5e-4;
            
            this->memory.init(max_memory_size);
            this->obs_dim = layers.front().input_dim;
            this->act_n = layers.back().output_dim;
            this->gamma = gamma;
            this->epsilon = epsilon;
            this->epsilon_decrease = epsilon_decrease;
            this->epsilon_lower = 0.05;

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
            if (randf() < this->epsilon) {
                action->front() = randd(0, this->act_n);
            } else {
                this->predict(obs, action);
            }
        }

        // 根据输入观测值，预测下一步动作
        void predict(const State &obs, Action *action) override
        {
            Vecf Q(this->act_n, 0);
            this->network.predict(obs, &Q);
            action->front() = argmax(Q);
        }

        void store(const State &state, const Action &action, Real reward, const State &next_state, bool done) override 
        {
            this->memory.store({state, action, reward, next_state, done});
        }

        Real learn() override 
        {
            this->memory.sample_onedim(this->batch_state, this->batch_action, this->batch_reward, this->batch_next_state, this->batch_done);
            unsigned batch_size = this->batch_reward.size();
            
            if (this->use_double_dqn)
            {
                this->target_network.predict(this->batch_next_state, &this->batch_target_Q);
            } else {
                this->network.predict(this->batch_next_state, &this->batch_target_Q);
            }
            Vecf target_values(batch_size);            
            for (int i = 0; i < batch_size; i++)
            {
                Real maxQ = *std::max_element(this->batch_target_Q.begin() + i * this->act_n, this->batch_target_Q.begin() + (i+1) * this->act_n);
                target_values[i] = this->batch_reward[i] + this->gamma * maxQ * (1 - this->batch_done[i]);
            }
            
            dynet::ComputationGraph cg;
            Expression batch_state_expr = dynet::input(cg, dynet::Dim({unsigned(this->obs_dim)}, batch_size), this->batch_state);
            Expression batch_Q_expr = this->network.nn.run(batch_state_expr, cg);
            Expression picked_values_expr = dynet::pick(batch_Q_expr, {this->batch_action.begin(), this->batch_action.end()});
            Expression target_values_expr = dynet::input(cg, dynet::Dim({1}, batch_size), target_values);
            Expression loss = dynet::sum_batches(dynet::squared_distance(picked_values_expr, target_values_expr));
            Real loss_value = dynet::as_scalar(cg.forward(loss)); 
            cg.backward(loss);
            this->trainer.update();
            this->epsilon = std::max(this->epsilon - this->epsilon_decrease, this->epsilon_lower);
            
            this->learn_step += 1;
            if (this->use_double_dqn && (this->learn_step % this->update_target_steps == 0))
            {
                this->target_network.update_weights_from(&this->network);
            }
            return loss_value;
        }
    public:
        Real epsilon;

    private:
        Int obs_dim; // dimension of observation space
        Int act_n;   // num. of action
        Real gamma;

        Real epsilon_decrease;
        Real epsilon_lower;

        size_t learn_step;
        size_t update_target_steps;

        Dynet_Network network;
        Dynet_Network target_network;
        bool use_double_dqn;
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

#endif // !__RLCPP_DQN_RANDOMREPLY_AGENT_H__
