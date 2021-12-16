#ifndef __BASIC_AGENT_H__
#define __BASIC_AGENT_H__

#include <algorithm>
#include "agent/agent.h"
#include "network/random_reply.h"
#include "tools/rand.h"
#include "network/dynet_network/dynet_network.h"

using dynet::Expression;

namespace rlcpp
{
    // observation space: continuous
    // action space: discrete
    class DQN_dynet_agent : Agent
    {
    public:
        DQN_dynet_agent(const std::vector<dynet::Layer>& layers, 
                  Int obs_dim, Int act_n,
                  Int max_memory_size, Int batch_size,
                  Int update_target_steps = 500, Float gamma = 0.99, 
                  Float epsilon = 1.0, Float epsilon_decrease = 1e-4)
        : network(obs_dim, act_n), target_network(obs_dim, act_n), trainer(network.model)
        {
            this->network.build_model(layers);
            this->target_network.build_model(layers);
            this->target_network.update_weights_from(&this->network);

            this->trainer.clip_threshold = 1.0;
            this->trainer.learning_rate = 5e-4;
            
            this->memory.init(max_memory_size);
            this->obs_dim = obs_dim;
            this->act_n = act_n;
            this->gamma = gamma;
            this->epsilon = epsilon;
            this->epsilon_decrease = epsilon_decrease;
            this->epsilon_lower = 0.05;

            this->learn_step = 0;
            this->update_target_steps = update_target_steps;

            this->batch_state.resize(batch_size * obs_dim);
            this->batch_action.resize(batch_size);
            this->batch_reward.resize(batch_size);
            this->batch_next_state.resize(batch_size * obs_dim);
            this->batch_done.resize(batch_size);
            this->batch_target_Q.resize(batch_size * act_n);
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
            Vecf Q(this->act_n);
            this->network.predict(obs, &Q);
            action->front() = std::max_element(Q.begin(), Q.end()) - Q.begin();
        }

        void store(const State &state, const Action &action, Float reward, const State &next_state, bool done)
        {
            this->memory.store(state, action, reward, next_state, done);
        }

        size_t memory_size() const 
        {
            return this->memory.size();
        }

        Float learn()
        {
            

            this->memory.sample_onedim(this->batch_state, this->batch_action, this->batch_reward, this->batch_next_state, this->batch_done);
            unsigned batch_size = this->batch_reward.size();

            // get max(Q') from target network
            this->target_network.predict(this->batch_next_state, &this->batch_target_Q);
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
            this->epsilon = std::max(this->epsilon - this->epsilon_decrease, this->epsilon_lower);
            
            this->learn_step += 1;
            if (this->learn_step % this->update_target_steps == 0)
            {
                this->target_network.update_weights_from(&this->network);
            }
            return loss_value;
        }
    public:
        Float epsilon;

    private:
        Int obs_dim; // dimension of observation space
        Int act_n;   // num. of action
        Float gamma;

        Float epsilon_decrease;
        Float epsilon_lower;

        size_t learn_step;
        size_t update_target_steps;

        Dynet_Network network;
        Dynet_Network target_network;
        dynet::AdamTrainer trainer;

        RandomReply memory;
        Vecf batch_state;
        Vecf batch_action;
        Vecf batch_reward;
        Vecf batch_next_state;
        std::vector<bool> batch_done;
        Vecf batch_target_Q;
    }; // !class Sarsa_agent

} // !namespace rlcpp

#endif // !__BASIC_AGENT_H__