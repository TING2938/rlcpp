#ifndef __DQN_TINYDNN_AGENT_H__
#define __DQN_TINYDNN_AGENT_H__

#include <algorithm>
#include "agent/agent.h"
#include "network/network.h"
#include "network/random_reply.h"
#include "tools/rand.h"

namespace rlcpp
{
    // observation space: continuous
    // action space: discrete
    class DQN_TinyDNN_agent : Agent
    {
    public:
        void init(Network *network, Network* target_network, 
                  Int obs_dim, Int act_n,
                  Int max_memory_size, Int batch_size,
                  Int update_target_steps = 500, Float gamma = 0.99, 
                  Float epsilon = 1.0, Float epsilon_decrease = 1e-4)
        {
            this->network = network;
            this->memory.init(max_memory_size);

            this->obs_dim = obs_dim;
            this->act_n = act_n;
            this->gamma = gamma;
            this->epsilon = epsilon;
            this->epsilon_decrease = epsilon_decrease;
            this->epsilon_lower = 0.05;

            this->learn_step = 0;

            this->batch_state.resize(batch_size);
            this->batch_action.resize(batch_size);
            this->batch_reward.resize(batch_size);
            this->batch_next_state.resize(batch_size);
            this->batch_done.resize(batch_size);
            this->batch_target_Q.resize(batch_size, Vecf(act_n, 0.0));
        }

        // 根据观测值，采样输出动作，带探索过程
        void sample(const State &obs, Action *action) override
        {
            if (randf() < this->epsilon) {
                action->front() = randd(0, this->act_n);
            }
            else {
                this->predict(obs, action);
            }
        }

        // 根据输入观测值，预测下一步动作
        void predict(const State &obs, Action *action) override
        {
            Vecf Q(this->act_n);
            this->network->predict(obs, &Q);
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
            this->memory.sample(this->batch_state, this->batch_action, this->batch_reward, this->batch_next_state, this->batch_done);
            this->network->predict(this->batch_next_state, &this->batch_target_Q); 
            for (int i = 0; i < this->batch_state.size(); i++)
            {
                Float maxQ = *std::max_element(this->batch_target_Q[i].begin(), this->batch_target_Q[i].end());
                this->batch_target_Q[i][this->batch_action[i].front()] = this->batch_reward[i] + this->gamma * maxQ * (1 - this->batch_done[i]);
            }
            auto loss_value = this->network->learn(this->batch_state, this->batch_target_Q);
            this->epsilon = std::max(this->epsilon - this->epsilon_decrease, this->epsilon_lower);

            this->learn_step += 1;
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

        Network *network;

        RandomReply memory;
        std::vector<State> batch_state;
        std::vector<Action> batch_action;
        Vecf batch_reward;
        std::vector<State> batch_next_state;
        std::vector<bool> batch_done;
        std::vector<Vecf> batch_target_Q;
    }; // !class Sarsa_agent

} // !namespace rlcpp

#endif // !__DQN_TINYDNN_AGENT_H__