#ifndef __BASIC_AGENT_H__
#define __BASIC_AGENT_H__

#include <algorithm>
#include "agent/agent.h"
#include "network/network.h"
#include "network/random_reply.h"

namespace rlcpp
{
    // observation space: continuous
    // action space: discrete
    class DQN_agent : Agent
    {
    public:
        void init(Network *network, Int obs_dim, Int act_n,
                  Int max_memory_size, Int batch_size,
                  Int update_target_steps = 200, Float learning_rate = 0.01,
                  Float gamma = 0.9, Float e_greed = 0.1)
        {
            this->network = network;
            this->target_network = network->deepCopy();
            this->memory.init(max_memory_size);
            this->obs_dim = obs_dim;
            this->act_n = act_n;
            this->learning_rate = learning_rate;
            this->gamma = gamma;
            this->e_greed = e_greed;
            this->global_step = 0;
            this->update_target_steps = update_target_steps;

            this->batch_state.resize(batch_size);
            this->batch_action.resize(batch_size);
            this->batch_reward.resize(batch_size);
            this->batch_next_state.resize(batch_size);
            this->batch_done.resize(batch_size);
        }

        // 根据观测值，采样输出动作，带探索过程
        void sample(const State &obs, Action *action) override
        {
            if (((Float)rand() / (Float)((unsigned)RAND_MAX + 1)) < (1.0 - this->e_greed))
            {
                this->predict(obs, action);
            }
            else
            {
                action->front() = rand() % (this->act_n);
            }
        }

        // 根据输入观测值，预测下一步动作
        void predict(const State &obs, Action *action) override
        {
            auto ret = this->network->predict({obs});
            action->front() = ret.front();
        }

        void store(const State &state, const Action &action, Float reward, const State &next_state, bool done)
        {
            this->memory.store(state, action, reward, next_state, done);
        }

        void learn(Int eposides)
        {
            if (this->global_step % this->update_target_steps == 0)
            {
                this->target_network->update_weights(this->network);
            }
            this->global_step++;

            for (int eposide = 0; eposide < eposides; eposide++)
            {
                this->memory.sample(this->batch_state, this->batch_action, this->batch_reward, this->batch_next_state, this->batch_done);
                // get Q predict
                auto pred_action_value = this->network->predict(batch_state);

                // get max(Q') from target network
                auto next_pred_value = this->target_network->predict(batch_next_state);
                

            }
        }

    private:
        Int obs_dim; // dimension of observation space
        Int act_n;   // num. of action
        Float gamma;
        Float learning_rate;
        Float e_greed;

        Int update_target_steps;
        size_t global_step;

        Network *network;
        std::shared_ptr<Network> target_network;

        RandomReply memory;
        std::vector<State> batch_state;
        std::vector<Action> batch_action;
        Vecf batch_reward;
        std::vector<State> batch_next_state;
        std::vector<bool> batch_done;
    }; // !class Sarsa_agent

} // !namespace rlcpp

#endif // !__BASIC_AGENT_H__