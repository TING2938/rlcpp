#ifndef __BASIC_AGENT_H__
#define __BASIC_AGENT_H__

#include <algorithm>
#include "agent/agent.h"
#include "network/random_reply.h"
#include "tools/rand.h"
#include "network/dynet_network/dynet_network.h"

namespace rlcpp
{
    // observation space: continuous
    // action space: discrete
    class DQN_agent : Agent
    {
    public:
        void init(const std::vector<dynet::Layer>& layers, 
                  Int obs_dim, Int act_n,
                  Int max_memory_size, Int batch_size,
                  Int update_target_steps = 200, Float gamma = 0.9, 
                  Float e_greed = 0.1, Float e_greed_decrement = 0.0)
        {
            this->network.build_model(layers);
            this->target_network.build_model(layers);

            this->memory.init(max_memory_size);
            this->obs_dim = obs_dim;
            this->act_n = act_n;
            this->gamma = gamma;
            this->e_greed = e_greed;
            this->e_greed_decrement = e_greed_decrement;

            this->global_step = 0;
            this->update_target_steps = update_target_steps;

            this->batch_state.resize(batch_size);
            this->batch_action.resize(batch_size);
            this->batch_reward.resize(batch_size);
            this->batch_next_state.resize(batch_size);
            this->batch_done.resize(batch_size);
            this->batch_Q.resize(batch_size, Vecf(act_n, 0.0));
            this->batch_target_Q.resize(batch_size, Vecf(act_n, 0.0));
        }

        // 根据观测值，采样输出动作，带探索过程
        void sample(const State &obs, Action *action) override
        {
            if (randf() < (1.0 - this->e_greed))
            {
                this->predict(obs, action);
            }
            else
            {
                action->front() = randd(0, this->act_n);
            }
            this->e_greed = std::max<Float>(0.01, this->e_greed - this->e_greed_decrement);
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

        void learn(Int eposides)
        {
            if (this->global_step % this->update_target_steps == 0)
            {
                this->target_network->update_weights_from(this->network);
            }
            this->global_step++;

            for (int eposide = 0; eposide < eposides; eposide++)
            {
                this->memory.sample(this->batch_state, this->batch_action, this->batch_reward, this->batch_next_state, this->batch_done);
                // get max(Q') from target network
                this->network->predict_batch(batch_state, &this->batch_target_Q); 
                this->target_network->predict_batch(batch_next_state, &this->batch_Q);
                for (int i = 0; i < this->batch_state.size(); i++)
                {
                    Float maxQ = *std::max_element(this->batch_Q[i].begin(), this->batch_Q[i].end());
                    this->batch_target_Q[i][this->batch_action[i].front()] = this->batch_reward[i] + this->gamma * maxQ * (1 - this->batch_done[i]);
                }
                this->network->learn(this->batch_state, this->batch_target_Q);
            }
        }

    public:
        Float e_greed;
    private:
        Int obs_dim; // dimension of observation space
        Int act_n;   // num. of action
        Float gamma;
        Float e_greed_decrement;

        Int update_target_steps;
        size_t global_step;

        Dynet_Network network;
        Dynet_Network target_network;

        RandomReply memory;
        std::vector<State> batch_state;
        std::vector<Action> batch_action;
        Vecf batch_reward;
        std::vector<State> batch_next_state;
        std::vector<bool> batch_done;
        std::vector<Vecf> batch_target_Q;
        std::vector<Vecf> batch_Q;
    }; // !class Sarsa_agent

} // !namespace rlcpp

#endif // !__BASIC_AGENT_H__