#ifndef __BASIC_AGENT_H__
#define __BASIC_AGENT_H__

#define RLCPP_STATE_TYPE 0
#define RLCPP_ACTION_TYPE 0

#include <algorithm>
#include "tools/random_tools.h"
#include "agent/agent.h"

namespace rlcpp
{
    class Sarsa_agent : public Agent 
    {
    public:
        void init(Int obs_n, Int act_n, Real learning_rate = 0.01, Real gamma = 0.9, Real e_greed = 0.1)
        {
            this->act_n = act_n;
            this->obs_n = obs_n;
            this->learning_rate = learning_rate;
            this->gamma = gamma;
            this->e_greed = e_greed;
            this->Q.resize(obs_n, Vecf(act_n, 0.0));
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
                *action = randd(0, this->act_n);
            }
        }

        // 根据输入观测值，预测下一步动作
        void predict(const State &obs, Action *action) override
        {
            auto &Q_list = this->Q[obs];
            auto maxQ = max(Q_list);
            Veci action_list;
            for (int i = 0; i < Q_list.size(); i++)
            {
                if (Q_list[i] == maxQ)
                    action_list.push_back(i);
            }
            *action = random_choise(action_list);
        }

        void store(const State &state, const Action &action, Real reward, const State &next_state, bool done) override {}

        Real learn() override { return 0.0f; }

        void learn(const State &obs, const Action &action, Real reward, const State &next_obs, const Action& next_action, bool done)
        {
            auto predict_Q = this->Q[obs][action];
            Real target_Q = 0.0;
            if (done)
            {
                target_Q = reward;
            }
            else
            {
                target_Q = reward + this->gamma * this->Q[next_obs][next_action];
            }
            this->Q[obs][action] += this->learning_rate * (target_Q - predict_Q);
        }

    private:
        Int act_n;
        Int obs_n;
        Real learning_rate;
        Real gamma;
        Real e_greed;
        std::vector<Vecf> Q;
    }; // !class Sarsa_agent

} // !namespace rlcpp

#endif // !__BASIC_AGENT_H__