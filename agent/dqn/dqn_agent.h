#ifndef __BASIC_AGENT_H__
#define __BASIC_AGENT_H__

#include <algorithm>
#include "agent/agent.h"

namespace rlcpp
{
    class DQN_agent : Agent
    {
    public:
        void init(Int obs_n, Int act_n, double learning_rate = 0.01, double gamma = 0.9, double e_greed = 0.1)
        {
            this->act_n = act_n;
            this->obs_n = obs_n;
            this->learning_rate = learning_rate;
            this->gamma = gamma;
            this->e_greed = e_greed;
            this->Q.resize(obs_n, Vecd(act_n, 0.0));
            srand((unsigned)time(NULL));
        }

        // 根据观测值，采样输出动作，带探索过程
        void sample(const State &obs, Action *action) override
        {
            if (((double)rand() / (double)((unsigned)RAND_MAX + 1)) < (1.0 - this->e_greed))
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
            auto &Q_list = this->Q[obs.front()];
            auto maxQ = *std::max_element(Q_list.begin(), Q_list.end());
            Veci action_list;
            for (int i = 0; i < Q_list.size(); i++)
            {
                if (Q_list[i] == maxQ)
                    action_list.push_back(i);
            }
            action->front() = action_list[rand() % action_list.size()];
        }

        void learn(const State &obs, const Action &action, double reward, const State &next_obs, const Action& next_action, bool done)
        {
            auto predict_Q = this->Q[obs.front()][action.front()];
            double target_Q = 0.0;
            if (done)
            {
                target_Q = reward;
            }
            else
            {
                target_Q = reward + this->gamma * this->Q[next_obs.front()][next_action.front()];
            }
            this->Q[obs.front()][action.front()] += this->learning_rate * (target_Q - predict_Q);
        }

    private:
        Int act_n;
        Int obs_n;
        double learning_rate;
        double gamma;
        double e_greed;
        std::vector<Vecd> Q;
    }; // !class Sarsa_agent

} // !namespace rlcpp

#endif // !__BASIC_AGENT_H__