#ifndef __BASIC_AGENT_H__
#define __BASIC_AGENT_H__

#include "agent/agent.h"

namespace rlcpp
{
    class Basic_agent : Agent
    {
    public:
        void init(Int obs_dim, Int act_n)
        {
            this->act_n = act_n;
            this->obs_dim = obs_dim;
        }

        // 根据观测值，采样输出动作，带探索过程
        void sample(const State &obs, Action *action) override
        {
            this->predict(obs, action);
        }

        // 根据输入观测值，预测下一步动作
        void predict(const State &obs, Action *action) override
        {
            srand((unsigned)time(NULL));
            *action = rand() % (this->act_n);
        }

    private:
        Int act_n;
        Int obs_dim;
    }; // !class Basic_agent
    
} // !namespace rlcpp
#endif // !__BASIC_AGENT_H__