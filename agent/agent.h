#ifndef __RL_AGENT_H__
#define __RL_AGENT_H__

#include "common/rl_config.h"
#include "common/state_action.h"

namespace rlcpp
{
    struct Agent
    {
        // 根据观测值，采样输出动作，带探索过程
        void virtual sample(const State &obs, Action *action) = 0;

        // 根据输入观测值，预测下一步动作
        void virtual predict(const State &obs, Action *action) = 0;
        
    }; // !sturct Agent

} // !namespace rlcpp

#endif // !__RL_AGENT_H__