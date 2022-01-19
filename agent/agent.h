/**
 * @file agent.h
 * @author Ting Ye (yeting2938@163.com)
 * @brief 所有agent的抽象基类
 * @version 0.1
 * @date 2022-01-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __RL_AGENT_H__
#define __RL_AGENT_H__

#include "common/rl_config.h"
#include "common/state_action.h"

namespace rlcpp
{
    struct Agent
    {
        /**
         * @brief  根据观测值，采样输出动作，带探索过程
         * @param[in]  obs 输入的环境状态
         * @param[out]  action 预测得到的动作
         * @return None
         */
        virtual void sample(const State &obs, Action *action) = 0;

        /**
         * @brief  根据输入观测值，预测下一步动作，不带探索过程
         * @note   
         * @param[in]  obs 输入的环境状态
         * @param[out] action 预测得到的动作
         * @return None
         */
        virtual void predict(const State &obs, Action *action) = 0;

        /**
         * @brief  存储一个TD的经验
         * @param[in]  state  状态
         * @param[in]  action 动作
         * @param[in]  reward 奖励
         * @param[in]  next_state 下一步状态
         * @param[in]  done 是否结束
         * @return None
         */
        virtual void store(const State &state, const Action &action, Float reward, const State &next_state, bool done) = 0;

        /**
         * @brief  学习一次，返回此次学习的error
         * @return 此次学习的error
         */
        virtual Float learn() = 0;

    }; // !sturct Agent

} // !namespace rlcpp

#endif // !__RL_AGENT_H__