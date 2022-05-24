/**
 * @file env.h
 * @author Ting Ye (yeting2938@163.com)
 * @brief 所有env的抽象基类
 * @version 0.1
 * @date 2022-01-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include "common/rl_config.h"
#include "common/state_action.h"

namespace rlcpp
{
struct Env
{
    /**
     * @brief  创建一个环境
     * @param[in]  gameName: 环境名称
     * @return None
     */
    virtual void make(const string& gameName) = 0;

    /**
     * @brief  获取动作空间
     * @return 动作空间
     */
    virtual Space action_space() const = 0;

    /**
     * @brief  获取状态空间
     * @return 状态空间
     */
    virtual Space obs_space() const = 0;

    /**
     * @brief  在环境中执行一次
     * @param[in]  action  此次执行下发的动作
     * @param[out]  next_obs 执行完后的状态
     * @param[out]  reward 执行获得的奖励
     * @param[out]  done  执行后，环境是否结束
     * @return None
     */
    virtual void step(const Action& action, State* next_obs, Real* reward, bool* done) = 0;

    /**
     * @brief  重置环境
     * @param[out]  obs 重置后的环境状态
     * @return None
     */
    virtual void reset(State* obs) = 0;

    /**
     * @brief  关闭环境
     * @return None
     */
    virtual void close() = 0;

    /**
     * @brief  渲染一帧
     * @return None
     */
    virtual void render() = 0;

    /**
     * @brief  最大回合数
     */
    size_t max_episode_steps;
};
}  // namespace rlcpp
