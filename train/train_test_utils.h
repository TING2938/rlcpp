/**
 * @file train_test_utils.h
 * @author Ting Ye (yeting2938@163.com)
 * @brief 训练与测试流程
 * @version 0.1
 * @date 2022-01-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __RLCPP_TRAIN_TEST_UTILS_H__
#define __RLCPP_TRAIN_TEST_UTILS_H__

#include "env/env.h"
#include "agent/agent.h"


using namespace rlcpp;
using std::vector;

void train_pipeline_progressive(Env &env, Agent &agent, Float score_threshold, Int n_episode, Int learn_start = 100, Int print_every = 10)
{
    auto obs = env.obs_space().getEmptyObs();
    auto next_obs = env.obs_space().getEmptyObs();
    auto action = env.action_space().getEmptyAction();
    Float rwd;
    bool done;

    Vecf rewards, losses;
    for (int i_episode = 0; i_episode < n_episode; i_episode++)
    {
        Float reward = 0.0;
        env.reset(&obs);

        for (int t = 0; t < env.max_episode_steps; t++)
        {
            agent.sample(obs, &action);
            env.step(action, &next_obs, &rwd, &done);
            agent.store(obs, action, rwd, next_obs, (t == env.max_episode_steps - 1) ? false : done);
            reward += rwd;
            if (i_episode > learn_start)
            {
                auto loss = agent.learn();
                losses.push_back(loss);
            }
            if (done)
                break;
            obs = next_obs;
        }
        rewards.push_back(reward);

        if (i_episode % print_every == 0)
        {
            auto len = std::min<size_t>(rewards.size(), 100);
            auto score = std::accumulate(rewards.end() - len, rewards.end(), Float(0.0)) / len;
            printf("===========================\n");
            printf("i_eposide: %d\n", i_episode);
            printf("100 games mean reward: %f\n", score);
            if (losses.size() > 0)
            {
                auto len = std::min<size_t>(losses.size(), 100);
                auto loss = std::accumulate(losses.end() - len, losses.end(), Float(0.0)) / len;
                printf("100 games mean loss: %f\n", loss);
            }
            printf("===========================\n\n");
            if (score >= score_threshold)
                break;
        }
    }
}

void train_pipeline_conservative(Env &env, Agent &agent, Float score_threshold, Int n_epoch = 500, Int n_rollout = 100, Int n_train = 1000, Int learn_start = 0, bool early_stop = true)
{
    auto obs = env.obs_space().getEmptyObs();
    auto next_obs = env.obs_space().getEmptyObs();
    auto action = env.action_space().getEmptyAction();
    Float rwd;
    bool done;

    for (int i_epoch = 0; i_epoch < n_epoch; i_epoch++)
    {
        Vecf rewards, losses;
        for (int rollout = 0; rollout < n_rollout; rollout++)
        {
            Float reward = 0.0;
            env.reset(&obs);
            for (int t = 0; t < env.max_episode_steps; t++)
            {
                agent.sample(obs, &action);
                env.step(action, &next_obs, &rwd, &done);
                agent.store(obs, action, rwd, next_obs, (t == env.max_episode_steps - 1) ? false : done);
                reward += rwd;
                if (done)
                {
                    break;
                }
                obs = next_obs;
            }
            rewards.push_back(reward);
        }

        if (i_epoch > learn_start)
        {
            for (int i = 0; i < n_train; i++)
            {
                auto loss = agent.learn();
                losses.push_back(loss);
            }
        }

        if (i_epoch % 1 == 0)
        {
            auto mean_reward = std::accumulate(rewards.begin(), rewards.end(), Float(0.0)) / rewards.size();
            printf("===========================\n");
            printf("i_epoch: %d\n", i_epoch);
            printf("Average score of %d rollout games: %f\n", n_rollout, mean_reward);
            if (i_epoch > learn_start)
            {
                auto mean_loss = std::accumulate(losses.begin(), losses.end(), Float(0.0)) / losses.size();
                printf("Average training loss: %f\n", mean_loss);
            }
            printf("===========================\n\n");
            if (early_stop && mean_reward >= score_threshold)
                break;
        }
    }
}

void test(Env &env, Agent &agent, Int n_turns, bool render = false)
{
    printf("Ready to test, Press any key to coninue...\n");
    getchar();

    auto obs = env.obs_space().getEmptyObs();
    auto next_obs = env.obs_space().getEmptyObs();
    auto action = env.action_space().getEmptyAction();
    Float reward;
    bool done;

    for (int i = 0; i < n_turns; i++)
    {
        Float score = 0.0;
        env.reset(&obs);
        for (int k = 0; k < env.max_episode_steps; k++)
        {
            agent.predict(obs, &action); // predict according to Q table
            env.step(action, &obs, &reward, &done);
            if (render)
            {
                env.render();
            }
            score += reward;
            if (done)
            {
                printf("The score is %f\n", score);
                break;
            }
        }
    }
}

#endif // !__RLCPP_TRAIN_TEST_UTILS_H__