#include "env/gym_env/gym_env.h"
#include "agent/dqn/dqn_agent.h"
#include <tuple>
#include "network/tiny_dnn/tiny_dnn_network.h"

using namespace rlcpp;
using fc = tiny_dnn::fully_connected_layer;
using relu = tiny_dnn::relu_layer;

/* ====================================== */
Float learning_rate = 0.001;
Float Gamma = 0.99;
Float e_greed = 0.1;
Float e_greed_decrement = 1e-6;
Int max_reply_memory_size = 20000;
Int memory_warmup_size = 200;
Int batch_size = 32;
Int learn_freq = 5;
/* ====================================== */

Float run_episode(Env &env, DQN_agent &agent, State &obs, State &next_obs, Action &action, Float &reward, bool &done, bool bRender = false)
{
    Float total_reward = 0.0;
    env.reset(&obs);
    Int step = 0;
    while (true)
    {
        step += 1;
        agent.sample(obs, &action);
        env.step(action, &next_obs, &reward, &done);
        agent.store(obs, action, reward, next_obs, done);

        // train model
        if (agent.memory_size() > memory_warmup_size && (step % learn_freq) == 0)
        {
            agent.learn(10);
        }

        total_reward += reward;
        obs = next_obs;

        if (done)
        {
            break;
        }
    }
    return total_reward;
}

Float test_episode(Env &env, DQN_agent &agent, State &obs, State &next_obs, Action &action, Float &reward, bool &done, bool bRender = false)
{
    Vecf total_reward(5);
    for (int i = 0; i < 5; i++)
    {
        env.reset(&obs);
        Float episode_reward = 0.0;
        while (true)
        {
            agent.predict(obs, &action); // predict according to Q table
            env.step(action, &obs, &reward, &done);
            episode_reward += reward;
            if (bRender)
            {
                sleep(1);
                env.render();
            }
            if (done)
            {
                break;
            }
        }
        total_reward[i] = episode_reward;
    }
    return std::accumulate(total_reward.begin(), total_reward.end(), Float(0.0)) / total_reward.size();
}

int main()
{

    Gym_Env env("localhost:50053");
    // MountainCar-v0
    // CartPole-v0
    // CliffWalking-v0
    env.make("CartPole-v0"); // 0 up, 1 right, 2 down, 3 left
    auto action_space = env.action_space();
    auto obs_space = env.obs_space();
    assert(action_space.bDiscrete);
    assert(!obs_space.bDiscrete);
    printf("action space: %d, obs_space: %d\n", action_space.n, obs_space.shape.front());

    rlcpp::TinyDNN_Network<tiny_dnn::mse> network;
    network.nn << fc(obs_space.shape.front(), 128) << relu()
               << fc(128, 128) << relu()
               << fc(128, action_space.n);

    rlcpp::TinyDNN_Network<tiny_dnn::mse> target_network;
    target_network.nn << fc(obs_space.shape.front(), 128) << relu()
                      << fc(128, 128) << relu()
                      << fc(128, action_space.n);

    DQN_agent agent;
    agent.init(&network, &target_network,
               obs_space.shape.front(), action_space.n,
               max_reply_memory_size, batch_size,
               200, Gamma, e_greed, e_greed_decrement);

    auto obs = obs_space.getEmptyObs();
    auto next_obs = obs_space.getEmptyObs();
    auto action = action_space.getEmptyAction();
    Float reward;
    bool done;

    while (agent.memory_size() < memory_warmup_size)
    {
        run_episode(env, agent, obs, next_obs, action, reward, done);
    }

    Int max_episode = 2000;
    Int episode = 0;
    while (episode < max_episode)
    {
        for (int i = 0; i < 50; i++)
        {
            auto ret = run_episode(env, agent, obs, next_obs, action, reward, done);
            episode += 1;
        }

        auto test_ret = test_episode(env, agent, obs, next_obs, action, reward, done);
        printf("episode: %d   e_greed: %.3f   test reward: %.2f", episode, agent.e_greed, test_ret);
    }
    env.close();
}
