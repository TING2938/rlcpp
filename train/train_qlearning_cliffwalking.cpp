#include "env/gym_env/gym_env.h"
#include "agent/qlearning/qlearning_agent.h"
#include <tuple>

using namespace rlcpp;

std::tuple<double, Int> run_episode(Env& env, Qlearning_agent& agent, State& obs, State& next_obs, Action &action, double &reward, bool &done, bool bRender = false)
{
    Int total_steps = 0;
    double total_reward = 0.0;
    env.reset(&obs);
    while (true)
    {
        agent.sample(obs, &action); // greedy sample
        env.step(action, &next_obs, &reward, &done);
        agent.learn(obs, action, reward, next_obs, done);

        obs = next_obs;
        total_reward += reward;
        total_steps += 1;
        if (bRender)
        {
            env.render();
        }
        if (done)
        {
            break;
        }
    }
    return {total_reward, total_steps};
}

void test_episode(Env& env, Qlearning_agent& agent, State& obs, State& next_obs, Action &action, double &reward, bool &done) 
{
    double total_reward = 0.0;
    env.reset(&obs);
    while (true)
    {
        agent.predict(obs, &action); // predict according to Q table
        env.step(action, &obs, &reward, &done);
        total_reward += reward;
        sleep(1);
        env.render();
        if (done)
        {
            printf("test reward = %.1f", total_reward);
            break;
        }
    }
}

int main()
{
    /* ====================================== */
    double learning_rate = 0.1;
    double gamma = 0.9;
    double e_greed = 0.1;
    /* ====================================== */

    Gym_Env env("192.168.0.105:50053");
    // MountainCar-v0
    // CartPole-v0
    // CliffWalking-v0
    env.make("CliffWalking-v0");  // 0 up, 1 right, 2 down, 3 left
    auto action_space = env.action_space();
    auto obs_space = env.obs_space();
    assert(action_space.bDiscrete);
    assert(obs_space.bDiscrete);
    printf("action space: %d, obs_space: %d\n", action_space.n, obs_space.n);

    Qlearning_agent agent;
    agent.init(obs_space.n, action_space.n, learning_rate, gamma, e_greed);

    auto obs = obs_space.getEmptyObs();
    auto next_obs = obs_space.getEmptyObs();
    auto action = action_space.getEmptyAction();
    double reward;
    bool done;

    bool bRender = false;
    for (int episode = 0; episode < 500; episode++)
    {
        auto ret = run_episode(env, agent, obs, next_obs, action, reward, done, bRender);
        printf("Episode %d: steps = %d, reward = %.1f\n", episode, std::get<1>(ret), std::get<0>(ret));

        if (episode % 20 == 0) {
            bRender = true;
        } else {
            bRender = false;
        }
    }
    test_episode(env, agent, obs, next_obs, action, reward, done);
    env.close();
}
