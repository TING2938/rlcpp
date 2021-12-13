#include "env/gym_env/gym_env.h"
#include "agent/dqn/dqn_agent.h"
#include <tuple>

using namespace rlcpp;

std::tuple<Float, Int> run_episode(Env& env, DQN_agent& agent, State& obs, State& next_obs, Action &action, Action& next_action, Float &reward, bool &done, bool bRender = false)
{
    Int total_steps = 0;
    Float total_reward = 0.0;
    env.reset(&obs);
    agent.sample(obs, &action);

    while (true)
    {
        env.step(action, &next_obs, &reward, &done);
        agent.sample(next_obs, &next_action);
        agent.learn(obs, action, reward, next_obs, next_action, done);

        action = next_action;
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

void test_episode(Env& env, DQN_agent& agent, State& obs, State& next_obs, Action &action, Float &reward, bool &done) 
{
    Float total_reward = 0.0;
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
    Float learning_rate = 0.1;
    Float gamma = 0.9;
    Float e_greed = 0.1;
    /* ====================================== */

    Gym_Env env("localhost:50053");
    // MountainCar-v0
    // CartPole-v0
    // CliffWalking-v0
    env.make("CliffWalking-v0");  // 0 up, 1 right, 2 down, 3 left
    auto action_space = env.action_space();
    auto obs_space = env.obs_space();
    assert(action_space.bDiscrete);
    assert(obs_space.bDiscrete);
    printf("action space: %d, obs_space: %d\n", action_space.n, obs_space.n);

    DQN_agent agent;
    agent.init(obs_space.n, action_space.n, learning_rate, gamma, e_greed);

    auto obs = obs_space.getEmptyObs();
    auto next_obs = obs_space.getEmptyObs();
    auto action = action_space.getEmptyAction();
    auto next_action = action_space.getEmptyAction();
    Float reward;
    bool done;

    bool bRender = false;
    for (int episode = 0; episode < 500; episode++)
    {
        auto ret = run_episode(env, agent, obs, next_obs, action, next_action, reward, done, bRender);
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
