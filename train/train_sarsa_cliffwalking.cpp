#include "env/gym_cpp/gymcpp.h"
#include "agent/sarsa/sarsa_agent.h"
#include <tuple>

using namespace rlcpp;

std::tuple<Real, Int> run_episode(Env& env, Sarsa_agent& agent, State& obs, State& next_obs, Action &action, Action& next_action, Real &reward, bool &done, bool bRender = false)
{
    Int total_steps = 0;
    Real total_reward = 0.0;
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

void test_episode(Env& env, Sarsa_agent& agent, State& obs, State& next_obs, Action &action, Real &reward, bool &done) 
{
    Real total_reward = 0.0;
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
            printf("test reward = %.1f\n", total_reward);
            break;
        }
    }
}

int main()
{
    /* ====================================== */
    Real learning_rate = 0.1;
    Real gamma = 0.9;
    Real e_greed = 0.1;
    /* ====================================== */

    Gym_cpp env;
    // MountainCar-v0
    // CartPole-v0
    // CliffWalking-v0
    env.make("CliffWalking-v0");  // 0 up, 1 right, 2 down, 3 left
    auto action_space = env.action_space();
    auto obs_space = env.obs_space();
    assert(action_space.bDiscrete);
    assert(obs_space.bDiscrete);
    printf("action space: %d, obs_space: %d\n", action_space.n, obs_space.n);

    Sarsa_agent agent;
    agent.init(obs_space.n, action_space.n, learning_rate, gamma, e_greed);

    auto obs = obs_space.getEmptyObs();
    auto next_obs = obs_space.getEmptyObs();
    auto action = action_space.getEmptyAction();
    auto next_action = action_space.getEmptyAction();
    Real reward;
    bool done;

    bool bRender = false;
    for (int episode = 0; episode < 5000; episode++)
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
