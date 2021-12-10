#include "env/gym_env/gym_env.h"
#include "agent/basic_agent/basic_agent.h"

using namespace rlcpp;

int main()
{
    Gym_Env env("10.227.6.189:50053");
    // CliffWalking-v0
    // MountainCar-v0
    // CartPole-v0
    env.make("CliffWalking-v0");
    auto action_space = env.action_space();
    auto obs_space = env.obs_space();
    
    Basic_agent agent;
    agent.init(obs_space.shape.front(), action_space.n);

    auto obs = obs_space.getEmptyObs();
    auto next_obs = obs_space.getEmptyObs();
    Action action = action_space.getEmptyAction();
    double reward;
    bool done;

    env.reset(&obs);
    for (int episode = 0; episode < 1000; episode++)
    {
        env.render();
        //sleep(0.5);
        env.step(action, &obs, &reward, &done);
    }
    env.close();
}
