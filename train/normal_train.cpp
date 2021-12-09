#include "../env/gym_env/gym_env.h"
#include "../agent/basic_agent/basic_agent.h"

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
    agent.init(obs_space.back(), action_space);

    State obs, next_obs;
    Action action;
    double reward;
    bool done;

    obs.resize(obs_space.back());
    next_obs.resize(obs_space.back());

    env.reset(&obs);
    for (int episode = 0; episode < 1000; episode++)
    {
        env.render();
        //sleep(0.5);
        env.step(action, &obs, &reward, &done);
    }
    env.close();
}
