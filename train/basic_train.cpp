#include "../env/gym_env/gym_env.h"
#include "../agent/basic_agent/basic_agent.h"

using namespace rlcpp;

int main()
{
    Int print_interval = 20;

    Gym_Env env("10.227.6.189:50053");
    // MountainCar-v0
    // CartPole-v0
    env.make("MountainCar-v0");
    auto action_space = env.action_space();
    auto obs_space = env.obs_space();
    
    Basic_agent agent;
    agent.init(obs_space.shape.front(), action_space.n);

    auto obs = obs_space.getEmptyObs();
    auto next_obs = obs_space.getEmptyObs();
    Action action = action_space.getEmptyAction();
    double reward;
    bool done;

    double score = 0;
    for (int episode = 0; episode < 100; episode++)
    {
        double total_reward = 0.0;
        env.reset(&obs);
        for (int j = 0; j < 600; j++)
        {
            agent.sample(obs, &action);
            env.step(action, &obs, &reward, &done);
            score += reward;
            if (done) 
                break;
            
            if ((episode % print_interval == 0) && (episode != 0))
            {
                std::cout << "# episode: " << episode 
                          << ", avg score: " << score / print_interval << std::endl; 
                score = 0.0;
            }
        }
    }
    env.close();
}
