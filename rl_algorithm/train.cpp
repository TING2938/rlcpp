#include "environment/gym_env/gym_env.h"
#include "agent/basic_agent.h"

int main()
{
    Int print_interval = 20;

    Gym_Env env("localhost:50053");
    env.make("CartPole-v0");
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
