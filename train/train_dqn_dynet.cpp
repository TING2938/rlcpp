#include "env/gym_env/gym_env.h"
#include "agent/dqn/dqn_dynet_agent.h"
#include "network/dynet_network/dynet_network.h"

using namespace rlcpp;
using std::vector;

void train_pipeline_conservative(Env &env, DQN_dynet_agent &agent, Float score_threshold, Int n_epoch = 500, Int n_rollout = 100, Int n_train = 1000, Int learn_start = 1000, bool early_stop = true)
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
                agent.store(obs, action, rwd, next_obs, t == env.max_episode_steps - 1 ? false : done);
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
            printf("epsilon: %f\n", agent.epsilon);
            printf("Average score of %d rollout games: %f\n", n_rollout, mean_reward);
            if (i_epoch > learn_start) {
                auto mean_loss = std::accumulate(losses.begin(), losses.end(), Float(0.0)) / losses.size();
                printf("Average training loss: %f\n", mean_loss);
            }
            printf("===========================\n\n");
            if (early_stop && mean_reward >= score_threshold) 
                break;
        }
    }
}

void test(Env &env, DQN_dynet_agent &agent, Int n_turns, bool render = false)
{
    printf("Ready to test., Press any key to coninue...\n");
    {
        string tmp;
        std::cin >> tmp;
    }

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
            if (render) {
                env.render();
            }
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
            if (render)
            {
                env.render();
            }
        }
    }
}

int main(int argc, char** argv)
{
    dynet::initialize(argc, argv);
    
    // ================================= //
    int env_id = 0;
    Int max_reply_memory_size = 20000;
    Int batch_size = 256;
    // ================================= //

    vector<string> ENVs = {"CartPole-v0", "Acrobot-v1", "MountainCar-v0"};
    vector<Int> score_thresholds = {499, -100, -100};
    Gym_Env env("localhost:50053");
    env.make(ENVs[env_id]);

    auto action_space = env.action_space();
    auto obs_space = env.obs_space();
    assert(action_space.bDiscrete);
    assert(!obs_space.bDiscrete);
    printf("action space: %d, obs_space: %d\n", action_space.n, obs_space.shape.front());

    std::vector<dynet::Layer> layers = {
        dynet::Layer(obs_space.shape.front(),            128, dynet::RELU  , /* dropout_rate */ 0.0), 
        dynet::Layer(128                    ,            128, dynet::RELU  , /* dropout_rate */ 0.0),
        dynet::Layer(128                    , action_space.n, dynet::LINEAR, /* dropout_rate */ 0.0)
    };

    DQN_dynet_agent agent(layers, obs_space.shape.front(), action_space.n, max_reply_memory_size, batch_size, 200, 0.99, 1, 5e-5);

    Int max_episode = 500;
    Int learn_start = 0;
    train_pipeline_conservative(env, agent, score_thresholds[env_id], max_episode, 100, 1000, learn_start);
    test(env, agent, 10, false);
    env.close();
}
