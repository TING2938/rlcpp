#include "env/gym_cpp/gymcpp.h"
#include "agent/dqn/dqn_dynet_agent.h"
#include "network/dynet_network/dynet_network.h"

using namespace rlcpp;
using std::vector;

void train_pipeline_progressive(Env &env, DQN_dynet_agent &agent, Float score_threshold, Int n_episode, Int learn_start = 100, Int print_every = 10)
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

void train_pipeline_conservative(Env &env, DQN_dynet_agent &agent, Float score_threshold, Int n_epoch = 500, Int n_rollout = 100, Int n_train = 1000, Int learn_start = 0, bool early_stop = true)
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
            printf("epsilon: %f\n", agent.epsilon);
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
            if (render)
            {
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

int main(int argc, char **argv)
{
    dynet::initialize(argc, argv);

    // ================================= //
    int env_id = 0;
    Int max_reply_memory_size = 50000;
    Int batch_size;
    bool use_double_dqn = true;
    // ================================= //
    if (env_id == 0)
    {
        batch_size = 256;
    } else
    {
        batch_size = 32;
    }

    vector<string> ENVs = {"CartPole-v1", "Acrobot-v1", "MountainCar-v0"};
    vector<Int> score_thresholds = {499, -100, -100};
    Gym_cpp env;
    env.make(ENVs[env_id]);

    auto action_space = env.action_space();
    auto obs_space = env.obs_space();
    assert(action_space.bDiscrete);
    assert(!obs_space.bDiscrete);
    printf("action space: %d, obs_space: %d\n", action_space.n, obs_space.shape.front());

    std::vector<dynet::Layer> layers = {
        dynet::Layer(obs_space.shape.front(), 128, dynet::RELU, /* dropout_rate */ 0.0),
        dynet::Layer(128, action_space.n, dynet::LINEAR, /* dropout_rate */ 0.0)
    };

    DQN_dynet_agent agent(layers, max_reply_memory_size, use_double_dqn, batch_size, 500, 0.99, 1, 5e-5);

    if (env_id == 0)
        train_pipeline_conservative(env, agent, score_thresholds[env_id], 500, 100, 1000, 0);
    if (env_id == 1 || env_id == 2)
    {
        train_pipeline_progressive(env, agent, score_thresholds[env_id], 2000, 100);
    }
    test(env, agent, 10, false);
    env.close();
}
