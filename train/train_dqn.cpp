#include <sstream>

#include "env/gym_cpp/gymcpp.h"

#include "agent/dqn/dqn_prioritizedReply_agent.h"
#include "agent/dqn/dqn_randomReply_agent.h"

#include "tools/dynet_network/dynet_network.h"

#include "tools/core_getopt.hpp"

using namespace rlcpp;

using State  = DQN_Base_Agent::State;
using Action = DQN_Base_Agent::Action;
using Env    = Gym_cpp<State, Action>;

void train_pipeline_progressive(Env& env,
                                DQN_Base_Agent& agent,
                                Real score_threshold,
                                const std::string& model_name,
                                Int n_episode,
                                Int learn_start = 100,
                                Int print_every = 10)
{
    auto seaborn = py::module_::import("seaborn");
    auto plt     = py::module_::import("matplotlib.pyplot");
    seaborn.attr("set")();

    State obs;
    State next_obs;
    Action action;
    Real rwd;
    bool done;

    RingVector<Real> rewards, losses, mean_rewards;
    rewards.init(100);
    losses.init(100);
    mean_rewards.init(200);

    for (int i_episode = 0; i_episode < n_episode; i_episode++) {
        Real reward = 0.0;
        env.reset(&obs);

        for (int t = 0; t < env.max_episode_steps; t++) {
            agent.sample(obs, &action);
            env.step(action, &next_obs, &rwd, &done);
            agent.store(obs, action, rwd, next_obs, (t == env.max_episode_steps - 1) ? false : done);
            reward += rwd;
            if (i_episode > learn_start) {
                auto loss = agent.learn();
                losses.store(loss);
            }
            if (t % 10 == 0) {
                env.render();
            }
            if (done)
                break;
            obs = next_obs;
        }
        rewards.store(reward);

        if (i_episode % print_every == 0) {
            auto score = rewards.mean();
            mean_rewards.store(score);
            plt.attr("clf")();
            plt.attr("plot")(mean_rewards.lined_vector(), "-o");
            plt.attr("ylabel")("Rewards");
            // plt.attr("ylim")(py::make_tuple(0, 500));
            plt.attr("pause")(0.1);

            printf("===========================\n");
            printf("i_eposide: %d\n", i_episode);
            printf("100 games mean reward: %f\n", score);
            printf("100 games mean loss: %f\n", losses.mean());
            printf("===========================\n\n");
            if (score >= score_threshold) {
                agent.save_model(model_name);
                break;
            }
        }
    }
    agent.save_model(model_name);
}

void train_pipeline_conservative(Env& env,
                                 DQN_Base_Agent& agent,
                                 Real score_threshold,
                                 const std::string& model_name,
                                 Int n_epoch     = 500,
                                 Int n_rollout   = 100,
                                 Int n_train     = 1000,
                                 Int learn_start = 0,
                                 bool early_stop = true)
{
    auto seaborn = py::module_::import("seaborn");
    auto plt     = py::module_::import("matplotlib.pyplot");
    seaborn.attr("set")();

    State obs;
    State next_obs;
    Action action;
    Real rwd;
    bool done;
    Vecf rewards, losses;

    RingVector<Real> mean_rewards;
    mean_rewards.init(200);

    for (int i_epoch = 0; i_epoch < n_epoch; i_epoch++) {
        rewards.clear();
        losses.clear();

        for (int rollout = 0; rollout < n_rollout; rollout++) {
            Real reward = 0.0;
            env.reset(&obs);
            for (int t = 0; t < env.max_episode_steps; t++) {
                agent.sample(obs, &action);
                env.step(action, &next_obs, &rwd, &done);
                agent.store(obs, action, rwd, next_obs, (t == env.max_episode_steps - 1) ? false : done);
                reward += rwd;
                if (done) {
                    break;
                }
                obs = next_obs;
            }
            rewards.push_back(reward);
        }

        if (i_epoch > learn_start) {
            for (int i = 0; i < n_train; i++) {
                auto loss = agent.learn();
                losses.push_back(loss);
            }
        }

        if (i_epoch % 1 == 0) {
            auto mean_reward = mean(rewards);

            mean_rewards.store(mean_reward);
            plt.attr("clf")();
            plt.attr("plot")(mean_rewards.lined_vector(), "-o");
            plt.attr("ylabel")("Rewards");
            // plt.attr("ylim")(py::make_tuple(0, 500));
            plt.attr("pause")(0.1);

            printf("===========================\n");
            printf("i_epoch: %d\n", i_epoch);
            printf("Average score of %d rollout games: %f\n", n_rollout, mean_reward);
            if (i_epoch > learn_start) {
                auto mean_loss = mean(losses);
                printf("Average training loss: %f\n", mean_loss);
            }
            printf("===========================\n\n");
            if (early_stop && mean_reward >= score_threshold) {
                agent.save_model(model_name);
                break;
            }
        }
    }
    agent.save_model(model_name);
}

void test(Env& env, DQN_Base_Agent& agent, Int n_turns, bool render = false)
{
    printf("Ready to test, Press any key to coninue...\n");
    getchar();

    State obs;
    State next_obs;
    Action action;
    Real reward;
    bool done;

    for (int i = 0; i < n_turns; i++) {
        Real score = 0.0;
        env.reset(&obs);
        for (int k = 0; k < env.max_episode_steps; k++) {
            agent.predict(obs, &action);  // predict according to Q table
            env.step(action, &obs, &reward, &done);
            if (render) {
                env.render();
            }
            score += reward;
            if (done) {
                printf("The score is %f\n", score);
                break;
            }
        }
        // printf("the score is %f\n", score);
    }
}


int main(int argc, char** argv)
{
    py::scoped_interpreter guard;
    // ================================= //
    int env_id                = 1;
    Int max_reply_memory_size = 50000;
    Int batch_size;
    bool use_double          = false;
    bool use_prioritized     = false;
    std::string dynet_memory = "1";
    std::string method       = "train";  // train/test
    unsigned int seed        = 0;        // random seed

    // ================================= //
    // get options from commandline
    itp::Getopt getopt(argc, argv, "Train RL with DQN algorithm (dynet nn lib)");

    getopt(env_id, "-id", false,
           "env id for train."
           "\n0: CartPole-v1, 1: Acrobot-v1, 2: MountainCar-v0\n");
    getopt(use_double, "-ddqn", false, "whether to use double dqn\n");
    getopt(use_prioritized, "-prioritized", false, "whether to use prioitized memory reply\n");
    getopt(method, "-method", false, "set to train or test model\n");
    getopt(seed, "-seed", false, "random seed to be set");
    getopt(dynet_memory, "-dynet_mem", false,
           "Memory used for dynet (MB).\n"
           "or set as FOR,BACK,PARAM,SCRATCH\n"
           "for the amount of memory for forward calculation, backward calculation, parameters, and scratch use\n"
           "by using comma separated variables");

    getopt.finish();

    if (seed == 0) {
        seed = time(nullptr);
    }

    // ================================= //
    // for dynet command line options
    dynet::DynetParams dynetParams;
    dynetParams.random_seed = seed;
    if (!dynet_memory.empty())
        dynetParams.mem_descriptor = dynet_memory;
    dynet::initialize(dynetParams);
    rlcpp::set_rand_seed(seed);

    if (env_id == 0) {
        batch_size = 256;
    } else {
        batch_size = 32;
    }

    std::vector<std::string> ENVs     = {"CartPole-v1", "Acrobot-v1", "MountainCar-v0"};
    std::vector<Int> score_thresholds = {499, -100, -100};
    Env env;
    env.make(ENVs[env_id]);

    auto action_space = env.action_space();
    auto obs_space    = env.obs_space();
    assert(action_space.bDiscrete);
    assert(!obs_space.bDiscrete);
    printf("action space: %d, obs_space: %d\n", action_space.n, obs_space.shape.front());

    std::vector<dynet::Layer> layers = {
        dynet::Layer(obs_space.shape.front(), 128, dynet::RELU, /* dropout_rate */ 0.0),
        dynet::Layer(128, action_space.n, dynet::LINEAR, /* dropout_rate */ 0.0),
    };

    DQN_Base_Agent* agent;
    if (use_prioritized) {
        agent =
            new DQN_PrioritizedReply_Agent(layers, max_reply_memory_size, use_double, batch_size, 500, 0.99, 1, 5e-5);
    } else {
        agent = new DQN_RandomReply_Agent(layers, max_reply_memory_size, use_double, batch_size, 500, 0.99, 1, 5e-5);
    }

    std::stringstream model_name;
    model_name << "DQN-" << ENVs[env_id] << "_"
               << "use_double-" << use_double << "_network-" << layers << ".params";
    std::cout << "model name: " << model_name.str() << std::endl;

    try {
        agent->load_model(model_name.str());
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    if (method == "train") {
        // for train
        if (env_id == 0)
            train_pipeline_conservative(env, *agent, score_thresholds[env_id], model_name.str(), 500, 100, 1000);
        if (env_id == 1 || env_id == 2) {
            train_pipeline_progressive(env, *agent, score_thresholds[env_id], model_name.str(), 2000);
        }
    }

    // for test
    test(env, *agent, 100, true);

    env.close();
    delete agent;
}
