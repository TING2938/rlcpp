#include <cpptools/ct_bits/ring_vector.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <cpptools/ct_bits/getopt.hpp>
#include <sstream>
#include "agent/ppo/PPO_Discrete_agent.h"
#include "env/gym_cpp/gymcpp.h"
#include "tools/dynet_network/dynet_network.h"

using namespace rlcpp;
namespace py = pybind11;

using State  = PPO_Discrete_Agent::State;
using Action = PPO_Discrete_Agent::Action;
using Env    = Gym_cpp<State, Action>;

void test(Env& env, PPO_Discrete_Agent& agent, Int n_turns, bool render = false)
{
    printf("Ready to test\n");

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
            score += reward;
            // printf("k: %d\n", k);
            if (done) {
                printf("k = %d, The score is %f\n", k, score);
                break;
            }
            if (render) {
                env.render();
            }
        }
        // printf("the score is %f\n", score);
    }
}


void train_pipeline_progressive(Env& env,
                                Env& test_env,
                                PPO_Discrete_Agent& agent,
                                const std::string& model_name,
                                Int n_episode,
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

    size_t steps = 0;

    ct::RingVector<Real> rewards, losses, mean_rewards;
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
            steps++;
            if (done) {
                if (agent.buffer().size() >= 32) {
                    auto loss = agent.learn();
                    losses.store(loss);
                }
                break;
            }
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
        }

        if (i_episode >= 100 && i_episode % 50 == 0) {
            test(test_env, agent, 2, true);
        }
    }
    agent.save_model(model_name);
}


int main(int argc, char** argv)
{
    py::scoped_interpreter guard{};

    // ================================= //
    int env_id               = 0;
    std::string dynet_memory = "1";
    std::string method       = "train";  // train/test
    unsigned int seed        = 0;
    // ================================= //
    // get options from commandline
    ct::Getopt getopt(argc, argv, "Train RL with DQN algorithm (dynet nn lib)");

    getopt(env_id, "-id", false,
           "env id for train."
           "\n0: CartPole-v1, 1: Acrobot-v1, 2: MountainCar-v0\n");
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
    ct::set_rand_seed(seed);

    std::vector<std::string> ENVs     = {"CartPole-v1", "Acrobot-v1", "MountainCar-v0"};
    std::vector<Int> score_thresholds = {499, -100, -100};
    Env env, test_env;
    env.make(ENVs[env_id]);
    env.env.attr("seed")(seed);

    test_env.make(ENVs[env_id]);

    auto action_space = env.action_space();
    auto obs_space    = env.obs_space();
    assert(action_space.bDiscrete);
    assert(!obs_space.bDiscrete);
    printf("action space: %d, obs_space: %d\n", action_space.n, obs_space.shape.front());

    int hidden                             = 128;
    std::vector<dynet::Layer> actor_layers = {
        dynet::Layer(obs_space.shape.front(), hidden, dynet::RELU, /* dropout_rate */ 0.0),
        dynet::Layer(hidden, hidden, dynet::RELU, /* dropout_rate */ 0.0),
        dynet::Layer(hidden, action_space.n, dynet::SOFTMAX, /* dropout_rate */ 0.0),
    };

    std::vector<dynet::Layer> critic_layers = {
        dynet::Layer(obs_space.shape.front(), hidden, dynet::RELU, /* dropout_rate */ 0.0),
        dynet::Layer(hidden, hidden, dynet::RELU, /* dropout_rate */ 0.0),
        dynet::Layer(hidden, 1, dynet::LINEAR, /* dropout_rate */ 0.0),
    };

    auto agent = new PPO_Discrete_Agent(actor_layers, critic_layers, 0.98);

    std::stringstream model_name;
    model_name << "PPO-" << ENVs[env_id] << "_"
               << "_actor-" << actor_layers << "_critic-" << critic_layers << ".params";
    std::cout << "model name: " << model_name.str() << std::endl;

    /*
        try {
            agent->load_model(model_name.str());
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
        }
    */


    if (method == "train") {
        // for train
        train_pipeline_progressive(env, test_env, *agent, model_name.str(), 5000);
    }

    env.close();
    delete agent;
}
