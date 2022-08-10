#include "env/gym_cpp/gymcpp.h"

#include "agent/policy_gradient/reinforce_agent.h"

#include "tools/dynet_network/dynet_network.h"

#include "tools/core_getopt.hpp"

using namespace rlcpp;

using State  = REINFORCE_Agent::State;
using Action = REINFORCE_Agent::Action;
using Env    = Gym_cpp<State, Action>;

void test(Env& env, REINFORCE_Agent& agent, Int n_turns, bool render = false)
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
    int env_id               = 0;
    rlcpp::Real gamma        = 0.99;
    std::string dynet_memory = "1";
    unsigned int seed        = 0;
    std::string method       = "train";  // train/test
    // ================================= //
    // get options from commandline
    itp::Getopt getopt(argc, argv, "Train RL with PG reinforce algorithm (dynet nn lib)");

    getopt(env_id, "-id", false,
           "env id for train."
           "\n0: CartPole-v1, 1: Acrobot-v1, 2: MountainCar-v0\n");
    getopt(gamma, "-gamma", false, "gamma for Gt");
    getopt(method, "-method", false, "set to train or test model\n");
    getopt(seed, "-seed", false, "random seed to be set");
    getopt(dynet_memory, "-dynet_mem", false,
           "Memory used for dynet (MB).\n"
           "or set as FOR,BACK,PARAM,SCRATCH\n"
           "for the amount of memory for forward calculation, backward calculation, parameters, and scratch use\n"
           "by using comma separated variables");

    getopt.finish();

    // ================================= //
    // for dynet command line options
    dynet::DynetParams dynetParams;
    dynetParams.random_seed = seed;
    if (!dynet_memory.empty())
        dynetParams.mem_descriptor = dynet_memory;
    dynet::initialize(dynetParams);
    rlcpp::set_rand_seed(seed);

    std::vector<std::string> ENVs = {"CartPole-v1", "Acrobot-v1", "MountainCar-v0"};
    Env env, test_env;
    env.make(ENVs[env_id]);
    env.env.attr("seed")(seed);

    test_env.make(ENVs[env_id]);
    test_env.env.attr("seed")(seed);

    auto action_space = env.action_space();
    auto obs_space    = env.obs_space();
    assert(action_space.bDiscrete);
    assert(!obs_space.bDiscrete);
    printf("action space: %d, obs_space: %d\n", action_space.n, obs_space.shape.front());

    int hidden                       = obs_space.shape.front() * 10;
    std::vector<dynet::Layer> layers = {
        dynet::Layer(obs_space.shape.front(), hidden, dynet::RELU, /* dropout_rate */ 0.0),
        dynet::Layer(hidden, action_space.n, dynet::SOFTMAX, /* dropout_rate */ 0.0),
    };

    REINFORCE_Agent agent(layers, gamma);

    std::stringstream model_name;
    model_name << "Reinforce-" << ENVs[env_id] << "_network-" << layers << ".params";
    std::cout << "model name: " << model_name.str() << std::endl;

    try {
        agent.load_model(model_name.str());
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    if (method == "train") {
        State obs;
        State next_obs;
        Action action;
        Real reward = 0.0;
        bool done   = false;

        for (int i_episode = 0; i_episode < 5000; i_episode++) {
            env.reset(&obs);
            Real tot_reward = 0;
            while (true) {
                agent.sample(obs, &action);
                env.step(action, &next_obs, &reward, &done);
                agent.store(obs, action, reward, next_obs, done);
                tot_reward += reward;
                if (done) {
                    auto loss = agent.learn();
                    if (i_episode % 50 == 0) {
                        std::cout << "episode: " << i_episode << ", loss: " << loss << ", reward: " << tot_reward
                                  << std::endl;
                    }
                    tot_reward = 0;
                    break;
                }
                obs = next_obs;
            }
        }
        agent.save_model(model_name.str());
    }

    test(env, agent, 100, true);

    env.close();
}
