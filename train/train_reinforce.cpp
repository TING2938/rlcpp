#define RLCPP_STATE_TYPE 1
#define RLCPP_ACTION_TYPE 0

#include "env/grpc_gym/gym_env.h"
#include "env/gym_cpp/gymcpp.h"

#include "agent/policy_gradient/reinforce_agent.h"

#include "network/dynet_network/dynet_network.h"

#include "tools/core_getopt.hpp"
#include "train/train_test_utils.h"

int main(int argc, char** argv)
{
    // ================================= //
    int env_id               = 1;
    rlcpp::Real gamma        = 0.99;
    std::string dynet_memory = "1";
    // ================================= //
    // get options from commandline
    itp::Getopt getopt(argc, argv, "Train RL with DQN algorithm (dynet nn lib)");

    getopt(env_id, "-id", false,
           "env id for train."
           "\n0: CartPole-v1, 1: Acrobot-v1, 2: MountainCar-v0\n");
    getopt(gamma, "-gamma", false, "gamma for Gt");
    getopt(dynet_memory, "-dynet_mem", false,
           "Memory used for dynet (MB).\n"
           "or set as FOR,BACK,PARAM,SCRATCH\n"
           "for the amount of memory for forward calculation, backward calculation, parameters, and scratch use\n"
           "by using comma separated variables");

    getopt.finish();

    // ================================= //
    // for dynet command line options
    dynet::DynetParams dynetParams;
    if (!dynet_memory.empty())
        dynetParams.mem_descriptor = dynet_memory;
    dynet::initialize(dynetParams);
    rlcpp::set_rand_seed();

    vector<string> ENVs = {"CartPole-v1", "Acrobot-v1", "MountainCar-v0"};
    Gym_cpp env;
    env.make(ENVs[env_id]);

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

    auto obs      = obs_space.getEmptyObs();
    auto action   = action_space.getEmptyAction();
    auto next_obs = obs_space.getEmptyObs();
    Real reward   = 0.0;
    bool done     = false;

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

    // grpc_env.close();
    env.close();
}
