#define RLCPP_STATE_TYPE 1
#define RLCPP_ACTION_TYPE 0

#include "env/grpc_gym/gym_env.h"
#include "env/gym_cpp/gymcpp.h"

#include "agent/dqn/dqn_prioritizedReply_agent.h"
#include "agent/dqn/dqn_randomReply_agent.h"

#include "tools/dynet_network/dynet_network.h"

#include "tools/core_getopt.hpp"
#include "train/train_test_utils.h"

int main(int argc, char** argv)
{
    // ================================= //
    int env_id                = 1;
    Int max_reply_memory_size = 50000;
    Int batch_size;
    bool use_double_dqn      = false;
    bool use_prioritized     = false;
    std::string dynet_memory = "1";
    // ================================= //
    // get options from commandline
    itp::Getopt getopt(argc, argv, "Train RL with DQN algorithm (dynet nn lib)");

    getopt(env_id, "-id", false,
           "env id for train."
           "\n0: CartPole-v1, 1: Acrobot-v1, 2: MountainCar-v0\n");
    getopt(use_double_dqn, "-ddqn", false, "whether to use double dqn\n");
    getopt(use_prioritized, "-prioritized", false, "whether to use prioitized memory reply\n");
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

    if (env_id == 0) {
        batch_size = 256;
    } else {
        batch_size = 32;
    }

    vector<string> ENVs          = {"CartPole-v1", "Acrobot-v1", "MountainCar-v0"};
    vector<Int> score_thresholds = {499, -100, -100};
    Gym_cpp env;
    env.make(ENVs[env_id]);

    auto action_space = env.action_space();
    auto obs_space    = env.obs_space();
    assert(action_space.bDiscrete);
    assert(!obs_space.bDiscrete);
    printf("action space: %d, obs_space: %d\n", action_space.n, obs_space.shape.front());

    std::vector<dynet::Layer> layers = {dynet::Layer(obs_space.shape.front(), 128, dynet::RELU, /* dropout_rate */ 0.0),
                                        dynet::Layer(128, action_space.n, dynet::LINEAR, /* dropout_rate */ 0.0)};

    Agent* agent;
    if (use_prioritized) {
        agent = new DQN_PrioritizedReply_Agent(layers, max_reply_memory_size, use_double_dqn, batch_size, 500, 0.99, 1,
                                               5e-5);
    } else {
        agent =
            new DQN_RandomReply_Agent(layers, max_reply_memory_size, use_double_dqn, batch_size, 500, 0.99, 1, 5e-5);
    }

    // for train
    if (env_id == 0)
        train_pipeline_conservative(env, *agent, score_thresholds[env_id], 500, 100, 1000, 0);
    if (env_id == 1 || env_id == 2) {
        train_pipeline_progressive(env, *agent, score_thresholds[env_id], 2000000, 100);
    }

    // for test
    // rlcpp::Gym_gRPC grpc_env("10.227.6.132:50248");
    // grpc_env.make(ENVs[env_id]);
    test(env, *agent, 100, true);

    // grpc_env.close();
    env.close();
    delete agent;
}
