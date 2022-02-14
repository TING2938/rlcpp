#define RLCPP_STATE_TYPE 1
#define RLCPP_ACTION_TYPE 1

#include "agent/ddpg/ddpg_agent.h"
#include "env/grpc_gym/gym_env.h"
#include "env/gym_cpp/gymcpp.h"
#include "tools/core_getopt.hpp"
#include "tools/dynet_network/dynet_network.h"
#include "train/train_test_utils_ddpg.h"

using namespace rlcpp;

int main(int argc, char** argv)
{
    // ================================= //
    int env_id                = 0;
    Int max_reply_memory_size = 1e6;
    Int batch_size            = 64;
    std::string dynet_memory  = "1";
    // ================================= //
    // get options from commandline
    itp::Getopt getopt(argc, argv, "Train RL with DQN algorithm (dynet nn lib)");

    getopt(env_id, "-id", false,
           "env id for train."
           "\n0: Pendulum-v1\n");
    getopt(batch_size, "-b", false, "the batch size");
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

    std::vector<std::string> ENVs = {"Pendulum-v1", "Walker2d-v2"};
    Gym_cpp env;
    env.make(ENVs[env_id]);

    auto action_space = env.action_space();
    auto obs_space    = env.obs_space();
    assert(!action_space.bDiscrete);
    assert(!obs_space.bDiscrete);
    auto obs_dim    = obs_space.shape.front();
    auto action_dim = action_space.shape.front();
    printf("action space: %d, obs_space: %d\n", action_dim, obs_dim);

    std::vector<dynet::Layer> actor_layers = {
        dynet::Layer(obs_dim, 64, dynet::TANH, /* dropout_rate */ 0.0),
        dynet::Layer(64, 64, dynet::TANH, /* dropout_rate */ 0.0),
        dynet::Layer(64, action_dim, dynet::TANH, /* dropout_rate */ 0.0),
    };

    std::vector<dynet::Layer> critic_layers = {
        dynet::Layer(obs_dim + action_dim, 64, dynet::TANH, /* dropout_rate */ 0.0),
        dynet::Layer(64, 64, dynet::TANH, /* dropout_rate */ 0.0),
        dynet::Layer(64, 1, dynet::LINEAR, /* dropout_rate */ 0.0),
    };

    DDPG_Agent agent(actor_layers, critic_layers, max_reply_memory_size, batch_size);


    // for train
    if (env_id == 1)
        train_pipeline_conservative(env, agent, 999, 5000, 100, 100);
    if (env_id == 0) {
        train_pipeline_progressive(env, agent, -200, 5000);
    }

    // for test
    // rlcpp::Gym_gRPC grpc_env("10.227.6.132:50248");
    // grpc_env.make(ENVs[env_id]);
    // grpc_env.close();
    test(env, agent, 100000, true);

    env.close();
}
