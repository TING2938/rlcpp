#define RLCPP_STATE_TYPE 1
#define RLCPP_ACTION_TYPE 1

#include <sstream>
#include "agent/ddpg/ddpg_agent.h"
#include "env/grpc_gym/gym_env.h"
#include "env/gym_cpp/gymcpp.h"
#include "tools/core_getopt.hpp"
#include "tools/dynet_network/dynet_network.h"
#include "train/train_test_utils.h"

using namespace rlcpp;
using namespace rlcpp::opt;

int main(int argc, char** argv)
{
    // ================================= //
    int env_id                = 0;
    Int max_reply_memory_size = 1e6;
    Int batch_size            = 64;
    std::string dynet_memory  = "1";
    std::string method        = "train";  // train/test
    // ================================= //
    // get options from commandline
    itp::Getopt getopt(argc, argv, "Train RL with DDPG algorithm (dynet nn lib)");

    getopt(env_id, "-id", false,
           "env id for train."
           "\n0: Pendulum-v1\n");
    getopt(batch_size, "-b", false, "the batch size");
    getopt(method, "-method", false, "set to train or test model\n");
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

    auto obs_dim    = env.obs_space().shape.front();
    auto action_dim = env.action_space().shape.front();
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

    std::stringstream model_name;
    model_name << "DDPG-" << ENVs[env_id] << "_actor-" << actor_layers << "_critic-" << critic_layers << ".params";
    std::cout << "model name: " << model_name.str() << std::endl;

    try {
        agent.load_model(model_name.str());
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    // scale action from [-1, 1] to [low, high]
    auto low     = env.action_space().low;
    auto high    = env.action_space().high;
    auto bound_a = (high - low) / 2.0f;
    auto bound_b = low + 1.0f;

    if (method == "train") {
        // for train
        if (env_id == 1)
            train_pipeline_conservative(env, agent, 999, model_name.str(), 5000, 100, 100, bound_a, bound_b);
        if (env_id == 0) {
            train_pipeline_progressive(env, agent, -200, model_name.str(), 5000, bound_a, bound_b);
        }
    }

    // for test
    test(env, agent, 100000, true, bound_a, bound_b);

    env.close();
}
