#define RLCPP_STATE_TYPE 1
#define RLCPP_ACTION_TYPE 0

#include <sstream>

#include "env/aifan_simple/aifan.h"

#include "agent/dqn/dqn_prioritizedReply_agent.h"
#include "agent/dqn/dqn_randomReply_agent.h"

#include "tools/dynet_network/dynet_network.h"

#include "tools/core_getopt.hpp"
#include "train/train_test_utils.h"

int main(int argc, char** argv)
{
    // ================================= //
    Int max_reply_memory_size = 50000;
    Int batch_size;
    bool use_double          = false;
    bool use_prioritized     = false;
    std::string dynet_memory = "1";
    std::string method       = "train";  // train/test
    // ================================= //
    // get options from commandline
    itp::Getopt getopt(argc, argv, "Train RL with DQN algorithm (dynet nn lib)");
    getopt(use_double, "-ddqn", false, "whether to use double dqn\n");
    getopt(use_prioritized, "-prioritized", false, "whether to use prioitized memory reply\n");
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

    batch_size = 32;

    AIFanSimple env;
    env.make("aifan");

    auto action_space = env.action_space();
    auto obs_space    = env.obs_space();
    assert(action_space.bDiscrete);
    assert(!obs_space.bDiscrete);
    printf("action space: %d, obs_space: %d\n", action_space.n, obs_space.shape.front());

    std::vector<dynet::Layer> layers = {
        dynet::Layer(obs_space.shape.front(), 256, dynet::RELU, /* dropout_rate */ 0.0),
        dynet::Layer(256, action_space.n, dynet::LINEAR, /* dropout_rate */ 0.0),
    };

    Agent* agent;
    if (use_prioritized) {
        agent =
            new DQN_PrioritizedReply_Agent(layers, max_reply_memory_size, use_double, batch_size, 500, 0.99, 1, 5e-5);
    } else {
        agent = new DQN_RandomReply_Agent(layers, max_reply_memory_size, use_double, batch_size, 500, 0.99, 1, 5e-5);
    }

    std::stringstream model_name;
    model_name << "DQN-"
               << "aifan"
               << "_"
               << "use_double-" << use_double << "_network-" << layers << ".params";
    std::cout << "model name: " << model_name.str() << std::endl;

    try {
        agent->load_model(model_name.str());
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    if (method == "train") {
        // for train
        // train_pipeline_conservative(env, *agent, 1000, model_name.str(), 500, 100, 1000);
        // if (env_id == 1 || env_id == 2) {
        train_pipeline_progressive(env, *agent, 800, model_name.str(), 2000000);
    }

    // for test
    test(env, *agent, 100, true);

    env.close();
    delete agent;
}
