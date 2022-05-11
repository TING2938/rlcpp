#define RLCPP_STATE_TYPE 1
#define RLCPP_ACTION_TYPE 0

#include <sstream>

#include "env/grpc_gym/gym_env.h"
#include "env/gym_cpp/gymcpp.h"

#include "agent/ac/AC_agent.h"

#include "tools/dynet_network/dynet_network.h"

#include "tools/core_getopt.hpp"
#include "train/train_test_utils.h"

int main(int argc, char** argv)
{
    py::scoped_interpreter guard{};

    // ================================= //
    int env_id               = 0;
    std::string dynet_memory = "1";
    std::string method       = "train";  // train/test
    // ================================= //
    // get options from commandline
    itp::Getopt getopt(argc, argv, "Train RL with DQN algorithm (dynet nn lib)");

    getopt(env_id, "-id", false,
           "env id for train."
           "\n0: CartPole-v1, 1: Acrobot-v1, 2: MountainCar-v0\n");
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

    vector<string> ENVs          = {"CartPole-v1", "Acrobot-v1", "MountainCar-v0"};
    vector<Int> score_thresholds = {499, -100, -100};
    Gym_cpp env;
    env.make(ENVs[env_id]);

    auto action_space = env.action_space();
    auto obs_space    = env.obs_space();
    assert(action_space.bDiscrete);
    assert(!obs_space.bDiscrete);
    printf("action space: %d, obs_space: %d\n", action_space.n, obs_space.shape.front());

    int hidden                             = 128;  // obs_space.shape.front() * 10;
    std::vector<dynet::Layer> actor_layers = {
        dynet::Layer(obs_space.shape.front(), hidden, dynet::RELU, /* dropout_rate */ 0.0),
        dynet::Layer(hidden, action_space.n, dynet::SOFTMAX, /* dropout_rate */ 0.0),
    };

    std::vector<dynet::Layer> critic_layers = {
        dynet::Layer(obs_space.shape.front(), hidden, dynet::RELU, /* dropout_rate */ 0.0),
        dynet::Layer(hidden, 1, dynet::LINEAR, /* dropout_rate */ 0.0),
    };

    Agent* agent = new AC_Agent(actor_layers, critic_layers);

    std::stringstream model_name;
    model_name << "AC-" << ENVs[env_id] << "_"
               << "_actor-" << actor_layers << "_critic-" << critic_layers << ".params";
    std::cout << "model name: " << model_name.str() << std::endl;

    try {
        agent->load_model(model_name.str());
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    if (method == "train") {
        // for train
        if (env_id == -1)
            train_pipeline_conservative(env, *agent, score_thresholds[env_id], model_name.str(), 500, 100, 1000);
        if (env_id == 0 || env_id == 1 || env_id == 2) {
            train_pipeline_progressive(env, *agent, score_thresholds[env_id], model_name.str(), 20000, {}, {}, 0);
        }
    }

    // for test
    test(env, *agent, 100, true);

    env.close();
    delete agent;
}
