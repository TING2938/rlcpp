#include <sstream>
#include "agent/ddpg/ddpg_agent.h"
#include "env/gym_cpp/gymcpp.h"
#include "tools/core_getopt.hpp"
#include "tools/dynet_network/dynet_network.h"

using namespace rlcpp;
using namespace rlcpp::opt;

using State  = DDPG_Agent::State;
using Action = DDPG_Agent::Action;
using Env    = Gym_cpp<State, Action>;

inline Action scale_action(const Action& action, const Vecf& scale_a, const Vecf& scale_b)
{
    if (scale_a.empty())
        return action;
    else {
        return scale(action, scale_a, scale_b);
    }
}

void train_pipeline_progressive(Env& env,
                                DDPG_Agent& agent,
                                Real score_threshold,
                                const std::string& model_name,
                                Int n_episode,
                                const Vecf& scale_action_a = {},
                                const Vecf& scale_action_b = {},
                                Int learn_start            = 100,
                                Int print_every            = 10)
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
            env.step(scale_action(action, scale_action_a, scale_action_b), &next_obs, &rwd, &done);
            agent.store(obs, action, rwd, next_obs, (t == env.max_episode_steps - 1) ? false : done);
            reward += rwd;
            if (i_episode > learn_start) {
                auto loss = agent.learn();
                losses.store(loss);
            }
            if (t % 1000 == 0) {
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
                                 DDPG_Agent& agent,
                                 Real score_threshold,
                                 const std::string& model_name,
                                 Int n_epoch                = 500,
                                 Int n_rollout              = 100,
                                 Int n_train                = 1000,
                                 const Vecf& scale_action_a = {},
                                 const Vecf& scale_action_b = {},
                                 Int learn_start            = 0,
                                 bool early_stop            = true)
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
                env.step(scale_action(action, scale_action_a, scale_action_b), &next_obs, &rwd, &done);
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

void test(Env& env,
          DDPG_Agent& agent,
          Int n_turns,
          bool render                = false,
          const Vecf& scale_action_a = {},
          const Vecf& scale_action_b = {})
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
            env.step(scale_action(action, scale_action_a, scale_action_b), &obs, &reward, &done);
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
    Env env;
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
            train_pipeline_conservative(env, agent, 1500, model_name.str(), 5000, 100, 100, bound_a, bound_b);
        if (env_id == 0) {
            train_pipeline_progressive(env, agent, -180, model_name.str(), 5000, bound_a, bound_b);
        }
    }

    // for test
    test(env, agent, 100000, true, bound_a, bound_b);

    env.close();
}
