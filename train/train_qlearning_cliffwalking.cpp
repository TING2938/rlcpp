#include <cpptools/ct_bits/getopt.hpp>
#include <tuple>
#include "agent/qlearning/qlearning_agent.h"
#include "env/gym_cpp/gymcpp.h"

using namespace rlcpp;

using State  = Qlearning_agent::State;
using Action = Qlearning_agent::Action;
using Env    = Gym_cpp<State, Action>;

std::tuple<Real, Int> run_episode(Env& env,
                                  Qlearning_agent& agent,
                                  State& obs,
                                  State& next_obs,
                                  Action& action,
                                  Real& reward,
                                  bool& done,
                                  bool bRender = false)
{
    Int total_steps   = 0;
    Real total_reward = 0.0;
    env.reset(&obs);
    while (true) {
        agent.sample(obs, &action);  // greedy sample
        env.step(action, &next_obs, &reward, &done);
        agent.learn(obs, action, reward, next_obs, done);

        obs = next_obs;
        total_reward += reward;
        total_steps += 1;
        if (bRender) {
            env.render();
        }
        if (done) {
            break;
        }
    }
    return {total_reward, total_steps};
}

void test_episode(Env& env,
                  Qlearning_agent& agent,
                  State& obs,
                  State& next_obs,
                  Action& action,
                  Real& reward,
                  bool& done)
{
    Real total_reward = 0.0;
    env.reset(&obs);
    while (true) {
        agent.predict(obs, &action);  // predict according to Q table
        env.step(action, &obs, &reward, &done);
        total_reward += reward;
        sleep(1);
        env.render();
        if (done) {
            printf("test reward = %.1f\n", total_reward);
            break;
        }
    }
}

int main(int argc, char** argv)
{
    py::scoped_interpreter guard;
    Real learning_rate            = 0.1;
    Real gamma                    = 0.9;
    Real e_greed                  = 0.1;
    int env_id                    = 1;
    std::string method            = "train";  // train/test
    std::vector<std::string> ENVs = {"CliffWalking-v0"};

    ct::Getopt getopt(argc, argv, "Train RL with Q-Learning algorithm");

    getopt(env_id, "-id", false,
           "env id for train."
           "\n0: CartPole-v1, 1: Acrobot-v1, 2: MountainCar-v0\n");
    getopt(method, "-method", false, "set to train or test model\n");

    getopt.finish();

    Env env;

    env.make(ENVs[env_id]);  // 0 up, 1 right, 2 down, 3 left
    auto action_space = env.action_space();
    auto obs_space    = env.obs_space();
    assert(action_space.bDiscrete);
    assert(obs_space.bDiscrete);
    printf("action space: %d, obs_space: %d\n", action_space.n, obs_space.n);

    Qlearning_agent agent;
    agent.init(obs_space.n, action_space.n, learning_rate, gamma, e_greed);

    std::stringstream model_name;
    model_name << "QLearning-" << ENVs[env_id] << "-" << obs_space.n << "-" << action_space.n << ".params";
    agent.load_model(model_name.str());

    State obs;
    State next_obs;
    Action action;
    Real reward;
    bool done;

    if (method == "train") {
        bool bRender = false;
        for (int episode = 0; episode < 5000; episode++) {
            auto ret = run_episode(env, agent, obs, next_obs, action, reward, done, bRender);
            printf("Episode %d: steps = %d, reward = %.1f\n", episode, std::get<1>(ret), std::get<0>(ret));

            if (episode % 20 == 0) {
                bRender = false;
            } else {
                bRender = false;
            }
        }
        agent.save_model(model_name.str());
    }

    test_episode(env, agent, obs, next_obs, action, reward, done);
    env.close();
}
