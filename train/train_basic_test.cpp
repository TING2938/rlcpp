#include <cpptools/ct.hpp>
#include <cpptools/ct_bits/random_tools.hpp>
#include "env/gym_cpp/gymcpp.h"

using namespace rlcpp;

using State  = Vecf;
using Action = Int;
using Env    = Gym_cpp<State, Action>;

int main()
{
    ct::set_rand_seed();
    py::scoped_interpreter guard;

    Env env;
    // CliffWalking-v0
    // MountainCar-v0
    // CartPole-v0
    env.make("CartPole-v1");
    auto action_space = env.action_space();
    auto obs_space    = env.obs_space();
    assert(action_space.bDiscrete);
    assert(!obs_space.bDiscrete);

    State obs;
    State next_obs;
    Action action;
    Real reward;
    bool done;

    ct::Timeit timeit;
    int total_episode = 2000;
    size_t count      = 0;

    timeit.start();

    for (int episode = 0; episode < total_episode; episode++) {
        env.reset(&obs);
        for (int t = 0; t < env.max_episode_steps; t++) {
            action = ct::randd(0, action_space.n);
            env.step(action, &obs, &reward, &done);
            count++;
            if (done)
                break;
        }
    }
    env.close();

    timeit.stop();

    timeit.printSpan("spent ", " s\n");
    std::cout << "step count: " << count << "\n"
              << "total episode: " << total_episode << std::endl;
}
