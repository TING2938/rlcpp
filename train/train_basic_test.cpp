#include "env/gym_cpp/gymcpp.h"
#include "tools/core_timer.hpp"
#include "tools/random_tools.h"

using namespace rlcpp;

int main()
{
    rlcpp::set_rand_seed();

    Gym_cpp env;
    // CliffWalking-v0
    // MountainCar-v0
    // CartPole-v0
    env.make("CartPole-v1");
    auto action_space = env.action_space();
    auto obs_space    = env.obs_space();
    assert(action_space.bDiscrete);
    assert(!obs_space.bDiscrete);

    auto obs      = obs_space.getEmptyObs();
    auto next_obs = obs_space.getEmptyObs();
    Action action = action_space.getEmptyAction();
    Real reward;
    bool done;

    itp::Timeit timeit;
    int total_episode = 2000;
    size_t count      = 0;

    timeit.start();

    for (int episode = 0; episode < total_episode; episode++) {
        env.reset(&obs);
        for (int t = 0; t < env.max_episode_steps; t++) {
            action.front() = randd(0, action_space.n);
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
