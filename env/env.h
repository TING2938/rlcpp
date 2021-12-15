#ifndef __RL_ENV_H__
#define __RL_ENV_H__

#include "common/rl_config.h"
#include "common/state_action.h"

namespace rlcpp
{
    struct Env
    {
        virtual void make(const string &gameName) = 0;

        virtual Space action_space() const = 0;

        virtual Space obs_space() const = 0;

        virtual void step(const Action &action, State *next_obs, Float *reward, bool *done) = 0;

        virtual void reset(State *obs) = 0;

        virtual void close() = 0;

        virtual void render() = 0;

        size_t max_episode_steps;
    };
}

#endif
