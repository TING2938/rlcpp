#ifndef __RL_ENV_H__
#define __RL_ENV_H__

#include "../common/rl_config.h"
#include "../common/state_action.h"

namespace rlcpp
{
    struct Env
    {
        void virtual make(const string &gameName) = 0;

        Space virtual action_space() const = 0;

        Space virtual obs_space() const = 0;

        void virtual step(const Action &action, State *next_obs, double *reward, bool *done) = 0;

        void virtual reset(State *obs) = 0;

        void virtual close() = 0;

        void virtual render() = 0;

        bool virtual bDiscrete_obs_space() = 0;

        bool virtual bDiscrete_action_space() = 0;
    };
}

#endif
