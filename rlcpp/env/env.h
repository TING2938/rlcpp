#ifndef __RL_ENV_H__
#define __RL_ENV_H__

#include "../common/rl_config.h"
#include "../common/state_action.h"

struct Env
{
    void virtual make(const string& gameName) = 0;

    Int virtual action_space() const = 0;

    Veci virtual obs_space() const = 0;

    void virtual step(const Action& action, State* next_obs, double* reward, bool* done) = 0;

    void virtual reset(State* obs) = 0;

    void virtual close() = 0;

    void virtual render() = 0;
};

#endif

