#pragma once

#include "common/rl_config.h"
#include "common/state_action.h"

namespace rlcpp
{
struct DQN_Base_Agent
{
public:
    using State  = Vecf;
    using Action = Int;

public:
    virtual void sample(const State& obs, Action* action) = 0;

    virtual void predict(const State& obs, Action* action) = 0;

    virtual void store(const State& state, const Action& action, Real reward, const State& next_state, bool done) = 0;

    virtual Real learn() = 0;

    virtual void save_model(const string& file_name) = 0;

    virtual void load_model(const string& file_name) = 0;

    virtual ~DQN_Base_Agent() {}
};

}  // namespace rlcpp
