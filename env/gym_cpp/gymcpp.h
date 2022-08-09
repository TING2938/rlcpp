#pragma once

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "common/rl_config.h"
#include "common/state_action.h"

namespace rlcpp
{
namespace
{
namespace py = pybind11;
}

template <typename State, typename Action>
class __attribute__((visibility("hidden"))) Gym_cpp
{
public:
    Gym_cpp() {}

    ~Gym_cpp() {}

    void make(const string& game_name)
    {
        auto gym  = py::module::import("gym");
        this->env = gym.attr("make")(game_name);
        try {
            auto mes = this->env.attr("_max_episode_steps");
            if (mes.is_none()) {
                this->max_episode_steps = -1;
            } else {
                this->max_episode_steps = py::cast<size_t>(mes);
            }
        } catch (const std::exception& e) {
            std::cout << e.what() << std::endl;
            this->max_episode_steps = -1;
        }
    }

    Space action_space() const
    {
        Space ret;
        auto act_space = this->env.attr("action_space");
        if (py::len(act_space.attr("shape")) == 0) {
            ret.bDiscrete = true;
            ret.n         = act_space.attr("n").cast<int>();
        } else {
            ret.bDiscrete = false;
            ret.high      = py::cast<Vecf>(act_space.attr("high"));
            ret.low       = py::cast<Vecf>(act_space.attr("low"));
            ret.shape     = py::cast<Veci>(act_space.attr("shape"));
        }
        return ret;
    }

    Space obs_space() const
    {
        Space ret;
        auto obs_space = this->env.attr("observation_space");
        if (py::len(obs_space.attr("shape")) == 0) {
            ret.bDiscrete = true;
            ret.n         = obs_space.attr("n").cast<int>();
        } else {
            ret.bDiscrete = false;
            ret.high      = py::cast<Vecf>(obs_space.attr("high"));
            ret.low       = py::cast<Vecf>(obs_space.attr("low"));
            ret.shape     = py::cast<Veci>(obs_space.attr("shape"));
        }
        return ret;
    }

    void reset(State* obs)
    {
        *obs = py::cast<State>(this->env.attr("reset")());
    }

    void close()
    {
        this->env.attr("close")();
    }

    void step(const Action& action, State* next_obs, Real* reward, bool* done)
    {
        py::tuple res = this->env.attr("step")(action);
        *next_obs     = py::cast<State>(res[0]);
        *reward       = py::cast<Real>(res[1]);
        *done         = py::cast<bool>(res[2]);
        // py::print(py::str("run step, action: {}, res: {}").format(action, res));
    }

    void render()
    {
        this->env.attr("render")();
    }

    py::object env;

    /**
     * @brief  最大回合数
     */
    size_t max_episode_steps;
};  // !class PyGym
}  // namespace rlcpp
