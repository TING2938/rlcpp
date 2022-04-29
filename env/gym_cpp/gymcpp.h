#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "env/env.h"

namespace rlcpp
{
namespace
{
namespace py = pybind11;
}

class __attribute__((visibility("hidden"))) Gym_cpp : public Env
{
public:
    Gym_cpp() {}

    ~Gym_cpp() {}

    void make(const string& game_name) override
    {
        auto gym  = py::module::import("gym");
        this->env = gym.attr("make")(game_name);
        auto mes  = this->env.attr("_max_episode_steps");
        if (mes.is_none()) {
            this->max_episode_steps = -1;
        } else {
            this->max_episode_steps = py::cast<size_t>(mes);
        }
    }

    Space action_space() const override
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

    Space obs_space() const override
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

    void reset(State* obs) override
    {
        *obs = py::cast<State>(this->env.attr("reset")());
    }

    void close() override
    {
        this->env.attr("close")();
    }

    void step(const Action& action, State* next_obs, Real* reward, bool* done) override
    {
        py::tuple res = this->env.attr("step")(action);
        *next_obs     = py::cast<State>(res[0]);
        *reward       = py::cast<Real>(res[1]);
        *done         = py::cast<bool>(res[2]);
        // py::print(py::str("run step, action: {}, res: {}").format(action, res));
    }

    void render() override
    {
        this->env.attr("render")();
    }

    py::object env;
};  // !class PyGym
}  // namespace rlcpp
