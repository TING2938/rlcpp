#pragma once

#include <cpptools/ct_bits/random_tools.h>
#include <cpptools/ct_bits/ring_vector.h>
#include <pybind11/embed.h>

#include "common/rl_config.h"
#include "common/state_action.h"

namespace py = pybind11;
using namespace pybind11::literals;

class SinGen
{
public:
    void init(double min, double max, double T)
    {
        this->min = min;
        this->max = max;
        this->T   = T;
        this->k   = 0;
    }
    /**
     * @brief return (max-min)/2 * sin(2pi/T*k) + (max+min)/2
     */
    double operator()()
    {
        return std::sin(2 * M_PI / this->T * this->k++) * (this->max - this->min) / 2.0 + (this->max + this->min) / 2.0;
    }

private:
    double min;
    double max;
    double T;
    size_t k;
};

namespace rlcpp
{
template <typename State, typename Action>
class __attribute__((visibility("hidden"))) AIFanSimple
{
public:
    void make(const string& gameName)
    {
        this->action_space_.bDiscrete = true;
        this->action_space_.n         = 100;
        this->obs_space_.bDiscrete    = false;
        this->obs_space_.shape        = {1};

        this->EnvTemp    = 32;
        this->FanNx      = 100;
        this->SensorTemp = 68;
        this->FanPwr     = this->_Nx2FanPwr(this->FanNx);
        this->BrdPwr     = 30;

        this->max_episode_steps = 100;

        this->target_temp = 50;

        this->memory_temp.init(100);
        this->memory_P.init(100);
        this->memory_Nx.init(100);
        this->singen.init(60, 130, 10);

        auto plt = py::module::import("matplotlib.pyplot");
        this->ax = plt.attr("subplots")(3, 1, "figsize"_a = py::make_tuple(15, 8));
    }

    Space action_space() const
    {
        return this->action_space_;
    }

    Space obs_space() const
    {
        return this->obs_space_;
    }

    void step(const Action& action, State* next_obs, Real* reward, bool* done)
    {
        this->FanNx  = this->_Action2Nx(action);
        auto u       = this->_Nx2U(this->FanNx, 200);
        auto steps   = 4 / 0.04;
        this->BrdPwr = this->singen() + ct::randf(-5., 5.);
        this->FanPwr = this->_Nx2FanPwr(this->FanNx);
        this->SensorTemp =
            this->_get_cpu_temp(steps, this->SensorTemp, this->BrdPwr, this->EnvTemp, u, 0.187572, 0.05, 0.618);

        this->memory_temp.store(this->SensorTemp);
        this->memory_Nx.store(this->FanNx);
        this->memory_P.store(this->BrdPwr);
        // printf("the temp is %f\n", this->SensorTemp);

        this->_fillState(next_obs);

        /*
            if (this->SensorTemp >= 80) {
                *reward = -200;
                *done   = false;
                return;
            }
        */

        //*reward = -(this->SensorTemp - (this->target_temp - 20)) * (this->SensorTemp - (this->target_temp + 20)) -
        // 399; *done   = false;
        if (std::abs(this->SensorTemp - this->target_temp) < 2.1) {
            *reward = 1.0;
            *done   = false;
        } else if (std::abs(this->SensorTemp - this->target_temp) < 10.1) {
            *reward = 0.0;
            *done   = false;
        } else {
            *reward = -1.0;
            *done   = false;
        }
        /*
        if (this->SensorTemp < this->target_temp) {
            *reward = -(this->target_temp - this->SensorTemp) / (this->target_temp - this->EnvTemp) / 24.0 + 2.0;
            *done   = false;
            return;
        } else {
            *reward = -(this->SensorTemp - this->target_temp) / (80 - this->target_temp) / 24.0 + 2.0;
            *done   = false;
            return;
        }
        */
    }

    void reset(State* obs)
    {
        this->EnvTemp    = 32;
        this->FanNx      = 100;
        this->SensorTemp = 68;
        this->FanPwr     = this->_Nx2FanPwr(this->FanNx);
        this->BrdPwr     = 30;

        this->_fillState(obs);
    }

    void close() {}

    void render()
    {
        for (int i = 0; i < 3; i++) {
            ax[i].attr("cla")();
        }

        ax[0].attr("plot")(this->memory_temp.lined_vector(), "--o");
        ax[0].attr("axhline")(this->target_temp, "color"_a = "r", "ls"_a = ":");
        ax[0].attr("set_ylim")(py::make_tuple(this->target_temp - 30, this->target_temp + 30));
        ax[0].attr("set_ylabel")("Sensor Temp");

        ax[1].attr("plot")(this->memory_P.lined_vector(), "--or");
        ax[1].attr("set_ylim")(py::make_tuple(30, 150));
        ax[1].attr("set_ylabel")("BrdPwr");

        ax[2].attr("plot")(this->memory_Nx.lined_vector(), "--ob");
        ax[2].attr("set_ylim")(py::make_tuple(0, 200));
        // ax[2].set_yticks({20, 40, 60, 80, 100});
        ax[2].attr("set_ylabel")("Nx");

        plt.attr("pause")(0.01);
    }

private:
    Real
    _get_cpu_temp(Real n, Real T0, Real P, Real T_env, Real u, Real k2 = 0.187572, Real k3 = 0.05, Real Gamma = 0.618)
    {
        /*
        auto p   = 1 - k2 * k3 * std::pow(u, Gamma);
        auto q   = k3 * P + (1 - p) * T_env;
        auto tmp = q / (p - 1);
        return (T0 + tmp) * std::pow(p, n) - tmp;
        */
        return T_env + P / (k2 * std::pow(u, Gamma));
    }

    Int _Action2Nx(const Action& action)
    {
        return action * 2;
    }

    Real _Nx2U(Int Nx, Real Umax = 100)
    {
        // return Umax * std::pow(Nx / 100.0, 0.7);
        return Umax * (Nx / 100.0);
    }

    Real _Nx2FanPwr(Int Nx)
    {
        return 528.107 * std::pow(Nx / 100.0, 2);
    }

    void _fillState(State* obs)
    {
        // (*obs)[0] = this->EnvTemp - 32;
        // (*obs)[0] = (this->FanNx - 60) / 23.0;
        // (*obs)[0] = (this->FanPwr - 215.55) / 147.63;
        (*obs)[0] = this->BrdPwr;  // (this->BrdPwr - 90) / 35.0;
        // (*obs)[0] = (this->SensorTemp - 60) / 10.0;
    }

private:
    Space action_space_;
    Space obs_space_;

    Real SensorTemp;
    Real EnvTemp;
    Int FanNx;
    Real FanPwr;
    Real BrdPwr;

    Real target_temp;

    SinGen singen;

    py::module_ plt;
    py::tuple ax;

    ct::RingVector<Real> memory_temp, memory_P, memory_Nx;

public:
    /**
     * @brief  最大回合数
     */
    size_t max_episode_steps;
};
}  // namespace rlcpp
