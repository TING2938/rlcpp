#ifndef __AIFAN_SIMPLE_H__
#define __AIFAN_SIMPLE_H__

#include "env/env.h"
#include "matplotlibcpp.h"
#include "tools/random_tools.h"
#include "tools/ring_vector.h"

namespace plt = matplotlibcpp;

namespace rlcpp
{
class AIFanSimple : public Env
{
public:
    void make(const string& gameName)
    {
        this->action_space_.bDiscrete = true;
        this->action_space_.n         = 20;
        this->obs_space_.bDiscrete    = false;
        this->obs_space_.shape        = {1};

        this->EnvTemp    = 32;
        this->FanNx      = 100;
        this->SensorTemp = 68;
        this->FanPwr     = this->_Nx2FanPwr(this->FanNx);
        this->BrdPwr     = 30;

        this->max_episode_steps = 100;

        this->target_temp = 60;

        this->memory_temp.init(100);
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
        this->BrdPwr = rlcpp::randf(30., 150.);
        this->FanPwr = this->_Nx2FanPwr(this->FanNx);
        this->SensorTemp =
            this->_get_cpu_temp(steps, this->SensorTemp, this->BrdPwr, this->EnvTemp, u, 0.187572, 0.05, 0.618);

        this->memory_temp.store(this->SensorTemp);
        // printf("the temp is %f\n", this->SensorTemp);

        this->_fillState(next_obs);

        if (this->SensorTemp >= 80) {
            *reward = -100;
            *done   = true;
            return;
        }


        //*reward = -(this->SensorTemp - (this->target_temp - 20)) * (this->SensorTemp - (this->target_temp + 20)) -
        // 399; *done   = false;

        if (this->SensorTemp < this->target_temp) {
            *reward = -(this->target_temp - this->SensorTemp) / (this->target_temp - this->EnvTemp) / 2.0 + 1.0;
            *done   = false;
            return;
        } else {
            *reward = -(this->SensorTemp - this->target_temp) / (80 - this->target_temp) / 2.0 + 1.0;
            *done   = false;
            return;
        }
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

    void close()
    {
        plt::detail::_interpreter::kill();
    }

    void render()
    {
        plt::clf();
        plt::plot(this->memory_temp.lined_vector(), "--o");
        plt::plot(std::vector<Real>(this->memory_temp.size(), this->target_temp), ":r");
        plt::ylim(this->target_temp - 30, this->target_temp + 30);
        plt::pause(0.01);
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
        if (action < 19)
            return action * 4 + 20;
        else
            return 100;
    }

    Real _Nx2U(Int Nx, Real Umax = 100)
    {
        return Umax * std::pow(Nx / 100.0, 0.7);
    }

    Real _Nx2FanPwr(Int Nx)
    {
        return 528.107 * std::pow(Nx / 100.0, 2);
    }

    void _fillState(State* obs)
    {
        (*obs)[0] = this->EnvTemp - 32;
        // (*obs)[0] = (this->FanNx - 60) / 23.0;
        // (*obs)[0] = (this->FanPwr - 215.55) / 147.63;
        // (*obs)[0] = this->BrdPwr;  // (this->BrdPwr - 90) / 35.0;
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

    RingVector<Real> memory_temp;
};
}  // namespace rlcpp


#endif  // !__AIFAN_SIMPLE_H__