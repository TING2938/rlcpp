/**
 * @file AC_agent.h
 * @author Ting Ye (yeting2938@163.com)
 * @brief AC algorithm
 * @version 0.1
 * @date 2022-03-22
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <algorithm>
#include <cpptools/ct_bits/random_tools.hpp>
#include "tools/dynet_network/dynet_network.h"
#include "tools/memory_reply.h"

#include "common/rl_config.h"
#include "common/state_action.h"

namespace rlcpp
{
using namespace ct::opt;

// observation space: continuous
// action space: discrete
class AC_Agent
{
    using Expression = dynet::Expression;

public:
    using State  = Vecf;
    using Action = Int;

public:
    AC_Agent(const std::vector<dynet::Layer>& actor_layers,
             const std::vector<dynet::Layer>& critic_layers,
             Real gamma = 0.99)
        : trainer_actor(actor.model), trainer_critic(critic.model)
    {
        this->actor.build_model(actor_layers);
        this->critic.build_model(critic_layers);

        this->trainer_actor.learning_rate  = 5e-4;
        this->trainer_critic.learning_rate = 5e-4;

        this->obs_dim = actor_layers.front().input_dim;
        this->act_n   = actor_layers.back().output_dim;

        this->gamma = gamma;

        this->learn_step = 0;
    }

    // 根据观测值，采样输出动作，带探索过程
    void sample(const State& obs, Action* action)
    {
        auto act_prob = this->actor.predict(obs);
        *action       = ct::random_choise(this->act_n, act_prob);
    }

    // 根据输入观测值，预测下一步动作
    void predict(const State& obs, Action* action)
    {
        auto act_prob = this->actor.predict(obs);
        *action       = ct::argmax(act_prob);
    }

    void store(const State& state, const Action& action, Real reward, const State& next_state, bool done)
    {
        this->trans = {state, action, reward, next_state, done};
    }

    Real learn()
    {
        Real loss_value = 0.0f;
        Real td_error   = 0.0f;

        dynet::ComputationGraph cg;

        // for critic update
        {
            auto state_expr      = dynet::input(cg, {unsigned(this->obs_dim)}, this->trans.state);
            auto next_state_expr = dynet::input(cg, {unsigned(this->obs_dim)}, this->trans.next_state);
            auto v_expr          = this->critic.nn.run(state_expr, cg);
            auto next_v_expr     = this->critic.nn.run(next_state_expr, cg);
            auto td_error_expr   = this->trans.reward + next_v_expr * this->gamma * (1 - this->trans.done) - v_expr;
            td_error             = dynet::as_scalar(cg.forward(td_error_expr));
            auto loss_expr       = dynet::square(td_error_expr);
            loss_value += dynet::as_scalar(cg.forward(loss_expr));
            cg.backward(loss_expr);
            this->trainer_critic.update();
        }

        // for actor update
        {
            cg.clear();
            auto state_expr    = dynet::input(cg, {unsigned(this->obs_dim)}, this->trans.state);
            auto prob_expr     = this->actor.nn.run(state_expr, cg);
            auto log_prob_expr = -dynet::log(dynet::pick(prob_expr, this->trans.action));
            auto loss_expr     = log_prob_expr * td_error;
            loss_value += dynet::as_scalar(cg.forward(loss_expr));
            cg.backward(loss_expr);
            this->trainer_actor.update();
        }

        return loss_value;
    }

    void save_model(const string& file_name)
    {
        this->actor.save(file_name, "/ac_actor_network", false);
        this->critic.save(file_name, "/ac_critic_network", true);
    }

    void load_model(const string& file_name)
    {
        this->actor.load(file_name, "/ac_actor_network");
        this->critic.load(file_name, "/ac_critic_network");
    }


private:
    Int obs_dim;  // dimension of observation space
    Int act_n;    // num. of action
    Real gamma;

    size_t learn_step;

    // input: state, output: probability of each action
    Dynet_Network actor;   // state -> prob. of action
    Dynet_Network critic;  // V function (state -> V)

    dynet::AdamTrainer trainer_actor;
    dynet::AdamTrainer trainer_critic;

    Transition<State, Action> trans;

};  // class AC_Agent

}  // namespace rlcpp
