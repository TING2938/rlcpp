/**
 * @file PPO_Discrete_agent.h
 * @author Ting Ye (yeting2938@163.com)
 * @brief PPO Discrete algorithm
 * @version 0.1
 * @date 2022-04-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __RLCPP_PPO_Discrete_AGENT_H__
#define __RLCPP_PPO_Discrete_AGENT_H__

#define RLCPP_STATE_TYPE 1
#define RLCPP_ACTION_TYPE 0

#include <algorithm>
#include "agent/agent.h"
#include "tools/dynet_network/dynet_network.h"
#include "tools/memory_reply.h"
#include "tools/random_tools.h"

namespace rlcpp
{
using namespace opt;

class PPORandomReply
{
public:
    std::vector<State> states;
    std::vector<Action> actions;
    Vecf probs;
    Vecf values;
    Vecf rewards;
    Vecf dones;

    void clear()
    {
        this->states.clear();
        this->actions.clear();
        this->probs.clear();
        this->values.clear();
        this->rewards.clear();
        this->dones.clear();
    }
};


// observation space: continuous
// action space: discrete
class PPO_Discrete_Agent : public Agent
{
    using Expression = dynet::Expression;

public:
    PPO_Discrete_Agent(const std::vector<dynet::Layer>& actor_layers,
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
    void sample(const State& obs, Action* action) override
    {
        auto act_prob = this->actor.predict(obs);
        *action       = random_choise(this->act_n, act_prob);
        this->memory.probs.push_back(act_prob[*action]);
        this->memory.values.push_back(this->critic.predict(obs).front());
    }

    // 根据输入观测值，预测下一步动作
    void predict(const State& obs, Action* action) override
    {
        auto act_prob = this->actor.predict(obs);
        *action       = argmax(act_prob);
    }

    void store(const State& state, const Action& action, Real reward, const State& next_state, bool done) override
    {
        this->memory.states.push_back(state);
        this->memory.actions.push_back(action);
        this->memory.rewards.push_back(reward);
        this->memory.dones.push_back(done);
    }

    Real learn() override
    {
        Real loss_value = 0.0f;
        Real td_error   = 0.0f;

        this->calc_norm_rewards();
        auto advantage = this->memory.rewards - this->memory.values;

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

    void save_model(const string& file_name) override
    {
        this->actor.save(file_name, "/ac_actor_network", false);
        this->critic.save(file_name, "/ac_critic_network", true);
    }

    void load_model(const string& file_name) override
    {
        this->actor.load(file_name, "/ac_actor_network");
        this->critic.load(file_name, "/ac_critic_network");
    }

private:
    Vecf calc_advantage()
    {
        Vecf advantage(this->memory.rewards.size(), 0);
        for (int t = 0; t < advantage.size() - 1; t++) {
            Real discount = 1.0;
            Real a_t      = 0.0;
            for (int k = t; k < advantage.size() - 1; k++) {
                a_t += discount * (this->memory.rewards[k] +
                                   this->gamma * this->memory.values[k + 1] * (1 - this->memory.dones[k]) -
                                   this->memory.values[k]);
                discount *= (this->gamma * 0.95);
            }
            advantage[t] = a_t;
        }
        return advantage;
    }

private:
    Int obs_dim;  // dimension of observation space
    Int act_n;    // num. of action
    Real gamma;

    size_t learn_step;

    // input: state, output: probability of each action
    Dynet_Network actor;   // state -> prob. of action
    Dynet_Network critic;  // V function (state -> V)

    PPORandomReply memory;

    dynet::AdamTrainer trainer_actor;
    dynet::AdamTrainer trainer_critic;
};  // class AC_Agent

}  // namespace rlcpp

#endif  // !__RLCPP_PPO_Discrete_AGENT_H__
