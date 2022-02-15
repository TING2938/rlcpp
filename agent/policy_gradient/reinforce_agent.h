/**
 * @file basic_pg_agent.h
 * @author Ting Ye (yeting2938@163.com)
 * @brief basic policy gradient algorithm
 * @version 0.1
 * @date 2022-01-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __RLCPP_REINFORCE_AGENT_H__
#define __RLCPP_REINFORCE_AGENT_H__

#define RLCPP_STATE_TYPE 1
#define RLCPP_ACTION_TYPE 0

#include <algorithm>
#include "agent/agent.h"
#include "tools/dynet_network/dynet_network.h"
#include "tools/random_tools.h"

namespace rlcpp
{
using namespace opt;

// observation space: continuous
// action space: discrete
class REINFORCE_Agent : public Agent
{
    using Expression = dynet::Expression;

public:
    REINFORCE_Agent(const std::vector<dynet::Layer>& layers, Real gamma = 0.99) : network(), trainer(network.model)
    {
        this->network.build_model(layers);
        this->trainer.clip_threshold = 1.0;
        this->trainer.learning_rate  = 5e-4;

        this->obs_dim = layers.front().input_dim;
        this->act_n   = layers.back().output_dim;
        this->gamma   = gamma;

        this->learn_step = 0;
    }

    // 根据观测值，采样输出动作，带探索过程
    void sample(const State& obs, Action* action) override
    {
        auto act_prob = this->network.predict(obs);
        *action       = random_choise(this->act_n, act_prob);
    }

    // 根据输入观测值，预测下一步动作
    void predict(const State& obs, Action* action) override
    {
        auto act_prob = this->network.predict(obs);
        *action       = argmax(act_prob);
    }

    void store(const State& state, const Action& action, Real reward, const State& next_state, bool done) override
    {
        this->op_states.insert(this->op_states.end(), state.begin(), state.end());
        this->op_actions.push_back(action);
        this->op_rewards.push_back(reward);
    }

    Real learn() override
    {
        this->calc_norm_rewards();
        dynet::ComputationGraph cg;
        Expression op_state_expr =
            dynet::input(cg, dynet::Dim({unsigned(this->obs_dim)}, this->op_actions.size()), this->op_states);
        Expression op_rewards_expr  = dynet::input(cg, dynet::Dim({1}, this->op_actions.size()), this->op_rewards);
        Expression op_act_prob_expr = this->network.nn.run(op_state_expr, cg);
        Expression op_picked_act_prob_expr = dynet::pick(op_act_prob_expr, this->op_actions);
        Expression log_prob                = -dynet::log(op_picked_act_prob_expr);
        Expression loss                    = dynet::mean_batches(dynet::cmult(log_prob, op_rewards_expr));
        Real loss_value                    = dynet::as_scalar(cg.forward(loss));
        cg.backward(loss);
        this->trainer.update();
        this->learn_step += 1;
        this->op_actions.clear();
        this->op_states.clear();
        this->op_rewards.clear();
        return loss_value;
    }

private:
    void calc_norm_rewards()
    {
        if (this->op_rewards.size() < 2)
            return;

        for (int i = this->op_rewards.size() - 2; i >= 0; i--) {
            // G(i) = r(i) + γ * G(i+1)
            this->op_rewards[i] += this->gamma * this->op_rewards[i + 1];
        }
        this->op_rewards -= Real(mean(this->op_rewards));
        this->op_rewards /= Real(stddev(this->op_rewards));
    }

private:
    Int obs_dim;  // dimension of observation space
    Int act_n;    // num. of action
    Real gamma;

    size_t learn_step;

    // input: state, output: probability of each action
    Dynet_Network network;
    dynet::AdamTrainer trainer;

    Vecf op_states;
    std::vector<unsigned> op_actions;
    Vecf op_rewards;

};  // !class

}  // namespace rlcpp

#endif  // !__RLCPP_REINFORCE_AGENT_H__
