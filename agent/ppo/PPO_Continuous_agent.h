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
#define RLCPP_ACTION_TYPE 1

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
    std::vector<Vecf> probs;
    Vecf rewards;
    Vecf dones;
    Int batch_size;

    std::vector<Veci> sample()
    {
        std::vector<Veci> ret(this->states.size() / this->batch_size, Veci(this->batch_size));
        Veci ind(this->states.size());
        std::iota(ind.begin(), ind.end(), 0);
        std::random_shuffle(ind.begin(), ind.end());
        for (Int i = 0; i < ret.size(); i++) {
            for (Int j = 0; j < this->batch_size; j++) {
                ret[i][j] = ind[i * batch_size + j];
            }
        }
        return ret;
    }

    void clear()
    {
        this->states.clear();
        this->actions.clear();
        this->probs.clear();
        this->rewards.clear();
        this->dones.clear();
    }

    size_t size()
    {
        return this->states.size();
    }
};


// observation space: continuous
// action space: discrete
class PPO_Discrete_Agent : public Agent
{
    using Expression = dynet::Expression;

public:
    /**
     * @brief Construct a new ppo discrete agent object
     *
     * @param actor_layers obs_dim -> hidden_n
     * @param critic_layers obs_dim -> 1
     * @param obs_dim
     * @param act_dim
     * @param gamma
     */
    PPO_Discrete_Agent(const std::vector<dynet::Layer>& actor_layers,
                       const std::vector<dynet::Layer>& critic_layers,
                       Int obs_dim,
                       Int act_dim,
                       Real gamma = 0.99)
        : trainer_actor(actor.model), trainer_critic(critic.model)
    {
        this->actor.build_model(actor_layers);
        this->critic.build_model(critic_layers);

        this->trainer_actor.learning_rate  = 3e-4;
        this->trainer_critic.learning_rate = 3e-4;

        this->obs_dim = obs_dim;
        this->act_dim = act_dim;

        auto actor_hidden_n = actor_layers.back().output_dim;
        this->actor_mu.append(this->actor.model, dynet::Layer(actor_hidden_n, this->act_dim, dynet::TANH, 0.0));
        this->actor_sigma.append(this->actor.model, dynet::Layer(actor_hidden_n, this->act_dim, dynet::SOFTPLUS, 0.0));

        this->memory.batch_size = 32;
        this->policy_clip       = 0.2;

        this->gamma = gamma;

        this->learn_step = 0;
    }

    // 根据观测值，采样输出动作，带探索过程
    void sample(const State& obs, Action* action) override
    {
        constexpr float SQRT_2PI = 2.5066282746310002;  // sqrt(2pi)
        dynet::ComputationGraph cg;
        auto obs_expr      = dynet::input(cg, {(unsigned)this->obs_dim}, obs);
        auto res           = this->run_actor(obs_expr, cg);  // res ==> [mu, sigma]
        auto dist_expr     = dynet::random_normal(cg, {(unsigned)this->act_dim}, 0.0, 1.0);
        auto action_expr   = dynet::cmult(dist_expr, res.second) + res.first;
        auto log_prob_expr = -dynet::cmult(dist_expr, dist_expr) / 2 - dynet::log(SQRT_2PI * res.second);
        this->memory.probs.push_back(dynet::as_vector(cg.forward(log_prob_expr)));
        *action = dynet::as_vector(cg.forward(action_expr));
        cg.print_graphviz();
    }

    // 根据输入观测值，预测下一步动作
    void predict(const State& obs, Action* action) override
    {
        constexpr float SQRT_2PI = 2.5066282746310002;  // sqrt(2pi)
        dynet::ComputationGraph cg;
        auto obs_expr = dynet::input(cg, {(unsigned)this->obs_dim}, obs);
        auto res      = this->run_actor(obs_expr, cg);  // res ==> [mu, sigma]
        *action       = dynet::as_vector(cg.forward(res.first));
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
        Vecf Rs(this->memory.rewards.size(), 0);
        Real R = 0.0;
        for (int t = Rs.size() - 1; t >= 0; t--) {
            R     = this->memory.rewards[t] + this->gamma * R * (1 - this->memory.dones[t]);
            Rs[t] = R;
        }

        dynet::Dim reward_dynet_dim({1}, this->memory.batch_size);
        dynet::Dim state_dynet_dim({unsigned(this->obs_dim)}, this->memory.batch_size);

        Real total_loss = 0.0;

        int n_epochs = std::round(10.0 * this->memory.size() / this->memory.batch_size);
        for (int epoch = 0; epoch < n_epochs; epoch++) {
            auto batches = this->memory.sample();
            for (auto&& batch : batches) {
                auto states    = rlcpp::flatten(rlcpp::gather(this->memory.states, batch));
                auto actions   = rlcpp::gather(this->memory.actions, batch);
                auto old_probs = rlcpp::gather(this->memory.probs, batch);
                auto v_target  = rlcpp::gather(Rs, batch);

                dynet::ComputationGraph cg;
                {
                    auto rn                  = dynet::random_normal(cg, {1}, 0, 1);
                    auto states_expr         = dynet::input(cg, state_dynet_dim, states);
                    auto critic_value_expr   = this->critic.nn.run(states_expr, cg);
                    auto dist_expr           = this->actor.nn.run(states_expr, cg);
                    auto new_probs_expr      = dynet::pick(dist_expr, {actions[0].begin(), actions[0].end()});
                    auto v_target_expr       = dynet::input(cg, reward_dynet_dim, v_target);
                    auto advs_expr           = v_target_expr - critic_value_expr;
                    auto old_probs_expr      = dynet::input(cg, reward_dynet_dim, states);
                    auto prob_ratio_expr     = new_probs_expr / old_probs_expr;
                    auto weighted_probs_expr = dynet::cmult(advs_expr, prob_ratio_expr);
                    auto cliped_prob_ratio_expr =
                        dynet::clip(prob_ratio_expr, 1 - this->policy_clip, 1 + this->policy_clip, cg);
                    auto weighted_clipped_probs_expr = dynet::cmult(cliped_prob_ratio_expr, advs_expr);
                    auto actor_loss_expr =
                        -dynet::mean_batches(dynet::min(weighted_probs_expr, weighted_clipped_probs_expr));
                    total_loss += dynet::as_scalar(cg.forward(actor_loss_expr));

                    if (0) {
                        std::cout << "dist_expr: " << dynet::as_vector(dist_expr.value()) << std::endl;
                        std::cout << "prob_ratio:             " << dynet::as_vector(prob_ratio_expr.value())
                                  << std::endl;
                        std::cout << "cliped_prob_ratio_expr: " << dynet::as_vector(cliped_prob_ratio_expr.value())
                                  << std::endl;
                    }
                    cg.backward(actor_loss_expr);
                    this->trainer_actor.update();

                    auto critic_loss_expr =
                        dynet::mean_batches(dynet::squared_distance(v_target_expr, critic_value_expr));
                    total_loss += dynet::as_scalar(cg.forward(critic_loss_expr));
                    cg.backward(critic_loss_expr);
                    this->trainer_critic.update();
                }
            }
        }
        this->memory.clear();
        return total_loss;
    }

    void save_model(const string& file_name) override
    {
        this->actor.save(file_name, "/ppo_actor_network", false);
        this->critic.save(file_name, "/ppo_critic_network", true);
    }

    void load_model(const string& file_name) override
    {
        this->actor.load(file_name, "/ppo_actor_network");
        this->critic.load(file_name, "/ppo_critic_network");
    }

    PPORandomReply& buffer()
    {
        return this->memory;
    }

private:
    /**
     * @brief run in computation graph and get mu and sigma
     *
     * @param x state
     * @param cg
     * @param device
     * @return std::pair<dynet::Expression, dynet::Expression> [mu, sigma]
     */
    std::pair<dynet::Expression, dynet::Expression> run_actor(const dynet::Expression& x,
                                                              dynet::ComputationGraph& cg,
                                                              dynet::Device* device = dynet::default_device)
    {
        auto l1 = this->actor.nn.run(x, cg);
        return {this->actor_mu.run(l1, cg), this->actor_sigma.run(l1, cg)};
    }

private:
    Int obs_dim;  // dimension of observation space
    Int act_dim;  // dimension of action
    Real gamma;

    size_t learn_step;

    // input: state, output: probability of each action
    Dynet_Network actor;     // state -> prob. of action
    dynet::MLP actor_mu;     // [-1, 1]
    dynet::MLP actor_sigma;  // softplus, > 0

    Dynet_Network critic;  // V function (state -> V)
    Real policy_clip;

    PPORandomReply memory;

    dynet::AdamTrainer trainer_actor;
    dynet::AdamTrainer trainer_critic;
};  // class AC_Agent

}  // namespace rlcpp

#endif  // !__RLCPP_PPO_Discrete_AGENT_H__
