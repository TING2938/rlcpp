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
    std::vector<State> new_states;
    Vecf value;
    Vecf adv;
    Vecf rewards;
    Vecf dones;

    void clear()
    {
        this->states.clear();
        this->actions.clear();
        this->new_states.clear();
        this->value.clear();
        this->adv.clear();
        this->rewards.clear();
        this->dones.clear();
    }

    size_t size()
    {
        return this->states.size();
    }
};

struct Actor_policy
{
    Actor_policy(const std::vector<dynet::Layer>& layers, unsigned int n_actions, Real lr = 4e-4)
        : trainer(network.model)
    {
        this->trainer.learning_rate = lr;
        this->network.build_model(layers);
        this->logstd = network.model.add_parameters({n_actions});
    }

    dynet::Expression get_logstd(dynet::ComputationGraph& cg)
    {
        return dynet::parameter(cg, this->logstd);
    }

    dynet::Expression get_mean(const dynet::Expression& x, dynet::ComputationGraph& cg)
    {
        return this->network.nn.run(x, cg);
    }

    void update()
    {
        this->trainer.update();
    }

private:
    Dynet_Network network;
    dynet::Parameter logstd;
    dynet::AdamTrainer trainer;
};

struct Value_network
{
    Value_network(const std::vector<dynet::Layer>& layers, Real lr = 1e-3) : trainer(network.model)
    {
        this->trainer.learning_rate = lr;
        this->network.build_model(layers);
    }

    dynet::Expression get_value(const dynet::Expression& x, dynet::ComputationGraph& cg)
    {
        return this->network.nn.run(x, cg);
    }

    void update()
    {
        this->trainer.update();
    }

private:
    Dynet_Network network;
    dynet::AdamTrainer trainer;
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
                       Real gamma      = 0.99,
                       Real gae_lambda = 0.95)
        : actor(actor_layers, act_dim, 4e-4), critic(critic_layers, 1e-3)
    {
        this->obs_dim = obs_dim;
        this->act_dim = act_dim;

        this->policy_clip = 0.2;

        this->gamma      = gamma;
        this->gae_lambda = gae_lambda;

        this->learn_step = 0;
    }

    // 根据观测值，采样输出动作，带探索过程
    void sample(const State& obs, Action* action) override
    {
        dynet::ComputationGraph cg;
        auto obs_expr    = dynet::input(cg, {(unsigned)this->obs_dim}, obs);
        auto mean_expr   = this->actor.get_mean(obs_expr, cg);
        auto logstd_expr = this->actor.get_logstd(cg);
        auto pi          = dynet::distributions::Normal(mean_expr, logstd_expr);
        auto action_expr = dynet::clip(pi.sample(), -1, 1);
        *action          = dynet::as_vector(cg.forward(action_expr));

        auto state_value = this->critic.get_value(obs_expr, cg);
        this->memory.value.push_back(dynet::as_scalar(cg.forward(state_value)));
    }

    // 根据输入观测值，预测下一步动作
    void predict(const State& obs, Action* action) override
    {
        dynet::ComputationGraph cg;
        auto obs_expr  = dynet::input(cg, {(unsigned)this->obs_dim}, obs);
        auto mean_expr = this->actor.get_mean(obs_expr, cg);
        *action        = dynet::as_vector(cg.forward(mean_expr));
    }

    void store(const State& state, const Action& action, Real reward, const State& next_state, bool done) override
    {
        this->memory.states.push_back(state);
        this->memory.new_states.push_back(next_state);
        this->memory.actions.push_back(action);
        this->memory.rewards.push_back(reward);
        this->memory.dones.push_back(done);
    }

    Real learn() override
    {
        this->generalized_advantage_estimation();

        auto old_log_policy = this->compute_log_policy_prob();  // [act_dim, memory.size]
        auto batch_adv      = this->memory.adv;                 // [1, memory.size]
        // normalize it to stabilize network
        batch_adv = (batch_adv - Real(rlcpp::mean(batch_adv))) / Real(rlcpp::stddev(batch_adv) + 1e-7);

        Real total_loss = 0.0;

        // execute PPO_EPOCHS epochs
        constexpr int PPO_EPOCHS = 7;
        constexpr int BATCH_SIZE = 64;
        for (int epoch = 0; epoch < PPO_EPOCHS; epoch++) {
            // compute the loss and optimize over mini batches of size BATCH_SIZE
            for (int mb = 0; mb < this->memory.size(); mb += BATCH_SIZE) {
                int mb_end         = std::min<int>(this->memory.size(), mb + BATCH_SIZE);
                int minibatch_size = mb_end - mb;

                dynet::Dim state_dynet_dim({unsigned(this->obs_dim)}, minibatch_size);
                dynet::Dim action_dynet_dim({unsigned(this->act_dim)}, minibatch_size);
                dynet::Dim reward_dynet_dim({1}, minibatch_size);

                Vecf minib_states         = rlcpp::flatten(std::vector<rlcpp::State>{
                    this->memory.states.begin() + mb, this->memory.states.begin() + mb_end});  // [obs_dim, b]
                Vecf minib_action         = rlcpp::flatten(std::vector<rlcpp::Action>{
                    this->memory.actions.begin() + mb, this->memory.actions.begin() + mb_end});  // [act_dim, b]
                Vecf minib_old_log_policy = {old_log_policy.begin() + mb * this->act_dim,
                                             old_log_policy.begin() + mb_end * this->act_dim};     // [act_dim, b]
                Vecf minib_adv            = {batch_adv.begin() + mb, batch_adv.begin() + mb_end};  // [1, b]
                Vecf minib_rewards        = {this->memory.rewards.begin() + mb,
                                      this->memory.rewards.begin() + mb_end};  // [1, b]

                dynet::ComputationGraph cg;
                // cg.set_immediate_compute(true);
                // cg.set_check_validity(true);
                {
                    auto action_expr       = dynet::input(cg, action_dynet_dim, minib_action);          // [act_dim, b]
                    auto states_expr       = dynet::input(cg, state_dynet_dim, minib_states);           // [obs_dim, b]
                    auto rewards_expr      = dynet::input(cg, reward_dynet_dim, minib_rewards);         // [1, b]
                    auto value_expr        = this->critic.get_value(states_expr, cg);                   // [1, b]
                    auto adv_expr          = dynet::input(cg, reward_dynet_dim, minib_adv);             // [1, b]
                    auto old_log_prob_expr = dynet::input(cg, action_dynet_dim, minib_old_log_policy);  // [act_dim, b]
                    auto vl_loss =
                        dynet::mean_batches(dynet::squared_distance(value_expr, rewards_expr));  // value loss

                    auto mean_expr         = this->actor.get_mean(states_expr, cg);  // [act_dim, b]
                    auto logstd_expr       = this->actor.get_logstd(cg);             // [act_dim, 1]
                    auto pi                = dynet::distributions::Normal(mean_expr, logstd_expr);
                    auto new_log_prob_expr = pi.log_prob(action_expr);                           // [act_dim, b]
                    auto rt_theta          = dynet::exp(new_log_prob_expr - old_log_prob_expr);  // [act_dim, b]
                    auto surr1             = rt_theta * adv_expr;
                    auto surr2   = dynet::clip(rt_theta, 1 - this->policy_clip, 1 + this->policy_clip) * adv_expr;
                    auto pg_loss = -dynet::mean_elems(dynet::mean_batches(dynet::min(surr1, surr2)));  // actor loss

                    total_loss += dynet::as_scalar(cg.forward(pg_loss));
                    cg.backward(pg_loss);
                    this->actor.update();

                    total_loss += dynet::as_scalar(cg.forward(vl_loss));
                    cg.backward(vl_loss);
                    this->critic.update();
                }
            }
        }
        this->memory.clear();
        return total_loss;
    }

    void save_model(const string& file_name) override
    {
        // this->actor.save(file_name, "/ppo_actor_network", false);
        // this->critic.save(file_name, "/ppo_critic_network", true);
    }

    void load_model(const string& file_name) override
    {
        // this->actor.load(file_name, "/ppo_actor_network");
        // this->critic.load(file_name, "/ppo_critic_network");
    }

    PPORandomReply& buffer()
    {
        return this->memory;
    }

private:
    // Calculate the advantage diuscounted reward
    void generalized_advantage_estimation()
    {
        this->memory.adv.reserve(this->memory.size() - 1);

        Vecf Rs(this->memory.size() - 1, 0);
        Real R       = 0.0;
        Real run_add = 0.0;
        for (int t = this->memory.size() - 2; t >= 0; t--) {
            if (this->memory.dones[t]) {
                run_add = this->memory.rewards[t];
            } else {
                Real sigma = this->memory.rewards[t] + this->gamma * this->memory.value[t + 1] - this->memory.value[t];
                run_add    = sigma + run_add * this->gamma * this->gae_lambda;
            }
            Rs[t] = run_add;
        }
        // the last memory is missing after doing this
        this->memory.adv = Rs;
        this->memory.value.pop_back();
        this->memory.rewards = Rs + this->memory.value;
        this->memory.states.pop_back();
        this->memory.actions.pop_back();
        this->memory.new_states.pop_back();
        this->memory.dones.pop_back();
    }

    // [act_dim, memory.size]
    Vecf compute_log_policy_prob()
    {
        dynet::ComputationGraph cg;
        auto states_expr       = dynet::input(cg, dynet::Dim({unsigned(this->obs_dim)}, this->memory.size()),
                                              rlcpp::flatten(this->memory.states));
        auto action_expr       = dynet::input(cg, dynet::Dim({unsigned(this->act_dim)}, this->memory.size()),
                                              rlcpp::flatten(this->memory.actions));
        auto mean_expr         = this->actor.get_mean(states_expr, cg);
        auto logstd_expr       = this->actor.get_logstd(cg);
        auto pi                = dynet::distributions::Normal(mean_expr, logstd_expr);
        auto old_log_prob_expr = pi.log_prob(action_expr);
        return dynet::as_vector(cg.forward(old_log_prob_expr));  // [act_dim, memory.size]
    }

private:
    Int obs_dim;  // dimension of observation space
    Int act_dim;  // dimension of action
    Real gamma;
    Real gae_lambda;

    size_t learn_step;

    // input: state, output: probability of each action
    Actor_policy actor;    // state -> prob. of action
    Value_network critic;  // V function (state -> V)
    Real policy_clip;

    PPORandomReply memory;

};  // class AC_Agent

}  // namespace rlcpp

#endif  // !__RLCPP_PPO_Discrete_AGENT_H__
