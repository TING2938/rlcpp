#ifndef __RLCPP_DDPG_AGENT_H__
#define __RLCPP_DDPG_AGENT_H__

#define RLCPP_STATE_TYPE 1
#define RLCPP_ACTION_TYPE 1

#include <algorithm>
#include "agent/agent.h"
#include "network/dynet_network/dynet_network.h"
#include "tools/memory_reply.h"
#include "tools/random_tools.h"

namespace rlcpp
{
// observation space: continuous
// action space: continuous
class DDPG_Agent : public Agent
{
    using Expression = dynet::Expression;

public:
    DDPG_Agent(const std::vector<dynet::Layer>& actor_layers,
               const std::vector<dynet::Layer>& critic_layers,
               const Vecf& action_low,
               const Vecf& action_high,
               Int max_memory_size,
               Int batch_size,
               Real gamma = 0.99)
        : trainer_actor(actor.model), trainer_critic(critic.model)
    {
        this->actor.build_model(actor_layers);
        this->critic.build_model(critic_layers);
        this->actor_target.build_model(actor_layers);
        this->critic_target.build_model(critic_layers);

        this->actor_target.update_weights_from(this->actor);
        this->critic_target.update_weights_from(this->critic);

        this->trainer_actor.learning_rate  = 1e-4;
        this->trainer_critic.learning_rate = 1e-3;

        this->memory.init(max_memory_size);
        this->obs_dim = actor_layers.front().input_dim;
        this->act_dim = actor_layers.back().output_dim;

        this->gamma                 = gamma;
        this->noise_stddev          = 1.0f;
        this->noise_stddev_decrease = 5e-4f;
        this->noise_stddev_lower    = 5e-2f;

        this->batch_state.resize(batch_size * this->obs_dim, 0);
        this->batch_action.resize(batch_size, 0);
        this->batch_reward.resize(batch_size, 0);
        this->batch_next_state.resize(batch_size * this->obs_dim, 0);
        this->batch_done.resize(batch_size, 0);
    }

    // 根据观测值，采样输出动作，带探索过程
    void sample(const State& obs, Action* action) override
    {
        auto norm_action = this->actor.predict(obs);
        // TODO
        *action = norm_action;
    }

    // 根据输入观测值，预测下一步动作
    void predict(const State& obs, Action* action) override
    {
        auto norm_action = this->actor.predict(obs);
        // TODO
        *action = norm_action;
    }

    void store(const State& state, const Action& action, Real reward, const State& next_state, bool done) override
    {
        this->memory.store({state, action, reward, next_state, done});
    }

    Real learn() override
    {
        this->memory.sample_onedim(this->batch_state, this->batch_action, this->batch_reward, this->batch_next_state,
                                   this->batch_done);
        unsigned batch_size = this->batch_reward.size();

        // update critic
        dynet::ComputationGraph cg1;
        auto batch_next_state_expr =
            dynet::input(cg1, dynet::Dim({unsigned(this->obs_dim)}, batch_size), this->batch_next_state);
        auto target_action_expr = this->actor_target.nn.run(batch_next_state_expr, cg1);
        auto target_values_expr =
            this->critic_target.nn.run(dynet::concatenate({batch_next_state_expr, target_action_expr}), cg1);
        auto target_values = dynet::as_vector(cg1.forward(target_values_expr));  // Q

        for (int i = 0; i < batch_size; i++) {
            // update Q
            target_values[i] = this->batch_reward[i] + this->gamma * target_values[i] * (1 - this->batch_done[i]);
        }


        Vecf batch_target_action = this->actor_target.predict(this->batch_next_state);
        dynet::concatenate({});
        Vecf target_values(batch_size);
        for (int i = 0; i < batch_size; i++) {
            Real maxQ        = *std::max_element(batch_target_Q.begin() + i * this->act_dim,
                                                 batch_target_Q.begin() + (i + 1) * this->act_dim);
            target_values[i] = this->batch_reward[i] + this->gamma * maxQ * (1 - this->batch_done[i]);
        }

        // update actor
        dynet::ComputationGraph cg;
        Expression batch_state_expr =
            dynet::input(cg, dynet::Dim({unsigned(this->obs_dim)}, batch_size), this->batch_state);
        Expression batch_Q_expr = this->network.nn.run(batch_state_expr, cg);
        Expression picked_values_expr =
            dynet::pick(batch_Q_expr, {this->batch_action.begin(), this->batch_action.end()});
        Expression target_values_expr = dynet::input(cg, dynet::Dim({1}, batch_size), target_values);
        Expression loss = dynet::sum_batches(dynet::squared_distance(picked_values_expr, target_values_expr));
        Real loss_value = dynet::as_scalar(cg.forward(loss));
        cg.backward(loss);
        this->trainer.update();


        return loss_value;
    }

private:
    Int obs_dim;  // dimension of observation space
    Int act_dim;  // dimension of action space
    Real gamma;

    Real noise_stddev;
    Real noise_stddev_decrease;
    Real noise_stddev_lower;

    Dynet_Network actor;
    Dynet_Network critic;
    Dynet_Network actor_target;
    Dynet_Network critic_target;

    dynet::AdamTrainer trainer_actor;
    dynet::AdamTrainer trainer_critic;

    RandomReply memory;  // memory reply
    Vecf batch_state;
    Vecf batch_action;
    Vecf batch_reward;
    Vecf batch_next_state;
    std::vector<bool> batch_done;
};  // !class

}  // namespace rlcpp

#endif  // !__RLCPP_DDPG_AGENT_H__
