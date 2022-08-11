#pragma once

#include <cpptools/ct_bits/random_tools.h>
#include <algorithm>
#include <random>
#include "tools/dynet_network/dynet_network.h"
#include "tools/memory_reply.h"

#include "common/rl_config.h"
#include "common/state_action.h"


namespace rlcpp
{
// observation space: continuous
// action space: continuous
class DDPG_Agent
{
    using Expression = dynet::Expression;

public:
    using State  = Vecf;
    using Action = Vecf;

public:
    DDPG_Agent(const std::vector<dynet::Layer>& actor_layers,
               const std::vector<dynet::Layer>& critic_layers,
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
        this->batch_action.resize(batch_size * this->act_dim, 0);
        this->batch_reward.resize(batch_size, 0);
        this->batch_next_state.resize(batch_size * this->obs_dim, 0);
        this->batch_done.resize(batch_size, 0);
    }

    // 根据观测值，采样输出动作[-1, 1]，带探索过程
    void sample(const State& obs, Action* action)
    {
        auto norm_action = this->actor.predict(obs);
        if (this->noise_stddev > 0) {
            for (int i = 0; i < this->act_dim; i++) {
                std::normal_distribution<Real> dist(norm_action[i], this->noise_stddev);
                norm_action[i] = dist(this->random_engine);
            }
            ct::clip_<Real>(norm_action, -1, 1);
        }
        *action = std::move(norm_action);
    }

    // 根据输入观测值，预测下一步动作[-1, 1]
    void predict(const State& obs, Action* action)
    {
        *action = this->actor.predict(obs);
    }

    // the action **must** stay between [-1, 1]
    void store(const State& state, const Action& action, Real reward, const State& next_state, bool done)
    {
        this->memory.store({state, action, reward, next_state, done});
    }

    Real learn()
    {
        this->memory.sample_onedim(this->batch_state, this->batch_action, this->batch_reward, this->batch_next_state,
                                   this->batch_done);
        unsigned batch_size = this->batch_reward.size();
        dynet::Dim state_dim({unsigned(this->obs_dim)}, batch_size);
        dynet::Dim action_dim({unsigned(this->act_dim)}, batch_size);
        dynet::Dim Q_dim({1}, batch_size);

        Real loss_value = 0.0f;
        dynet::ComputationGraph cg;

        /// First, update critic network
        {
            // 1. calculate target Q
            auto batch_next_state_expr = dynet::input(cg, state_dim, this->batch_next_state);
            auto target_values_expr    = this->critic_target.nn.run(
                   dynet::concatenate({batch_next_state_expr, this->actor_target.nn.run(batch_next_state_expr, cg)}), cg);
            auto target_Q_values = dynet::as_vector(cg.forward(target_values_expr));  // get Q value
            for (int i = 0; i < batch_size; i++) {
                // update target_Q
                target_Q_values[i] =
                    this->batch_reward[i] + this->gamma * target_Q_values[i] * (1 - this->batch_done[i]);
            }

            // 2. calculate predicted Q
            cg.clear();
            auto batch_state_expr  = dynet::input(cg, state_dim, this->batch_state);
            auto batch_action_expr = dynet::input(cg, action_dim, this->batch_action);
            auto pred_Q_expr       = this->critic.nn.run(dynet::concatenate({batch_state_expr, batch_action_expr}), cg);
            auto pred_Q            = dynet::as_vector(cg.forward(pred_Q_expr));

            // 3. minimize the loss between predicted Q and target Q
            auto loss =
                dynet::mean_batches(dynet::squared_distance(pred_Q_expr, dynet::input(cg, Q_dim, target_Q_values)));
            loss_value += dynet::as_scalar(cg.forward(loss));
            cg.backward(loss);
            this->trainer_critic.update();
        }

        /// Next, update actor network
        {
            cg.clear();
            auto batch_state_expr  = dynet::input(cg, state_dim, this->batch_state);
            auto batch_action_expr = this->actor.nn.run(batch_state_expr, cg);
            auto Q                 = this->critic.nn.run(dynet::concatenate({batch_state_expr, batch_action_expr}), cg);
            auto loss              = dynet::mean_batches(-Q);
            loss_value += dynet::as_scalar(cg.forward(loss));
            cg.backward(loss);
            this->trainer_actor.update();
        }

        this->noise_stddev = std::max(this->noise_stddev - this->noise_stddev_decrease, this->noise_stddev_lower);

        this->actor_target.update_weights_from(this->actor, 0.1);
        this->critic_target.update_weights_from(this->critic, 0.1);

        return loss_value;
    }

    void save_model(const string& file_name)
    {
        this->actor.save(file_name, "/ddpg_actor", false);
        this->critic.save(file_name, "/ddpg_critic", true);
        this->actor_target.save(file_name, "/ddpg_actor_target", true);
        this->critic_target.save(file_name, "/ddpg_critic_target", true);
    }

    void load_model(const string& file_name)
    {
        this->actor.load(file_name, "/ddpg_actor");
        this->critic.load(file_name, "/ddpg_critic");
        this->actor_target.load(file_name, "/ddpg_actor_target");
        this->critic_target.load(file_name, "/ddpg_critic_target");
    }

    RandomReply<State, Action>& memory_reply()
    {
        return this->memory;
    }

private:
    Int obs_dim;  // dimension of observation space
    Int act_dim;  // dimension of action space
    Real gamma;

    std::default_random_engine random_engine;
    Real noise_stddev;
    Real noise_stddev_decrease;
    Real noise_stddev_lower;

    Dynet_Network actor;
    Dynet_Network critic;
    Dynet_Network actor_target;
    Dynet_Network critic_target;

    dynet::AdamTrainer trainer_actor;
    dynet::AdamTrainer trainer_critic;

    RandomReply<State, Action> memory;  // memory reply
    Vecf batch_state;
    Vecf batch_action;
    Vecf batch_reward;
    Vecf batch_next_state;
    std::vector<bool> batch_done;
};  // !class

}  // namespace rlcpp
