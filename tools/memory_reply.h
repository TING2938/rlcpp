#ifndef __RL_RANDOM_REPLY_H__
#define __RL_RANDOM_REPLY_H__

#include <random>
#include "common/state_action.h"
#include "tools/random_tools.h"
#include "tools/reduce_tree.h"
#include "tools/ring_vector.h"
#include "tools/utility.hpp"
#include "tools/vector_tools.h"

namespace rlcpp
{
struct Transition
{
    State state;
    Action action;
    Real reward;
    State next_state;
    bool done;
};

class RandomReply : public RingVector<Transition>
{
    static_assert(RLCPP_STATE_TYPE == 1, "state must be box type");

public:
    void sample_onedim(Vecf& batch_state,
                       Vecf& batch_action,
                       Vecf& batch_reward,
                       Vecf& batch_next_state,
                       std::vector<bool>& batch_done) const
    {
        size_t batch_size = batch_reward.size();
        size_t state_dim  = batch_state.size() / batch_size;
        size_t action_dim = batch_action.size() / batch_size;
        size_t len        = this->size();

        for (size_t i = 0; i < batch_size; i++) {
            auto& tmp = this->memory[randd(0, len)];
            std::copy_n(tmp.state.begin(), state_dim, batch_state.begin() + i * state_dim);

#if RLCPP_ACTION_TYPE == 0
            batch_action[i] = tmp.action;
#elif RLCPP_ACTION_TYPE == 1
            std::copy_n(tmp.action.begin(), action_dim, batch_action.begin() + i * action_dim);
#endif

            batch_reward[i] = tmp.reward;
            std::copy_n(tmp.next_state.begin(), state_dim, batch_next_state.begin() + i * state_dim);
            batch_done[i] = tmp.done;
        }
    }

    void sample(std::vector<State>& batch_state,
                std::vector<Action>& batch_action,
                Vecf& batch_reward,
                std::vector<State>& batch_next_state,
                std::vector<bool>& batch_done) const
    {
        size_t batch_size = batch_state.size();
        size_t len        = this->size();

        for (size_t i = 0; i < batch_size; i++) {
            auto& tmp           = this->memory[randd(0, len)];
            batch_state[i]      = tmp.state;
            batch_action[i]     = tmp.action;
            batch_reward[i]     = tmp.reward;
            batch_next_state[i] = tmp.next_state;
            batch_done[i]       = tmp.done;
        }
    }

    bool is_fresh()
    {
        return this->idx == 0 && this->bFull;
    }

    friend std::ostream& operator<<(std::ostream& os, const RandomReply& reply);
    friend std::istream& operator>>(std::istream& is, RandomReply& reply);
};

class PrioritizedReply
{
public:
    void init(size_t max_size, Real alpha = 0.6)
    {
        this->max_size = std::pow(2, std::ceil(std::log2(max_size)));
        this->memory.resize(this->max_size);
        this->sum_tree.init(this->max_size);
        this->min_tree.init(this->max_size);
        this->idx             = 0;
        this->max_value       = 1.0;
        this->max_value_upper = 1000.0;
        this->alpha           = alpha;
    }

    void store(const State& state, const Action& action, Real reward, const State& next_state, bool done)
    {
        this->memory[this->idx] = {state, action, reward, next_state, done};
        this->sum_tree.setItem(idx, this->max_value);
        this->min_tree.setItem(idx, this->max_value);
        if (this->memory.empty()) {
            printf("empty memory relpy!");
            std::exit(-1);
        }
        if (this->idx == this->memory.size() - 1) {
            this->idx   = 0;
            this->bFull = true;
        } else {
            this->idx++;
        }
    }

    void sample_onedim(Real beta,
                       Veci& indices,
                       Vecf& batch_state,
                       Vecf& batch_action,
                       Vecf& batch_reward,
                       Vecf& batch_next_state,
                       std::vector<bool>& batch_done,
                       Vecf& weights)
    {
        size_t batch_size = batch_reward.size();
        size_t state_dim  = batch_state.size() / batch_size;
        size_t action_dim = batch_action.size() / batch_size;
        auto max_value    = this->sum_tree.root();

        for (size_t i = 0; i < batch_size; i++) {
            indices[i] = this->sum_tree.sample(randd(0, max_value));
            auto& tmp  = this->memory[indices[i]];
            std::copy_n(tmp.state.begin(), state_dim, batch_state.begin() + i * state_dim);

#if RLCPP_ACTION_TYPE == 0
            batch_action[i] = tmp.action;
#elif RLCPP_ACTION_TYPE == 1
            std::copy_n(tmp.action.begin(), action_dim, batch_action.begin() + i * action_dim);
#endif

            batch_reward[i] = tmp.reward;
            std::copy_n(tmp.next_state.begin(), state_dim, batch_next_state.begin() + i * state_dim);
            batch_done[i] = tmp.done;
        }
        auto min_value = this->min_tree.root();
        for (int i = 0; i < weights.size(); i++) {
            weights[i] = std::pow(this->sum_tree.getItem(indices[i]) / (min_value + 1e-4), -beta);
        }
    }

    void update(const Veci& indices, const Vecf& values)
    {
        for (Int i = 0; i < indices.size(); i++) {
            auto values_modified = std::pow(values[i], this->alpha);
            this->sum_tree.setItem(indices[i], values_modified);
            this->min_tree.setItem(indices[i], values_modified);
        }
        this->max_value = std::max(this->max_value, *std::max_element(values.begin(), values.end()));
        this->max_value = std::min(this->max_value, this->max_value_upper);
    }

    bool is_full() const
    {
        return this->bFull;
    }

private:
    struct Func_min
    {
        Real operator()(Real __x, Real __y) const
        {
            return std::min(__x, __y);
        }
    };

private:
    size_t max_size;
    std::vector<Transition> memory;
    SumTree sum_tree;
    ReduceTree<Func_min> min_tree;
    Real alpha;
    size_t idx;
    Real max_value;
    Real max_value_upper;
    bool bFull;
};


std::ostream& operator<<(std::ostream& os, const RandomReply& reply)
{
    os << "# random_memory_reply_data, total_size: " << reply.size() << " saved_time: " << rlcpp::localTime() << '\n'
       << "# state_type: " << RLCPP_STATE_TYPE << " state_length: " << state_len(reply.memory.front().state) << '\n'
       << "# action_type: " << RLCPP_ACTION_TYPE << " action_length: " << action_len(reply.memory.front().action)
       << '\n'
       << "# state \t next_state \t action \t reward \t done\n";

    for (size_t i = 0; i < reply.size(); i++) {
        auto& trans = reply.memory[i];
#if RLCPP_STATE_TYPE == 0
        os << trans.state << '\t' << trans.next_state << '\t';
#elif RLCPP_STATE_TYPE == 1
        for (auto&& s : trans.state)
            os << s << ' ';
        os << '\t';
        for (auto&& ns : trans.next_state)
            os << ns << ' ';
        os << '\t';
#endif
#if RLCPP_ACTION_TYPE == 0
        os << trans.action << '\t';
#elif RLCPP_ACTION_TYPE == 1
        for (auto&& a : trans.action)
            os << a << ' ';
        os << '\t';
#endif
        os << trans.reward << '\t' << trans.done << '\n';
    }
    return os;
}

std::istream& operator>>(std::istream& is, RandomReply& reply)
{
    std::string tmp, line;
    std::stringstream ss;
    size_t total_size;
    int state_type;
    int action_type;
    int state_length;
    int action_length;

    // #1
    std::getline(is, line);
    ss.str(line);
    ss >> tmp >> tmp >> tmp >> total_size;
    ss.clear();
    if (total_size >= reply.memory.size()) {
        reply.memory.resize(total_size);
        reply.bFull = true;
        reply.idx   = 0;
    } else {
        reply.bFull = false;
        reply.idx   = total_size;
    }

    // #2
    std::getline(is, line);
    ss.str(line);
    ss >> tmp >> tmp >> state_type >> tmp >> state_length;
    ss.clear();
    assert(state_type == RLCPP_STATE_TYPE);

    // #3
    std::getline(is, line);
    ss.str(line);
    ss >> tmp >> tmp >> action_type >> tmp >> action_length;
    ss.clear();
    assert(action_type == RLCPP_ACTION_TYPE);

    // #4
    std::getline(is, line);

    // # loop for next line
    for (size_t i = 0; i < total_size; i++) {
        std::getline(is, line);
        ss.str(line);
        auto& trans = reply.memory[i];

#if RLCPP_STATE_TYPE == 0
        ss >> trans.state >> trans.next_state;
#elif RLCPP_STATE_TYPE == 1
        trans.state.resize(state_length);
        trans.next_state.resize(state_length);
        for (int m = 0; m < state_length; m++) {
            ss >> trans.state[m];
        }
        for (int m = 0; m < state_length; m++) {
            ss >> trans.next_state[m];
        }
#endif

#if RLCPP_ACTION_TYPE == 0
        ss >> trans.action;
#elif RLCPP_ACTION_TYPE == 1
        trans.action.resize(action_length);
        for (int m = 0; m < action_length; m++) {
            ss >> trans.action[m];
        }
#endif

        ss >> trans.reward >> trans.done;
        ss.clear();
    }
    return is;
}


}  // namespace rlcpp

#endif  // !__RL_RANDOM_REPLY_H__