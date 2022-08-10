#pragma once

#include <random>
#include "common/state_action.h"
#include "tools/random_tools.h"
#include "tools/reduce_tree.h"
#include "tools/ring_vector.h"
#include "tools/utility.hpp"
#include "tools/vector_tools.h"

namespace rlcpp
{
template <typename State, typename Action>
struct Transition
{
    State state;
    Action action;
    Real reward;
    State next_state;
    bool done;
};

template <typename State, typename Action>
class RandomReply : public RingVector<Transition<State, Action>>
{
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
            this->copy_value(tmp.state, batch_state.begin() + i * state_dim);
            this->copy_value(tmp.action, batch_action.begin() + i * action_dim);
            this->copy_value(tmp.next_state, batch_next_state.begin() + i * state_dim);
            batch_reward[i] = tmp.reward;
            batch_done[i]   = tmp.done;
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

private:
    void copy_value(const Vecf& src, Vecf::iterator it) const
    {
        std::copy(src.begin(), src.end(), it);
    }

    void copy_value(Int src, Vecf::iterator it) const
    {
        *it = src;
    }

    template <typename S, typename A>
    friend std::ostream& operator<<(std::ostream& os, const RandomReply<S, A>& reply);

    template <typename S, typename A>
    friend std::istream& operator>>(std::istream& is, RandomReply<State, Action>& reply);
};

class PrioritizedReply
{
public:
    using State  = Vecf;
    using Action = Int;

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
            batch_action[i] = tmp.action;
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
    std::vector<Transition<State, Action>> memory;
    SumTree sum_tree;
    ReduceTree<Func_min> min_tree;
    Real alpha;
    size_t idx;
    Real max_value;
    Real max_value_upper;
    bool bFull;
};

namespace detail
{
std::ostream& print_var(std::ostream& os, const Vecf& var)
{
    for (auto&& s : var)
        os << s << ' ';
    return os;
}

std::ostream& print_var(std::ostream& os, Int var)
{
    os << var << ' ';
    return os;
}

void parse_var(std::istream& is, Vecf& var, Int length)
{
    var.resize(length);
    for (int m = 0; m < length; m++) {
        is >> var[m];
    }
}

void parse_var(std::istream& is, Int& var, Int length)
{
    is >> var;
}

}  // namespace detail

template <typename S, typename A>
std::ostream& operator<<(std::ostream& os, const RandomReply<S, A>& reply)
{
    os << "# random_memory_reply_data, total_size: " << reply.size() << " saved_time: " << rlcpp::localTime() << '\n'
       << "# state_type: " << !is_scalar_type<S>() << " state_length: " << type_size(reply.memory.front().state) << '\n'
       << "# action_type: " << !is_scalar_type<A>() << " action_length: " << type_size(reply.memory.front().action)
       << '\n'
       << "# state \t next_state \t action \t reward \t done\n";

    for (size_t i = 0; i < reply.size(); i++) {
        auto& trans = reply.memory[i];
        detail::print_var(os, trans.state) << '\t';
        detail::print_var(os, trans.next_state) << '\t';
        detail::print_var(os, trans.action) << '\t';
        os << trans.reward << '\t';
        os << trans.done << '\n';
    }
    return os;
}

template <typename S, typename A>
std::istream& operator>>(std::istream& is, RandomReply<S, A>& reply)
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

    // #3
    std::getline(is, line);
    ss.str(line);
    ss >> tmp >> tmp >> action_type >> tmp >> action_length;
    ss.clear();

    // #4
    std::getline(is, line);

    // # loop for next line
    for (size_t i = 0; i < total_size; i++) {
        std::getline(is, line);
        ss.str(line);
        auto& trans = reply.memory[i];
        detail::parse_var(ss, trans.state, state_length);
        detail::parse_var(ss, trans.next_state, state_length);
        detail::parse_var(ss, trans.action, action_length);
        ss >> trans.reward;
        ss >> trans.done;
        ss.clear();
    }
    return is;
}


}  // namespace rlcpp
