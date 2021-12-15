#ifndef __RL_RANDOM_REPLY_H__
#define __RL_RANDOM_REPLY_H__

#include <random>
#include "common/state_action.h"
#include "tools/rand.h"

namespace rlcpp
{
    class RandomReply
    {
    public:
        struct Transition
        {
            State state;
            Action action;
            Float reward;
            State next_state;
            bool done;
        };

        void init(size_t max_size)
        {
            this->idx = 0;
            this->memory.resize(max_size);
            this->bFull = false;
        }

        void store(const State &state, const Action &action, Float reward, const State &next_state, bool done)
        {
            this->memory[this->idx] = {state, action, reward, next_state, done};
            if (this->memory.empty()) {
                printf("empty memory relpy!");
                std::exit(-1);
            }
            if (this->idx == this->memory.size() - 1) {
                this->idx = 0;
                this->bFull = true;
            } else {
                this->idx++;
            }
        }

        size_t size() const 
        {
            if (this->bFull) {
                return this->memory.size();
            } else {
                return this->idx;
            }
        }

        void sample(std::vector<State> &batch_state,
                    std::vector<Action> &batch_action,
                    Vecf &batch_reward,
                    std::vector<State> &batch_next_state,
                    std::vector<bool> &batch_done)
        {
            size_t batch_size = batch_state.size();
            size_t len = this->size();
            std::vector<size_t> index(batch_size);
            for (size_t i = 0; i < batch_size; i++)
            {
                index[i] = i % len;
            }
            std::shuffle(index.begin(), index.end(), engine);
            for (size_t i = 0; i < batch_size; i++)
            {
                auto &tmp = memory[index[i]];
                batch_state[i] = tmp.state;
                batch_action[i] = tmp.action;
                batch_reward[i] = tmp.reward;
                batch_next_state[i] = tmp.next_state;
                batch_done[i] = tmp.done;
            }
        }

        void sample_onedim(Vecf &batch_state,
                    Vecf &batch_action,
                    Vecf &batch_reward,
                    Vecf &batch_next_state,
                    std::vector<bool> &batch_done)
        {
            size_t batch_size = batch_reward.size();
            size_t state_dim = batch_state.size() / batch_size;
            size_t action_dim = batch_action.size() / batch_size;

            size_t len = this->size();
            std::vector<size_t> index(batch_size);
            for (size_t i = 0; i < batch_size; i++)
            {
                index[i] = i % len;
            }
            std::shuffle(index.begin(), index.end(), engine);
            for (size_t i = 0; i < batch_size; i++)
            {
                auto &tmp = memory[index[i]];
                std::copy_n(tmp.state.begin(), state_dim, batch_state.begin() + i * state_dim);
                std::copy_n(tmp.action.begin(), action_dim, batch_action.begin() + i * action_dim); 
                batch_reward[i] = tmp.reward;
                std::copy_n(tmp.next_state.begin(), state_dim, batch_next_state.begin() + i * state_dim);
                batch_done[i] = tmp.done;
            }
        }

    private:
        bool bFull;
        size_t idx = 0;
        std::vector<Transition> memory;
    };
}

#endif // !__RL_RANDOM_REPLY_H__