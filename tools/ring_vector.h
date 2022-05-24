#pragma once

#include "common/rl_config.h"
#include "tools/vector_tools.h"

namespace rlcpp
{
template <typename T>
class RingVector
{
public:
    void init(size_t max_size)
    {
        this->idx = 0;
        this->memory.resize(max_size);
        this->bFull = false;
    }

    void store(const T& value)
    {
        this->memory[this->idx] = value;
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

    size_t size() const
    {
        if (this->bFull) {
            return this->memory.size();
        } else {
            return this->idx;
        }
    }

    bool is_full() const
    {
        return this->bFull;
    }

    Real mean() const
    {
        if (this->bFull) {
            return rlcpp::mean(memory);
        } else {
            if (this->idx == 0)
                return 0.0f;
            else
                return std::accumulate(memory.begin(), memory.begin() + this->idx, 0.0f) / this->idx;
        }
    }

    T sum() const
    {
        if (this->bFull) {
            return rlcpp::sum(memory);
        } else {
            return std::accumulate(memory.begin(), memory.begin() + this->idx, 0.0f);
        }
    }

    std::vector<T> lined_vector() const
    {
        std::vector<T> ret(this->size());
        if (!this->bFull) {
            std::copy(this->memory.begin(), this->memory.begin() + this->idx, ret.begin());
        } else {
            std::copy(this->memory.begin() + this->idx, this->memory.end(), ret.begin());
            std::copy(this->memory.begin(), this->memory.begin() + this->idx,
                      ret.begin() + this->memory.size() - this->idx);
        }
        return ret;
    }

    std::vector<T>& data()
    {
        return this->memory;
    }

protected:
    bool bFull;
    size_t idx = 0;
    std::vector<T> memory;
};
}  // namespace rlcpp
