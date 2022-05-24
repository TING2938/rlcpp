#pragma once

#include <cstdio>
#include "common/rl_config.h"

namespace rlcpp
{
template <typename Operator>
class ReduceTree
{
public:
    void init(size_t size)
    {
        if (size & (size - 1)) {
            printf("size must be a power of 2.\n");
            std::exit(-1);
        }
        this->size = size;
        this->values.resize(2 * this->size, 0);
    }

    Real root()
    {
        return this->values[1];
    }

    void setItem(size_t idx, Real val)
    {
        idx += this->size;
        this->values[idx] = val;
        this->percolate_up(idx);
    }

    template <typename T>
    void setItem(const std::vector<T>& idx, const Vecf& val)
    {
        for (size_t i = 0; i < idx.size(); i++) {
            this->setItem(idx[i], val[i]);
        }
    }

    Real getItem(size_t idx)
    {
        return this->values[idx + this->size];
    }

    template <typename T>
    Vecf getItem(const std::vector<T>& idx)
    {
        Vecf ret(idx.size());
        for (size_t i = 0; i < idx.size(); i++) {
            ret[i] = this->getItem(idx[i]);
        }
        return ret;
    }

private:
    void percolate_up(size_t idx)
    {
        idx /= 2;
        while (idx > 0) {
            this->values[idx] = this->op(this->values[2 * idx], this->values[2 * idx + 1]);
            idx /= 2;
        }
    }

protected:
    size_t size;
    Vecf values;
    Operator op;
};


class SumTree : public ReduceTree<std::plus<Real>>
{
public:
    size_t sample(Real value)
    {
        size_t idx = 1;
        size_t child;
        while (idx < this->size) {
            child = 2 * idx;
            if (value <= this->values[child]) {
                idx = child;
            } else {
                value -= this->values[child];
                idx = child + 1;
            }
        }
        return idx - this->size;
    }
};
}  // namespace rlcpp
