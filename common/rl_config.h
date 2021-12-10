#ifndef __RL_CONDIG_H__
#define __RL_CONDIG_H__

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

namespace rlcpp
{
    using Int = int32_t;

    using std::string;
    using Veci = std::vector<Int>;
    using Vecd = std::vector<double>;

    using DiscreteType = Int;
    using BoxType = Vecd;
}

#endif // !__RL_CONDIG_H__