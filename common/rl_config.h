#ifndef __RL_CONDIG_H__
#define __RL_CONDIG_H__

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
// #include <tiny_dnn/tiny_dnn.h>

namespace rlcpp
{
    using Int = int32_t;
    using Float = float;

    using std::string;
    using Veci = std::vector<Int>;
    using Vecd = std::vector<double>;
    // using Vecf = tiny_dnn::vec_t;
    using Vecf = std::vector<Float>;
}

#endif // !__RL_CONDIG_H__
