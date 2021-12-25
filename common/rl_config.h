#ifndef __RL_CONDIG_H__
#define __RL_CONDIG_H__

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#ifdef NN_TINYDNN
  #include <tiny_dnn/tiny_dnn.h>
#endif

namespace rlcpp
{
    using Int = int32_t;
    using Float = float;

    using std::string;
    using Veci = std::vector<Int>;
    using Vecd = std::vector<double>;

    #ifdef NN_TINYDNN
        using Vecf = tiny_dnn::vec_t;
    #else
        using Vecf = std::vector<Float>;
    #endif
}

#endif // !__RL_CONDIG_H__