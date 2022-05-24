#pragma once

#include <string>
#include <vector>

namespace rlcpp
{
using Int  = int;
using Real = float;

using std::string;
using Veci = std::vector<Int>;
using Vecf = std::vector<Real>;
}  // namespace rlcpp

// print dynet Expression info, for debug
#define DBG_VECTOR(expr) #expr << expr.dim() << ": " << dynet::as_vector(expr.value()) << "\n"
#define DBG_SCALAR(expr) #expr ": [" << expr.dim() << "] " << dynet::as_scalar(expr.value()) << "\n"
