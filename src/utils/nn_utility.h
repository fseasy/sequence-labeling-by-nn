#ifndef SLNN_UTILS_NN_UTILITY_H_
#define SLNN_UTILS_NN_UTILITY_H_
#include "dynet/expr.h"
namespace slnn{
namespace utils{

auto get_nonlinear_function_from_name(const std::string &name) -> dynet::expr::Expression(*)(const dynet::expr::Expression &);

} // end of namespace utils
} // end of namespace slnn


#endif