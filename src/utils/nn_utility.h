#ifndef SLNN_UTILS_NN_UTILITY_H_
#define SLNN_UTILS_NN_UTILITY_H_
#include <vector>
#include <string>
#include "dynet/expr.h"
namespace slnn{
namespace utils{

auto get_nonlinear_function_from_name(const std::string &name) -> dynet::expr::Expression(*)(const dynet::expr::Expression &);

std::vector<unsigned> parse_mlp_hidden_dim_list(const std::string& hidden_dim_list_str);

} // end of namespace utils
} // end of namespace slnn


#endif