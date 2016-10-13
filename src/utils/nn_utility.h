#ifndef SLNN_UTILS_NN_UTILITY_H_
#define SLNN_UTILS_NN_UTILITY_H_
#include "cnn/expr.h"
namespace slnn{
namespace utils{

auto get_nonlinear_function_from_name(const std::string &name) -> cnn::expr::Expression(*)(const cnn::expr::Expression &);

} // end of namespace utils
} // end of namespace slnn


#endif