#include "nn_utility.h"
namespace slnn{
namespace utils{

auto get_nonlinear_function_from_name(const std::string &name) -> cnn::expr::Expression(*)(const cnn::expr::Expression &)
{
    std::string lower_name(name);
    for( char &c : lower_name ){ c = ::tolower(c); }
    if( lower_name == "relu" || lower_name == "rectify" ){ return &cnn::expr::rectify; }
    else if( lower_name == "sigmoid" || lower_name == "softmax" ){ return &cnn::expr::softmax; } // a bit strange...
    else if( lower_name == "tanh" ){ return &cnn::expr::tanh; }
    else
    {
        std::ostringstream oss;
        oss << "not supported non-linear funtion: " << name << "\n"  
            <<"Exit!\n";
        throw std::invalid_argument(oss.str());
    }
}

} // end of namespace utils
} // end of namespace slnn
