#include <algorithm>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include "nn_utility.h"

namespace slnn{
namespace utils{

auto get_nonlinear_function_from_name(const std::string &name) -> dynet::expr::Expression(*)(const dynet::expr::Expression &)
{
    std::string lower_name(name);
    for( char &c : lower_name ){ c = ::tolower(c); }
    if( lower_name == "relu" || lower_name == "rectify" ){ return &dynet::expr::rectify; }
    else if( lower_name == "sigmoid" || lower_name == "softmax" ){ return &dynet::expr::softmax; } // a bit strange...
    else if( lower_name == "tanh" ){ return &dynet::expr::tanh; }
    else
    {
        std::ostringstream oss;
        oss << "not supported non-linear funtion: " << name << "\n"  
            <<"Exit!\n";
        throw std::invalid_argument(oss.str());
    }
}

std::vector<unsigned> parse_mlp_hidden_dim_list(const std::string& hidden_dim_list_str)
{
    std::string dim_list_str_copy = hidden_dim_list_str;
    boost::trim_if(dim_list_str_copy, boost::is_any_of("\", ")); 
    std::vector<std::string> dim_str_container;
    boost::split(dim_str_container, dim_list_str_copy, boost::is_any_of(", ")); // split by space or comma
    std::vector<unsigned> dim_list(dim_str_container.size());
    try
    {
        std::transform(dim_str_container.begin(), dim_str_container.end(), dim_list.begin(),
            [](const std::string &num_str){ return std::stoul(num_str); });
    }
    catch( std::invalid_argument &e )
    {
        std::cerr << "bad argument for 'mlp_hidden_dim_list' : " << hidden_dim_list_str << "\n";
        throw e ;
    }
    return dim_list;
}

} // end of namespace utils
} // end of namespace slnn
