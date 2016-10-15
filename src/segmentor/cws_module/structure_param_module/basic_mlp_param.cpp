#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include "basic_mlp_param.h"
namespace slnn{
namespace segmentor{
namespace structure_param_module{

void SegmentorBasicMlpParam::set_param_from_user_defined(const boost::program_options::variables_map &args)
{
    // Input
    corpus_token_embedding_dim = args["word_embedding_dim"].as<unsigned>();
    window_size = args["window_size"].as<unsigned>();
    // Mlp
    mlp_input_dim = corpus_token_embedding_dim * window_size;
    //  - parse mlp hidden layer dim list
    std::vector<std::string> dim_str_cont;
    std::string hidden_dim_list_str = args["mlp_hidden_dim_list"].as<std::string>();
    boost::trim_if(hidden_dim_list_str, boost::is_any_of("\", ")); 
    boost::split(dim_str_cont, hidden_dim_list_str, boost::is_any_of(", ")); // split by space of comma
    mlp_hidden_dim_list.resize(dim_str_cont.size());
    try
    {
        std::transform(dim_str_cont.begin(), dim_str_cont.end(), mlp_hidden_dim_list.begin(),
            [](const std::string &num_str){ return std::stoul(num_str); });
    }
    catch( std::invalid_argument &e )
    {
        std::cerr << "bad argument for 'mlp_hidden_dim_list' : " << hidden_dim_list_str << "\n";
        throw e ;
    }
    mlp_dropout_rate = args["dropout_rate"].as<slnn::type::real>() ;
    mlp_nonlinear_function_str = args["nonlinear_func"].as<std::string>();
    // Others
    replace_freq_threshold = args["replace_freq_threshold"].as<unsigned>();
    replace_prob_threshold = args["replace_prob_threshold"].as<float>();
}

std::string SegmentorBasicMlpParam::get_structure_info()
{
    std::ostringstream oss;
    oss << "+ Model info: \n"
        << "| input: " << "charset-size(" << corpus_token_dict_size << ") embedding-dim(" << corpus_token_embedding_dim
        << ") window-size(" << window_size << ")\n"
        << "| mlp: " << "input-dim(" << mlp_input_dim << ") hidden-dim-list(";
    if( !mlp_hidden_dim_list.empty() )
    {
        oss << mlp_hidden_dim_list[0];
        for( unsigned i = 1; i < mlp_hidden_dim_list.size(); ++i ){ oss << ", " << mlp_hidden_dim_list[i]; }
    }
    oss << ") mlp-nonlinear-func(" << mlp_nonlinear_function_str << ") mlp-dropout-rate("
        << mlp_dropout_rate << ")" << "\n"
        << "| output: " << "output-dim(" << output_dim << ")" << "\n"
        << "| others: " << "replace-frequent-threshold(" << replace_freq_threshold << ") "
        << "replace-probability-threshold(" << replace_prob_threshold << ")\n"
        << "= - - - - -";
    return oss.str();
}


} // end of namespace structure_param_module
} // end of namespace segmentor
} // end of namespace slnn
