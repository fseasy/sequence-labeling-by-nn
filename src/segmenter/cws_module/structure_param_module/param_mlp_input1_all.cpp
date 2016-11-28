#include "param_mlp_input1_all.h"
#include "utils/nn_utility.h"
namespace slnn{
namespace segmenter{
namespace structure_param_module{

void ParamSegmenterMlpInput1All::set_param_from_user_defined(const boost::program_options::variables_map &args)
{
    // Input
    enable_unigram = args["enable_unigram"].as<bool>();
    enable_bigram = args["enable_bigram"].as<bool>();
    enable_lexicon = args["enable_lexicon"].as<bool>();
    enable_type = args["enable_type"].as<bool>();
    
    lexicon_feature_max_len = args["lexicon_feature_max_len"].as<unsigned>();
    
    unigram_embedding_dim = args["unigram_embedding_dim"].as<unsigned>();
    bigram_embedding_dim = args["bigram_embedding_dim"].as<unsigned>();
    lexicon_embedding_dim = args["lexicon_embedding_dim"].as<unsigned>();
    type_embedding_dim = args["type_embedding_dim"].as<unsigned>();

    window_sz = args["window_size"].as<unsigned>();
    
    // Mlp
    window_process_method = args["window_process_method"].as<std::string>();
    mlp_hidden_dim_list = slnn::utils::parse_mlp_hidden_dim_list(args["mlp_hidden_dim_list"].as<std::string>());
    mlp_dropout_rate = args["dropout_rate"].as<slnn::type::real>();
    mlp_nonlinear_func_str = args["nonlinear_func"].as<std::string>();

    // Output
    tag_embedding_dim = args["tag_embedding_dim"].as<unsigned>();
    output_layer_type = args["output_layer_type"].as<std::string>();

    // Others
    replace_freq_threshold = args["replace_freq_threshold"].as<unsigned>();
    replace_prob_threshold = args["replace_prob_threshold"].as<float>();
}

std::string ParamSegmenterMlpInput1All::get_stucture_info()
{
    std::ostringstream oss;
    oss << "+ Model Info: \n"
        << std::boolalpha
        << "| input: " << "enable unigram(" << enable_unigram << ") enable bigram(" << enable_bigram << ")"
        << " enable lexicon(" << enable_lexicon << ") enable type(" << enable_type << ")\n"
        << "|        unigram dict size(" << unigram_dict_sz << ") embedding dim(" << unigram_embedding_dim << ")"
        << " bigram dict size(" << bigram_dict_sz << ") embedding dim(" << bigram_embedding_dim << ")"
        << " lexicon dict size(" << lexicon_dict_sz << ") embedding dim(" << lexicon_embedding_dim << ")"
        << " type dict size(" << type_dict_sz << ") embedding_dim(" << type_embedding_dim << ")\n"
        << "|        window size(" << window_sz << ")\n"
        << "| mlp: " <<"window process method(" << window_process_method << ") mlp hidden dim list(";
    if( !mlp_hidden_dim_list.empty() )
    {
        oss << mlp_hidden_dim_list[0];
        for( unsigned i = 1; i < mlp_hidden_dim_list.size(); ++i ){ oss << ", " << mlp_hidden_dim_list[i]; }
    }
    else{ oss << "None"; }
    oss << ") mlp nonlinear function(" << mlp_nonlinear_func_str << ") dropout rate(" << mlp_dropout_rate << ")\n"
        << "| output: output type(" << output_layer_type << ") tag dict size(" << tag_dict_sz << ") embedding_dim("
        << tag_embedding_dim << ")\n"
        << "| others: " << "replace-frequent-threshold(" << replace_freq_threshold << ") "
        << "replace-probability-threshold(" << replace_prob_threshold << ")\n"
        << "= - - - - -";
    return oss.str();
}

} // end of namespace token module
} // end of namespace segmenter
} // end of namespace slnn