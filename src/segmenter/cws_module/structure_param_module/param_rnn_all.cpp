#include "param_rnn_all.h"
#include "utils/nn_utility.h"

namespace slnn{
namespace segmenter{
namespace structure_param_module{

void ParamSegmenterRnnAll::set_param_from_user_defined(const boost::program_options::variables_map &args)
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

    // Rnn
    rnn_nr_stack_layer = args["nr_stack_layer"].as<unsigned>();
    rnn_h_dim = args["h_dim"].as<unsigned>();
    rnn_dropout_rate = args["dropout_rate"].as<slnn::type::real>() ;
    // Output
    output_layer_type = args["output_layer_type"].as<std::string>();
    tag_embedding_dim = args["tag_embedding_dim"].as<unsigned>();
    // Others
    replace_freq_threshold = args["replace_freq_threshold"].as<unsigned>();
    replace_prob_threshold = args["replace_prob_threshold"].as<float>();
}

std::string ParamSegmenterRnnAll::get_structure_info()
{
    std::ostringstream oss;
    oss << "+ Model info: \n"
        << std::boolalpha
        << "| input: " << "enable unigram(" << enable_unigram << ") enable bigram(" << enable_bigram << ")"
        << " enable lexicon(" << enable_lexicon << ") enable type(" << enable_type << ")\n"
        << "|        unigram dict size(" << unigram_dict_sz << ") embedding dim(" << unigram_embedding_dim << ")"
        << " bigram dict size(" << bigram_dict_sz << ") embedding dim(" << bigram_embedding_dim << ")\n"
        << "|        lexicon dict size(" << lexicon_dict_sz << ") embedding dim(" << lexicon_embedding_dim << ")"
        << " type dict size(" << type_dict_sz << ") embedding_dim(" << type_embedding_dim << ")\n"
        << "| rnn: h dim(" << rnn_h_dim << ") stacked layer num(" << rnn_nr_stack_layer << ") rnn-dropout-rate("
        << rnn_dropout_rate << ")" << "\n"
        << "| output: output type(" << output_layer_type << ") tag dict size(" << tag_dict_sz << ") embedding_dim("
        << tag_embedding_dim << ")\n"
        << "| others: " << "replace-frequent-threshold(" << replace_freq_threshold << ") "
        << "replace-probability-threshold(" << replace_prob_threshold << ")\n"
        << "= - - - - -";
    return oss.str();
}


} // end of namespace structure_param_module
} // end of segmenter
} // end of slnn
