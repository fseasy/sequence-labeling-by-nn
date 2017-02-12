#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include "rnn_input1_param.h"
namespace slnn{
namespace segmenter{
namespace structure_param_module{

void SegmenterRnnInput1Param::set_param_from_user_defined(const boost::program_options::variables_map &args)
{
    // Input
    corpus_token_embedding_dim = args["word_embedding_dim"].as<unsigned>();
    // Rnn
    rnn_nr_stack_layer = args["nr_stack_layer"].as<unsigned>();
    rnn_h_dim = args["h_dim"].as<unsigned>();
    rnn_dropout_rate = args["dropout_rate"].as<slnn::type::real>() ;
    // Output
    output_layer_type = args["output_layer_type"].as<std::string>();
    // Others
    replace_freq_threshold = args["replace_freq_threshold"].as<unsigned>();
    replace_prob_threshold = args["replace_prob_threshold"].as<float>();
}

std::string SegmenterRnnInput1Param::get_structure_info()
{
    std::ostringstream oss;
    oss << "+ Model info: \n"
        << "| input: " << "charset-size(" << corpus_token_dict_size << ") embedding-dim(" << corpus_token_embedding_dim
        << ") \n"
        << "| rnn: h dim(" << rnn_h_dim << ") stacked layer num(" << rnn_nr_stack_layer << ") rnn-dropout-rate("
        << rnn_dropout_rate << ")" << "\n"
        << "| output: " << "output-dim(" << output_dim << ") type(" << output_layer_type << ")\n"
        << "| others: " << "replace-frequent-threshold(" << replace_freq_threshold << ") "
        << "replace-probability-threshold(" << replace_prob_threshold << ")\n"
        << "= - - - - -";
    return oss.str();
}


} // end of namespace structure_param_module
} // end of namespace segmenter
} // end of namespace slnn
