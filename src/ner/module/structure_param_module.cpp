#include "structure_param_module.h"
#include "utils/nn_utility.h"

namespace slnn{
namespace ner{
namespace structure_param{

std::string StructureParam::get_structure_info() const noexcept
{
    std::ostringstream oss;
    oss << "+ Model Info: \n"
        << "|        word dict size(" << word_dict_sz << ") embedding dim(" << word_embed_dim << ")"
        << " postag dict size(" << pos_tag_dict_sz << ") embedding dim(" << pos_tag_embed_dim << ")\n"
        << "|        window size(" << window_sz << ")\n"
        << "| mlp: " <<"window process method(" << window_process_method << ") mlp hidden dim list(";
    if( !mlp_hidden_dim_list.empty() )
    {
        oss << mlp_hidden_dim_list[0];
        for( unsigned i = 1; i < mlp_hidden_dim_list.size(); ++i ){ oss << ", " << mlp_hidden_dim_list[i]; }
    }
    else{ oss << "None"; }
    oss << ") mlp nonlinear function(" << mlp_nonlinear_func_str << ") dropout rate(" << mlp_dropout_rate << ")\n"
        << "| output: output type(" << output_layer_type << ") ner tag dict size(" 
        << ner_tag_dict_sz << ") embedding_dim(" << ner_tag_embed_dim << ")\n"
        << "| others: " << "replace-frequent-threshold(" << replace_freq_threshold << ") "
        << "replace-probability-threshold(" << replace_prob_threshold << ")\n"
        << "= - - - - -";
    return oss.str();

}

void set_param_from_token_dict(StructureParam& param, const token_module::TokenDict& dict)
{
    param.word_dict_sz = dict.word_num_with_unk();
    param.pos_tag_dict_sz = dict.pos_tag_num();
    param.ner_tag_dict_sz = dict.ner_tag_num();
}

void set_param_from_cmdline(StructureParam& param, const boost::program_options::variables_map &args)
{
    // Input
    param.word_embed_dim = args["word_embed_dim"].as<unsigned>();
    param.pos_tag_embed_dim = args["pos_tag_embed_dim"].as<unsigned>();

    param.window_sz = args["window_size"].as<unsigned>();

    // Mlp
    param.window_process_method = args["window_process_method"].as<std::string>();
    param.mlp_hidden_dim_list = slnn::utils::parse_mlp_hidden_dim_list(args["mlp_hidden_dim_list"].as<std::string>());
    param.mlp_dropout_rate = args["dropout_rate"].as<slnn::type::real>();
    param.mlp_nonlinear_func_str = args["nonlinear_func"].as<std::string>();

    // Output
    param.ner_tag_embed_dim = args["ner_tag_embed_dim"].as<unsigned>();
    param.output_layer_type = args["output_layer_type"].as<std::string>();

    // Others
    param.replace_freq_threshold = args["replace_freq_threshold"].as<unsigned>();
    param.replace_prob_threshold = args["replace_prob_threshold"].as<float>();
}


} // end of namespace structure param
} // end of namespace ner
} // end of namespace slnn