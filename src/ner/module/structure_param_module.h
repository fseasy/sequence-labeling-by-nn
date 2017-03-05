#ifndef SLNN_NER_MLP_STRUCTURE_PARAM_INCLUDE_
#define SLNN_NER_MLP_STRUCTURE_PARAM_INCLUDE_

#include <vector>
#include <string>
#include <boost/program_options/variables_map.hpp>

#include "token_module.h"
#include "utils/typedeclaration.h"

namespace slnn{
namespace ner{
namespace structure_param{
struct StructureParam
{
    friend class boost::serialization::access;
    // 1. input
    unsigned word_dict_sz;
    unsigned pos_tag_dict_sz;
    unsigned word_embed_dim;
    unsigned pos_tag_embed_dim;
    unsigned window_sz;

    // 2. Mlp
    std::string window_process_method;
    std::vector<unsigned> mlp_hidden_dim_list;
    slnn::type::real mlp_dropout_rate;
    std::string mlp_nonlinear_func_str;

    // 3. output
    unsigned ner_tag_dict_sz; // (init from token module, target is nn module) 
    unsigned ner_tag_embed_dim; // (init from command line, target is nn module) ..|
    std::string output_layer_type;
    // 4. others
    unsigned replace_freq_threshold; // (init from command line, target is the token module)
    slnn::type::real replace_prob_threshold;

    std::string get_structure_info() const noexcept;
    // serialization
    template <class Archive>
    void serialize(Archive &ar, const unsigned int);
};

void set_param_from_token_dict(StructureParam&, const token_module::TokenDict&);
void set_param_from_cmdline(StructureParam&, const boost::program_options::variables_map& varmap);


/***********************
 * inline/template implementation
 ***********************/
template<class Archive>
void StructureParam::serialize(Archive& ar, const unsigned int)
{
    ar &word_dict_sz &word_embed_dim &pos_tag_dict_sz &pos_tag_embed_dim &window_sz
        &window_process_method &mlp_hidden_dim_list &mlp_dropout_rate &mlp_nonlinear_func_str
        &ner_tag_dict_sz &ner_tag_embed_dim &output_layer_type
        &replace_freq_threshold &replace_prob_threshold;
}


} // end of namespace structure_param
} // end of namespace ner
} // end of namespace slnn


#endif