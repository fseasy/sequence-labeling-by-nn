#ifndef SLNN_SEGMENTER_CWS_MODULE_STRUCTURE_PARAM_MODULE_PARAM_MLP_INPUT1_ALL_H_
#define SLNN_SEGMENTER_CWS_MODULE_STRUCTURE_PARAM_MODULE_PARAM_MLP_INPUT1_ALL_H_
#include <string>
#include <vector>
#include <boost/serialization/access.hpp>
#include <boost/program_options/variables_map.hpp>
#include "utils/typedeclaration.h"
namespace slnn{
namespace segmenter{
namespace structure_param_module{

struct ParamSegmenterMlpInput1All
{
    friend class boost::serialization::access;
    // Data

    // 1. input
    bool enable_unigram; // (init from command line, target is the token module)
    bool enable_bigram;
    bool enable_lexicon;
    bool enable_type;
    
    unsigned lexicon_feature_max_len; // (init from command line, target is token module)
    unsigned unigram_dict_sz; // (init from token module. target is the nn module) ...|
    unsigned bigram_dict_sz;
    unsigned lexicon_dict_sz;
    unsigned type_dict_sz; 

    unsigned unigram_embedding_dim; // (init from command line, target is nn module) ...|
    unsigned bigram_embedding_dim;
    unsigned lexicon_embedding_dim;
    unsigned type_embedding_dim;
    

    unsigned window_sz; // (init from command line, target is nn module) ...|

    // 2. Mlp
    std::string window_process_method; // (init from command line, target is the nn module) ...|
    std::vector<unsigned> mlp_hidden_dim_list;
    slnn::type::real mlp_dropout_rate;
    std::string mlp_nonlinear_func_str;
    // 3. output
    unsigned tag_dict_sz; // (init from token module, target is nn module) 
    unsigned tag_embedding_dim; // (init from command line, target is nn module) ..|
    std::string output_layer_type;
    // 4. others
    unsigned replace_freq_threshold; // (init from command line, target is the token module)
    slnn::type::real replace_prob_threshold;

    // Interface
    void set_param_from_user_defined(const boost::program_options::variables_map &args);
    template<typename TokenModuleT>
    void set_param_from_token_module(const TokenModuleT&);
    std::string get_stucture_info();

    // serialization
    template <class Archive>
    void serialize(Archive &ar, const unsigned int);
};


/**************************
 * Inline / template implementation
 **************************/

template<typename TokenModuleT>
void ParamSegmenterMlpInput1All::set_param_from_token_module(const TokenModuleT& token_module)
{
    auto &state = token_module.get_token_state();
    unigram_dict_sz = state.unigram_dict_sz;
    bigram_dict_sz = state.bigram_dict_sz;
    lexicon_dict_sz = state.lexicon_dict_sz;
    type_dict_sz = state.type_dict_sz;
    tag_dict_sz = state.tag_dict_sz;
}

template <class Archive>
void ParamSegmenterMlpInput1All::serialize(Archive &ar, const unsigned int)
{
    ar &enable_unigram &enable_bigram &enable_lexicon &enable_type
        &lexicon_feature_max_len &unigram_dict_sz &bigram_dict_sz &lexicon_dict_sz &type_dict_sz
        &unigram_embedding_dim &bigram_embedding_dim &lexicon_embedding_dim &type_embedding_dim
        &window_sz
        &window_processing_method &mlp_hidden_dim_list &mlp_dropout_rate &mlp_nonlinear_func_str
        &tag_dict_sz &tag_embedding_dim &output_layer_type
        &replace_freq_threshold &replace_prob_threshold;
}


} // end of namespace structure param module
} // end of namespace segmenter
} // end of namespace slnn




#endif