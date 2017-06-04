#ifndef SLNN_SEGMENTER_CWS_MODULE_STRUCTURE_PARAM_MODULE_PARAM_RNN_ALL_H_
#define SLNN_SEGMENTER_CWS_MODULE_STRUCTURE_PARAM_MODULE_PARAM_RNN_ALL_H_
#include <vector>
#include <string>
#include <boost/serialization/access.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/program_options/variables_map.hpp>
#include "utils/typedeclaration.h"
namespace slnn{
namespace segmenter{
namespace structure_param_module{


/**
 * Basic segmenter MLP structure param module.
 * for segmenter with sigle word token and no feature.
 * using this to decouple the (frontend-param, token-module) and the nn module
 */

struct ParamSegmenterRnnAll
{
    friend class boost::serialization::access;
    // Data
    
    //   - input
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

    //   - Rnn
    unsigned rnn_nr_stack_layer;
    unsigned rnn_h_dim;
    slnn::type::real rnn_dropout_rate;
    //   - Ouptut
    unsigned tag_dict_sz; // (init from token module, target is nn module) 
    unsigned tag_embedding_dim; // (init from command line, target is nn module) ..|
    std::string output_layer_type;
    //   - Others
    unsigned replace_freq_threshold;
    float replace_prob_threshold;
    // Interface
    void set_param_from_user_defined(const boost::program_options::variables_map &args);
    template<typename TokenModuleT>
    void set_param_from_token_module(const TokenModuleT &token_module);
    std::string get_structure_info();
    // Serialization
    template<class Archive>
    void serialize(Archive& ar, const unsigned int);
};

/**********************************************
 *  Inline Implementation
 **********************************************/

template<typename TokenModuleT>
void ParamSegmenterRnnAll::set_param_from_token_module(const TokenModuleT &token_module)
{
    auto &state = token_module.get_token_state();
    unigram_dict_sz = state.unigram_dict_sz;
    bigram_dict_sz = state.bigram_dict_sz;
    lexicon_dict_sz = state.lexicon_dict_sz;
    type_dict_sz = state.type_dict_sz;
    tag_dict_sz = state.tag_dict_sz;
}


template <class Archive>
void ParamSegmenterRnnAll::serialize(Archive &ar, const unsigned int)
{
    ar &enable_unigram &enable_bigram &enable_lexicon &enable_type
        &lexicon_feature_max_len &unigram_dict_sz &bigram_dict_sz &lexicon_dict_sz &type_dict_sz
        &unigram_embedding_dim &bigram_embedding_dim &lexicon_embedding_dim &type_embedding_dim
        &rnn_nr_stack_layer &rnn_h_dim &rnn_dropout_rate
        &tag_dict_sz &tag_embedding_dim &output_layer_type
        &replace_freq_threshold &replace_prob_threshold;
}

} // end of namespace structure_param_module
} // end of namespace segmenter
} // end of namespace slnn

#endif