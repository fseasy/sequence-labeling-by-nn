#ifndef SLNN_SEGMENTER_CWS_MODULE_STRUCTURE_PARAM_MODULE_RNN_INPUT1_PARAM_H_
#define SLNN_SEGMENTER_CWS_MODULE_STRUCTURE_PARAM_MODULE_RNN_INPUT1_PARAM_H_
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

struct SegmenterRnnInput1Param
{
    friend class boost::serialization::access;
    // Data
    //   - Input
    unsigned corpus_token_embedding_dim;
    unsigned corpus_token_dict_size;
    //   - Rnn
    unsigned rnn_nr_stack_layer;
    unsigned rnn_h_dim;
    slnn::type::real rnn_dropout_rate;
    //   - Ouptut
    std::string output_layer_type;
    unsigned output_dim;
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
void SegmenterRnnInput1Param::set_param_from_token_module(const TokenModuleT &token_module)
{
    // Input - dict size
    corpus_token_dict_size = token_module.get_charset_size();
    // Output
    output_dim = token_module.get_tagset_size();
}


template <class Archive>
void SegmenterRnnInput1Param::serialize(Archive &ar, const unsigned int)
{
    ar &corpus_token_embedding_dim &corpus_token_dict_size
        &rnn_nr_stack_layer &rnn_h_dim &rnn_dropout_rate
        &output_layer_type  &output_dim
        &replace_freq_threshold &replace_prob_threshold;
}

} // end of namespace structure_param_module
} // end of namespace segmenter
} // end of namespace slnn

#endif