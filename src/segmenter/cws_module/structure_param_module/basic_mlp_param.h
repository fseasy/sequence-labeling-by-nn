#ifndef SLNN_SEGMENTER_CWS_MODULE_STRUCTURE_PARAM_MODULE_BASIC_MLP_PARAM_H_
#define SLNN_SEGMENTER_CWS_MODULE_STRUCTURE_PARAM_MODULE_BASIC_MLP_PARAM_H_
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

struct SegmentorBasicMlpParam
{
    friend class boost::serialization::access;
    // Data
    //   - Input
    unsigned corpus_token_embedding_dim;
    unsigned corpus_token_dict_size;
    unsigned window_size;
    //   - Mlp
    std::string window_process_method;
    std::vector<unsigned> mlp_hidden_dim_list;
    slnn::type::real mlp_dropout_rate;
    std::string mlp_nonlinear_function_str;
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
void SegmentorBasicMlpParam::set_param_from_token_module(const TokenModuleT &token_module)
{
    // Input - dict size
    corpus_token_dict_size = token_module.get_charset_size();
    // Output
    output_dim = token_module.get_tagset_size();
}


template <class Archive>
void SegmentorBasicMlpParam::serialize(Archive &ar, const unsigned int)
{
    ar &corpus_token_embedding_dim &corpus_token_dict_size &window_size
        &window_process_method &mlp_hidden_dim_list &mlp_dropout_rate &mlp_nonlinear_function_str
        &output_layer_type  &output_dim
        &replace_freq_threshold &replace_prob_threshold;
}

} // end of namespace structure_param_module
} // end of namespace segmenter
} // end of namespace slnn

#endif