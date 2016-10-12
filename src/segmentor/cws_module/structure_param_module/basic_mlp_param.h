#ifndef SLNN_SEGMENTOR_CWS_MODULE_STRUCTURE_PARAM_MODULE_BASIC_MLP_PARAM_H_
#define SLNN_SEGMENTOR_CWS_MODULE_STRUCTURE_PARAM_MODULE_BASIC_MLP_PARAM_H_
#include <vector>
#include <string>
#include <boost/serialization/access.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/program_options/variables_map.hpp>
#include "utils/typedeclaration.h"
namespace slnn{
namespace segmentor{
namespace structure_param_module{


/**
 * Basic segmentor MLP structure param module.
 * for segmentor with sigle word token and no feature.
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
    unsigned mlp_input_dim;
    std::vector<unsigned> mlp_hidden_dim_list;
    slnn::type::real mlp_dropout_rate;
    std::string mlp_nonlinear_function_str;
    //   - Ouptut
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

template <class Archive>
void SegmentorBasicMlpParam::serialize(Archive &ar, const unsigned int)
{
    ar &corpus_token_embedding_dim &corpus_token_dict_size &window_size
        &mlp_input_dim &mlp_hidden_dim_list &mlp_dropout_rate &mlp_nonlinear_function_str
        &output_dim
        &replace_freq_threshold &replace_prob_threshold;
}

} // end of namespace structure_param_module
} // end of namespace segmentor
} // end of namespace slnn

#endif