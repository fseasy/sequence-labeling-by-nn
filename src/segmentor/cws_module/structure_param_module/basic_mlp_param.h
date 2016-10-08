#ifndef SLNN_SEGMENTOR_CWS_MODULE_STRUCTURE_PARAM_MODULE_BASIC_MLP_PARAM_H_
#define SLNN_SEGMENTOR_CWS_MODULE_STRUCTURE_PARAM_MODULE_BASIC_MLP_PARAM_H_
#include <vector>
#include <string>
#include <boost/serialization/access.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
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
    // Input
    unsigned corpus_token_embedding_dim;
    unsigned corpus_token_dict_size;

    // Mlp
    unsigned mlp_input_dim;
    std::vector<unsigned> mlp_hidden_dim_list;
    slnn::type::real mlp_dropout_rate;
    std::string mlp_nonlinear_function_str;
    // Ouptut
    unsigned output_dim;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int);
};

/**********************************************
 *  Inline Implementation
 **********************************************/

template <class Archive>
void BasicMlpParam::serialize(Archive &ar, const unsigned int)
{
    ar &corpus_token_embedding_dim &corpus_token_dict_size
        &mlp_input_dim &mlp_hidden_dim_list &mlp_dropout_rate &mlp_nonlinear_function_str
        &output_dim;
}

} // end of namespace structure_param_module
} // end of namespace segmentor
} // end of namespace slnn

#endif