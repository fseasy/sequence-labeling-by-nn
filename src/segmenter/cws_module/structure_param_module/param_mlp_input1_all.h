#ifndef SLNN_SEGMENTER_CWS_MODULE_STRUCTURE_PARAM_MODULE_PARAM_MLP_INPUT1_ALL_H_
#define SLNN_SEGMENTER_CWS_MODULE_STRUCTURE_PARAM_MODULE_PARAM_MLP_INPUT1_ALL_H_
#include <string>
#include <vector>
#include <boost/serialization/access.hpp>
#include "utils/typedeclaration.h"
namespace slnn{
namespace segmenter{
namespace structure_param_module{

struct ParamSegmenterMlpInput1All
{
    friend class boost::serialization::access;
    // Data

    // 1. input
    bool enable_unigram; // (init from commindline)
    bool enable_bigram;
    bool enable_lexicon;
    bool enable_type;
    
    unsigned lexicon_feature_max_len; // (init from commindline.)
    unsigned unigram_dict_sz; // (init from token module. target is the nn module) ...|
    unsigned bigram_dict_sz;
    unsigned lexicon_dict_sz;
    unsigned type_dict_sz; 

    unsigned unigram_embedding_dim; // (init from commindline) ...|
    unsigned bigram_embedding_dim;
    unsigned lexicon_embeddign_dim;
    unsigned type_embedding_dim;
    

    unsigned window_sz; // (init from commindline) ...|

    // 2. Mlp
    std::string window_processing_method;
    std::vector<unsigned> mlp_hidden_dim_list;
    slnn::type::real mlp_dropout_rate;
    std::string mlp_nonlinear_func_str;
    // 3. output
    unsigned tag_dict_sz;
    unsigned tag_embedding_dim;
    std::string output_layer_type;
    // 4. others
    unsigned replace_freq_threshold;
    slnn::type::real replace_prob_threshold;
};


} // end of namespace structure param module
} // end of namespace segmenter
} // end of namespace slnn




#endif