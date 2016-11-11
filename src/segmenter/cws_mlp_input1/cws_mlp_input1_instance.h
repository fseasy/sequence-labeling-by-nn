#ifndef SLNN_SEGMENTER_CWS_MLP_INPUT1_CWS_MLP_INPUT1_INSTANCE_H_
#define SLNN_SEGMENTER_CWS_MLP_INPUT1_CWS_MLP_INPUT1_INSTANCE_H_
#include "segmenter/cws_mlp_input1/cws_mlp_input1_template.h"
#include "segmenter/cws_module/token_module/input1/token_input1_unigram.h"
#include "segmenter/cws_module/token_module/input1/token_input1_bigram.h"
#include "segmenter/cws_module/structure_param_module/basic_mlp_param.h"
#include "segmenter/cws_module/nn_module/mlp_input1/nn_cws_mlp_input1_abstract.h"
namespace slnn{
namespace segmenter{
namespace mlp_input1{

extern template class SegmenterMlpInput1Template<
    token_module::TokenSegmenterInput1Unigram,
    structure_param_module::SegmenterBasicMlpParam,
    nn_module::NnSegmenterInput1MlpAbstract>;

using MlpInput1Unigram = SegmenterMlpInput1Template<
    token_module::TokenSegmenterInput1Unigram,
    structure_param_module::SegmenterBasicMlpParam,
    nn_module::NnSegmenterInput1MlpAbstract>;

extern template class SegmenterMlpInput1Template<
    token_module::TokenSegmenterInput1Bigram,
    structure_param_module::SegmenterBasicMlpParam,
    nn_module::NnSegmenterInput1MlpAbstract>;

using MlpInput1Bigram = SegmenterMlpInput1Template<
    token_module::TokenSegmenterInput1Bigram,
    structure_param_module::SegmenterBasicMlpParam,
    nn_module::NnSegmenterInput1MlpAbstract > ;

} // enf of namespace mlp-input1
} // enf of namespace segmenter
} // end of namespace slnn


#endif
