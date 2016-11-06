#ifndef SLNN_SEGMENTER_CWS_MLP_INPUT1_CWS_MLP_INPUT1_INSTANCE_H_
#define SLNN_SEGMENTER_CWS_MLP_INPUT1_CWS_MLP_INPUT1_INSTANCE_H_
#include "segmenter/cws_mlp_input1/cws_mlp_input1_template.h"
#include "segmenter/cws_module/token_module/input1/token_input1_unigram.h"
#include "segmenter/cws_module/structure_param_module/basic_mlp_param.h"
#include "segmenter/cws_module/nn_module/mlp_input1/nn_cws_mlp_input1_abstract.h"
namespace slnn{
namespace segmenter{
namespace mlp_input1{

extern template class SegmentorMlpInput1Template<
    token_module::TokenSegmenterInput1Unigram,
    structure_param_module::SegmentorBasicMlpParam,
    nn_module::NnSegmenterInput1Abstract>;

using MlpInput1 = SegmentorMlpInput1Template<
    token_module::TokenSegmenterInput1Unigram,
    structure_param_module::SegmentorBasicMlpParam,
    nn_module::NnSegmenterInput1Abstract>;

} // enf of namespace mlp-input1
} // enf of namespace segmenter
} // end of namespace slnn


#endif
