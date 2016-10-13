#ifndef SLNN_SEGMENTOR_CWS_MLP_INPUT1_CLASSIFICATION_CWS_MLP_INPUT1_CL_MODEL_H_
#define SLNN_SEGMENTOR_CWS_MLP_INPUT1_ClASSIFICATION_CWS_MLP_INPUT1_CL_MODEL_H_
#include "segmentor/cws_mlp_input1/cws_mlp_input1_template.h"
#include "segmentor/cws_module/nn_module/mlp_input1/nn_cws_mlp_input1_cl.h"
#include "segmentor/cws_module/token_module/cws_basic_token_module.h"
#include "segmentor/cws_module/structure_param_module/basic_mlp_param.h"

namespace slnn{
namespace segmentor{
namespace mlp_input1{

extern template class SegmentorMlpInput1Template<
    token_module::SegmentorBasicTokenModule,
    structure_param_module::SegmentorBasicMlpParam,
    nn_module::NnSegmentorInput1Cl>;

using MlpInput1Cl = SegmentorMlpInput1Template<
    token_module::SegmentorBasicTokenModule,
    structure_param_module::SegmentorBasicMlpParam,
    nn_module::NnSegmentorInput1Cl>;

} // enf of namespace mlp-input1
} // enf of namespace segmentor
} // end of namespace slnn


#endif
