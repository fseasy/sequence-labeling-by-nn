#include "cws_mlp_input1_cl_model.h"
namespace slnn{
namespace segmenter{
namespace mlp_input1{

template class SegmentorMlpInput1Template<
    token_module::SegmentorBasicTokenModule,
    structure_param_module::SegmentorBasicMlpParam,
    nn_module::NnSegmentorInput1Cl>;

} // enf of namespace mlp-input1
} // enf of namespace segmenter
} // end of namespace slnn