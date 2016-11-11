#include "cws_mlp_input1_instance.h"
namespace slnn{
namespace segmenter{
namespace mlp_input1{

template class SegmenterMlpInput1Template<
    token_module::TokenSegmenterInput1Unigram,
    structure_param_module::SegmenterBasicMlpParam,
    nn_module::NnSegmenterInput1MlpAbstract>;

template class SegmenterMlpInput1Template<
    token_module::TokenSegmenterInput1Bigram,
    structure_param_module::SegmenterBasicMlpParam,
    nn_module::NnSegmenterInput1MlpAbstract>;

} // enf of namespace mlp-input1
} // enf of namespace segmenter
} // end 