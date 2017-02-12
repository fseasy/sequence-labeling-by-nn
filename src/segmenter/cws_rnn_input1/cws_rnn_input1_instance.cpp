#include "cws_rnn_input1_instance.h"
namespace slnn{
namespace segmenter{
namespace rnn_input1{

template class SegmenterRnnInput1Template<
    token_module::TokenSegmenterInput1Unigram,
    structure_param_module::SegmenterRnnInput1Param,
    nn_module::NnSegmenterRnnInput1Abstract>;

template class SegmenterRnnInput1Template<
    token_module::TokenSegmenterInput1Bigram,
    structure_param_module::SegmenterRnnInput1Param,
    nn_module::NnSegmenterRnnInput1Abstract>;

} // enf of namespace rnn-input1
} // enf of namespace segmenter
} // end 