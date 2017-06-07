#ifndef SLNN_SEGMENTER_CWS_MLP_INPUT1_CWS_RNN_INPUT1_INSTANCE_H_
#define SLNN_SEGMENTER_CWS_MLP_INPUT1_CWS_RNN_INPUT1_INSTANCE_H_
#include "segmenter/cws_rnn_input1/cws_rnn_input1_template.h"
#include "segmenter/cws_module/token_module/input1/token_input1_unigram.h"
#include "segmenter/cws_module/token_module/input1/token_input1_bigram.h"
#include "segmenter/cws_module/token_module/input1/token_input1_all.h"
#include "segmenter/cws_module/structure_param_module/rnn_input1_param.h"
#include "segmenter/cws_module/structure_param_module/param_rnn_all.h"
#include "segmenter/cws_module/nn_module/rnn_input1/nn_cws_rnn_input1_abstract.h"
#include "segmenter/cws_module/nn_module/rnn_input1/nn_cws_rnn_all.h"
namespace slnn{
namespace segmenter{
namespace rnn_input1{

extern template class SegmenterRnnInput1Template<
    token_module::TokenSegmenterInput1Unigram,
    structure_param_module::SegmenterRnnInput1Param,
    nn_module::NnSegmenterRnnInput1Abstract>;

using RnnInput1Unigram = SegmenterRnnInput1Template<
    token_module::TokenSegmenterInput1Unigram,
    structure_param_module::SegmenterRnnInput1Param,
    nn_module::NnSegmenterRnnInput1Abstract>;

extern template class SegmenterRnnInput1Template<
    token_module::TokenSegmenterInput1Bigram,
    structure_param_module::SegmenterRnnInput1Param,
    nn_module::NnSegmenterRnnInput1Abstract>;

using RnnInput1Bigram = SegmenterRnnInput1Template<
    token_module::TokenSegmenterInput1Bigram,
    structure_param_module::SegmenterRnnInput1Param,
    nn_module::NnSegmenterRnnInput1Abstract > ;

extern template class SegmenterRnnInput1Template<
    token_module::TokenSegmenterInput1All,
    structure_param_module::ParamSegmenterRnnAll,
    nn_module::NnSegmenterRnnAll >;

using RnnAll = SegmenterRnnInput1Template<
    token_module::TokenSegmenterInput1All,
    structure_param_module::ParamSegmenterRnnAll,
    nn_module::NnSegmenterRnnAll >;


} // enf of namespace rnn-input1
} // enf of namespace segmenter
} // end of namespace slnn


#endif
