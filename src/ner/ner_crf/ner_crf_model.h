#ifndef NER_CRF_H_INCLUDED_
#define NER_CRF_H_INCLUDED_

#include <string>
#include <sstream>

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/rnn.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"

#include <boost/program_options.hpp>

#include "utils/typedeclaration.h"
#include "modelmodule/layers.h"
#include "utils/dict_wrapper.hpp"
#include "utils/stat.hpp"

namespace slnn
{
struct NERCRFModelHandler;

struct NERCRFModel
{
    friend struct NERCRFModelHandler;
    // Model structure param 
    unsigned word_embedding_dim,
        postag_embedding_dim,
        ner_embedding_dim;
    unsigned word_embedding_dict_size,
        postag_embedding_dict_size,
        ner_embedding_dict_size;
    unsigned nr_lstm_stacked_layer,
        lstm_x_dim,
        lstm_h_dim,
        emit_hidden_layer_dim;

    // Model param
    dynet::Model *m;

    Merge2Layer *merge_input_layer;
    BILSTMLayer *bilstm_layer;
    Merge3Layer *emit_hidden_layer;
    DenseLayer *emit_output_layer;

    dynet::LookupParameter words_lookup_param;
    dynet::LookupParameter postag_lookup_param;
    dynet::LookupParameter ner_lookup_param;
    
    dynet::LookupParameter init_score_lookup_param;
    dynet::LookupParameter trans_score_lookup_param;


    // Dict
    dynet::Dict word_dict;
    dynet::Dict postag_dict;
    dynet::Dict ner_dict;
    DictWrapper word_dict_wrapper;
    
    static const std::string UNK_STR ; 

    /******************functions********************/

    NERCRFModel();
    ~NERCRFModel();

    void build_model_structure();
    void print_model_info();


    dynet::expr::Expression viterbi_train(dynet::ComputationGraph *p_cg, 
        const IndexSeq *p_sent, const IndexSeq *p_postag_seq,
        const IndexSeq *p_ner_seq ,
        float dropout_rate ,
        Stat *p_stat = nullptr);
    void viterbi_predict(dynet::ComputationGraph *p_cg, 
        const IndexSeq *p_sent, const IndexSeq *p_postag_seq ,
        IndexSeq *p_predict_ner_seq);

};


} // end of namespace


#endif
