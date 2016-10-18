#ifndef BILSTMCRF_H_INCLUDED_
#define BILSTMCRF_H_INCLUDED_

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
#include "utils/utf8processing.hpp" 
#include "utils/dict_wrapper.hpp"
#include "utils/stat.hpp"

namespace slnn
{
struct BILSTMCRFModelHandler;

struct BILSTMCRFModel4POSTAG
{
    friend struct BILSTMCRFModelHandler;
    // Model structure param 
    // - set by outer
    unsigned word_embedding_dim,
        postag_embedding_dim,
        nr_lstm_stacked_layer ,
        lstm_h_dim,
        merge_hidden_dim;
    // - set from inner (dict)
    unsigned word_dict_size ,
        postag_dict_size;

    // Model param
    dynet::Model *m;

    BILSTMLayer *bilstm_layer;
    Merge3Layer *merge_hidden_layer;
    DenseLayer *emit_layer;

    dynet::LookupParameters *words_lookup_param;
    dynet::LookupParameters *postags_lookup_param;
    
    dynet::LookupParameters *trans_score_lookup_param; // trans score , that is , TAG_A -> TAG_B 's score
    dynet::LookupParameters *init_score_lookup_param; // init score , that is , the init TAG score


    // Dict
    dynet::Dict word_dict;
    dynet::Dict postag_dict;
    DictWrapper word_dict_wrapper;
    static const std::string UNK_STR ; 

    BILSTMCRFModel4POSTAG();
    ~BILSTMCRFModel4POSTAG();

    void build_model_structure();
    void print_model_info();


    dynet::expr::Expression viterbi_train(dynet::ComputationGraph *p_cg, 
        const IndexSeq *p_sent, const IndexSeq *p_tag_seq,
        Stat *p_stat = nullptr);
    void viterbi_predict(dynet::ComputationGraph *p_cg, 
        const IndexSeq *p_sent, IndexSeq *p_predict_tag_seq);

};


} // end of namespace


#endif
