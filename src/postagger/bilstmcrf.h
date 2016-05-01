#ifndef BILSTMMODEL_4_TAGGING_DOUBLECHANNEL_H_INCLUDED_
#define BILSTMMODEL_4_TAGGING_DOUBLECHANNEL_H_INCLUDED_

#include <string>
#include <sstream>

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

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
    unsigned dynamic_embedding_dim,
        postag_embedding_dim,
        nr_lstm_stacked_layer ,
        lstm_x_dim,
        lstm_h_dim,
        fixed_embedding_dim,
        fixed_embedding_dict_size;
    // - set from inner (dict)
    unsigned dynamic_embedding_dict_size,
        postag_dict_size;

    // Model param
    cnn::Model *m;

    Merge2Layer *merge_doublechannel_layer;
    BILSTMLayer *bilstm_layer;
    Merge3Layer *emit_layer;

    cnn::LookupParameters *dynamic_words_lookup_param;
    cnn::LookupParameters *fixed_words_lookup_param;
    cnn::LookupParameters *postags_lookup_param;
    
    cnn::LookupParameters *trans_score_lookup_param; // trans score , that is , TAG_A -> TAG_B 's score
    cnn::LookupParameters *init_score_lookup_param; // init score , that is , the init TAG score


    // Dict
    cnn::Dict dynamic_dict;
    cnn::Dict fixed_dict;
    cnn::Dict postag_dict;
    DictWrapper dynamic_dict_wrapper;
    static const std::string UNK_STR ; 

    BILSTMCRFModel4POSTAG();
    ~BILSTMCRFModel4POSTAG();

    void build_model_structure();
    void print_model_info();


    cnn::expr::Expression viterbi_train(cnn::ComputationGraph *p_cg, 
        const IndexSeq *p_dynamic_sent, const IndexSeq *p_fixed_sent, const IndexSeq *p_tag_seq,
        Stat *p_stat = nullptr);
    void viterbi_predict(cnn::ComputationGraph *p_cg, 
        const IndexSeq *p_dynamic_sent, const IndexSeq *p_fixed_sent, IndexSeq *p_predict_tag_seq);

};


} // end of namespace


#endif
