#ifndef NER_DC_H_INCLUDED_
#define NER_DC_H_INCLUDED_

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
struct NERDCModelHandler;

struct NERDCModel
{
    friend struct NERDCModelHandler;
    // Model structure param 
    unsigned dynamic_embedding_dim,
        fixed_embedding_dim,
        postag_embedding_dim,
        ner_embedding_dim;
    unsigned dynamic_embedding_dict_size,
        fixed_embedding_dict_size,
        postag_embedding_dict_size,
        ner_embedding_dict_size;
    unsigned nr_lstm_stacked_layer,
        lstm_x_dim,
        lstm_h_dim,
        tag_layer_hidden_dim;

    // Model param
    dynet::Model *m;

    Merge3Layer *merge_doublechannel_layer;
    BILSTMLayer *bilstm_layer;
    Merge3Layer *merge_bilstm_and_pretag_layer;
    DenseLayer *tag_output_linear_layer;

    dynet::LookupParameter dynamic_words_lookup_param;
    dynet::LookupParameter fixed_words_lookup_param;
    dynet::LookupParameter postag_lookup_param;
    dynet::LookupParameter ner_lookup_param;
    
    dynet::Parameter TAG_SOS_param; // for tag_hidden_layer , pre-tag


    // Dict
    dynet::Dict dynamic_dict;
    dynet::Dict fixed_dict;
    dynet::Dict postag_dict;
    dynet::Dict ner_dict;
    DictWrapper dynamic_dict_wrapper;
    
    static const std::string UNK_STR ; 

    /******************functions********************/

    NERDCModel();
    ~NERDCModel();

    void build_model_structure();
    void print_model_info();


    dynet::expr::Expression negative_loglikelihood(dynet::ComputationGraph *p_cg, 
        const IndexSeq *p_dynamic_sent, const IndexSeq *p_fixed_sent, const IndexSeq *p_postag_seq,
        const IndexSeq *p_ner_seq ,
        Stat *p_stat = nullptr);
    void do_predict(dynet::ComputationGraph *p_cg, 
        const IndexSeq *p_dynamic_sent, const IndexSeq *p_fixed_sent, const IndexSeq *p_postag_seq ,
        IndexSeq *p_predict_ner_seq);

};


} // end of namespace


#endif
