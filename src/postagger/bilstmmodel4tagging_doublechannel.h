#ifndef BILSTMMODEL_4_TAGGING_DOUBLECHANNEL_H_INCLUDED_
#define BILSTMMODEL_4_TAGGING_DOUBLECHANNEL_H_INCLUDED_

#include <string>

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
struct DoubleChannelModelHandler;

struct DoubleChannelModel4POSTAG
{
    friend struct DoubleChannelModelHandler;
    // Model structure param 
    // - set by outer
    unsigned dynamic_embedding_dim,
        postag_embedding_dim,
        nr_lstm_stacked_layer ,
        lstm_x_dim,
        lstm_h_dim,
        tag_layer_hidden_dim ,
        fixed_embedding_dim,
        fixed_embedding_dict_size;
    // - set from inner (dict)
    unsigned dynamic_embedding_dict_size,
        tag_layer_output_dim;

    // Model param
    cnn::Model *m;
    Merge2Layer *merge_doublechannel_layer;
    BILSTMLayer *bilstm_layer;
    Merge3Layer *merge_bilstm_and_pretag_layer;
    DenseLayer *tag_output_linear_layer;

    cnn::LookupParameters *dynamic_words_lookup_param;
    cnn::LookupParameters *fixed_words_lookup_param;
    cnn::LookupParameters *postags_lookup_param;
    
    cnn::Parameters *TAG_SOS_param;


    // Dict
    cnn::Dict dynamic_dict;
    cnn::Dict fixed_dict;
    cnn::Dict postag_dict;
    DictWrapper dynamic_dict_wrapper;
    static const std::string UNK_STR ; 

    DoubleChannelModel4POSTAG();
    ~DoubleChannelModel4POSTAG();

    void freeze_dict_and_add_UNK();
    void set_partial_model_structure_param_from_outer(boost::program_options::variables_map &varmap);
    void set_partial_model_structure_param_from_inner();
    void build_model_structure();
    void print_model_info();

    void save_model(std::ostream &os);
    void load_model(std::istream &is);

    cnn::expr::Expression negative_loglikelihood(const IndexSeq *p_dynamic_sent, const IndexSeq *p_fixed_sent, const IndexSeq *p_tag_seq,
        cnn::ComputationGraph *p_cg, Stat *p_stat = nullptr);
    void do_predict(const IndexSeq *p_sent, const IndexSeq *p_fixed_sent, 
        cnn::ComputationGraph *p_cg, IndexSeq *p_predict_tag_seq);

};

const std::string DoubleChannelModel4POSTAG::UNK_STR = "<UNK_REPR>" ;
} // end of namespace


#endif