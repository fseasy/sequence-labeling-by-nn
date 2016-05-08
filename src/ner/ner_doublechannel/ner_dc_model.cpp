#include <fstream>
#include <vector>
#include <string>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "ner_dc_model.h"

using namespace std;
using namespace cnn;
namespace slnn{

const std::string NERDCModel::UNK_STR = "<UNK_REPR>";

NERDCModel::NERDCModel()
    : m(nullptr),
    merge_doublechannel_layer(nullptr) ,
    bilstm_layer(nullptr),
    merge_bilstm_and_pretag_layer(nullptr),
    tag_output_linear_layer(nullptr),
    dynamic_dict_wrapper(dynamic_dict)
{}

NERDCModel::~NERDCModel()
{
    delete m; // delete nullptr is also valid 
    delete merge_doublechannel_layer;
    delete bilstm_layer;
    delete merge_bilstm_and_pretag_layer;
    delete tag_output_linear_layer;
}

void NERDCModel::build_model_structure()
{
    assert(dynamic_dict.is_frozen() && fixed_dict.is_frozen() && postag_dict.is_frozen() && ner_dict.is_frozen()); // Assert all frozen
    m = new Model();
    merge_doublechannel_layer = new Merge3Layer(m, dynamic_embedding_dim, fixed_embedding_dim, postag_embedding_dim , lstm_x_dim);
    bilstm_layer = new BILSTMLayer(m, nr_lstm_stacked_layer, lstm_x_dim, lstm_h_dim);
    merge_bilstm_and_pretag_layer = new Merge3Layer(m, lstm_h_dim, lstm_h_dim, postag_embedding_dim, tag_layer_hidden_dim);
    tag_output_linear_layer = new DenseLayer(m, tag_layer_hidden_dim, ner_embedding_dict_size);

    dynamic_words_lookup_param = m->add_lookup_parameters(dynamic_embedding_dict_size, { dynamic_embedding_dim });
    fixed_words_lookup_param = m->add_lookup_parameters(fixed_embedding_dict_size, { fixed_embedding_dim });
    postag_lookup_param = m->add_lookup_parameters(postag_embedding_dict_size ,  { postag_embedding_dim });
    ner_lookup_param = m->add_lookup_parameters(ner_embedding_dict_size, { ner_embedding_dim });

    TAG_SOS_param = m->add_parameters({ ner_embedding_dim });

}

void NERDCModel::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- Model Structure Info ----------------\n"
        << "dynamic vocabulary size : " << dynamic_embedding_dict_size << " with dimention : " << dynamic_embedding_dim << "\n"
        << "fixed vocabulary size : " << fixed_embedding_dict_size << " with dimension : " << fixed_embedding_dim << "\n"
        << "postag number : " << postag_embedding_dict_size << " with dimension : " << postag_embedding_dim << "\n"
        << "ner tag number : " << ner_embedding_dict_size << " with dimension : " << ner_embedding_dim << "\n"
        << "bilstm stacked layer num : " << nr_lstm_stacked_layer << " , x dim  : " << lstm_x_dim << " , h dim : " << lstm_h_dim << "\n"
        << "tag layer hidden dimension : " << tag_layer_hidden_dim << "\n"
        << "tag layer output dimension : " << ner_embedding_dict_size;
}

Expression NERDCModel::negative_loglikelihood(ComputationGraph *p_cg, 
    const IndexSeq *p_dynamic_sent, const IndexSeq *p_fixed_sent, const IndexSeq *p_postag_seq,
    const IndexSeq *p_ner_seq , 
    Stat *p_stat)
{
    const unsigned sent_len = p_dynamic_sent->size();
    ComputationGraph &cg = *p_cg;
    // New graph , ready for new sentence
    merge_doublechannel_layer->new_graph(cg);
    bilstm_layer->new_graph(cg);
    merge_bilstm_and_pretag_layer->new_graph(cg);
    tag_output_linear_layer->new_graph(cg);

    bilstm_layer->start_new_sequence();

    // Some container
    vector<Expression> err_exp_cont(sent_len); // for storing every error expression in each tag prediction
    vector<Expression> merge_dc_exp_cont(sent_len);
    vector<Expression> l2r_lstm_output_exp_cont; // for storing left to right lstm output(deepest hidden layer) expression for every timestep
    vector<Expression> r2l_lstm_output_exp_cont; // right to left 
    vector<Expression> pretag_lookup_exp_cont(sent_len); // ADD for PRE_TAG

    // 1. build input , and merge
    for (unsigned i = 0; i < sent_len; ++i)
    {
        Expression dynamic_word_lookup_exp = lookup(cg, dynamic_words_lookup_param, p_dynamic_sent->at(i));
        Expression fixed_word_lookup_exp = const_lookup(cg, fixed_words_lookup_param, p_fixed_sent->at(i)); // const look up
        Expression postag_lookup_exp = lookup(cg, postag_lookup_param, p_postag_seq->at(i));
        Expression merge_dc_exp = merge_doublechannel_layer->build_graph(dynamic_word_lookup_exp, fixed_word_lookup_exp ,
            postag_lookup_exp);
        merge_dc_exp_cont[i] = rectify(merge_dc_exp); // rectify for merged expression
    }

    // 2. Build bi-lstm
    bilstm_layer->build_graph(merge_dc_exp_cont, l2r_lstm_output_exp_cont, r2l_lstm_output_exp_cont);

    // 3. prepare for PRE_TAG embedding
    Expression TAG_SOS_exp = parameter(cg, TAG_SOS_param);
    pretag_lookup_exp_cont[0] = TAG_SOS_exp;
    for (unsigned i = 1; i < sent_len; ++i)
    {
        pretag_lookup_exp_cont[i] = lookup(cg, ner_lookup_param, p_ner_seq->at(i - 1));
    }

    // build tag network , calc loss Expression of every timestep 
    for (unsigned i = 0; i < sent_len; ++i)
    {
        // rectify is suggested as activation function
        Expression merge_bilstm_pretag_exp = merge_bilstm_and_pretag_layer->build_graph(l2r_lstm_output_exp_cont[i],
            r2l_lstm_output_exp_cont[i], pretag_lookup_exp_cont[i]);
        Expression tag_hidden_layer_output_at_timestep_t = rectify(merge_bilstm_pretag_exp); 
        Expression tag_output_layer_output_at_timestep_t = tag_output_linear_layer->build_graph(tag_hidden_layer_output_at_timestep_t);
        // if statistic , calc output at timestep t
        if (p_stat != nullptr)
        {
            vector<float> output_values = as_vector(cg.incremental_forward());
            Index tag_id_with_max_value = distance(output_values.begin(), max_element(output_values.begin(), output_values.end()));
            ++(p_stat->total_tags); // == ++stat->total_tags ;
            if (tag_id_with_max_value == p_ner_seq->at(i)) ++(p_stat->correct_tags);
        }
        err_exp_cont[i] = pickneglogsoftmax(tag_output_layer_output_at_timestep_t, p_ner_seq->at(i));
    }

    // build the finally loss 
    return sum(err_exp_cont); // in fact , no need to return . just to avoid a warning .
}

void NERDCModel::do_predict(ComputationGraph *p_cg, 
    const IndexSeq *p_dynamic_sent, const IndexSeq *p_fixed_sent, const IndexSeq *p_postag_seq ,
    IndexSeq *p_predict_ner_seq)
{
    // The main structure is just a copy from build_bilstm4tagging_graph2train! 
    const unsigned sent_len = p_dynamic_sent->size();
    ComputationGraph &cg = *p_cg;
    // New graph , ready for new sentence 
    // New graph , ready for new sentence
    merge_doublechannel_layer->new_graph(cg);
    bilstm_layer->new_graph(cg);
    merge_bilstm_and_pretag_layer->new_graph(cg);
    tag_output_linear_layer->new_graph(cg);

    bilstm_layer->start_new_sequence();

    // Some container
    vector<Expression> merge_dc_exp_cont(sent_len);
    vector<Expression> l2r_lstm_output_exp_cont; // for storing left to right lstm output(deepest hidden layer) expression for every timestep
    vector<Expression> r2l_lstm_output_exp_cont; // right to left                                                  
    
    // 1. get word embeddings for sent 
    for (unsigned i = 0; i < sent_len; ++i)
    {
        Expression dynamic_word_lookup_exp = lookup(cg, dynamic_words_lookup_param, p_dynamic_sent->at(i));
        Expression fixed_word_lookup_exp = const_lookup(cg, fixed_words_lookup_param, p_fixed_sent->at(i)); // const look up
        Expression postag_lookup_exp = lookup(cg, postag_lookup_param, p_postag_seq->at(i));
        Expression merge_dc_exp = merge_doublechannel_layer->build_graph(dynamic_word_lookup_exp, fixed_word_lookup_exp , postag_lookup_exp);
        merge_dc_exp_cont[i] = rectify(merge_dc_exp); // rectify for merged expression
    }
    // 2. calc Expression of every timestep of BI-LSTM
    bilstm_layer->build_graph(merge_dc_exp_cont, l2r_lstm_output_exp_cont, r2l_lstm_output_exp_cont);
    // 3. set previous tag lookup expression

    Expression pretag_lookup_exp = parameter(cg, TAG_SOS_param);

    // build tag network , calc loss Expression of every timestep 
    IndexSeq tmp_predict_tag_seq(sent_len);
    for (unsigned i = 0; i < sent_len; ++i)
    {
        Expression merge_bilstm_pretag_exp = merge_bilstm_and_pretag_layer->build_graph(l2r_lstm_output_exp_cont[i],
            r2l_lstm_output_exp_cont[i], pretag_lookup_exp);
        Expression tag_hidden_layer_output_at_timestep_t = rectify(merge_bilstm_pretag_exp);
        tag_output_linear_layer->build_graph(tag_hidden_layer_output_at_timestep_t);
        vector<float> output_values = as_vector(cg.incremental_forward());
        unsigned tag_id_with_max_value = distance(output_values.cbegin(), max_element(output_values.cbegin(), output_values.cend()));
        tmp_predict_tag_seq.at(i) = tag_id_with_max_value ;
        // set pretag_lookup_exp for next timestep 
        pretag_lookup_exp = lookup(cg, ner_lookup_param, tag_id_with_max_value);
    }
    swap(tmp_predict_tag_seq, *p_predict_ner_seq);
}


} // end of namespace
