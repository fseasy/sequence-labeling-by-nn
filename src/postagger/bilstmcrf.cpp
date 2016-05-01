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

#include "bilstmcrf.h"

using namespace std;
using namespace cnn;
namespace slnn{

const std::string BILSTMCRFModel4POSTAG::UNK_STR = "<UNK_REPR>";

BILSTMCRFModel4POSTAG::BILSTMCRFModel4POSTAG()
    : m(nullptr),
    merge_doublechannel_layer(nullptr) ,
    bilstm_layer(nullptr),
    emit_layer(nullptr),
    dynamic_dict_wrapper(dynamic_dict)
{}

BILSTMCRFModel4POSTAG::~BILSTMCRFModel4POSTAG()
{
    if (m) delete m;
    if (merge_doublechannel_layer) delete merge_doublechannel_layer;
    if (bilstm_layer) delete bilstm_layer;
    if (emit_layer) delete emit_layer;
}

void BILSTMCRFModel4POSTAG::build_model_structure()
{
    assert(dynamic_dict.is_frozen() && fixed_dict.is_frozen() && postag_dict.is_frozen()); // Assert all frozen
    m = new Model();
    merge_doublechannel_layer = new Merge2Layer(m, dynamic_embedding_dim, fixed_embedding_dim, lstm_x_dim);
    bilstm_layer = new BILSTMLayer(m, nr_lstm_stacked_layer, lstm_x_dim, lstm_h_dim);
    emit_layer = new Merge3Layer(m, lstm_h_dim, lstm_h_dim, postag_embedding_dim, 1);

    dynamic_words_lookup_param = m->add_lookup_parameters(dynamic_embedding_dict_size, { dynamic_embedding_dim });
    fixed_words_lookup_param = m->add_lookup_parameters(fixed_embedding_dict_size, { fixed_embedding_dim });
    postags_lookup_param = m->add_lookup_parameters(postag_dict_size , { postag_embedding_dim });

    trans_score_lookup_param = m->add_lookup_parameters(postag_dict_size * postag_dict_size, { 1 });
    init_score_lookup_param = m->add_lookup_parameters(postag_dict_size, { 1 });

}

void BILSTMCRFModel4POSTAG::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- Model Structure Info ----------------\n"
        << "dynamic vocabulary size : " << dynamic_embedding_dict_size << " with dimention : " << dynamic_embedding_dim << "\n"
        << "fixed vocabulary size : " << fixed_embedding_dict_size << " with dimension : " << fixed_embedding_dim << "\n"
        << "bilstm stacked layer num : " << nr_lstm_stacked_layer << " , x dim  : " << lstm_x_dim << " , h dim : " << lstm_h_dim << "\n"
        << "postag num : " << postag_dict_size << " with dimension : " << postag_embedding_dim;
}

Expression BILSTMCRFModel4POSTAG::viterbi_train(ComputationGraph *p_cg, 
    const IndexSeq *p_dynamic_sent, const IndexSeq *p_fixed_sent, const IndexSeq *p_tag_seq,
    Stat *p_stat)
{
    const unsigned sent_len = p_dynamic_sent->size();
    ComputationGraph &cg = *p_cg;
    // New graph , ready for new sentence
    merge_doublechannel_layer->new_graph(cg);
    bilstm_layer->new_graph(cg);
    emit_layer->new_graph(cg);

    bilstm_layer->start_new_sequence();

    // Some container
    vector<Expression> err_exp_cont(sent_len); // for storing every error expression in each tag prediction
    vector<Expression> merge_dc_exp_cont(sent_len);
    vector<Expression> l2r_lstm_output_exp_cont; // for storing left to right lstm output(deepest hidden layer) expression for every timestep
    vector<Expression> r2l_lstm_output_exp_cont; // right to left 

    // 1. build input , and merge
    for (unsigned i = 0; i < sent_len; ++i)
    {
        Expression dynamic_word_lookup_exp = lookup(cg, dynamic_words_lookup_param, p_dynamic_sent->at(i));
        Expression fixed_word_lookup_exp = const_lookup(cg, fixed_words_lookup_param, p_fixed_sent->at(i)); // const look up
        Expression merge_dc_exp = merge_doublechannel_layer->build_graph(dynamic_word_lookup_exp, fixed_word_lookup_exp);
        merge_dc_exp_cont[i] = rectify(merge_dc_exp); // rectify for merged expression
    }

    // 2. Build bi-lstm
    bilstm_layer->build_graph(merge_dc_exp_cont, l2r_lstm_output_exp_cont, r2l_lstm_output_exp_cont);

    // viterbi
    vector<Expression> all_postag_exp_cont(postag_dict_size);
    vector<vector<Expression>> lattice(sent_len, vector<Expression>(postag_dict_size));
    vector<Expression> init_score(postag_dict_size);
    vector<Expression> trans_score(postag_dict_size * postag_dict_size);
    vector<vector<Expression>> emit_score(sent_len, vector<Expression>(postag_dict_size));
    // init all_postag_exp_cont , init_score , trans_score , emit_score
    for (size_t postag_idx = 0; postag_idx < postag_dict_size; ++postag_idx)
    {
        all_postag_exp_cont[postag_idx] = lookup(cg, postags_lookup_param, postag_idx);
        init_score[postag_idx] = lookup(cg, init_score_lookup_param, postag_idx);
    }
    /*
    for (size_t from_postag_idx = 0; from_postag_idx < postag_dict_size; ++from_postag_idx)
    {
        for (size_t to_postag_idx = 0; to_postag_idx < postag_dict_size; ++to_postag_idx)
        {
            size_t flat_idx = from_postag_idx * postag_dict_size + to_postag_idx;
            trans_score[flat_idx] = lookup(cg, trans_score_lookup_param, flat_idx);
        }
    }
    */
    for (size_t flat_idx = 0; flat_idx < postag_dict_size * postag_dict_size; ++flat_idx)
    {
        trans_score[flat_idx] = lookup(cg, trans_score_lookup_param, flat_idx);
    }
    for (size_t time_step = 0; time_step < sent_len; ++time_step)
    {
        for (size_t postag_idx = 0; postag_idx < postag_dict_size; ++postag_idx)
        {
            emit_score[time_step][postag_idx] = emit_layer->build_graph(l2r_lstm_output_exp_cont[time_step],
                r2l_lstm_output_exp_cont[time_step], all_postag_exp_cont[postag_idx]);
        }
    }
    // 1. the time 0
    for (size_t postag_idx = 0; postag_idx < postag_dict_size; ++postag_idx)
    {
        // init_score + emit_score
        lattice[0][postag_idx] = init_score[postag_idx] + emit_score[0][postag_idx];
    }
    // 2. the continues time
    for(size_t time_step = 1; time_step < sent_len; ++time_step)
    {
        // for every tag(to-tag)
        for (size_t postag_idx = 0; postag_idx < postag_dict_size; ++postag_idx)
        {
            // for every possible trans
            vector<Expression> trans_score_from_pre_tag_cont(postag_dict_size);
            for (size_t from_postag_idx = 0; from_postag_idx < postag_dict_size; ++from_postag_idx)
            {
                size_t flat_idx = from_postag_idx * postag_dict_size + postag_dict_size;
                // from-tag score + trans_score
                trans_score_from_pre_tag_cont[from_postag_idx] = lattice[time_step - 1][from_postag_idx] +
                    trans_score[flat_idx];
            }
            lattice[time_step][postag_idx] = logsumexp(trans_score_from_pre_tag_cont) + emit_score[time_step][postag_idx];
        }
    }
    Expression predict_score = logsumexp(lattice[sent_len - 1]);

    // gold score
    Expression gold_score = lattice[0][p_tag_seq->at(0)];
    for (size_t time_step = 1 ; time_step < sent_len; ++time_step)
    {
        gold_score =  gold_score + lattice[time_step][p_tag_seq->at(time_step)];
    }
    // predict is the max-score of lattice
    // if totally correct , loss = 0 (predict_score = gold_score , that is , predict sequence equal to gold sequence)
    // else , loss = predict_score - gold_score
    return predict_score - gold_score;
}

void BILSTMCRFModel4POSTAG::do_predict(ComputationGraph *p_cg, 
    const IndexSeq *p_dynamic_sent, const IndexSeq *p_fixed_sent, IndexSeq *p_predict_tag_seq)
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
        Expression merge_dc_exp = merge_doublechannel_layer->build_graph(dynamic_word_lookup_exp, fixed_word_lookup_exp);
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
        pretag_lookup_exp = lookup(cg, postags_lookup_param, tag_id_with_max_value);
    }
    swap(tmp_predict_tag_seq, *p_predict_tag_seq);
}


} // end of namespace
