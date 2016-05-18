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

#include "ner_crf_model.h"

using namespace std;
using namespace cnn;
namespace slnn{

const std::string NERCRFModel::UNK_STR = "<UNK_REPR>";

NERCRFModel::NERCRFModel()
    : m(nullptr),
    merge_input_layer(nullptr) ,
    bilstm_layer(nullptr),
    emit_hidden_layer(nullptr),
    emit_output_layer(nullptr),
    word_dict_wrapper(word_dict)
{}

NERCRFModel::~NERCRFModel()
{
    delete m; // delete nullptr is also valid 
    delete merge_input_layer;
    delete bilstm_layer;
    delete emit_hidden_layer;
    delete emit_output_layer;
}

void NERCRFModel::build_model_structure()
{
    assert(word_dict.is_frozen()  && postag_dict.is_frozen() && ner_dict.is_frozen()); // Assert all frozen
    m = new Model();
    merge_input_layer = new Merge2Layer(m, word_embedding_dim, postag_embedding_dim , lstm_x_dim);
    bilstm_layer = new BILSTMLayer(m, nr_lstm_stacked_layer, lstm_x_dim, lstm_h_dim);
    emit_hidden_layer = new Merge3Layer(m, lstm_h_dim, lstm_h_dim, ner_embedding_dim, emit_hidden_layer_dim);
    emit_output_layer = new DenseLayer(m, emit_hidden_layer_dim , 1);

    words_lookup_param = m->add_lookup_parameters(word_embedding_dict_size, { word_embedding_dim });
    postag_lookup_param = m->add_lookup_parameters(postag_embedding_dict_size ,  { postag_embedding_dim });
    ner_lookup_param = m->add_lookup_parameters(ner_embedding_dict_size, { ner_embedding_dim });

    init_score_lookup_param = m->add_lookup_parameters(ner_embedding_dict_size, { 1 });
    trans_score_lookup_param = m->add_lookup_parameters(ner_embedding_dict_size * ner_embedding_dict_size , { 1 });
}

void NERCRFModel::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- Model Structure Info ----------------\n"
        << "vocabulary size : " << word_embedding_dict_size << " with dimention : " << word_embedding_dim << "\n"
        << "postag number : " << postag_embedding_dict_size << " with dimension : " << postag_embedding_dim << "\n"
        << "ner tag number : " << ner_embedding_dict_size << " with dimension : " << ner_embedding_dim << "\n"
        << "bilstm stacked layer num : " << nr_lstm_stacked_layer << " , x dim  : " << lstm_x_dim << " , h dim : " << lstm_h_dim << "\n"
        << "emit hidden layer dimension : " << emit_hidden_layer_dim ;
}

Expression NERCRFModel::viterbi_train(ComputationGraph *p_cg, 
    const IndexSeq *p_sent, const IndexSeq *p_postag_seq,
    const IndexSeq *p_ner_seq , 
    float dropout_rate , 
    Stat *p_stat)
{
    const unsigned sent_len = p_sent->size();
    ComputationGraph &cg = *p_cg;
    // New graph , ready for new sentence
    merge_input_layer->new_graph(cg);
    bilstm_layer->new_graph(cg);
    emit_hidden_layer->new_graph(cg);
    emit_output_layer->new_graph(cg);

    bilstm_layer->set_dropout(dropout_rate) ;
    bilstm_layer->start_new_sequence();
    // Some container
    vector<Expression> merge_dc_exp_cont(sent_len);
    vector<Expression> l2r_lstm_output_exp_cont; // for storing left to right lstm output(deepest hidden layer) expression for every timestep
    vector<Expression> r2l_lstm_output_exp_cont; // right to left 

    // 1. build input , and merge
    for (unsigned i = 0; i < sent_len; ++i)
    {
        Expression word_lookup_exp = lookup(cg, words_lookup_param, p_sent->at(i));
        Expression postag_lookup_exp = lookup(cg, postag_lookup_param, p_postag_seq->at(i));
        Expression merge_dc_exp = merge_input_layer->build_graph(word_lookup_exp, postag_lookup_exp);
        merge_dc_exp_cont[i] = rectify(merge_dc_exp); // rectify for merged expression
    }

    // 2. Build bi-lstm
    bilstm_layer->build_graph(merge_dc_exp_cont, l2r_lstm_output_exp_cont, r2l_lstm_output_exp_cont);

    // viterbi data preparation
    vector<Expression> all_ner_exp_cont(ner_embedding_dict_size);
    vector<Expression> init_score(ner_embedding_dict_size);
    vector<Expression> trans_score(ner_embedding_dict_size * ner_embedding_dict_size);
    vector<vector<Expression>> emit_score(sent_len, vector<Expression>(ner_embedding_dict_size));
    vector<Expression> cur_score_exp_cont(ner_embedding_dict_size),
        pre_score_exp_cont(ner_embedding_dict_size);
    shared_ptr<vector<vector<size_t>>> p_path_matrix ;
    //   - allocate memory only when using p_stat
    if (p_stat) p_path_matrix.reset(new vector<vector<size_t>>(sent_len, vector<size_t>(ner_embedding_dict_size)));
    Expression gold_score_exp;
    // init all_ner_exp_cont , init_score together
    for (size_t ner_idx = 0; ner_idx < ner_embedding_dict_size; ++ner_idx)
    {
        all_ner_exp_cont[ner_idx] = lookup(cg, ner_lookup_param, ner_idx);
        init_score[ner_idx] = lookup(cg, init_score_lookup_param, ner_idx);
    }
    // init translation score
    for (size_t flat_idx = 0; flat_idx < ner_embedding_dict_size * ner_embedding_dict_size; ++flat_idx)
    {
        trans_score[flat_idx] = lookup(cg, trans_score_lookup_param, flat_idx);
    }
    // init emit score
    for (size_t time_step = 0; time_step < sent_len; ++time_step)
    {
        for (size_t ner_idx = 0; ner_idx < ner_embedding_dict_size; ++ner_idx)
        {
            Expression emit_hidden_out_exp = emit_hidden_layer->build_graph(l2r_lstm_output_exp_cont[time_step],
                r2l_lstm_output_exp_cont[time_step], all_ner_exp_cont[ner_idx]);
            Expression non_linear_exp = rectify(emit_hidden_out_exp) ;
            Expression dropout_exp = dropout(non_linear_exp , dropout_rate) ;
            emit_score[time_step][ner_idx] = emit_output_layer->build_graph(dropout_exp);
        }
    }
    // viterbi docoding
    // 1. the time 0
    for (size_t ner_idx = 0; ner_idx < ner_embedding_dict_size; ++ner_idx)
    {
        // init_score + emit_score
        cur_score_exp_cont[ner_idx] = init_score[ner_idx] + emit_score[0][ner_idx];
    }
    gold_score_exp = cur_score_exp_cont[p_ner_seq->at(0)];
    // 2. the continues time
    for (size_t time_step = 1; time_step < sent_len; ++time_step)
    {
        swap(cur_score_exp_cont, pre_score_exp_cont);
        for (size_t cur_ner_idx = 0; cur_ner_idx < ner_embedding_dict_size; ++cur_ner_idx)
        {
            // for every possible trans
            vector<Expression> partial_score_exp_cont(ner_embedding_dict_size);
            for (size_t pre_ner_idx = 0; pre_ner_idx < ner_embedding_dict_size; ++pre_ner_idx)
            {
                size_t flatten_idx = pre_ner_idx * ner_embedding_dict_size + cur_ner_idx;
                // from-tag score + trans_score
                partial_score_exp_cont[pre_ner_idx] = pre_score_exp_cont[pre_ner_idx] +
                    trans_score[flatten_idx];
            }
            cur_score_exp_cont[cur_ner_idx] = logsumexp(partial_score_exp_cont) +
                emit_score[time_step][cur_ner_idx];
            if (p_stat)
            {
                size_t pre_ner_idx_with_max_score = 0;
                cnn::real max_score_value = as_scalar(cg.get_value(partial_score_exp_cont[pre_ner_idx_with_max_score]));
                for (size_t pre_ner_idx = 1; pre_ner_idx < ner_embedding_dict_size; ++pre_ner_idx)
                {
                    cnn::real score_value = as_scalar(cg.get_value(partial_score_exp_cont[pre_ner_idx]));
                    if (score_value > max_score_value)
                    {
                        max_score_value = score_value;
                        pre_ner_idx_with_max_score = pre_ner_idx;
                    }
                }
                (*p_path_matrix)[time_step][cur_ner_idx] = pre_ner_idx_with_max_score;
            }
        }
        // calc gold 
        size_t gold_trans_flatten_idx = p_ner_seq->at(time_step - 1) * ner_embedding_dict_size + p_ner_seq->at(time_step);
        gold_score_exp = gold_score_exp + trans_score[gold_trans_flatten_idx] +
            emit_score[time_step][p_ner_seq->at(time_step)];
    }
    Expression predict_score_exp = logsumexp(cur_score_exp_cont);

    // predict is the max-score of lattice
    // if totally correct , loss = 0 (predict_score = gold_score , that is , predict sequence equal to gold sequence)
    // else , loss = predict_score - gold_score
    Expression loss = predict_score_exp - gold_score_exp;

    if (p_stat)
    {
        Index end_predicted_idx = 0;
        cnn::real max_score_value = as_scalar(cg.get_value(cur_score_exp_cont[end_predicted_idx]));
        for (size_t ner_idx = 1; ner_idx < ner_embedding_dict_size; ++ner_idx)
        {
            cnn::real score_value = as_scalar(cg.get_value(cur_score_exp_cont[ner_idx]));
            if (score_value > max_score_value)
            {
                max_score_value = score_value;
                end_predicted_idx = ner_idx;
            }
        }
        ++p_stat->total_tags;
        if (end_predicted_idx == p_ner_seq->at(sent_len - 1)) ++p_stat->correct_tags;
        Index pre_predicted_tag = end_predicted_idx;
        for (unsigned backtrace_idx = sent_len - 1; backtrace_idx >= 1; --backtrace_idx)
        {
            pre_predicted_tag = (*p_path_matrix)[backtrace_idx][pre_predicted_tag];
            ++p_stat->total_tags;
            if (pre_predicted_tag == p_ner_seq->at(backtrace_idx - 1)) ++p_stat->correct_tags;
        }
    }
    return loss;
}

void NERCRFModel::viterbi_predict(ComputationGraph *p_cg, 
    const IndexSeq *p_sent, const IndexSeq *p_postag_seq ,
    IndexSeq *p_predict_ner_seq)
{
    // The main structure is just a copy from build_bilstm4tagging_graph2train! 
    const unsigned sent_len = p_sent->size();
    ComputationGraph &cg = *p_cg;
    // New graph , ready for new sentence 
    // New graph , ready for new sentence
    merge_input_layer->new_graph(cg);
    bilstm_layer->new_graph(cg);
    emit_hidden_layer->new_graph(cg);
    emit_output_layer->new_graph(cg);

    bilstm_layer->disable_dropout() ;
    bilstm_layer->start_new_sequence();

    // Some container
    vector<Expression> merge_dc_exp_cont(sent_len);
    vector<Expression> l2r_lstm_output_exp_cont; // for storing left to right lstm output(deepest hidden layer) expression for every timestep
    vector<Expression> r2l_lstm_output_exp_cont; // right to left                                                  
    
    // 1. get word embeddings for sent 
    for (unsigned i = 0; i < sent_len; ++i)
    {
        Expression word_lookup_exp = lookup(cg, words_lookup_param, p_sent->at(i));
        Expression postag_lookup_exp = lookup(cg, postag_lookup_param, p_postag_seq->at(i));
        Expression merge_dc_exp = merge_input_layer->build_graph(word_lookup_exp, postag_lookup_exp);
        merge_dc_exp_cont[i] = rectify(merge_dc_exp); // rectify for merged expression
    }
    // 2. calc Expression of every timestep of BI-LSTM
    bilstm_layer->build_graph(merge_dc_exp_cont, l2r_lstm_output_exp_cont, r2l_lstm_output_exp_cont);
   
    //viterbi - preparing score
    vector<Expression> all_ner_exp_cont(ner_embedding_dict_size);
    vector<cnn::real> init_score(ner_embedding_dict_size);
    vector<cnn::real> trans_score(ner_embedding_dict_size * ner_embedding_dict_size);
    vector<vector<cnn::real>> emit_score(sent_len, vector<cnn::real>(ner_embedding_dict_size));

    // get init score
    for (size_t ner_idx = 0; ner_idx < ner_embedding_dict_size; ++ner_idx)
    {
        Expression init_exp = lookup(cg, init_score_lookup_param, ner_idx);
        init_score[ner_idx] = as_scalar(cg.get_value(init_exp));
    }
    // get trans score
    for (size_t flat_idx = 0; flat_idx < ner_embedding_dict_size * ner_embedding_dict_size; ++flat_idx)
    {
        Expression trans_exp = lookup(cg, trans_score_lookup_param, flat_idx);
        trans_score[flat_idx] = as_scalar(cg.get_value(trans_exp));
    }
    // get emit score
    for (size_t ner_idx = 0; ner_idx < ner_embedding_dict_size; ++ner_idx)
    {
        all_ner_exp_cont[ner_idx] = lookup(cg, ner_lookup_param, ner_idx);
    }
    for (size_t time_step = 0; time_step < sent_len; ++time_step)
    {
        for (size_t ner_idx = 0; ner_idx < ner_embedding_dict_size; ++ner_idx)
        {
            Expression emit_hidden_out_exp = emit_hidden_layer->build_graph(l2r_lstm_output_exp_cont[time_step],
                r2l_lstm_output_exp_cont[time_step], all_ner_exp_cont[ner_idx]);
            Expression emit_score_exp = emit_output_layer->build_graph(rectify(emit_hidden_out_exp));
            emit_score[time_step][ner_idx] = as_scalar(cg.get_value(emit_score_exp));
        }

    }
    // viterbi - process
    vector<vector<size_t>> path_matrix(sent_len, vector<size_t>(ner_embedding_dict_size));
    vector<cnn::real> current_scores(ner_embedding_dict_size); 
    // time 0
    for (size_t ner_idx = 0; ner_idx < ner_embedding_dict_size; ++ner_idx)
    {
        current_scores[ner_idx] = init_score[ner_idx] + emit_score[0][ner_idx];
    }
    // continues time
    vector<cnn::real>  pre_timestep_scores(current_scores.size());
    for (size_t time_step = 1; time_step < sent_len; ++time_step)
    {
        swap(pre_timestep_scores, current_scores); // move current_score -> pre_timestep_score
        for (size_t ner_idx = 0; ner_idx < ner_embedding_dict_size; ++ner_idx)
        {
            size_t pre_tag_with_max_score = 0;
            cnn::real max_score = pre_timestep_scores[pre_tag_with_max_score] + 
                trans_score[pre_tag_with_max_score * ner_embedding_dict_size + ner_idx];
            for (size_t pre_ner_idx = 1 ; pre_ner_idx < ner_embedding_dict_size; ++pre_ner_idx)
            {
                size_t flat_idx = pre_ner_idx * ner_embedding_dict_size + ner_idx;
                cnn::real score = pre_timestep_scores[pre_ner_idx] + trans_score[flat_idx];
                if (score > max_score)
                {
                    pre_tag_with_max_score = pre_ner_idx;
                    max_score = score;
                }
            }
            path_matrix[time_step][ner_idx] = pre_tag_with_max_score;
            current_scores[ner_idx] = max_score + emit_score[time_step][ner_idx];
        }
    }
    // get result 
    IndexSeq tmp_predict_ner_seq(sent_len);
    Index end_predicted_idx = distance(current_scores.begin(), max_element(current_scores.begin(), current_scores.end()));
    tmp_predict_ner_seq[sent_len - 1] = end_predicted_idx;
    Index pre_predicted_idx = end_predicted_idx;
    for (size_t reverse_idx = sent_len - 1; reverse_idx >= 1; --reverse_idx)
    {
        pre_predicted_idx = path_matrix[reverse_idx][pre_predicted_idx]; // backtrace
        tmp_predict_ner_seq[reverse_idx - 1] = pre_predicted_idx;
    }
    swap(tmp_predict_ner_seq, *p_predict_ner_seq);
}


} // end of namespace
