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
    bilstm_layer(nullptr),
    merge_hidden_layer(nullptr),
    emit_layer(nullptr),
    word_dict_wrapper(word_dict)
{}

BILSTMCRFModel4POSTAG::~BILSTMCRFModel4POSTAG()
{
    if (m) delete m;
    if (bilstm_layer) delete bilstm_layer;
    if (merge_hidden_layer) delete merge_hidden_layer;
    if (emit_layer) delete emit_layer;
}

void BILSTMCRFModel4POSTAG::build_model_structure()
{
    assert(word_dict.is_frozen() && postag_dict.is_frozen()); // Assert all frozen
    m = new Model();
    bilstm_layer = new BILSTMLayer(m, nr_lstm_stacked_layer, word_embedding_dim , lstm_h_dim);
    merge_hidden_layer = new Merge3Layer(m, lstm_h_dim, lstm_h_dim, postag_embedding_dim, merge_hidden_dim);
    emit_layer = new DenseLayer(m, merge_hidden_dim, 1);

    words_lookup_param = m->add_lookup_parameters(word_dict_size, { word_embedding_dim });
    postags_lookup_param = m->add_lookup_parameters(postag_dict_size , { postag_embedding_dim });

    trans_score_lookup_param = m->add_lookup_parameters(postag_dict_size * postag_dict_size, { 1 });
    init_score_lookup_param = m->add_lookup_parameters(postag_dict_size, { 1 });

}

void BILSTMCRFModel4POSTAG::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- Model Structure Info ----------------\n"
        << "vocabulary size : " << word_dict_size << " with dimention : " << word_embedding_dim << "\n"
        << "bilstm stacked layer num : " << nr_lstm_stacked_layer << " , x dim  : " << word_embedding_dim << " , h dim : " << lstm_h_dim << "\n"
        << "postag num : " << postag_dict_size << " with dimension : " << postag_embedding_dim << "\n" 
        << "merge hidden dimension : " << merge_hidden_dim ;
}

Expression BILSTMCRFModel4POSTAG::viterbi_train(ComputationGraph *p_cg, 
    const IndexSeq *p_sent, const IndexSeq *p_tag_seq,
    Stat *p_stat)
{
    const unsigned sent_len = p_sent->size();
    ComputationGraph &cg = *p_cg;
    // New graph , ready for new sentence
    bilstm_layer->new_graph(cg);
    merge_hidden_layer->new_graph(cg);
    emit_layer->new_graph(cg);

    bilstm_layer->start_new_sequence();

    // Some container
    vector<Expression> word_exp_cont(sent_len);
    vector<Expression> l2r_lstm_output_exp_cont; // for storing left to right lstm output(deepest hidden layer) expression for every timestep
    vector<Expression> r2l_lstm_output_exp_cont; // right to left 

    // 1. build input , and merge
    for (unsigned i = 0; i < sent_len; ++i)
    {
        word_exp_cont[i] = lookup(cg, words_lookup_param, p_sent->at(i));
    }

    // 2. Build bi-lstm
    bilstm_layer->build_graph(word_exp_cont, l2r_lstm_output_exp_cont, r2l_lstm_output_exp_cont);

    // viterbi data preparation
    vector<Expression> all_postag_exp_cont(postag_dict_size);
    vector<Expression> init_score(postag_dict_size);
    vector<Expression> trans_score(postag_dict_size * postag_dict_size);
    vector<vector<Expression>> emit_score(sent_len, vector<Expression>(postag_dict_size));
    vector<Expression> cur_score_exp_cont(postag_dict_size)  ,
                       pre_score_exp_cont(postag_dict_size) ;
    vector<vector<size_t>> *p_path_matrix = nullptr ;
        // allocate memory only when using p_stat
    if(p_stat) p_path_matrix = new vector<vector<size_t>>(sent_len , vector<size_t>(postag_dict_size)) ;
    Expression gold_score_exp ;
    
    // init all_postag_exp_cont , init_score together
    for (size_t postag_idx = 0; postag_idx < postag_dict_size; ++postag_idx)
    {
        all_postag_exp_cont[postag_idx] = lookup(cg, postags_lookup_param, postag_idx);
        init_score[postag_idx] = lookup(cg, init_score_lookup_param, postag_idx);
    }
    // init translation score
    for (size_t flat_idx = 0; flat_idx < postag_dict_size * postag_dict_size; ++flat_idx)
    {
        trans_score[flat_idx] = lookup(cg, trans_score_lookup_param, flat_idx);
    }
    // init emit score
    for (size_t time_step = 0; time_step < sent_len; ++time_step)
    {
        for (size_t postag_idx = 0; postag_idx < postag_dict_size; ++postag_idx)
        {
            Expression hidden_out_exp = merge_hidden_layer->build_graph(l2r_lstm_output_exp_cont[time_step],
                r2l_lstm_output_exp_cont[time_step], all_postag_exp_cont[postag_idx]);
            emit_score[time_step][postag_idx] = emit_layer->build_graph( rectify(hidden_out_exp) );
        }
    }
    // viterbi docoding
    // 1. the time 0
    for (size_t postag_idx = 0; postag_idx < postag_dict_size; ++postag_idx)
    {
        // init_score + emit_score
        cur_score_exp_cont[postag_idx] = init_score[postag_idx] + emit_score[0][postag_idx];
    }
    gold_score_exp = cur_score_exp_cont[p_tag_seq->at(0)];
    // 2. the continues time
    for(size_t time_step = 1; time_step < sent_len; ++time_step)
    {
        // for every tag(to-tag)
        swap(cur_score_exp_cont , pre_score_exp_cont) ;
        for (size_t cur_postag_idx = 0; cur_postag_idx < postag_dict_size; ++cur_postag_idx)
        {
            // for every possible trans
            vector<Expression> partial_score_exp_cont(postag_dict_size);
            for (size_t pre_postag_idx = 0; pre_postag_idx < postag_dict_size; ++pre_postag_idx)
            {
                size_t flatten_idx = pre_postag_idx * postag_dict_size + cur_postag_idx ;
                // from-tag score + trans_score
                partial_score_exp_cont[pre_postag_idx] = pre_score_exp_cont[pre_postag_idx] +
                    trans_score[flatten_idx];
            }
            cur_score_exp_cont[cur_postag_idx] = logsumexp(partial_score_exp_cont) + 
                                                 emit_score[time_step][cur_postag_idx];
            if (p_stat)
            {
                size_t pre_tag_value = 0;
                cnn::real max_score_value = as_scalar(cg.get_value(partial_score_exp_cont[pre_tag_value]));
                for (size_t pre_tag_idx = 1; pre_tag_idx < postag_dict_size; ++pre_tag_idx)
                {
                    cnn::real score_value = as_scalar(cg.get_value(partial_score_exp_cont[pre_tag_idx]));
                    if (score_value > max_score_value)
                    {
                        max_score_value = score_value;
                        pre_tag_value = pre_tag_idx;
                    }
                }
                (*p_path_matrix)[time_step][cur_postag_idx] = pre_tag_value;
            }
        }
        // calc gold 
        size_t gold_trans_flatten_idx = p_tag_seq->at(time_step - 1) * postag_dict_size + p_tag_seq->at(time_step);
        gold_score_exp = gold_score_exp + trans_score[gold_trans_flatten_idx] + 
                                          emit_score[time_step][p_tag_seq->at(time_step)];
    }
    Expression predict_score_exp = logsumexp(cur_score_exp_cont);

    // predict is the max-score of lattice
    // if totally correct , loss = 0 (predict_score = gold_score , that is , predict sequence equal to gold sequence)
    // else , loss = predict_score - gold_score
    Expression loss =  predict_score_exp - gold_score_exp;
    if (p_stat)
    {
        Index end_predicted_idx = 0;
        cnn::real max_score_value = as_scalar(cg.get_value(cur_score_exp_cont[end_predicted_idx]));
        for (size_t postag_idx = 1; postag_idx < postag_dict_size; ++postag_idx)
        {
            cnn::real score_value = as_scalar(cg.get_value(cur_score_exp_cont[postag_idx]));
            if (score_value > max_score_value)
            {
               max_score_value = score_value;
               end_predicted_idx = postag_idx;
            }
        }
        ++p_stat->total_tags;
        if (end_predicted_idx == p_tag_seq->at(sent_len - 1)) ++p_stat->correct_tags;
        Index pre_predicted_tag = end_predicted_idx;
        for (unsigned backtrace_idx = sent_len - 1; backtrace_idx >= 1; --backtrace_idx)
        {
            pre_predicted_tag = (*p_path_matrix)[backtrace_idx][pre_predicted_tag];
            ++p_stat->total_tags;
            if (pre_predicted_tag == p_tag_seq->at(backtrace_idx - 1)) ++p_stat->correct_tags;
        }
        if(p_path_matrix) delete p_path_matrix ;
    }
    return loss;
}

void BILSTMCRFModel4POSTAG::viterbi_predict(ComputationGraph *p_cg, 
    const IndexSeq *p_sent, IndexSeq *p_predict_tag_seq)
{
    // The main structure is just a copy from build_bilstm4tagging_graph2train! 
    const unsigned sent_len = p_sent->size();
    ComputationGraph &cg = *p_cg;
    // New graph , ready for new sentence 
    bilstm_layer->new_graph(cg);
    merge_hidden_layer->new_graph(cg);
    emit_layer->new_graph(cg);

    bilstm_layer->start_new_sequence();

    // Some container
    vector<Expression> word_exp_cont(sent_len);
    vector<Expression> l2r_lstm_output_exp_cont; // for storing left to right lstm output(deepest hidden layer) expression for every timestep
    vector<Expression> r2l_lstm_output_exp_cont; // right to left                                                  
    
    // 1. get word embeddings for sent 
    for (unsigned i = 0; i < sent_len; ++i)
    {
        word_exp_cont[i] = lookup(cg, words_lookup_param, p_sent->at(i));
    }
    // 2. calc Expression of every timestep of BI-LSTM

    bilstm_layer->build_graph(word_exp_cont, l2r_lstm_output_exp_cont, r2l_lstm_output_exp_cont);
    
    

    //viterbi - preparing score
    vector<Expression> all_postag_exp_cont(postag_dict_size);
    vector<cnn::real> init_score(postag_dict_size);
    vector<cnn::real> trans_score(postag_dict_size * postag_dict_size);
    vector<vector<cnn::real>> emit_score(sent_len, vector<cnn::real>(postag_dict_size));

    // get init score
    for (size_t postag_idx = 0; postag_idx < postag_dict_size; ++postag_idx)
    {
        Expression init_exp = lookup(cg, init_score_lookup_param, postag_idx);
        init_score[postag_idx] = as_scalar(cg.get_value(init_exp));
    }
    // get trans score
    for (size_t flat_idx = 0; flat_idx < postag_dict_size * postag_dict_size; ++flat_idx)
    {
        Expression trans_exp = lookup(cg, trans_score_lookup_param, flat_idx);
        trans_score[flat_idx] = as_scalar(cg.get_value(trans_exp));
    }
    // get emit score
    for (size_t postag_idx = 0; postag_idx < postag_dict_size; ++postag_idx)
    {
        all_postag_exp_cont[postag_idx] = lookup(cg, postags_lookup_param, postag_idx);
    }
    for (size_t time_step = 0; time_step < sent_len; ++time_step)
    {
        for (size_t postag_idx = 0; postag_idx < postag_dict_size; ++postag_idx)
        {
            Expression hidden_out_exp = merge_hidden_layer->build_graph(l2r_lstm_output_exp_cont[time_step],
                r2l_lstm_output_exp_cont[time_step], all_postag_exp_cont[postag_idx]);
            Expression emit_score_exp = emit_layer->build_graph(rectify(hidden_out_exp));
            emit_score[time_step][postag_idx] =  as_scalar(cg.get_value(emit_score_exp));
        }
            
    }
    // viterbi - process
    vector<vector<size_t>> path_matrix(sent_len, vector<size_t>(postag_dict_size));
    vector<cnn::real> current_scores(postag_dict_size);
    // time 0
    for (size_t postag_idx = 0; postag_idx < postag_dict_size; ++postag_idx)
    {
        current_scores[postag_idx] = init_score[postag_idx] + emit_score[0][postag_idx];
    }
    // continues time
    vector<cnn::real>  pre_timestep_scores(current_scores);
    for (size_t time_step = 1; time_step < sent_len; ++time_step)
    {
        swap(pre_timestep_scores, current_scores); // move current_score -> pre_timestep_score
        for (size_t postag_idx = 0; postag_idx < postag_dict_size; ++postag_idx)
        {
            size_t pre_tag_with_max_score = 0;
            cnn::real max_score = pre_timestep_scores[0] + trans_score[postag_idx]; // 0*postag_dict_size + postag_idx
            for (size_t pre_tag_idx = 1; pre_tag_idx < postag_dict_size; ++pre_tag_idx)
            {
                size_t flat_idx = pre_tag_idx * postag_dict_size + postag_idx;
                cnn::real score = pre_timestep_scores[pre_tag_idx] + trans_score[flat_idx];
                if (score > max_score)
                {
                    pre_tag_with_max_score = pre_tag_idx;
                    max_score = score;
                }
            }
            path_matrix[time_step][postag_idx] = pre_tag_with_max_score;
            current_scores[postag_idx] = max_score + emit_score[time_step][postag_idx];
        }
    }
    // get result 
    IndexSeq tmp_predict_tag_seq(sent_len);
    Index end_predicted_idx = distance(current_scores.begin() ,  max_element(current_scores.begin(), current_scores.end()));
    tmp_predict_tag_seq[sent_len - 1] = end_predicted_idx;
    Index pre_predicted_idx = end_predicted_idx;
    for (size_t reverse_idx = sent_len - 1; reverse_idx >= 1 ; --reverse_idx)
    {
        pre_predicted_idx = path_matrix[reverse_idx][pre_predicted_idx]; // backtrace
        tmp_predict_tag_seq[reverse_idx-1] = pre_predicted_idx;

    }
    swap(tmp_predict_tag_seq, *p_predict_tag_seq);
}


} // end of namespace
