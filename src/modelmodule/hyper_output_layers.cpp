#include "hyper_output_layers.h"

namespace slnn{

/************ OutputBase ***************/

OutputBase::OutputBase(dynet::real dropout_rate, NonLinearFunc *nonlinear_func) : 
    dropout_rate(dropout_rate),
    nonlinear_func(nonlinear_func)
{}

OutputBase::~OutputBase(){} // base class should implement deconstructor , even for pure virtual function

/************ SimpleOutput *************/

SimpleOutput::SimpleOutput(dynet::Model *m, unsigned input_dim1, unsigned input_dim2 ,
    unsigned hidden_dim, unsigned output_dim , 
    dynet::real dropout_rate,
    NonLinearFunc *nonlinear_func)
    : OutputBase(dropout_rate, nonlinear_func),
    hidden_layer(m , input_dim1 , input_dim2 , hidden_dim) ,
    output_layer(m , hidden_dim , output_dim) 
{}

SimpleOutput::~SimpleOutput() {};

/*************** PretagOutput **************/

PretagOutput::PretagOutput(dynet::Model *m,
    unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
    unsigned hidden_dim, unsigned output_dim , 
    dynet::real dropout_rate, NonLinearFunc *nonlinear_func)
    :OutputBase(dropout_rate, nonlinear_func),
    hidden_layer(m , input_dim1 , input_dim2 , tag_embedding_dim , hidden_dim) ,
    output_layer(m , hidden_dim , output_dim) ,
    tag_lookup_param(m->add_lookup_parameters(output_dim , {tag_embedding_dim})) ,
    TAG_SOS(m->add_parameters({tag_embedding_dim})) 
{}

PretagOutput::~PretagOutput(){} 

/****************** CRFOutput ****************/
CRFOutput::CRFOutput(dynet::Model *m,
    unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
    unsigned hidden_dim,
    unsigned tag_num,
    dynet::real dropout_rate ,
    NonLinearFunc *nonlinear_func)
    :OutputBase(dropout_rate, nonlinear_func),
    hidden_layer(m , input_dim1 , input_dim2 , tag_embedding_dim , hidden_dim) ,
    emit_layer(m , hidden_dim , 1) ,
    tag_lookup_param(m->add_lookup_parameters(tag_num , {tag_embedding_dim})) ,
    trans_score_lookup_param(m->add_lookup_parameters(tag_num * tag_num , {1})) ,
    init_score_lookup_param(m->add_lookup_parameters(tag_num , {1})) ,
    tag_num(tag_num) 
{}

CRFOutput::~CRFOutput(){} 

dynet::expr::Expression
CRFOutput::build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
    const std::vector<dynet::expr::Expression> &expr_cont2,
    const IndexSeq &gold_seq)
{
    size_t len = expr_cont1.size() ;
    // viterbi data preparation
    std::vector<dynet::expr::Expression> all_tag_expr_cont(tag_num);
    std::vector<dynet::expr::Expression> init_score(tag_num);
    std::vector<dynet::expr::Expression> trans_score(tag_num * tag_num);
    std::vector<std::vector<dynet::expr::Expression>> emit_score(len,
        std::vector<dynet::expr::Expression>(tag_num));
    std::vector<dynet::expr::Expression> cur_score_expr_cont(tag_num),
        pre_score_expr_cont(tag_num);
    std::vector<dynet::expr::Expression> gold_score_expr_cont(len) ;
    // init tag expr , init score
    for( size_t i = 0; i < tag_num ; ++i )
    {
        all_tag_expr_cont[i] = dynet::expr::lookup(*pcg, tag_lookup_param, i);
        init_score[i] = dynet::expr::lookup(*pcg, init_score_lookup_param, i);
    }
    // init translation score
    for( size_t flat_idx = 0; flat_idx < tag_num * tag_num ; ++flat_idx )
    {
        trans_score[flat_idx] = lookup(*pcg, trans_score_lookup_param, flat_idx);
    }
    // init emit score
    for( size_t time_step = 0; time_step < len; ++time_step )
    {
        for( size_t i = 0; i < tag_num; ++i )
        {
            dynet::expr::Expression hidden_out_expr = hidden_layer.build_graph(expr_cont1[time_step],
                expr_cont2[time_step], all_tag_expr_cont[i]);
            dynet::expr::Expression non_linear_expr = (*nonlinear_func)(hidden_out_expr) ;
            dynet::expr::Expression dropout_expr = dropout(non_linear_expr, dropout_rate) ;
            emit_score[time_step][i] = emit_layer.build_graph(dropout_expr);
        }
    }
    // viterbi docoding
    // 1. the time 0
    for( size_t i = 0; i < tag_num ; ++i )
    {
        // init_score + emit_score
        cur_score_expr_cont[i] = init_score[i] + emit_score[0][i];
    }
    gold_score_expr_cont[0] = cur_score_expr_cont[gold_seq.at(0)];
    // 2. the continues time
    for( size_t time_step = 1; time_step < len; ++time_step )
    {
        std::swap(cur_score_expr_cont, pre_score_expr_cont);
        for( size_t cur_idx = 0; cur_idx < tag_num ; ++cur_idx )
        {
            // for every possible trans
            std::vector<dynet::expr::Expression> partial_score_expr_cont(tag_num);
            for( size_t pre_idx = 0; pre_idx < tag_num; ++pre_idx )
            {
                size_t flatten_idx = pre_idx * tag_num + cur_idx;
                // from-tag score + trans_score
                partial_score_expr_cont[pre_idx] = pre_score_expr_cont[pre_idx] +
                    trans_score[flatten_idx];
            }
            cur_score_expr_cont[cur_idx] = dynet::expr::logsumexp(partial_score_expr_cont) +
                emit_score[time_step][cur_idx];
        }
        // calc gold 
        size_t gold_trans_flatten_idx = gold_seq.at(time_step - 1) * tag_num + gold_seq.at(time_step);
        gold_score_expr_cont[time_step] = trans_score[gold_trans_flatten_idx] +
            emit_score[time_step][gold_seq.at(time_step)];
    }
    dynet::expr::Expression predict_score_expr = dynet::expr::logsumexp(cur_score_expr_cont);

    // if totally correct , loss = 0 (predict_score = gold_score , that is , predict sequence equal to gold sequence)
    // else , loss = predict_score - gold_score
    dynet::expr::Expression loss = predict_score_expr - dynet::expr::sum(gold_score_expr_cont);
    return loss;
}

void CRFOutput::build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
    const std::vector<dynet::expr::Expression> &expr_cont2,
    IndexSeq &pred_seq)
{
    size_t len = expr_cont1.size() ;
    // viterbi data preparation
    std::vector<dynet::expr::Expression> all_tag_expr_cont(tag_num);
    std::vector<dynet::real> init_score(tag_num);
    std::vector < dynet::real> trans_score(tag_num * tag_num);
    std::vector<std::vector<dynet::real>> emit_score(len, std::vector<dynet::real>(tag_num));
    // get initial score
    for( size_t i = 0 ; i < tag_num ; ++i )
    {
        dynet::expr::Expression init_score_expr = dynet::expr::lookup(*pcg, init_score_lookup_param, i);
        init_score[i] = dynet::as_scalar(pcg->get_value(init_score_expr)) ;
    }
    // get translation score
    for( size_t flat_idx = 0; flat_idx < tag_num * tag_num ; ++flat_idx )
    {
        dynet::expr::Expression trans_score_expr = lookup(*pcg, trans_score_lookup_param, flat_idx);
        trans_score[flat_idx] = dynet::as_scalar(pcg->get_value(trans_score_expr)) ;
    }
    // get emit score
    for( size_t i = 0; i < tag_num ; ++i )
    {
        all_tag_expr_cont[i] = dynet::expr::lookup(*pcg, tag_lookup_param, i);
    }
    for( size_t time_step = 0; time_step < len; ++time_step )
    {
        for( size_t i = 0; i < tag_num; ++i )
        {
            dynet::expr::Expression hidden_out_expr = hidden_layer.build_graph(expr_cont1[time_step],
                expr_cont2[time_step], all_tag_expr_cont[i]);
            dynet::expr::Expression non_linear_expr = (*nonlinear_func)(hidden_out_expr) ;
            dynet::expr::Expression emit_expr = emit_layer.build_graph(non_linear_expr);
            emit_score[time_step][i] = dynet::as_scalar( pcg->get_value(emit_expr) );
        }
    }
    // viterbi - process
    std::vector<std::vector<size_t>> path_matrix(len, std::vector<size_t>(tag_num));
    std::vector<dynet::real> current_scores(tag_num); 
    // time 0
    for (size_t i = 0; i < tag_num ; ++i)
    {
        current_scores[i] = init_score[i] + emit_score[0][i];
    }
    // continues time
    std::vector<dynet::real>  pre_timestep_scores(tag_num);
    for (size_t time_step = 1; time_step < len ; ++time_step)
    {
        std::swap(pre_timestep_scores, current_scores); // move current_score -> pre_timestep_score
        for (size_t i = 0; i < tag_num ; ++i )
        {
            size_t pre_tag_with_max_score = 0;
            dynet::real max_score = pre_timestep_scores[pre_tag_with_max_score] + 
                trans_score[pre_tag_with_max_score * tag_num + i];
            for (size_t pre_i = 1 ; pre_i < tag_num ; ++pre_i)
            {
                size_t flat_idx = pre_i * tag_num + pre_i ;
                dynet::real score = pre_timestep_scores[pre_i] + trans_score[flat_idx];
                if (score > max_score)
                {
                    pre_tag_with_max_score = pre_i;
                    max_score = score;
                }
            }
            path_matrix[time_step][i] = pre_tag_with_max_score;
            current_scores[i] = max_score + emit_score[time_step][i];
        }
    }
    // get result 
    IndexSeq tmp_predict_ner_seq(len);
    Index end_predicted_idx = std::distance(current_scores.cbegin(),
        max_element(current_scores.cbegin(), current_scores.cend()));
    tmp_predict_ner_seq[len - 1] = end_predicted_idx;
    Index pre_predicted_idx = end_predicted_idx;
    for (size_t reverse_idx = len - 1; reverse_idx >= 1; --reverse_idx)
    {
        pre_predicted_idx = path_matrix[reverse_idx][pre_predicted_idx]; // backtrace
        tmp_predict_ner_seq[reverse_idx - 1] = pre_predicted_idx;
    }
    std::swap(tmp_predict_ner_seq, pred_seq);
}


/****************
 * Bare Output base class.
 * 
 *****************/

BareOutputBase::~BareOutputBase(){}

/*********************************
 * Bare Output : [Simple] Bare Output 
 *********************************/
SimpleBareOutput::SimpleBareOutput(dynet::Model *m, unsigned input_dim, unsigned output_dim)
    :softmax_layer(m, input_dim, output_dim)
{}


/*********************
 * crf output class.
 *
 **/

CrfBareOutput::CrfBareOutput(dynet::Model *m, unsigned input_dim, unsigned output_dim)
    :tagdict_sz(output_dim),
    state_score_layer(m, input_dim, output_dim),
    init_score_lookup_param(m->add_lookup_parameters(output_dim, { 1 })),
    transition_score_lookup_param(m->add_lookup_parameters(output_dim * output_dim, {1}))
{}


dynet::expr::Expression
CrfBareOutput::build_output_loss(const std::vector<dynet::expr::Expression>& input_expr_seq,
    const std::vector<Index>& gold_tag_seq)
{
    unsigned seq_len = input_expr_seq.size();
    if( seq_len == 0 ){ return dynet::expr::Expression(); }
    // get init score expr
    std::vector<dynet::expr::Expression> init_score_expr(tagdict_sz);
    for( unsigned i = 0; i < tagdict_sz; ++i )
    {
        init_score_expr[i] = dynet::expr::lookup(*pcg, init_score_lookup_param, i);
    }
    // get trans score expr
    std::vector<dynet::expr::Expression> transition_score_expr(tagdict_sz * tagdict_sz);
    for( unsigned flat_idx = 0; flat_idx < tagdict_sz * tagdict_sz; ++flat_idx )
    {
        transition_score_expr[flat_idx] = dynet::expr::lookup(*pcg, transition_score_lookup_param, flat_idx);
    }
    // calculate all state score.
    std::vector<dynet::expr::Expression> state_score_expr(seq_len);
    for( unsigned i = 0; i < seq_len; ++i )
    {
        state_score_expr[i] = state_score_layer.build_graph(input_expr_seq[i]);
    }
    // viterbi
    std::vector<dynet::expr::Expression> cur_time_score_list(tagdict_sz),
        pre_time_score_list(tagdict_sz);
    std::vector<dynet::expr::Expression> gold_score_container;
    gold_score_container.reserve(2 * seq_len);
    // 1. init
    for( unsigned tagidx = 0; tagidx < tagdict_sz; ++tagidx )
    {
        cur_time_score_list[tagidx] = init_score_expr[tagidx] + dynet::expr::pick(state_score_expr[0], tagidx);
    }
    gold_score_container.push_back(init_score_expr[gold_tag_seq[0]]);
    gold_score_container.push_back(dynet::expr::pick(state_score_expr[0], gold_tag_seq[0]));
    // 2. continues
    for( unsigned t = 1; t < seq_len; ++t )
    {
        swap(cur_time_score_list, pre_time_score_list);
        for( unsigned tagidx = 0; tagidx < tagdict_sz; ++tagidx )
        {
            std::vector<dynet::expr::Expression> pre2cur_score(tagdict_sz);
            for( unsigned pre_tagidx = 0; pre_tagidx < tagdict_sz; ++pre_tagidx )
            {
                unsigned flat_idx = flat_transition_index(pre_tagidx, tagidx);
                pre2cur_score[pre_tagidx] = pre_time_score_list[pre_tagidx] + transition_score_expr[flat_idx];
            }
            cur_time_score_list[tagidx] = dynet::expr::logsumexp(pre2cur_score) + dynet::expr::pick(state_score_expr[t], tagidx);
           
        }
        unsigned gold_flat_idx = flat_transition_index(gold_tag_seq[t - 1], gold_tag_seq[t]);
        gold_score_container.push_back(transition_score_expr[gold_flat_idx]);
        gold_score_container.push_back(dynet::expr::pick(state_score_expr[t], gold_tag_seq[t]));
    }
    dynet::expr::Expression pred_score_expr = dynet::expr::logsumexp(cur_time_score_list),
        gold_score_expr = dynet::expr::sum(gold_score_container);
    return pred_score_expr - gold_score_expr;
}


void
CrfBareOutput::build_output(const std::vector<dynet::expr::Expression>& input_expr_seq,
    std::vector<Index>& pred_tagseq)
{
    unsigned seq_len = input_expr_seq.size();
    // init score
    std::vector<slnn::type::real> init_score_list(tagdict_sz);
    for( unsigned tagidx = 0; tagidx < tagdict_sz; ++tagidx )
    {
        auto cur_init_score_expr = dynet::expr::lookup(*pcg, init_score_lookup_param, tagidx);
        init_score_list[tagidx] = dynet::as_scalar(pcg->get_value(cur_init_score_expr));
    }
    // transition score
    std::vector<slnn::type::real> transition_score_list(tagdict_sz * tagdict_sz);
    for( unsigned flat_idx = 0; flat_idx < tagdict_sz * tagdict_sz; ++flat_idx )
    {
        auto cur_score_expr = dynet::expr::lookup(*pcg, transition_score_lookup_param, flat_idx);
        transition_score_list[flat_idx] = dynet::as_scalar(pcg->get_value(cur_score_expr));
    }
    // state score
    std::vector<std::vector<slnn::type::real>> state_score_list(seq_len);
    for( unsigned t = 0; t < seq_len; ++t )
    {
        auto cur_score_expr = state_score_layer.build_graph(input_expr_seq[t]);
        state_score_list[t] = dynet::as_vector(pcg->get_value(cur_score_expr));
    }
    // viterbi
    // - time 0 no trace back
    std::vector<std::vector<unsigned>> traceback_matrix(seq_len - 1, std::vector<unsigned>(tagdict_sz));
    std::vector<slnn::type::real> cur_time_score_list(tagdict_sz),
        pre_time_score_list(tagdict_sz);
    // - time 0
    for( unsigned tagidx = 0; tagidx < tagdict_sz; ++tagidx )
    {
        cur_time_score_list[tagidx] = init_score_list[tagidx] + state_score_list[0][tagidx];
    }
    // - continues
    for( unsigned t = 1; t < seq_len; ++t )
    {
        swap(cur_time_score_list, pre_time_score_list);
        for( unsigned tagidx = 0; tagidx < tagdict_sz; ++tagidx )
        {
            std::vector<slnn::type::real> pre2cur_tag_score_list(tagdict_sz);
            for( unsigned pre_tagidx = 0; pre_tagidx < tagdict_sz; ++pre_tagidx )
            {
                unsigned flat_idx = flat_transition_index(pre_tagidx, tagidx);
                pre2cur_tag_score_list[pre_tagidx] = pre_time_score_list[pre_tagidx] + transition_score_list[flat_idx];
            }
            auto max_pre2cur_score_iter = std::max_element(pre2cur_tag_score_list.begin(), pre2cur_tag_score_list.end());
            unsigned pre_tagidx_with_max_score = std::distance(pre2cur_tag_score_list.begin(), max_pre2cur_score_iter);
            cur_time_score_list[tagidx] = *max_pre2cur_score_iter + state_score_list[t][tagidx];
            traceback_matrix[t - 1][tagidx] = pre_tagidx_with_max_score;
        }
    }
    // - find the path with max score.
    auto end_tagidx_with_max_score = std::distance(cur_time_score_list.begin(), 
        std::max_element(cur_time_score_list.begin(), cur_time_score_list.end()) );
    std::vector<Index> tmp_pred_tagseq(seq_len);
    tmp_pred_tagseq.back() = end_tagidx_with_max_score;
    unsigned pre_tag = tmp_pred_tagseq.back();
    for( int t = seq_len - 1; t >= 1; --t )
    {
        pre_tag = traceback_matrix[t - 1][pre_tag]; // traceback matrix record from time 1.
        tmp_pred_tagseq[t - 1] = pre_tag;
    }
    swap(tmp_pred_tagseq, pred_tagseq);
}


/************* SoftmaxLayer ***********/

SoftmaxLayer::SoftmaxLayer(dynet::Model *model, unsigned input_dim, unsigned output_dim)
    :output_layer(model, input_dim, output_dim)
{}

/************* OutputBaseWithFeature ***************/

OutputBaseWithFeature::OutputBaseWithFeature(dynet::real dropout_rate, NonLinearFunc *nonlinear_func)
    :dropout_rate(dropout_rate),
    nonlinear_func(nonlinear_func)
{}

OutputBaseWithFeature::~OutputBaseWithFeature(){}

/************ SimpleOutputWithFeature *************/

SimpleOutputWithFeature::SimpleOutputWithFeature(dynet::Model *m, unsigned input_dim1, unsigned input_dim2,
    unsigned feature_dim, unsigned hidden_dim, unsigned output_dim,
    dynet::real dropout_rate,
    NonLinearFunc *nonlinear_func)
    :OutputBaseWithFeature(dropout_rate, nonlinear_func),
    hidden_layer(m, input_dim1, input_dim2, feature_dim, hidden_dim),
    output_layer(m, hidden_dim, output_dim)
{}

SimpleOutputWithFeature::~SimpleOutputWithFeature() {};

/*  PretagOutputWithFeature */
PretagOutputWithFeature::PretagOutputWithFeature(dynet::Model *m, unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2, 
    unsigned feature_dim,
    unsigned hidden_dim, unsigned output_dim , 
    dynet::real dropout_rate, NonLinearFunc *nonlinear_func)
    :OutputBaseWithFeature(dropout_rate, nonlinear_func),
    hidden_layer(m, input_dim1, input_dim2, feature_dim, tag_embedding_dim, hidden_dim),
    output_layer(m, hidden_dim, output_dim),
    tag_lookup_param(m->add_lookup_parameters(output_dim , {tag_embedding_dim})) ,
    TAG_SOS(m->add_parameters({tag_embedding_dim})) 
{}

PretagOutputWithFeature::~PretagOutputWithFeature(){}

/* CRFOutputWithFeature */

CRFOutputWithFeature::CRFOutputWithFeature(dynet::Model *m,
    unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
    unsigned feature_dim,
    unsigned hidden_dim,
    unsigned tag_num,
    dynet::real dropout_rate ,
    NonLinearFunc *nonlinear_func)
    :OutputBaseWithFeature(dropout_rate, nonlinear_func),
    hidden_layer(m , input_dim1 , input_dim2 , feature_dim, tag_embedding_dim , hidden_dim) ,
    emit_layer(m , hidden_dim , 1) ,
    tag_lookup_param(m->add_lookup_parameters(tag_num , {tag_embedding_dim})) ,
    trans_score_lookup_param(m->add_lookup_parameters(tag_num * tag_num , {1})) ,
    init_score_lookup_param(m->add_lookup_parameters(tag_num , {1})) ,
    tag_num(tag_num) 
{}

CRFOutputWithFeature::~CRFOutputWithFeature(){} 

/* Totally copy from CRFOutput , just add feature_expr */
dynet::expr::Expression
CRFOutputWithFeature::build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
    const std::vector<dynet::expr::Expression> &expr_cont2,
    const std::vector<dynet::expr::Expression> &feature_expr_cont,
    const IndexSeq &gold_seq)
{
    size_t len = expr_cont1.size() ;
    // viterbi data preparation
    std::vector<dynet::expr::Expression> all_tag_expr_cont(tag_num);
    std::vector<dynet::expr::Expression> init_score(tag_num);
    std::vector<dynet::expr::Expression> trans_score(tag_num * tag_num);
    std::vector<std::vector<dynet::expr::Expression>> emit_score(len,
        std::vector<dynet::expr::Expression>(tag_num));
    std::vector<dynet::expr::Expression> cur_score_expr_cont(tag_num),
        pre_score_expr_cont(tag_num);
    std::vector<dynet::expr::Expression> gold_score_expr_cont(len) ;
    // init tag expr , init score
    for( size_t i = 0; i < tag_num ; ++i )
    {
        all_tag_expr_cont[i] = dynet::expr::lookup(*pcg, tag_lookup_param, i);
        init_score[i] = dynet::expr::lookup(*pcg, init_score_lookup_param, i);
    }
    // init translation score
    for( size_t flat_idx = 0; flat_idx < tag_num * tag_num ; ++flat_idx )
    {
        trans_score[flat_idx] = lookup(*pcg, trans_score_lookup_param, flat_idx);
    }
    // init emit score
    for( size_t time_step = 0; time_step < len; ++time_step )
    {
        for( size_t i = 0; i < tag_num; ++i )
        {
            dynet::expr::Expression hidden_out_expr = hidden_layer.build_graph(expr_cont1.at(time_step),
                expr_cont2.at(time_step), feature_expr_cont.at(time_step), all_tag_expr_cont.at(i));
            dynet::expr::Expression non_linear_expr = (*nonlinear_func)(hidden_out_expr) ;
            dynet::expr::Expression dropout_expr = dropout(non_linear_expr, dropout_rate) ;
            emit_score[time_step][i] = emit_layer.build_graph(dropout_expr);
        }
    }
    // viterbi docoding
    // 1. the time 0
    for( size_t i = 0; i < tag_num ; ++i )
    {
        // init_score + emit_score
        cur_score_expr_cont[i] = init_score[i] + emit_score[0][i];
    }
    gold_score_expr_cont[0] = cur_score_expr_cont[gold_seq.at(0)];
    // 2. the continues time
    for( size_t time_step = 1; time_step < len; ++time_step )
    {
        std::swap(cur_score_expr_cont, pre_score_expr_cont);
        for( size_t cur_idx = 0; cur_idx < tag_num ; ++cur_idx )
        {
            // for every possible trans
            std::vector<dynet::expr::Expression> partial_score_expr_cont(tag_num);
            for( size_t pre_idx = 0; pre_idx < tag_num; ++pre_idx )
            {
                size_t flatten_idx = pre_idx * tag_num + cur_idx;
                // from-tag score + trans_score
                partial_score_expr_cont[pre_idx] = pre_score_expr_cont[pre_idx] +
                    trans_score[flatten_idx];
            }
            cur_score_expr_cont[cur_idx] = dynet::expr::logsumexp(partial_score_expr_cont) +
                emit_score[time_step][cur_idx];
        }
        // calc gold 
        size_t gold_trans_flatten_idx = gold_seq.at(time_step - 1) * tag_num + gold_seq.at(time_step);
        gold_score_expr_cont[time_step] = trans_score[gold_trans_flatten_idx] +
            emit_score[time_step][gold_seq.at(time_step)];
    }
    dynet::expr::Expression predict_score_expr = dynet::expr::logsumexp(cur_score_expr_cont);

    // if totally correct , loss = 0 (predict_score = gold_score , that is , predict sequence equal to gold sequence)
    // else , loss = predict_score - gold_score
    dynet::expr::Expression loss = predict_score_expr - dynet::expr::sum(gold_score_expr_cont);
    return loss;
}

/* Totally copy from CRFOutput */
void CRFOutputWithFeature::build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
    const std::vector<dynet::expr::Expression> &expr_cont2,
    const std::vector<dynet::expr::Expression> &feature_expr_cont,
    IndexSeq &pred_seq)
{
    size_t len = expr_cont1.size() ;
    // viterbi data preparation
    std::vector<dynet::expr::Expression> all_tag_expr_cont(tag_num);
    std::vector<dynet::real> init_score(tag_num);
    std::vector < dynet::real> trans_score(tag_num * tag_num);
    std::vector<std::vector<dynet::real>> emit_score(len, std::vector<dynet::real>(tag_num));
    // get initial score
    for( size_t i = 0 ; i < tag_num ; ++i )
    {
        dynet::expr::Expression init_score_expr = dynet::expr::lookup(*pcg, init_score_lookup_param, i);
        init_score[i] = dynet::as_scalar(pcg->get_value(init_score_expr)) ;
    }
    // get translation score
    for( size_t flat_idx = 0; flat_idx < tag_num * tag_num ; ++flat_idx )
    {
        dynet::expr::Expression trans_score_expr = lookup(*pcg, trans_score_lookup_param, flat_idx);
        trans_score[flat_idx] = dynet::as_scalar(pcg->get_value(trans_score_expr)) ;
    }
    // get emit score
    for( size_t i = 0; i < tag_num ; ++i )
    {
        all_tag_expr_cont[i] = dynet::expr::lookup(*pcg, tag_lookup_param, i);
    }
    for( size_t time_step = 0; time_step < len; ++time_step )
    {
        for( size_t i = 0; i < tag_num; ++i )
        {
            dynet::expr::Expression hidden_out_expr = hidden_layer.build_graph(expr_cont1.at(time_step),
                expr_cont2.at(time_step), feature_expr_cont.at(time_step), all_tag_expr_cont.at(i));
            dynet::expr::Expression non_linear_expr = (*nonlinear_func)(hidden_out_expr) ;
            dynet::expr::Expression emit_expr = emit_layer.build_graph(non_linear_expr);
            emit_score[time_step][i] = dynet::as_scalar( pcg->get_value(emit_expr) );
        }
    }
    // viterbi - process
    std::vector<std::vector<size_t>> path_matrix(len, std::vector<size_t>(tag_num));
    std::vector<dynet::real> current_scores(tag_num); 
    // time 0
    for (size_t i = 0; i < tag_num ; ++i)
    {
        current_scores[i] = init_score[i] + emit_score[0][i];
    }
    // continues time
    std::vector<dynet::real>  pre_timestep_scores(tag_num);
    for (size_t time_step = 1; time_step < len ; ++time_step)
    {
        std::swap(pre_timestep_scores, current_scores); // move current_score -> pre_timestep_score
        for (size_t i = 0; i < tag_num ; ++i )
        {
            size_t pre_tag_with_max_score = 0;
            dynet::real max_score = pre_timestep_scores[pre_tag_with_max_score] + 
                trans_score[pre_tag_with_max_score * tag_num + i];
            for (size_t pre_i = 1 ; pre_i < tag_num ; ++pre_i)
            {
                size_t flat_idx = pre_i * tag_num + pre_i ;
                dynet::real score = pre_timestep_scores[pre_i] + trans_score[flat_idx];
                if (score > max_score)
                {
                    pre_tag_with_max_score = pre_i;
                    max_score = score;
                }
            }
            path_matrix[time_step][i] = pre_tag_with_max_score;
            current_scores[i] = max_score + emit_score[time_step][i];
        }
    }
    // get result 
    IndexSeq tmp_predict_ner_seq(len);
    Index end_predicted_idx = std::distance(current_scores.cbegin(),
        max_element(current_scores.cbegin(), current_scores.cend()));
    tmp_predict_ner_seq[len - 1] = end_predicted_idx;
    Index pre_predicted_idx = end_predicted_idx;
    for (size_t reverse_idx = len - 1; reverse_idx >= 1; --reverse_idx)
    {
        pre_predicted_idx = path_matrix[reverse_idx][pre_predicted_idx]; // backtrace
        tmp_predict_ner_seq[reverse_idx - 1] = pre_predicted_idx;
    }
    std::swap(tmp_predict_ner_seq, pred_seq);
}
} // end of namespace slnn