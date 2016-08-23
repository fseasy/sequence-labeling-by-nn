#include "hyper_output_layers.h"

namespace slnn{

/************ OutputBase ***************/

OutputBase::OutputBase(cnn::real dropout_rate, NonLinearFunc *nonlinear_func) : 
    dropout_rate(dropout_rate),
    nonlinear_func(nonlinear_func)
{}

OutputBase::~OutputBase(){} // base class should implement deconstructor , even for pure virtual function

/************ SimpleOutput *************/

SimpleOutput::SimpleOutput(cnn::Model *m, unsigned input_dim1, unsigned input_dim2 ,
    unsigned hidden_dim, unsigned output_dim , 
    cnn::real dropout_rate,
    NonLinearFunc *nonlinear_func)
    : OutputBase(dropout_rate, nonlinear_func),
    hidden_layer(m , input_dim1 , input_dim2 , hidden_dim) ,
    output_layer(m , hidden_dim , output_dim) 
{}

SimpleOutput::~SimpleOutput() {};

/*************** PretagOutput **************/

PretagOutput::PretagOutput(cnn::Model *m,
    unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
    unsigned hidden_dim, unsigned output_dim , 
    cnn::real dropout_rate, NonLinearFunc *nonlinear_func)
    :OutputBase(dropout_rate, nonlinear_func),
    hidden_layer(m , input_dim1 , input_dim2 , tag_embedding_dim , hidden_dim) ,
    output_layer(m , hidden_dim , output_dim) ,
    tag_lookup_param(m->add_lookup_parameters(output_dim , {tag_embedding_dim})) ,
    TAG_SOS(m->add_parameters({tag_embedding_dim})) 
{}

PretagOutput::~PretagOutput(){} 

/****************** CRFOutput ****************/
CRFOutput::CRFOutput(cnn::Model *m,
    unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
    unsigned hidden_dim,
    unsigned tag_num,
    cnn::real dropout_rate ,
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

cnn::expr::Expression
CRFOutput::build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
    const std::vector<cnn::expr::Expression> &expr_cont2,
    const IndexSeq &gold_seq)
{
    size_t len = expr_cont1.size() ;
    // viterbi data preparation
    std::vector<cnn::expr::Expression> all_tag_expr_cont(tag_num);
    std::vector<cnn::expr::Expression> init_score(tag_num);
    std::vector<cnn::expr::Expression> trans_score(tag_num * tag_num);
    std::vector<std::vector<cnn::expr::Expression>> emit_score(len,
        std::vector<cnn::expr::Expression>(tag_num));
    std::vector<cnn::expr::Expression> cur_score_expr_cont(tag_num),
        pre_score_expr_cont(tag_num);
    std::vector<cnn::expr::Expression> gold_score_expr_cont(len) ;
    // init tag expr , init score
    for( size_t i = 0; i < tag_num ; ++i )
    {
        all_tag_expr_cont[i] = cnn::expr::lookup(*pcg, tag_lookup_param, i);
        init_score[i] = cnn::expr::lookup(*pcg, init_score_lookup_param, i);
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
            cnn::expr::Expression hidden_out_expr = hidden_layer.build_graph(expr_cont1[time_step],
                expr_cont2[time_step], all_tag_expr_cont[i]);
            cnn::expr::Expression non_linear_expr = (*nonlinear_func)(hidden_out_expr) ;
            cnn::expr::Expression dropout_expr = dropout(non_linear_expr, dropout_rate) ;
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
            std::vector<cnn::expr::Expression> partial_score_expr_cont(tag_num);
            for( size_t pre_idx = 0; pre_idx < tag_num; ++pre_idx )
            {
                size_t flatten_idx = pre_idx * tag_num + cur_idx;
                // from-tag score + trans_score
                partial_score_expr_cont[pre_idx] = pre_score_expr_cont[pre_idx] +
                    trans_score[flatten_idx];
            }
            cur_score_expr_cont[cur_idx] = cnn::expr::logsumexp(partial_score_expr_cont) +
                emit_score[time_step][cur_idx];
        }
        // calc gold 
        size_t gold_trans_flatten_idx = gold_seq.at(time_step - 1) * tag_num + gold_seq.at(time_step);
        gold_score_expr_cont[time_step] = trans_score[gold_trans_flatten_idx] +
            emit_score[time_step][gold_seq.at(time_step)];
    }
    cnn::expr::Expression predict_score_expr = cnn::expr::logsumexp(cur_score_expr_cont);

    // if totally correct , loss = 0 (predict_score = gold_score , that is , predict sequence equal to gold sequence)
    // else , loss = predict_score - gold_score
    cnn::expr::Expression loss = predict_score_expr - cnn::expr::sum(gold_score_expr_cont);
    return loss;
}

void CRFOutput::build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
    const std::vector<cnn::expr::Expression> &expr_cont2,
    IndexSeq &pred_seq)
{
    size_t len = expr_cont1.size() ;
    // viterbi data preparation
    std::vector<cnn::expr::Expression> all_tag_expr_cont(tag_num);
    std::vector<cnn::real> init_score(tag_num);
    std::vector < cnn::real> trans_score(tag_num * tag_num);
    std::vector<std::vector<cnn::real>> emit_score(len, std::vector<cnn::real>(tag_num));
    // get initial score
    for( size_t i = 0 ; i < tag_num ; ++i )
    {
        cnn::expr::Expression init_score_expr = cnn::expr::lookup(*pcg, init_score_lookup_param, i);
        init_score[i] = cnn::as_scalar(pcg->get_value(init_score_expr)) ;
    }
    // get translation score
    for( size_t flat_idx = 0; flat_idx < tag_num * tag_num ; ++flat_idx )
    {
        cnn::expr::Expression trans_score_expr = lookup(*pcg, trans_score_lookup_param, flat_idx);
        trans_score[flat_idx] = cnn::as_scalar(pcg->get_value(trans_score_expr)) ;
    }
    // get emit score
    for( size_t i = 0; i < tag_num ; ++i )
    {
        all_tag_expr_cont[i] = cnn::expr::lookup(*pcg, tag_lookup_param, i);
    }
    for( size_t time_step = 0; time_step < len; ++time_step )
    {
        for( size_t i = 0; i < tag_num; ++i )
        {
            cnn::expr::Expression hidden_out_expr = hidden_layer.build_graph(expr_cont1[time_step],
                expr_cont2[time_step], all_tag_expr_cont[i]);
            cnn::expr::Expression non_linear_expr = (*nonlinear_func)(hidden_out_expr) ;
            cnn::expr::Expression emit_expr = emit_layer.build_graph(non_linear_expr);
            emit_score[time_step][i] = cnn::as_scalar( pcg->get_value(emit_expr) );
        }
    }
    // viterbi - process
    std::vector<std::vector<size_t>> path_matrix(len, std::vector<size_t>(tag_num));
    std::vector<cnn::real> current_scores(tag_num); 
    // time 0
    for (size_t i = 0; i < tag_num ; ++i)
    {
        current_scores[i] = init_score[i] + emit_score[0][i];
    }
    // continues time
    std::vector<cnn::real>  pre_timestep_scores(tag_num);
    for (size_t time_step = 1; time_step < len ; ++time_step)
    {
        std::swap(pre_timestep_scores, current_scores); // move current_score -> pre_timestep_score
        for (size_t i = 0; i < tag_num ; ++i )
        {
            size_t pre_tag_with_max_score = 0;
            cnn::real max_score = pre_timestep_scores[pre_tag_with_max_score] + 
                trans_score[pre_tag_with_max_score * tag_num + i];
            for (size_t pre_i = 1 ; pre_i < tag_num ; ++pre_i)
            {
                size_t flat_idx = pre_i * tag_num + pre_i ;
                cnn::real score = pre_timestep_scores[pre_i] + trans_score[flat_idx];
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

/* Bare Output Base */
BareOutputBase::BareOutputBase(cnn::Model *m, unsigned input_dim, unsigned output_dim)
    :softmax_layer(m, input_dim, output_dim)
{}

BareOutputBase::~BareOutputBase(){}

/* Simple Bare Output */
SimpleBareOutput::SimpleBareOutput(cnn::Model *m, unsigned inputs_total_dim, unsigned output_dim)
    :BareOutputBase(m, inputs_total_dim, output_dim)
{}

/************* SoftmaxLayer ***********/

SoftmaxLayer::SoftmaxLayer(cnn::Model *model, unsigned input_dim, unsigned output_dim)
    :output_layer(model, input_dim, output_dim)
{}

/************* OutputBaseWithFeature ***************/

OutputBaseWithFeature::OutputBaseWithFeature(cnn::real dropout_rate, NonLinearFunc *nonlinear_func)
    :dropout_rate(dropout_rate),
    nonlinear_func(nonlinear_func)
{}

OutputBaseWithFeature::~OutputBaseWithFeature(){}

/************ SimpleOutputWithFeature *************/

SimpleOutputWithFeature::SimpleOutputWithFeature(cnn::Model *m, unsigned input_dim1, unsigned input_dim2,
    unsigned feature_dim, unsigned hidden_dim, unsigned output_dim,
    cnn::real dropout_rate,
    NonLinearFunc *nonlinear_func)
    :OutputBaseWithFeature(dropout_rate, nonlinear_func),
    hidden_layer(m, input_dim1, input_dim2, feature_dim, hidden_dim),
    output_layer(m, hidden_dim, output_dim)
{}

SimpleOutputWithFeature::~SimpleOutputWithFeature() {};

/*  PretagOutputWithFeature */
PretagOutputWithFeature::PretagOutputWithFeature(cnn::Model *m, unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2, 
    unsigned feature_dim,
    unsigned hidden_dim, unsigned output_dim , 
    cnn::real dropout_rate, NonLinearFunc *nonlinear_func)
    :OutputBaseWithFeature(dropout_rate, nonlinear_func),
    hidden_layer(m, input_dim1, input_dim2, feature_dim, tag_embedding_dim, hidden_dim),
    output_layer(m, hidden_dim, output_dim),
    tag_lookup_param(m->add_lookup_parameters(output_dim , {tag_embedding_dim})) ,
    TAG_SOS(m->add_parameters({tag_embedding_dim})) 
{}

PretagOutputWithFeature::~PretagOutputWithFeature(){}

/* CRFOutputWithFeature */

CRFOutputWithFeature::CRFOutputWithFeature(cnn::Model *m,
    unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
    unsigned feature_dim,
    unsigned hidden_dim,
    unsigned tag_num,
    cnn::real dropout_rate ,
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
cnn::expr::Expression
CRFOutputWithFeature::build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
    const std::vector<cnn::expr::Expression> &expr_cont2,
    const std::vector<cnn::expr::Expression> &feature_expr_cont,
    const IndexSeq &gold_seq)
{
    size_t len = expr_cont1.size() ;
    // viterbi data preparation
    std::vector<cnn::expr::Expression> all_tag_expr_cont(tag_num);
    std::vector<cnn::expr::Expression> init_score(tag_num);
    std::vector<cnn::expr::Expression> trans_score(tag_num * tag_num);
    std::vector<std::vector<cnn::expr::Expression>> emit_score(len,
        std::vector<cnn::expr::Expression>(tag_num));
    std::vector<cnn::expr::Expression> cur_score_expr_cont(tag_num),
        pre_score_expr_cont(tag_num);
    std::vector<cnn::expr::Expression> gold_score_expr_cont(len) ;
    // init tag expr , init score
    for( size_t i = 0; i < tag_num ; ++i )
    {
        all_tag_expr_cont[i] = cnn::expr::lookup(*pcg, tag_lookup_param, i);
        init_score[i] = cnn::expr::lookup(*pcg, init_score_lookup_param, i);
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
            cnn::expr::Expression hidden_out_expr = hidden_layer.build_graph(expr_cont1.at(time_step),
                expr_cont2.at(time_step), feature_expr_cont.at(time_step), all_tag_expr_cont.at(i));
            cnn::expr::Expression non_linear_expr = (*nonlinear_func)(hidden_out_expr) ;
            cnn::expr::Expression dropout_expr = dropout(non_linear_expr, dropout_rate) ;
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
            std::vector<cnn::expr::Expression> partial_score_expr_cont(tag_num);
            for( size_t pre_idx = 0; pre_idx < tag_num; ++pre_idx )
            {
                size_t flatten_idx = pre_idx * tag_num + cur_idx;
                // from-tag score + trans_score
                partial_score_expr_cont[pre_idx] = pre_score_expr_cont[pre_idx] +
                    trans_score[flatten_idx];
            }
            cur_score_expr_cont[cur_idx] = cnn::expr::logsumexp(partial_score_expr_cont) +
                emit_score[time_step][cur_idx];
        }
        // calc gold 
        size_t gold_trans_flatten_idx = gold_seq.at(time_step - 1) * tag_num + gold_seq.at(time_step);
        gold_score_expr_cont[time_step] = trans_score[gold_trans_flatten_idx] +
            emit_score[time_step][gold_seq.at(time_step)];
    }
    cnn::expr::Expression predict_score_expr = cnn::expr::logsumexp(cur_score_expr_cont);

    // if totally correct , loss = 0 (predict_score = gold_score , that is , predict sequence equal to gold sequence)
    // else , loss = predict_score - gold_score
    cnn::expr::Expression loss = predict_score_expr - cnn::expr::sum(gold_score_expr_cont);
    return loss;
}

/* Totally copy from CRFOutput */
void CRFOutputWithFeature::build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
    const std::vector<cnn::expr::Expression> &expr_cont2,
    const std::vector<cnn::expr::Expression> &feature_expr_cont,
    IndexSeq &pred_seq)
{
    size_t len = expr_cont1.size() ;
    // viterbi data preparation
    std::vector<cnn::expr::Expression> all_tag_expr_cont(tag_num);
    std::vector<cnn::real> init_score(tag_num);
    std::vector < cnn::real> trans_score(tag_num * tag_num);
    std::vector<std::vector<cnn::real>> emit_score(len, std::vector<cnn::real>(tag_num));
    // get initial score
    for( size_t i = 0 ; i < tag_num ; ++i )
    {
        cnn::expr::Expression init_score_expr = cnn::expr::lookup(*pcg, init_score_lookup_param, i);
        init_score[i] = cnn::as_scalar(pcg->get_value(init_score_expr)) ;
    }
    // get translation score
    for( size_t flat_idx = 0; flat_idx < tag_num * tag_num ; ++flat_idx )
    {
        cnn::expr::Expression trans_score_expr = lookup(*pcg, trans_score_lookup_param, flat_idx);
        trans_score[flat_idx] = cnn::as_scalar(pcg->get_value(trans_score_expr)) ;
    }
    // get emit score
    for( size_t i = 0; i < tag_num ; ++i )
    {
        all_tag_expr_cont[i] = cnn::expr::lookup(*pcg, tag_lookup_param, i);
    }
    for( size_t time_step = 0; time_step < len; ++time_step )
    {
        for( size_t i = 0; i < tag_num; ++i )
        {
            cnn::expr::Expression hidden_out_expr = hidden_layer.build_graph(expr_cont1.at(time_step),
                expr_cont2.at(time_step), feature_expr_cont.at(time_step), all_tag_expr_cont.at(i));
            cnn::expr::Expression non_linear_expr = (*nonlinear_func)(hidden_out_expr) ;
            cnn::expr::Expression emit_expr = emit_layer.build_graph(non_linear_expr);
            emit_score[time_step][i] = cnn::as_scalar( pcg->get_value(emit_expr) );
        }
    }
    // viterbi - process
    std::vector<std::vector<size_t>> path_matrix(len, std::vector<size_t>(tag_num));
    std::vector<cnn::real> current_scores(tag_num); 
    // time 0
    for (size_t i = 0; i < tag_num ; ++i)
    {
        current_scores[i] = init_score[i] + emit_score[0][i];
    }
    // continues time
    std::vector<cnn::real>  pre_timestep_scores(tag_num);
    for (size_t time_step = 1; time_step < len ; ++time_step)
    {
        std::swap(pre_timestep_scores, current_scores); // move current_score -> pre_timestep_score
        for (size_t i = 0; i < tag_num ; ++i )
        {
            size_t pre_tag_with_max_score = 0;
            cnn::real max_score = pre_timestep_scores[pre_tag_with_max_score] + 
                trans_score[pre_tag_with_max_score * tag_num + i];
            for (size_t pre_i = 1 ; pre_i < tag_num ; ++pre_i)
            {
                size_t flat_idx = pre_i * tag_num + pre_i ;
                cnn::real score = pre_timestep_scores[pre_i] + trans_score[flat_idx];
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