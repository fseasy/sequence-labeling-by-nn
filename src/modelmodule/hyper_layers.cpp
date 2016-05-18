#include "hyper_layers.h"

#include <exception>

namespace slnn
{

/********** Input1 *********/

Input1::Input1(cnn::Model *m, unsigned vocab_size, unsigned embedding_dim)
    :word_lookup_param(m->add_lookup_parameters(vocab_size, {embedding_dim}))
{}

Input1::~Input1()
{}

void Input1::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg;
}

void Input1::build_inputs(const IndexSeq &sent, std::vector<cnn::expr::Expression> &inputs_exprs)
{
    if (nullptr == pcg) throw std::runtime_error("cg should be set .");
    size_t sent_len = sent.size();
    std::vector<cnn::expr::Expression> tmp_inputs(sent_len);
    for (size_t i = 0; i < sent_len; ++i)
    {
        tmp_inputs[i] = lookup(*pcg, word_lookup_param, sent[i]);
    }
    std::swap(inputs_exprs, tmp_inputs);
}

/********** Input2D ***********/

Input2D::Input2D(cnn::Model *m, unsigned vocab_size1, unsigned embedding_dim1,
    unsigned vocab_size2, unsigned embedding_dim2,
    unsigned mergeout_dim ,
    NonLinearFunc *nonlinear_func)
    : dynamic_lookup_param1(m->add_lookup_parameters(vocab_size1, {embedding_dim1})) ,
    dynamic_lookup_param2(m->add_lookup_parameters(vocab_size2 , {embedding_dim2})) ,
    m2_layer(m , embedding_dim1 , embedding_dim2 , mergeout_dim) ,
    nonlinear_func(nonlinear_func)
{}

Input2D::~Input2D() {}

void Input2D::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg;
    m2_layer.new_graph(cg);
}

void Input2D::build_inputs(const IndexSeq &seq1, const IndexSeq &seq2, std::vector<cnn::expr::Expression> &inputs_exprs )
{
    size_t seq_len = seq1.size();
    std::vector<cnn::expr::Expression> tmp_inputs(seq_len);
    for (size_t i = 0; i < seq_len; ++i)
    {
        cnn::expr::Expression expr1 = lookup(*pcg, dynamic_lookup_param1, seq1.at(i));
        cnn::expr::Expression expr2 = lookup(*pcg, dynamic_lookup_param2, seq2.at(i));
        cnn::expr::Expression linear_merge_expr = m2_layer.build_graph(expr1, expr2);
        cnn::expr::Expression nonlinear_expr = nonlinear_func(linear_merge_expr);
        tmp_inputs[i] = nonlinear_expr;
    }
    std::swap(inputs_exprs, tmp_inputs);
 }

/************* Input2 ***************/

Input2::Input2(cnn::Model *m, unsigned dynamic_vocab_size, unsigned dynamic_embedding_dim,
unsigned fixed_vocab_size, unsigned fixed_embedding_dim,
unsigned mergeout_dim ,
NonLinearFunc *nonlinear_func) 
    :dynamic_lookup_param(m->add_lookup_parameters(dynamic_vocab_size, {dynamic_embedding_dim})) ,
    fixed_lookup_param(m->add_lookup_parameters(fixed_vocab_size , {fixed_embedding_dim})) ,
    m2_layer(m , dynamic_embedding_dim , fixed_embedding_dim , mergeout_dim) ,
    nonlinear_func(nonlinear_func)
{}

Input2::~Input2() {};

void Input2::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg;
    m2_layer.new_graph(cg);
}

void Input2::build_inputs(const IndexSeq &dynamic_seq, const IndexSeq &fixed_seq, std::vector<cnn::expr::Expression> &inputs_exprs)
{
    size_t seq_len = dynamic_seq.size();
    std::vector<cnn::expr::Expression> tmp_inputs(seq_len);
    for (size_t i = 0; i < seq_len; ++i)
    {
        cnn::expr::Expression expr1 = lookup(*pcg, dynamic_lookup_param, dynamic_seq.at(i));
        cnn::expr::Expression expr2 = lookup(*pcg, fixed_lookup_param, fixed_seq.at(i));
        cnn::expr::Expression linear_merge_expr = m2_layer.build_graph(expr1, expr2);
        cnn::expr::Expression nonlinear_expr = nonlinear_func(linear_merge_expr);
        tmp_inputs[i] = nonlinear_expr;
    }
    std::swap(inputs_exprs, tmp_inputs);
}

/*********** Input3 *********/

Input3::Input3(cnn::Model *m, unsigned dynamic_vocab_size1, unsigned dynamic_embedding_dim1,
    unsigned dynamic_vocab_size2, unsigned dynamic_embedding_dim2,
    unsigned fixed_vocab_size, unsigned fixed_embedding_dim,
    unsigned mergeout_dim,
    NonLinearFunc *nonlinear_func)
    : dynamic_lookup_param1(m->add_lookup_parameters(dynamic_vocab_size1 , {dynamic_embedding_dim1})) ,
    dynamic_lookup_param2(m->add_lookup_parameters(dynamic_vocab_size2 , {dynamic_embedding_dim2})) ,
    fixed_lookup_param(m->add_lookup_parameters(fixed_vocab_size , {fixed_embedding_dim})) ,
    m3_layer(m , dynamic_embedding_dim1 , dynamic_embedding_dim2 , fixed_embedding_dim , mergeout_dim) ,
    nonlinear_func(nonlinear_func)
{}

Input3::~Input3(){}

void Input3::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg;
    m3_layer.new_graph(cg);
}

void Input3::build_inputs(const IndexSeq &dseq1, const IndexSeq &dseq2, const IndexSeq &fseq,
    std::vector<cnn::expr::Expression> &inputs_exprs)
{
    size_t seq_len = dseq1.size();
    std::vector<cnn::expr::Expression> tmp_inputs(seq_len);
    for (size_t i = 0; i < seq_len; ++i)
    {
        cnn::expr::Expression dexpr1 = lookup(*pcg, dynamic_lookup_param1, dseq1.at(i));
        cnn::expr::Expression dexpr2 = lookup(*pcg, dynamic_lookup_param2, dseq2.at(i));
        cnn::expr::Expression fexpr = const_lookup(*pcg, fixed_lookup_param, fseq.at(i));
        cnn::expr::Expression linear_merge_expr = m3_layer.build_graph(dexpr1, dexpr2, fexpr);
        tmp_inputs[i] = nonlinear_func(linear_merge_expr);
    }
    std::swap(inputs_exprs, tmp_inputs);
}


/************ SimpleOutput *************/

SimpleOutput::SimpleOutput(cnn::Model *m, unsigned input_dim1, unsigned input_dim2 ,
    unsigned hidden_dim, unsigned output_dim , 
    NonLinearFunc *nonlinear_func)
    : hidden_layer(m , input_dim1 , input_dim2 , hidden_dim) ,
    output_layer(m , hidden_dim , output_dim) ,
    nonlinear_func(nonlinear_func)
{}

SimpleOutput::~SimpleOutput() {};

void SimpleOutput::new_graph(cnn::ComputationGraph &cg)
{
    hidden_layer.new_graph(cg);
    output_layer.new_graph(cg);
    pcg = &cg;
}

Expression SimpleOutput::build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
    const std::vector<cnn::expr::Expression> &expr_cont2 , const IndexSeq &gold_seq)
{
    size_t len = expr_cont1.size();
    std::vector<cnn::expr::Expression> loss_cont(len);
    for (size_t i = 0; i < len; ++i)
    {
        cnn::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i]);
        cnn::expr::Expression nonlinear_expr = nonlinear_func(merge_out_expr);
        cnn::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        loss_cont[i] = cnn::expr::pickneglogsoftmax(out_expr, gold_seq.at(i));
    }
    return cnn::expr::sum(loss_cont);
}

void SimpleOutput::build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
    const std::vector<cnn::expr::Expression> &expr_cont2,
    IndexSeq &pred_out_seq)
{
    size_t len = expr_cont1.size();
    std::vector<Index> tmp_pred_out(len);
    for (size_t i = 0; i < len; ++i)
    {
        cnn::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i]);
        cnn::expr::Expression nonlinear_expr = nonlinear_func(merge_out_expr);
        cnn::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<cnn::real> out_probs = cnn::as_vector(pcg->get_value(out_expr));
        Index idx_of_max_prob = std::distance(out_probs.cbegin(),
            std::max_element(out_probs.cbegin(), out_probs.cend()));
        tmp_pred_out[i] = idx_of_max_prob;
    }
    std::swap(pred_out_seq, tmp_pred_out);
}

/*************** PretagOutput **************/
PretagOutput::PretagOutput(cnn::Model *m,
                           unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
                           unsigned hidden_dim, unsigned output_dim ,
                           NonLinearFunc *nonlinear_func)
    :hidden_layer(m , input_dim1 , input_dim2 , tag_embedding_dim , hidden_dim) ,
    output_layer(m , hidden_dim , output_dim) ,
    nonlinear_func(nonlinear_func) ,
    TAG_SOS(m->add_parameters({tag_embedding_dim})) ,
    tag_lookup_param(m->add_lookup_parameters(output_dim , {tag_embedding_dim}))
{}

PretagOutput::~PretagOutput(){} 

void PretagOutput::new_graph(cnn::ComputationGraph &cg)
{
    hidden_layer.new_graph(cg) ;
    output_layer.new_graph(cg) ;
    pcg = &cg ;
}

cnn::expr::Expression
PretagOutput::build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
                                const std::vector<cnn::expr::Expression> &expr_cont2,
                                const IndexSeq &gold_seq)
{
    size_t len = expr_cont1.size() ;
    std::vector<cnn::expr::Expression> loss_cont(len);
    cnn::expr::Expression pretag_exp = parameter(*pcg, TAG_SOS) ;
    for( size_t i = 0; i < len; ++i )
    {
        cnn::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i] , pretag_exp);
        cnn::expr::Expression nonlinear_expr = (*nonlinear_func)(merge_out_expr);
        cnn::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        loss_cont[i] = cnn::expr::pickneglogsoftmax(out_expr, gold_seq.at(i));
        pretag_exp = lookup(*pcg, tag_lookup_param, gold_seq.at(i)) ;
    }
    return cnn::expr::sum(loss_cont);
}


void PretagOutput::build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
                                const std::vector<cnn::expr::Expression> &expr_cont2,
                                IndexSeq &pred_seq)
{
    size_t len = expr_cont1.size() ;
    IndexSeq tmp_pred(len) ;
    cnn::expr::Expression pretag_exp = parameter(*pcg, TAG_SOS) ;
    for( size_t i = 0; i < len; ++i )
    {
        cnn::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i], pretag_exp);
        cnn::expr::Expression nonlinear_expr = (*nonlinear_func)(merge_out_expr);
        cnn::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<cnn::real> dist = as_vector(pcg->get_value(out_expr)) ;
        Index id_of_max_prob = std::distance(dist.cbegin(), std::max_element(dist.cbegin(), dist.cend())) ;
        tmp_pred[i] = id_of_max_prob ;
        pretag_exp = lookup(*pcg, tag_lookup_param,id_of_max_prob) ;
    }
    std::swap(pred_seq, tmp_pred) ;
}

/****************** CRFOutput ****************/
CRFOutput::CRFOutput(cnn::Model *m,
          unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
          unsigned hidden_dim,
          unsigned tag_num,
          cnn::real dropout_rate ,
          NonLinearFunc *nonlinear_func)
    :hidden_layer(m , input_dim1 , input_dim2 , tag_embedding_dim , hidden_dim) ,
    emit_layer(m , hidden_dim , 1) ,
    trans_score_lookup_param(m->add_lookup_parameters(tag_num * tag_num , {1})) ,
    init_score_lookup_param(m->add_lookup_parameters(tag_num , {1})) ,
    tag_lookup_param(m->add_lookup_parameters(tag_num , {tag_embedding_dim})) ,
    tag_num(tag_num) ,
    dropout_rate(dropout_rate) ,
    nonlinear_func(nonlinear_func)
{}

CRFOutput::~CRFOutput(){} 

void CRFOutput::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg ;
    hidden_layer.new_graph(cg) ;
    emit_layer.new_graph(cg) ;
}

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

} // end of namespace slnn