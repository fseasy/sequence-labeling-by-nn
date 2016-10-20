#include <limits>
#include "cws_output_layer.h"
#include "segmenter/cws_module/token_module/cws_tag_definition.h"
#include "segmenter/cws_module/token_module/cws_tag_utility.h"
namespace slnn{

CWSSimpleOutput::CWSSimpleOutput(dynet::Model *m,
    unsigned input_dim1, unsigned input_dim2,
    unsigned hidden_dim, unsigned output_dim,
    CWSTaggingSystem &tag_sys,
    dynet::real dropout_rate,
    NonLinearFunc *nonlinear_func)
    : SimpleOutput(m, input_dim1, input_dim2, hidden_dim, output_dim, dropout_rate, nonlinear_func),
    tag_sys(tag_sys)
{}

void CWSSimpleOutput::build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
                                   const std::vector<dynet::expr::Expression> &expr_cont2,
                                   IndexSeq &pred_out_seq)
{
    size_t len = expr_cont1.size();
    if( 1 == len ) // Special Condition 
    {
        pred_out_seq = { tag_sys.S_ID } ;
        return ;
    }
    std::vector<Index> tmp_pred_out(len);
    Index pre_tag_id = -1 ;
    for (size_t i = 0; i < len - 1; ++i)
    {
        dynet::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i]);
        dynet::expr::Expression nonlinear_expr = nonlinear_func(merge_out_expr);
        dynet::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<dynet::real> out_probs = dynet::as_vector(pcg->get_value(out_expr));
        
        Index max_prob_tag_in_constrain = select_pred_tag_in_constrain(out_probs , i , pre_tag_id );
        tmp_pred_out[i] = max_prob_tag_in_constrain ;
        pre_tag_id = max_prob_tag_in_constrain ;
    }
    if( pre_tag_id == tag_sys.M_ID || pre_tag_id == tag_sys.B_ID ){ tmp_pred_out[len - 1] = tag_sys.E_ID ; }
    else { tmp_pred_out[len - 1] = tag_sys.S_ID ; }
    std::swap(pred_out_seq, tmp_pred_out);
}

Index CWSSimpleOutput::select_pred_tag_in_constrain(std::vector<dynet::real> &dist, size_t pos , Index pre_tag_id)
{
    // dist value must bigger than zero
    dynet::real max_prob = std::numeric_limits<dynet::real>::lowest() ;
    Index selected_tag = -1 ;
    for( size_t cur_tag_id = 0 ; cur_tag_id < dist.size() ; ++cur_tag_id )
    {
        if( !tag_sys.can_emit(pos, cur_tag_id) ) continue ;
        if( pos > 0 && !tag_sys.can_trans(pre_tag_id, cur_tag_id) ) continue ;
        if( dist[cur_tag_id] > max_prob )
        {
            max_prob = dist[cur_tag_id] ;
            selected_tag = cur_tag_id ;
        }
    }
    assert(selected_tag != -1) ;
    return selected_tag ;
}

/**************** CWS Pretag Output ***************/

CWSPretagOutput::CWSPretagOutput(dynet::Model *m,
    unsigned tag_embedding_dim,
    unsigned input_dim1, unsigned input_dim2,
    unsigned hidden_dim, unsigned output_dim,
    CWSTaggingSystem &tag_sys,
    dynet::real dropout_rate,
    NonLinearFunc *nonlinear_fun)
    :PretagOutput(m, tag_embedding_dim, input_dim1, input_dim2, hidden_dim, output_dim, dropout_rate, nonlinear_fun),
    tag_sys(tag_sys)
{}

void CWSPretagOutput::build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
                                   const std::vector<dynet::expr::Expression> &expr_cont2,
                                   IndexSeq &pred_seq)
{
    
    size_t len = expr_cont1.size() ;
    if( 1 == len ) // Special Condition
    {
        pred_seq = { tag_sys.S_ID } ;
        return ;
    }
    IndexSeq tmp_pred(len) ;
    dynet::expr::Expression pretag_exp = parameter(*pcg, TAG_SOS) ;
    Index pre_tag_id = -1 ;
    for( size_t i = 0; i < len-1; ++i )
    {
        dynet::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i], pretag_exp);
        dynet::expr::Expression nonlinear_expr = (*nonlinear_func)(merge_out_expr);
        dynet::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<dynet::real> dist = as_vector(pcg->get_value(out_expr)) ;
        Index id_of_constrained_max_prob = select_pred_tag_in_constrain(dist, i, pre_tag_id) ;
        tmp_pred[i] = id_of_constrained_max_prob ;
        pretag_exp = lookup(*pcg, tag_lookup_param,id_of_constrained_max_prob) ;
        pre_tag_id = id_of_constrained_max_prob ;
    }
    // the last tag has already been determined . (pre_tag can't be -1 ! assert in `select_pred_tag_in_constrain` has ensured it !)
    if( pre_tag_id == tag_sys.B_ID || pre_tag_id == tag_sys.M_ID ){ tmp_pred[len - 1] = tag_sys.E_ID ; }
    else { tmp_pred[len - 1] = tag_sys.S_ID ; }
    std::swap(pred_seq, tmp_pred) ;
}

Index CWSPretagOutput::select_pred_tag_in_constrain(std::vector<dynet::real> &dist, size_t pos , Index pre_tag_id)
{
    // dist value must bigger than zero
    dynet::real max_prob = std::numeric_limits<dynet::real>::lowest() ;
    Index selected_tag = -1 ;
    for( size_t cur_tag_id = 0 ; cur_tag_id < dist.size() ; ++cur_tag_id )
    {
        if( !tag_sys.can_emit(pos, cur_tag_id) ) continue ;
        if( pos > 0 && !tag_sys.can_trans(pre_tag_id, cur_tag_id) ) continue ;
        if( dist[cur_tag_id] > max_prob )
        {
            max_prob = dist[cur_tag_id] ;
            selected_tag = cur_tag_id ;
        }
    }
    assert(selected_tag != -1) ;
    return selected_tag ;
}

/************** CWS CRF OUTPUT *****************/

CWSCRFOutput::CWSCRFOutput(dynet::Model *m,
                           unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
                           unsigned hidden_dim,
                           unsigned tag_num,
                           dynet::real dropout_rate,
                           CWSTaggingSystem &tag_sys,
                           NonLinearFunc *nonlinear_func)
    :CRFOutput(m , tag_embedding_dim , input_dim1 , input_dim2 ,hidden_dim , tag_num, dropout_rate, nonlinear_func) ,
    tag_sys(tag_sys)
{}

dynet::expr::Expression 
CWSCRFOutput::build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
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
    for( size_t pre_tag_id = 0 ; pre_tag_id < tag_num ; ++pre_tag_id )
    {
        for( size_t cur_tag_id = 0 ; cur_tag_id < tag_num; ++cur_tag_id )
        {
            if( !tag_sys.can_trans(pre_tag_id, cur_tag_id) ) continue ; // skip not valid transition
            size_t flat_idx = pre_tag_id * tag_num + cur_tag_id ;
            trans_score[flat_idx] = lookup(*pcg, trans_score_lookup_param, flat_idx);
        }
    }
    // init emit score
    for( size_t time_step = 0; time_step < len; ++time_step )
    {
        for( size_t i = 0; i < tag_num; ++i )
        {
            if( !tag_sys.can_emit(time_step, i) ) continue ;
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
        if( !tag_sys.can_emit(0, i) ) continue ; // skip will be ok .
        // init_score + emit_score
        cur_score_expr_cont[i] = init_score[i] + emit_score[0][i];
    }
    gold_score_expr_cont[0] = cur_score_expr_cont[gold_seq.at(0)];

    // Special condition : if len is 1 , only TAG `S` is valid . (the following logic can't handle it )
    if( 1 == len ) return ( cur_score_expr_cont[tag_sys.S_ID] - gold_score_expr_cont[0] ) ; 

    // 2. the continues time
    for( size_t time_step = 1; time_step < len; ++time_step )
    {
        std::swap(cur_score_expr_cont, pre_score_expr_cont);
        for( size_t cur_idx = 0; cur_idx < tag_num ; ++cur_idx )
        {
            // for every possible trans
            std::vector<dynet::expr::Expression> partial_score_expr_cont;
            for( size_t pre_idx = 0; pre_idx < tag_num; ++pre_idx )
            {
                // constrain : if pre-tag can't emit , or no from pre_tag to cur_tag transition , skip . 
                if(! tag_sys.can_emit(time_step -1 , pre_idx) ||  !tag_sys.can_trans(pre_idx, cur_idx) ) continue ; 
                size_t flatten_idx = pre_idx * tag_num + cur_idx;
                // from-tag score + trans_score
                partial_score_expr_cont.push_back(pre_score_expr_cont[pre_idx] + trans_score[flatten_idx]);
            }
            cur_score_expr_cont[cur_idx] = dynet::expr::logsumexp(partial_score_expr_cont) +
                emit_score[time_step][cur_idx];
        }
        // calc gold 
        size_t gold_trans_flatten_idx = gold_seq.at(time_step - 1) * tag_num + gold_seq.at(time_step);
        gold_score_expr_cont[time_step] = trans_score[gold_trans_flatten_idx] +
            emit_score[time_step][gold_seq.at(time_step)];
    }
    // the last position , only TAG `E` or `S` is valid .
    std::vector<dynet::expr::Expression> valid_expr = {  
        cur_score_expr_cont[tag_sys.E_ID] ,
        cur_score_expr_cont[tag_sys.S_ID]
    };
    dynet::expr::Expression predict_score_expr = dynet::expr::logsumexp(valid_expr);

    // if totally correct , loss = 0 (predict_score = gold_score , that is , predict sequence equal to gold sequence)
    // else , loss = predict_score - gold_score
    dynet::expr::Expression loss = predict_score_expr - dynet::expr::sum(gold_score_expr_cont);
    return loss;
}

void CWSCRFOutput::build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
                                const std::vector<dynet::expr::Expression> &expr_cont2,
                                IndexSeq &pred_seq)
{
    size_t len = expr_cont1.size() ;
    if( 1 == len ) // Special condition . 
    {
        pred_seq = { tag_sys.S_ID } ;
        return ;
    }
    // viterbi data preparation
    std::vector<dynet::expr::Expression> all_tag_expr_cont(tag_num);
    std::vector<dynet::real> init_score(tag_num , std::numeric_limits<dynet::real>::min());
    std::vector < dynet::real> trans_score(tag_num * tag_num);
    std::vector<std::vector<dynet::real>> emit_score(len, std::vector<dynet::real>(tag_num));
    // get initial score
    for( size_t i = 0 ; i < tag_num ; ++i )
    {
        if( !tag_sys.can_emit(0, i) ) continue ;
        dynet::expr::Expression init_score_expr = dynet::expr::lookup(*pcg, init_score_lookup_param, i);
        init_score[i] = dynet::as_scalar(pcg->get_value(init_score_expr)) ;
    }
    // get translation score
    for( size_t pre_idx = 0 ; pre_idx < tag_num ; ++pre_idx )
    {
        for( size_t cur_idx = 0 ; cur_idx < tag_num ; ++cur_idx )
        {
            if( !tag_sys.can_trans(pre_idx, cur_idx) ) continue ;
            size_t flat_idx = pre_idx * tag_num + cur_idx ;
            dynet::expr::Expression trans_score_expr = lookup(*pcg, trans_score_lookup_param, flat_idx);
            trans_score[flat_idx] = dynet::as_scalar(pcg->get_value(trans_score_expr)) ;
        }
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
            if( !tag_sys.can_emit(time_step, i) ) continue ;
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
        if( !tag_sys.can_emit(0, i) ) continue ;
        current_scores[i] = init_score[i] + emit_score[0][i];
    }
    // continues time
    std::vector<dynet::real>  pre_timestep_scores(tag_num);
    for (size_t time_step = 1; time_step < len ; ++time_step)
    {
        std::swap(pre_timestep_scores, current_scores); // move current_score -> pre_timestep_score
        for (size_t i = 0; i < tag_num ; ++i )
        {
            
            Index pre_tag_with_max_score = -1;
            dynet::real max_score = std::numeric_limits<dynet::real>::lowest() ;
            for (size_t pre_i = 0 ; pre_i < tag_num ; ++pre_i)
            {
                if( !tag_sys.can_emit(time_step - 1, pre_i) || !tag_sys.can_trans(pre_i, i) ) continue ;
                size_t flat_idx = pre_i * tag_num + pre_i ;
                dynet::real score = pre_timestep_scores[pre_i] + trans_score[flat_idx];
                if (score > max_score)
                {
                    pre_tag_with_max_score = pre_i;
                    max_score = score;
                }
            }
            assert(pre_tag_with_max_score != -1) ;
            path_matrix[time_step][i] = pre_tag_with_max_score;
            current_scores[i] = max_score + emit_score[time_step][i];
        }
    }
    // get result 
    IndexSeq tmp_predict_ner_seq(len);
    // from TAG `E` or `S` to select the id with the max score 
    Index end_predicted_idx ;
    dynet::real end_of_E_score = current_scores.at(tag_sys.E_ID) ;
    dynet::real end_of_S_score = current_scores.at(tag_sys.S_ID) ;
    if( end_of_E_score <= end_of_S_score ){ end_predicted_idx = tag_sys.S_ID ; }
    else { end_predicted_idx = tag_sys.E_ID ; }
    tmp_predict_ner_seq[len - 1] = end_predicted_idx;
    Index pre_predicted_idx = end_predicted_idx;
    for (size_t reverse_idx = len - 1; reverse_idx >= 1; --reverse_idx)
    {
        pre_predicted_idx = path_matrix[reverse_idx][pre_predicted_idx]; // backtrace
        tmp_predict_ner_seq[reverse_idx - 1] = pre_predicted_idx;
    }
    std::swap(tmp_predict_ner_seq, pred_seq);
}

/* CWSSimpleOutputWithFeature */

CWSSimpleOutputWithFeature::CWSSimpleOutputWithFeature(dynet::Model *m, unsigned input_dim1, unsigned input_dim2, unsigned feature_dim,
    unsigned hidden_dim, unsigned output_dim,
    dynet::real dropout_rate, NonLinearFunc *nonlinear_func)
    : SimpleOutputWithFeature(m, input_dim1, input_dim2, feature_dim, hidden_dim, output_dim, dropout_rate, nonlinear_func)
{}

void CWSSimpleOutputWithFeature::build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
    const std::vector<dynet::expr::Expression> &expr_cont2,
    const std::vector<dynet::expr::Expression> &feature_expr_cont,
    IndexSeq &pred_out_seq)
{
    using std::swap;
    size_t len = expr_cont1.size();
    if( 1 == len ) // Special Condition 
    {
        pred_out_seq = { CWSTaggingSystem::STATIC_S_ID };
        return ;
    }
    std::vector<Index> tmp_pred_out(len);
    Index pre_tag_id = CWSTaggingSystem::STATIC_NONE_ID ;
    for (size_t i = 0; i < len - 1; ++i)
    {
        dynet::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i], feature_expr_cont[i]);
        dynet::expr::Expression nonlinear_expr = nonlinear_func(merge_out_expr);
        dynet::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<dynet::real> out_probs = dynet::as_vector(pcg->get_value(out_expr));

        Index max_prob_tag_in_constrain = CWSTaggingSystem::static_select_tag_constrained(out_probs , i , pre_tag_id );
        tmp_pred_out[i] = max_prob_tag_in_constrain ;
        pre_tag_id = max_prob_tag_in_constrain ;
    }
    if( pre_tag_id == CWSTaggingSystem::STATIC_M_ID || 
        pre_tag_id == CWSTaggingSystem::STATIC_B_ID )
    { 
        tmp_pred_out[len - 1] = CWSTaggingSystem::STATIC_E_ID ; 
    }
    else { tmp_pred_out[len - 1] = CWSTaggingSystem::STATIC_S_ID; }
    swap(pred_out_seq, tmp_pred_out);
}

/* CWSSimpleOutput NEW (0628) */
CWSSimpleOutputNew::CWSSimpleOutputNew(dynet::Model *m,
    unsigned input_dim1, unsigned input_dim2,
    unsigned hidden_dim, unsigned output_dim,
    dynet::real dropout_rate,
    NonLinearFunc *nonlinear_func)
    : SimpleOutput(m , input_dim1 , input_dim2 , hidden_dim , output_dim, dropout_rate, nonlinear_func)
{}

void CWSSimpleOutputNew::build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
    const std::vector<dynet::expr::Expression> &expr_cont2,
    IndexSeq &pred_out_seq)
{
    using std::swap;
    size_t len = expr_cont1.size();
    if( 1 == len ) // Special Condition 
    {
        pred_out_seq = { CWSTaggingSystem::STATIC_S_ID };
        return ;
    }
    std::vector<Index> tmp_pred_out(len);
    Index pre_tag_id = CWSTaggingSystem::STATIC_NONE_ID ;
    for (size_t i = 0; i < len - 1; ++i)
    {
        dynet::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i]);
        dynet::expr::Expression nonlinear_expr = nonlinear_func(merge_out_expr);
        dynet::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<dynet::real> out_probs = dynet::as_vector(pcg->get_value(out_expr));

        Index max_prob_tag_in_constrain = CWSTaggingSystem::static_select_tag_constrained(out_probs , i , pre_tag_id );
        tmp_pred_out[i] = max_prob_tag_in_constrain ;
        pre_tag_id = max_prob_tag_in_constrain ;
    }
    if( pre_tag_id == CWSTaggingSystem::STATIC_M_ID || 
        pre_tag_id == CWSTaggingSystem::STATIC_B_ID )
    { 
        tmp_pred_out[len - 1] = CWSTaggingSystem::STATIC_E_ID ; 
    }
    else { tmp_pred_out[len - 1] = CWSTaggingSystem::STATIC_S_ID; }
    swap(pred_out_seq, tmp_pred_out);
}

/***************************************************
 * Segmentor output :  Simple Bare output (re-write @2016-10-13)
 ***************************************************/

CWSSimpleBareOutput::CWSSimpleBareOutput(dynet::Model *m, unsigned input_dim, unsigned output_dim)
    :SimpleBareOutput(m, input_dim, output_dim)
{}

void CWSSimpleBareOutput::
build_output(const std::vector<dynet::expr::Expression> &input_expr_seq, std::vector<Index> &out_pred_seq)
{
    using std::swap;
    std::size_t len = input_expr_seq.size();
    if( 1 == len ) // Special Condition : len = 1
    {
        out_pred_seq = { segmenter::Tag::TAG_S_ID };
        return ;
    }
    std::vector<Index> tmp_pred_out(len);
    Index pre_tag_id = segmenter::Tag::TAG_NONE_ID;
    for (std::size_t i = 0; i < len - 1; ++i) // select first and middle tags
    {
        dynet::expr::Expression out_expr = softmax_layer.build_graph(input_expr_seq[i]) ;
        std::vector<dynet::real> out_probs = dynet::as_vector(pcg->get_value(out_expr));
        Index max_prob_tag_in_constrain = segmenter::token_module::select_best_tag_constrained(out_probs , i , pre_tag_id );
        tmp_pred_out[i] = max_prob_tag_in_constrain ;
        pre_tag_id = max_prob_tag_in_constrain ;
    }
    if( pre_tag_id == segmenter::Tag::TAG_M_ID||  // select last tag
        pre_tag_id == segmenter::Tag::TAG_B_ID)
    { 
        tmp_pred_out[len - 1] = segmenter::Tag::TAG_E_ID; 
    }
    else { tmp_pred_out[len - 1] = segmenter::Tag::TAG_S_ID; }
    swap(out_pred_seq, tmp_pred_out);
}

} // end of namespace slnn
