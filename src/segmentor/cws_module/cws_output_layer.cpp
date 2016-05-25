#include <limits>
#include "cws_output_layer.h"

namespace slnn{

CWSSimpleOutput::CWSSimpleOutput(cnn::Model *m,
                                 unsigned input_dim1, unsigned input_dim2,
                                 unsigned hidden_dim, unsigned output_dim,
                                 CWSTaggingSystem &tag_sys,
                                 NonLinearFunc *nonlinear_func)
    : SimpleOutput(m , input_dim1 , input_dim2 , hidden_dim , output_dim , nonlinear_func) ,
    tag_sys(tag_sys)
{}

void CWSSimpleOutput::build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
                                   const std::vector<cnn::expr::Expression> &expr_cont2,
                                   IndexSeq &pred_out_seq)
{
    size_t len = expr_cont1.size();
    std::vector<Index> tmp_pred_out(len);
    Index pre_tag_id = -1 ;
    for (size_t i = 0; i < len; ++i)
    {
        cnn::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i]);
        cnn::expr::Expression nonlinear_expr = nonlinear_func(merge_out_expr);
        cnn::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<cnn::real> out_probs = cnn::as_vector(pcg->get_value(out_expr));
        
        Index max_prob_tag_in_constrain = select_pred_tag_in_constrain(out_probs , i , pre_tag_id );
        tmp_pred_out[i] = max_prob_tag_in_constrain ;
        pre_tag_id = max_prob_tag_in_constrain ;
    }
    std::swap(pred_out_seq, tmp_pred_out);
}

Index CWSSimpleOutput::select_pred_tag_in_constrain(std::vector<cnn::real> &dist, int pos , Index pre_tag_id)
{
    // dist value must bigger than zero
    cnn::real max_prob = std::numeric_limits<cnn::real>::min() ;
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
    return selected_tag ;
}

CWSPretagOutput::CWSPretagOutput(cnn::Model *m,
                                 unsigned tag_embedding_dim,
                                 unsigned input_dim1, unsigned input_dim2,
                                 unsigned hidden_dim, unsigned output_dim,
                                 CWSTaggingSystem &tag_sys,
                                 NonLinearFunc *nonlinear_fun)
    :PretagOutput(m , tag_embedding_dim , input_dim1 , input_dim2 , hidden_dim , output_dim , nonlinear_fun) ,
    tag_sys(tag_sys)
{}

void CWSPretagOutput::build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
                                   const std::vector<cnn::expr::Expression> &expr_cont2,
                                   IndexSeq &pred_seq)
{
    size_t len = expr_cont1.size() ;
    IndexSeq tmp_pred(len) ;
    cnn::expr::Expression pretag_exp = parameter(*pcg, TAG_SOS) ;
    Index pre_tag_id = -1 ;
    for( size_t i = 0; i < len; ++i )
    {
        cnn::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i], pretag_exp);
        cnn::expr::Expression nonlinear_expr = (*nonlinear_func)(merge_out_expr);
        cnn::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<cnn::real> dist = as_vector(pcg->get_value(out_expr)) ;
        Index id_of_constrained_max_prob = select_pred_tag_in_constrain(dist, i, pre_tag_id) ;
        tmp_pred[i] = id_of_constrained_max_prob ;
        pretag_exp = lookup(*pcg, tag_lookup_param,id_of_constrained_max_prob) ;
        pre_tag_id = id_of_constrained_max_prob ;
    }
    std::swap(pred_seq, tmp_pred) ;
}

Index CWSPretagOutput::select_pred_tag_in_constrain(std::vector<cnn::real> &dist, int pos , Index pre_tag_id)
{
    // dist value must bigger than zero
    cnn::real max_prob = std::numeric_limits<cnn::real>::min() ;
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
    return selected_tag ;
}

} // end of namespace slnn