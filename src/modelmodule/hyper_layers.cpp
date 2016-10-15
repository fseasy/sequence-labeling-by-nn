#include <deque>
#include "hyper_layers.h"

namespace slnn{

Index2ExprLayer::Index2ExprLayer(cnn::Model *m, unsigned vocab_size, unsigned embedding_dim)
    :lookup_param(m->add_lookup_parameters(vocab_size, { embedding_dim }))
{}

ShiftedIndex2ExprLayer::ShiftedIndex2ExprLayer(cnn::Model *m, unsigned vocab_size, unsigned embedding_dim, ShiftDirection direction,
    unsigned shift_distance)
    : Index2ExprLayer(m, vocab_size, embedding_dim),
    shift_direction(direction),
    shift_distance(shift_distance),
    padding_parameters(shift_distance)
{
    for( unsigned i = 0 ; i < shift_distance; ++i )
    {
        padding_parameters[i] = m->add_parameters({ embedding_dim });
    }
}

void ShiftedIndex2ExprLayer::index_seq2expr_seq(const IndexSeq &indexSeq, std::vector<cnn::expr::Expression> &exprs)
{
    using std::swap;
    unsigned sz = indexSeq.size();
    std::vector<cnn::expr::Expression> tmp_exprs(sz);
    if( shift_direction == LeftShift )
    {
        for( unsigned i = shift_distance ; i < sz; ++i )
        {
            tmp_exprs[i - shift_distance] = lookup(*pcg, lookup_param, indexSeq[i]);
        }
        unsigned padding_pos = shift_distance > sz ? 0 : sz - shift_distance ;
        for( unsigned i = padding_pos; i < sz; ++i )
        {
            tmp_exprs[i] = parameter(*pcg, padding_parameters[i - padding_pos]);
        }
    }
    else
    {
        unsigned padding_end_pos = std::min(shift_distance, sz);
        for( unsigned i = 0; i < padding_end_pos; ++i )
        {
            tmp_exprs[i] = parameter(*pcg, padding_parameters[i]);
        }
        for( unsigned i = padding_end_pos; i < sz; ++i )
        {
            tmp_exprs[i] = lookup(*pcg, lookup_param, indexSeq[i - padding_end_pos]);
        }
    }
    swap(exprs, tmp_exprs);
}


/***********************************************
 * WindowExprGenerateLayer
 ***********************************************/

WindowExprGenerateLayer::WindowExprGenerateLayer(cnn::Model *cnn_model, unsigned window_sz, unsigned embedding_dim)
    :sos_param(cnn_model->add_parameters({ embedding_dim })),
    eos_param(cnn_model->add_parameters({embedding_dim})),
    window_sz(window_sz)
{}

std::vector<cnn::expr::Expression>
WindowExprGenerateLayer::generate_window_expr_by_concatenating(const std::vector<cnn::expr::Expression> &unit_exprs)
{
    std::deque<cnn::expr::Expression> window_expr_list(window_sz);
    unsigned len = unit_exprs.size();
    // init
    unsigned half_sz = window_sz / 2;
    for( unsigned i = 0; i < half_sz; ++i ){ window_expr_list[i] = sos_expr; }
    for( unsigned i = half_sz; i < window_sz; ++i )
    {
        window_expr_list[i] = i < len ? unit_exprs[i] : eos_expr;
    }
    // generate window concatenated expr 
    std::vector<cnn::expr::Expression> concat_expr_list(len);
    concat_expr_list[0] = cnn::expr::concatenate(window_expr_list);
    for( unsigned i = 1; i < len; ++i )
    {
        // scroll
        window_expr_list.pop_front();
        window_expr_list.push_back(i + half_sz < len ? unit_exprs[i + half_sz] : eos_expr);
        concat_expr_list[i] = cnn::expr::concatenate(window_expr_list);
    }
    return concat_expr_list;
}

} // end of namespace slnn

