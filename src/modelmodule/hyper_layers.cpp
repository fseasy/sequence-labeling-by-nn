#include "hyper_layers.h"

namespace slnn{

Index2ExprLayer::Index2ExprLayer(cnn::Model *m, unsigned vocab_size, unsigned embedding_dim)
    :lookup_param(m->add_lookup_parameters(vocab_size, {embedding_dim}))
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
            tmp_exprs[i-shift_distance] = lookup(*pcg, lookup_param, indexSeq[i]);
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

} // end of namespace slnn

