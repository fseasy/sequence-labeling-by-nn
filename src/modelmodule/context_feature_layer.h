#ifndef MODELMODULE_CONTEXT_FEATURE_LAYER_H_
#define MODELMODULE_CONTEXT_FEATURE_LAYER_H_

#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "context_feature.h"
namespace slnn{

struct ContextFeatureLayer
{
    ContextFeatureLayer(cnn::Model *m, cnn::LookupParameters * &word_lookup_param);
    void new_graph(cnn::ComputationGraph &cg);
    template <size_t N>
    cnn::expr::Expression
    build_feature_expr(const typename ContextFeature<N>::ContextFeatureIndexGroup & context_feature_gp);
    
    template <size_t N>
    void build_feature_exprs(const typename ContextFeature<N>::ContextFeatureIndexGroupSeq &context_feature_gp_seq,
        std::vector<cnn::expr::Expression> &context_feature_exprs);
private:
    template <size_t N>
    cnn::expr::Expression build_word_expr(Index word_id);
    cnn::LookupParameters *&word_lookup_param;
    cnn::ComputationGraph *pcg;
    cnn::Parameters *word_sos_param;
    cnn::Parameters *word_eos_param;
    cnn::expr::Expression word_sos_expr;
    cnn::expr::Expression word_eos_expr;
};

inline 
void ContextFeatureLayer::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg;
    word_sos_expr = cnn::expr::parameter(cg, word_sos_param);
    word_eos_expr = cnn::expr::parameter(cg, word_eos_param);
}

template <size_t N>
inline 
cnn::expr::Expression ContextFeatureLayer::build_word_expr(Index word_id)
{
    if( word_id == ContextFeature<N>::WordSOSId ){ return word_sos_expr ; }
    else if( word_id == ContextFeature<N>::WordEOSId ){ return word_eos_expr; }
    else { return cnn::expr::lookup(*pcg, word_lookup_param, word_id); }
}

template <size_t N>
inline 
cnn::expr::Expression ContextFeatureLayer::build_feature_expr(typename const ContextFeature<N>::ContextFeatureIndexGroup &context_feature_gp)
{
    std::vector<cnn::expr::Expression> context_word_exprs(ContextFeature<N>::ContextSize);
    for( unsigned i = 0 ; i < ContextFeature<N>::ContextSize ; ++i )
    {
        context_word_exprs.at(i) = build_word_expr<N>(context_feature_gp.at(i));
    }
    return cnn::expr::concatenate(context_word_exprs);
}

template <size_t N>
inline 
void ContextFeatureLayer::build_feature_exprs(typename const ContextFeature<N>::ContextFeatureIndexGroupSeq &context_feature_gp_seq,
    std::vector<cnn::expr::Expression> &context_feature_exprs)
{
    using std::swap;
    unsigned seq_len = context_feature_gp_seq.size();
    std::vector<cnn::expr::Expression> tmp_context_feature_exprs(seq_len);
    for( unsigned i = 0; i < seq_len; ++i )
    {
        tmp_context_feature_exprs.at(i) = bulid_feature_expr<N>(context_feature_gp_seq.at(i));
    }
    swap(contxt_feature_exprs, tmp_context_feature_exprs);
}

} // end of namespace slnn
#endif