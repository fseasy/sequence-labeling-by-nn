#ifndef MODELMODULE_CONTEXT_FEATURE_LAYER_H_
#define MODELMODULE_CONTEXT_FEATURE_LAYER_H_

#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "context_feature.h"
namespace slnn{

struct ContextFeatureLayer
{
    ContextFeatureLayer(dynet::Model *m, dynet::LookupParameters *word_lookup_param);
    void new_graph(dynet::ComputationGraph &cg);
    dynet::expr::Expression build_feature_expr(const ContextFeatureData &context_feature_data);
    
    void build_feature_exprs(const ContextFeatureDataSeq &context_feature_data_seq,
        std::vector<dynet::expr::Expression> &context_feature_exprs);
private:
    dynet::expr::Expression build_word_expr(Index word_id);
    dynet::LookupParameters *word_lookup_param;
    dynet::ComputationGraph *pcg;
    dynet::Parameters *word_sos_param;
    dynet::Parameters *word_eos_param;
    dynet::expr::Expression word_sos_expr;
    dynet::expr::Expression word_eos_expr;
};

inline 
void ContextFeatureLayer::new_graph(dynet::ComputationGraph &cg)
{
    pcg = &cg;
    word_sos_expr = dynet::expr::parameter(cg, word_sos_param);
    word_eos_expr = dynet::expr::parameter(cg, word_eos_param);
}

inline 
dynet::expr::Expression ContextFeatureLayer::build_word_expr(Index word_id)
{
    if( word_id == ContextFeature::WordSOSId ){ return word_sos_expr ; }
    else if( word_id == ContextFeature::WordEOSId ){ return word_eos_expr; }
    else { return dynet::expr::lookup(*pcg, word_lookup_param, word_id); }
}

inline 
dynet::expr::Expression ContextFeatureLayer::build_feature_expr(const ContextFeatureData &context_feature_data)
{
    // this may can be optimized ! 
    // context_feature_data.size() always the same , and contxt_word_exprs may be declaration as static or member variable
    // but we give up ! 
    // it should not be the key factor of speed .
    // for static and member variable , we may get some diffeculties when multi-threads
    // for constant context_feature_data.size() , we need the information from the ContextFeature instance.
    size_t sz = context_feature_data.size();
    std::vector<dynet::expr::Expression> context_word_exprs(sz); 
    for( unsigned i = 0 ; i < sz ; ++i )
    {
        context_word_exprs.at(i) = build_word_expr(context_feature_data.at(i));
    }
    return dynet::expr::concatenate(context_word_exprs);
}

inline 
void ContextFeatureLayer::build_feature_exprs(const ContextFeatureDataSeq &context_feature_data_seq,
    std::vector<dynet::expr::Expression> &context_feature_exprs)
{
    using std::swap;
    unsigned seq_len = context_feature_data_seq.size();
    std::vector<dynet::expr::Expression> tmp_context_feature_exprs(seq_len);
    for( unsigned i = 0; i < seq_len; ++i )
    {
        tmp_context_feature_exprs.at(i) = build_feature_expr(context_feature_data_seq.at(i));
    }
    swap(context_feature_exprs, tmp_context_feature_exprs);
}

} // end of namespace slnn
#endif
