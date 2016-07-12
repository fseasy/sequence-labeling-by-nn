#ifndef MODELMODULE_CONTEXT_FEATURE_LAYER_H_
#define MODELMODULE_CONTEXT_FEATURE_LAYER_H_

#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "context_feature.h"
namespace slnn{

struct ContextFeatureLayer
{
    ContextFeatureLayer(cnn::Model *m, cnn::LookupParameters *word_lookup_param);
    void new_graph(cnn::ComputationGraph &cg);
    cnn::expr::Expression build_feature_expr(const ContextFeatureData &context_feature_data);
    
    void build_feature_exprs(const ContextFeatureDataSeq &context_feature_data_seq,
        std::vector<cnn::expr::Expression> &context_feature_exprs);
private:
    cnn::expr::Expression build_word_expr(Index word_id);
    cnn::LookupParameters *word_lookup_param;
    cnn::ComputationGraph *pcg;
    cnn::Parameters *word_sos_param;
    cnn::Parameters *word_eos_param;
    cnn::expr::Expression word_sos_expr;
    cnn::expr::Expression word_eos_expr;
};

ContextFeatureLayer::ContextFeatureLayer(cnn::Model *m, cnn::LookupParameters *word_lookup_param)
    :word_lookup_param(word_lookup_param),
    pcg(nullptr),
    word_sos_param(m->add_parameters(word_lookup_param->dim)),
    word_eos_param(m->add_parameters(word_lookup_param->dim))
{}

inline 
void ContextFeatureLayer::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg;
    word_sos_expr = cnn::expr::parameter(cg, word_sos_param);
    word_eos_expr = cnn::expr::parameter(cg, word_eos_param);
}

inline 
cnn::expr::Expression ContextFeatureLayer::build_word_expr(Index word_id)
{
    if( word_id == ContextFeature::WordSOSId ){ return word_sos_expr ; }
    else if( word_id == ContextFeature::WordEOSId ){ return word_eos_expr; }
    else { return cnn::expr::lookup(*pcg, word_lookup_param, word_id); }
}

inline 
cnn::expr::Expression ContextFeatureLayer::build_feature_expr(const ContextFeatureData &context_feature_data)
{
    // this may can be optimized ! 
    // context_feature_data.size() always the same , and contxt_word_exprs may be declaration as static or member variable
    // but we give up ! 
    // it should not be the key factor of speed .
    // for static and member variable , we may get some diffeculties when multi-threads
    // for constant context_feature_data.size() , we need the information from the ContextFeature instance.
    size_t sz = context_feature_data.size();
    std::vector<cnn::expr::Expression> context_word_exprs(sz); 
    for( unsigned i = 0 ; i < sz ; ++i )
    {
        context_word_exprs.at(i) = build_word_expr(context_feature_data.at(i));
    }
    return cnn::expr::concatenate(context_word_exprs);
}

inline 
void ContextFeatureLayer::build_feature_exprs(const ContextFeatureDataSeq &context_feature_data_seq,
    std::vector<cnn::expr::Expression> &context_feature_exprs)
{
    using std::swap;
    unsigned seq_len = context_feature_data_seq.size();
    std::vector<cnn::expr::Expression> tmp_context_feature_exprs(seq_len);
    for( unsigned i = 0; i < seq_len; ++i )
    {
        tmp_context_feature_exprs.at(i) = build_feature_expr(context_feature_data_seq.at(i));
    }
    swap(context_feature_exprs, tmp_context_feature_exprs);
}

} // end of namespace slnn
#endif
