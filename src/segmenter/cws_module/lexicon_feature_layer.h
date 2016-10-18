#ifndef SLNN_SEGMENTOR_CWS_MODULE_LEXICON_FEATURE_LAYER_H_
#define SLNN_SEGMENTOR_CWS_MODULE_LEXICON_FEATURE_LAYER_H_
#include "lexicon_feature.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
namespace slnn{

class LexiconFeatureLayer
{
public :
    LexiconFeatureLayer(dynet::Model *dynet_m, unsigned start_here_dict_size, unsigned start_here_dim,
        unsigned pass_here_dict_size, unsigned pass_here_dim,
        unsigned end_here_dict_size, unsigned end_here_dim);
    LexiconFeatureLayer(dynet::Model *dynet_m, const LexiconFeature &lexicon_feature);
    void new_graph(dynet::ComputationGraph &cg);
    dynet::expr::Expression build_lexicon_feature(const LexiconFeatureData &lexicon_feature);
    void build_lexicon_feature(const LexiconFeatureDataSeq &lexicon_feature_seq,
        std::vector<dynet::expr::Expression> &lexicon_feature_exprs);

private:
    dynet::LookupParameters *start_here_lookup_param;
    dynet::LookupParameters *pass_here_lookup_param;
    dynet::LookupParameters *end_here_lookup_param;
    dynet::ComputationGraph *pcg;
};

inline
void LexiconFeatureLayer::new_graph(dynet::ComputationGraph &cg)
{
    pcg = &cg;
}

inline 
dynet::expr::Expression LexiconFeatureLayer::build_lexicon_feature(const LexiconFeatureData &lexicon_feature_data)
{
    return dynet::expr::concatenate({
        dynet::expr::lookup(*pcg, start_here_lookup_param, lexicon_feature_data.get_start_here_feature_index()),
        dynet::expr::lookup(*pcg, pass_here_lookup_param, lexicon_feature_data.get_pass_here_feature_index()),
        dynet::expr::lookup(*pcg, end_here_lookup_param, lexicon_feature_data.get_end_here_feature_index())
    });
}

inline
void LexiconFeatureLayer::build_lexicon_feature(const LexiconFeatureDataSeq &lexicon_feature_seq,
    std::vector<dynet::expr::Expression> &lexicon_feature_exprs)
{
    using std::swap;
    size_t seq_len = lexicon_feature_seq.size();
    std::vector<dynet::expr::Expression> tmp_lexicon_feature_exprs(seq_len);
    for( size_t i = 0; i < seq_len; ++i )
    {
        tmp_lexicon_feature_exprs[i] = build_lexicon_feature(lexicon_feature_seq[i]);
    }
    swap(lexicon_feature_exprs, tmp_lexicon_feature_exprs);
}

} // end of namespace slnn

#endif