#ifndef SLNN_SEGMENTOR_CWS_MODULE_CWS_FEATURE_LAYER_H_
#define SLNN_SEGMENTOR_CWS_MODULE_CWS_FEATURE_LAYER_H_

#include "cws_feature.h"
#include "lexicon_feature_layer.h"
#include "modelmodule/context_feature_layer.h"
#include "modelmodule/hyper_layers.h"

namespace slnn{

class CWSFeatureLayer
{
public:
    CWSFeatureLayer(dynet::Model *dynet_m, unsigned start_here_dict_size, unsigned start_here_dim,
        unsigned pass_here_dict_size, unsigned pass_here_dim,
        unsigned end_here_dict_size, unsigned end_here_dim,
        dynet::LookupParameters *word_lookup_param,
        unsigned chartype_category_num, unsigned chartype_dim);
    CWSFeatureLayer(dynet::Model *dynet_m, const CWSFeature &cws_feature, dynet::LookupParameters *word_lookup_param);
    void new_graph(dynet::ComputationGraph &cg);
    void build_cws_feature(const CWSFeatureDataSeq &cws_feature_data_seq, std::vector<dynet::expr::Expression> &cws_feature_exprs);

private:
    LexiconFeatureLayer lexicon_feature_layer;
    ContextFeatureLayer context_feature_layer;
    Index2ExprLayer chartype_feature_layer;
};

inline
void CWSFeatureLayer::new_graph(dynet::ComputationGraph &cg)
{
    lexicon_feature_layer.new_graph(cg);
    context_feature_layer.new_graph(cg);
    chartype_feature_layer.new_graph(cg);
}

inline
void CWSFeatureLayer::build_cws_feature(const CWSFeatureDataSeq &cws_feature_data_seq, std::vector<dynet::expr::Expression> &cws_feature_exprs)
{
    using std::swap;
    size_t seq_len = cws_feature_data_seq.size();
    std::vector<dynet::expr::Expression> tmp_cws_exprs(seq_len);
    const LexiconFeatureDataSeq &lexicon_data_seq = cws_feature_data_seq.get_lexicon_feature_data_seq();
    const ContextFeatureDataSeq &context_data_seq = cws_feature_data_seq.get_context_feature_data_seq();
    const CharTypeFeatureDataSeq &chartype_data_seq = cws_feature_data_seq.get_chartype_feature_data_seq();
    for( size_t i = 0; i < seq_len; ++i )
    {
        tmp_cws_exprs[i] = dynet::expr::concatenate({
            lexicon_feature_layer.build_lexicon_feature(lexicon_data_seq[i]),
            context_feature_layer.build_feature_expr(context_data_seq[i]),
            chartype_feature_layer.index2expr(chartype_data_seq[i])
        });
    }
    swap(cws_feature_exprs, tmp_cws_exprs);
}


} // end of namespace slnn
#endif