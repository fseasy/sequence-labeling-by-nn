#ifndef SLNN_SEGMENTOR_CWS_MODULE_CWS_FEATURE_LAYER_H_
#define SLNN_SEGMENTOR_CWS_MODULE_CWS_FEATURE_LAYER_H_

#include "cws_feature.h"
#include "lexicon_feature_layer.h"

namespace slnn{

class CWSFeatureLayer
{
public:
    CWSFeatureLayer(cnn::Model *cnn_m, unsigned start_here_dict_size, unsigned start_here_dim,
        unsigned pass_here_dict_size, unsigned pass_here_dim,
        unsigned end_here_dict_size, unsigned end_here_dim);
    CWSFeatureLayer(cnn::Model *cnn_m, const CWSFeature &cws_feature);
    void new_graph(cnn::ComputationGraph &cg);
    void build_cws_feature(const CWSFeatureDataSeq &cws_feature_data_seq, std::vector<cnn::expr::Expression> &cws_feature_exprs);

private:
    LexiconFeatureLayer lexicon_feature_layer;
};

inline
void CWSFeatureLayer::new_graph(cnn::ComputationGraph &cg)
{
    lexicon_feature_layer.new_graph(cg);
}

inline
void CWSFeatureLayer::build_cws_feature(const CWSFeatureDataSeq &cws_feature_data_seq, std::vector<cnn::expr::Expression> &cws_feature_exprs)
{
    lexicon_feature_layer.build_lexicon_feature(cws_feature_data_seq.get_lexicon_feature_data_seq(), cws_feature_exprs);
}


} // end of namespace slnn
#endif