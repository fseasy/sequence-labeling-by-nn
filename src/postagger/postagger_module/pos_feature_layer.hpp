#ifndef POSTAGGER_POSTAGGER_MODULE_POS_FEATURE_LAYER_HPP_
#define POSTAGGER_POSTAGGER_MODULE_POS_FEATURE_LAYER_HPP_

#include <array>
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "pos_feature_extractor.hpp"

namespace slnn{
class POSFeatureLayer
{
public:
    POSFeatureLayer(cnn::Model *model, size_t prefix_suffix_fdict_size, size_t prefix_suffix_feature_dim,
                    size_t length_fdict_size, size_t length_feature_dim);
    size_t get_concatenate_feature_dim();
    void new_graph(cnn::ComputationGraph &cg);
    cnn::expr::Expression build_feature_expr(const POSFeatureExtractor::POSFeaturesIndex &pos_features);
    void build_feature_exprs(const POSFeatureExtractor::POSFeaturesIndexSeq &pos_features_seq,
                                              std::vector<cnn::expr::Expression> &pos_featrues_exprs);

private:
    cnn::LookupParameters *prefix_suffix_feature_lookup_param;
    cnn::LookupParameters *length_lookup_feature_param;
    std::array <cnn::expr::Expression, POSFeatureExtractor::PREFIX_SUFFIX_MAX_LEN > prefix_exprs;
    std::array<cnn::expr::Expression, POSFeatureExtractor::PREFIX_SUFFIX_MAX_LEN> suffix_exprs;
    cnn::expr::Expression length_expr;
    cnn::ComputationGraph *pcg ;
};

POSFeatureLayer::POSFeatureLayer(cnn::Model *model, size_t prefix_suffix_fdict_size, size_t prefix_suffix_feature_dim,
                                 size_t length_fdict_size, size_t length_feature_dim)
    :prefix_suffix_feature_lookup_param(model->add_lookup_parameters(prefix_suffix_fdict_size, {prefix_suffix_feature_dim})),
    length_lookup_feature_param(model->add_lookup_parameters(length_fdict_size, {length_feature_dim}))
{}

size_t POSFeatureLayer::get_concatenate_feature_dim()
{
    // [ prefix_feature * PREFIX_SUFFIX_MAX_LEN, suffix_feature * PREFIX_SUFFIX_MAX_LEN, length ]
    return (prefix_suffix_feature_lookup_param->dim.rows() * POSFeatureExtractor::PREFIX_SUFFIX_MAX_LEN * 2 +
            length_lookup_feature_param->dim.rows()); 
}

void POSFeatureLayer::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg;
}

inline
cnn::expr::Expression POSFeatureLayer::build_feature_expr(const POSFeatureExtractor::POSFeaturesIndex &pos_features)
{
    std::array<cnn::expr::Expression, POSFeatureExtractor::NR_FEATURES> features_exprs;
    for( size_t i = 0; i < POSFeatureExtractor::PREFIX_SUFFIX_MAX_LEN * 2; ++i )
    {
        if( pos_features[i] != POSFeatureExtractor::FEATURE_NONE_IDX )
        {
            features_exprs[i] = cnn::expr::lookup(*pcg, prefix_suffix_feature_lookup_param, pos_features[i]);
        }
        else
        {
            features_exprs[i] = cnn::expr::zeroes(*pcg, prefix_suffix_feature_lookup_param->dim);
        }
    }
    features_exprs[POSFeatureExtractor::NR_FEATURES - 1] = cnn::expr::lookup(*pcg, length_lookup_feature_param, 
                                                                  pos_features[POSFeatureExtractor::NR_FEATURES - 1]);
    return cnn::expr::concatenate({
        features_exprs[0], features_exprs[1], features_exprs[2], // prefix
        features_exprs[3], features_exprs[4], features_exprs[5], // suffix
        features_exprs[6]  // length
    }); 
}

void POSFeatureLayer::build_feature_exprs(const POSFeatureExtractor::POSFeaturesIndexSeq &pos_features_seq,
                                          std::vector<cnn::expr::Expression> &pos_features_exprs)
{
    using std::swap;
    size_t len = pos_features_seq.size();
    std::vector<cnn::expr::Expression> tmp_pos_features_exprs(len);
    for( size_t i = 0; i < len; ++i )
    {
        tmp_pos_features_exprs[i] = build_feature_expr(pos_features_seq[i]);
    }
    swap(tmp_pos_features_exprs, pos_features_exprs);
}

} // end of namespace slnn

#endif