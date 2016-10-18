#ifndef POSTAGGER_POSTAGGER_MODULE_POS_FEATURE_LAYER_H_
#define POSTAGGER_POSTAGGER_MODULE_POS_FEATURE_LAYER_H_

#include <array>
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "pos_feature.h"

namespace slnn{
class POSFeatureLayer
{
public:
    POSFeatureLayer(dynet::Model *model, 
                    size_t prefix_suffix_len1_dict_size, unsigned prefix_suffix_len1_embedding_dim,
                    size_t prefix_suffix_len2_dict_size, unsigned prefix_suffix_len2_embedding_dim,
                    size_t prefix_suffix_len3_dict_size, unsigned prefix_suffix_len3_embedding_dim,
                    size_t char_length_dict_size, unsigned char_length_embedding_dim);
    POSFeatureLayer(dynet::Model *model, POSFeature &pos_feature);
    void new_graph(dynet::ComputationGraph &cg);
    dynet::expr::Expression build_feature_expr(const POSFeature::POSFeatureIndexGroup &pos_feature_gp);
    void build_feature_exprs(const POSFeature::POSFeatureIndexGroupSeq &pos_feature_gp_seq,
                                              std::vector<dynet::expr::Expression> &pos_featrues_exprs);

private:
    dynet::LookupParameters *prefix_suffix_len1_lookup_param;
    dynet::LookupParameters *prefix_suffix_len2_lookup_param;
    dynet::LookupParameters *prefix_suffix_len3_lookup_param;
    dynet::LookupParameters *char_length_lookup_param;
    dynet::ComputationGraph *pcg ;

    dynet::expr::Expression build_prefix_suffix_feature_expr(dynet::LookupParameters* lookup_param, Index idx);
    dynet::expr::Expression build_char_length_feature_expr(Index idx);
};


inline
void POSFeatureLayer::new_graph(dynet::ComputationGraph &cg)
{
    pcg = &cg;
}

inline
dynet::expr::Expression POSFeatureLayer::build_prefix_suffix_feature_expr(dynet::LookupParameters* lookup_param, Index idx)
{
    if( idx == POSFeature::FeatureEmptyIndexPlaceholder )
    {
        return dynet::expr::zeroes(*pcg, lookup_param->dim);
    }
    else
    {
        return dynet::expr::lookup(*pcg, lookup_param, idx);
    }
}

inline
dynet::expr::Expression POSFeatureLayer::build_char_length_feature_expr(Index idx)
{
    return dynet::expr::lookup(*pcg, char_length_lookup_param, idx);
}

inline
dynet::expr::Expression POSFeatureLayer::build_feature_expr(const POSFeature::POSFeatureIndexGroup &feature_gp)
{
    return dynet::expr::concatenate({
        build_prefix_suffix_feature_expr(prefix_suffix_len1_lookup_param, feature_gp[0]),
        build_prefix_suffix_feature_expr(prefix_suffix_len2_lookup_param, feature_gp[1]),
        build_prefix_suffix_feature_expr(prefix_suffix_len3_lookup_param, feature_gp[2]),
        build_prefix_suffix_feature_expr(prefix_suffix_len1_lookup_param, feature_gp[3]),
        build_prefix_suffix_feature_expr(prefix_suffix_len2_lookup_param, feature_gp[4]),
        build_prefix_suffix_feature_expr(prefix_suffix_len3_lookup_param, feature_gp[5]),
        build_char_length_feature_expr(feature_gp[6])
    });
}

inline
void POSFeatureLayer::build_feature_exprs(const POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq,
                                          std::vector<dynet::expr::Expression> &pos_features_exprs)
{
    using std::swap;
    size_t len = feature_gp_seq.size();
    std::vector<dynet::expr::Expression> tmp_pos_features_exprs(len);
    for( size_t i = 0; i < len; ++i )
    {
        tmp_pos_features_exprs[i] = build_feature_expr(feature_gp_seq[i]);
    }
    swap(tmp_pos_features_exprs, pos_features_exprs);
}

} // end of namespace slnn

#endif
