#ifndef POS_POS_MODULE_POS_FEATURE_HPP_
#define POS_POS_MODULE_POS_FEATURE_HPP_
#include <string>
#include "cnn/cnn.h"
#include "utils/dict_wrapper.hpp"
#include "utils/typedeclaration.h"
namespace slnn{

struct POSFeature
{
    static const size_t NrFeature = 7 ;
    static const size_t FeatureCharLengthLimit = 5 ;

    static const std::string FeatureEmptyStrPlaceholder;
    static const Index FeatureEmptyIndexPlaceholder;
    static const std::string FeatureUnkStr;

    const size_t prefix_suffix_len1_embedding_dim;
    const size_t prefix_suffix_len2_embedding_dim;
    const size_t prefix_suffix_len3_embedding_dim;
    const size_t char_length_embedding_dim;
    const size_t concatenated_feature_embedding_dim; 

    cnn::dict prefix_suffix_len1_dict;
    cnn::dict prefix_suffix_len2_dict;
    cnn::dict prefix_suffix_len3_dict;

    using POSFeatureIndexGroup = FeaturesIndex<NrFeature>;
    using POSFeatureIndexGroupSeq = FeaturesIndexSeq<NrFeature>;
    using POSFeatureGroup = FeatureGroup<NrFeature>;
    using POSFeatureGroupSeq = FeatureGroupSeq<NrFeature>;

    POSFeature(size_t prefix_suffix_len1_embedding_dim,
               size_t prefix_suffix_len2_embedding_dim,
               size_t prefix_suffix_len3_embedding_dim,
               size_t char_length_embedding_dim);
    void feature_group2feature_index_group(const POSFeatureGroup &feature_gp,
                                           POSFeatureIndexGroup &feature_index_gp);
    void feature_group_seq2feature_index_group_seq(const POSFeatureGroupSeq &feature_gp_seq,
                                                   POSFeatureIndexGroupSeq &feature_index_gp_seq);
};

const std::string POSFeature::FeatureEmptyStrPlaceholder = "feature_empty_str_placeholder";
const Index POSFeature::FeatureEmptyIndexPlaceholder = -1;
const std::string POSFeature::FeatureUnkStr = "feature_unk_str";

POSFeature::POSFeature(size_t prefix_suffix_len1_embedding_dim,
                       size_t prefix_suffix_len2_embedding_dim,
                       size_t prefix_suffix_len3_embedding_dim,
                       size_t char_length_embedding_dim)
    :prefix_suffix_len1_embedding_dim(prefix_suffix_len1_embedding_dim),
    prefix_suffix_len2_embedding_dim(prefix_suffix_len2_embedding_dim),
    prefix_suffix_len3_embedding_dim(prefix_suffix_len3_embedding_dim),
    char_length_embedding_dim(char_length_embedding_dim),
    concatenated_feature_embedding_dim(
    prefix_suffix_len1_embedding_dim * 2 + prefix_suffix_len2_embedding_dim * 2 +
    prefix_suffix_len3_embedding_dim * 2 + char_length_embedding_dim
    )
{}

}
#endif