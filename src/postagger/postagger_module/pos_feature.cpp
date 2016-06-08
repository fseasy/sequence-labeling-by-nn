#include "pos_feature.h"

namespace slnn{

const size_t POSFeature::NrFeature;
const size_t POSFeature::FeatureCharLengthLimit;
const size_t POSFeature::PrefixSuffixMaxLen;

const std::string POSFeature::FeatureEmptyStrPlaceholder = "";
const Index POSFeature::FeatureEmptyIndexPlaceholder = -1;
const std::string POSFeature::FeatureUnkStr = "feature_unk_str";

POSFeature::POSFeature()
    : prefix_suffix_len1_dict_wrapper(prefix_suffix_len1_dict),
    prefix_suffix_len2_dict_wrapper(prefix_suffix_len2_dict),
    prefix_suffix_len3_dict_wrapper(prefix_suffix_len3_dict)
{}

template <typename Archive>
void POSFeature::serialize(Archive &ar, const unsigned versoin)
{
    ar & prefix_suffix_len1_embedding_dim
        & prefix_suffix_len2_embedding_dim
        & prefix_suffix_len3_embedding_dim
        & char_length_embedding_dim
        & concatenated_feature_embedding_dim
        & prefix_suffix_len1_dict
        & prefix_suffix_len2_dict
        & prefix_suffix_len3_dict ;
}

}