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


std::string POSFeature::get_feature_info()
{
    std::ostringstream oss;

    oss << "prefix and suffix dict size : [ " << prefix_suffix_len1_dict.size() << ", " << prefix_suffix_len2_dict.size() << ", "
        << prefix_suffix_len3_dict.size() << " ]\n"
        << "prefix and suffix embedding dim : [ " << prefix_suffix_len1_embedding_dim << ", " << prefix_suffix_len2_embedding_dim << ", "
        << prefix_suffix_len3_embedding_dim << " ]\n"
        << "character length feature dict size : " << get_char_length_dict_size() << " , dimension : " << char_length_embedding_dim << "\n"
        << "total pos feature dimension : " << get_pos_feature_dim() ;
    return oss.str();
}
}
