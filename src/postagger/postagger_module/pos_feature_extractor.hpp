#ifndef POSTAGGER_POSTAGGER_MODULE_POS_FEATURE_EXTRACTOR_HPP_
#define POSTAGGER_POSTAGGER_MODULE_POS_FEATURE_EXTRACTOR_HPP_
#include <string>

#include "pos_feature.hpp"
#include "utils/utf8processing.hpp"

namespace slnn{

class POSFeatureExtractor
{
    static void extract(const Seq &raw_inputs, POSFeature::POSFeatureGroupSeq &feature_seq);

};

/* *
 *  extract features sequence from raw inputs .
 *  For postagger , there are NR_FEATURES for every words of raw inputs , 
 *  they are storing in the array , ordering like following :
 *          0 -> PREFIX_SUFFIX_MAX_LEN - 1                        : prefix feature index
 *          PREFIX_SUFFIX_MAX_LEN -> PREFIX_SUFFIX_MAX_LEN * 2 -1 : suffix feature index
 *          PREFIX_SUFFIX_MAX_LEN * 2                             : length feature index
*/
void POSFeatureExtractor::extract(const Seq &raw_inputs, POSFeature::POSFeatureGroupSeq &features_seq)
{
    using std::swap;
    size_t nr_tokens = raw_inputs.size();
    POSFeature::POSFeatureGroupSeq tmp_features_seq(nr_tokens);
    for( size_t i = 0; i < nr_tokens; ++i )
    {
        POSFeature::POSFeatureGroup &cur_f = tmp_features_seq[i];
        const std::string &word = raw_inputs[i] ;
        Seq utf8_chars ;
        UTF8Processing::utf8_str2char_seq(word, utf8_chars);
        size_t utf8_chars_len = utf8_chars.size() ;
        // prefix , suffix
        std::string prefix_chars, suffix_chars ;
        size_t min_len = std::min(POSFeature::PrefixSuffixMaxLen, utf8_chars_len);
        for( size_t len = 1 ; len <= min_len ; ++len )
        {
            prefix_chars += word[len - 1] ;
            suffix_chars = word[utf8_chars_len - len] + suffix_chars;
            cur_f[len - 1] = prefix_chars;
            cur_f[len - 1 + POSFeature::PrefixSuffixMaxLen] = suffix_chars;
            
        }
        // padding
        for( size_t len = min_len + 1 ; len <= POSFeature::PrefixSuffixMaxLen; ++len )
        {
            cur_f[len - 1] = POSFeature::FeatureEmptyStrPlaceholder;
            cur_f[len - 1 + POSFeature::PrefixSuffixMaxLen] = POSFeature::FeatureEmptyStrPlaceholder;
        }
        cur_f[POSFeature::NrFeature - 1] = std::to_string(utf8_chars_len);
    }
    swap(features_seq, tmp_features_seq);
}

} // end of namespace slnn


#endif