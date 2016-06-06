#ifndef POSTAGGER_POSTAGGER_MODULE_POS_FEATURE_EXTRACTOR_HPP_
#define POSTAGGER_POSTAGGER_MODULE_POS_FEATURE_EXTRACTOR_HPP_
#include <string>

#include <boost/serialization/serialization.hpp>

#include "cnn/dict.h"
#include "utils/dict_wrapper.hpp"
#include "utils/typedeclaration.h"
#include "utils/utf8processing.hpp"
namespace slnn{

class POSFeatureExtractor
{
    friend class boost::serialization::access;
public:
    static const std::string FEATURE_UNK_STR ;
    static const Index FEATURE_NONE_IDX; // for padding of prefix , suffix feature when no enough chars
    static const size_t MAX_CHARS_LENGTH = 5 ; // word length feature upper bound
    static const size_t PREFIX_SUFFIX_MAX_LEN = 3 ;
    static const size_t NR_FEATURES = PREFIX_SUFFIX_MAX_LEN * 2 + 1 ; // 3 prefix features , 3 suffix features , 1 length feature.
    using POSFeaturesIndex = FeaturesIndex<NR_FEATURES>;
    using POSFeaturesIndexSeq = FeaturesIndexSeq<NR_FEATURES>;
public:
    POSFeatureExtractor();

    void extract(const Seq &raw_inputs, POSFeaturesIndexSeq &features_seq);
    void freeze_dict();

    size_t get_nr_features() { return NR_FEATURES; }
    size_t get_prefix_suffix_dict_size(){ return prefix_suffix_fdict.size() ; }
    size_t get_length_dict_size(){ return MAX_CHARS_LENGTH; }
    cnn::Dict& get_prefix_suffix_dict(){ return prefix_suffix_fdict; }

private:
    cnn::Dict prefix_suffix_fdict; // UNK

    DictWrapper prefix_suffix_fdict_wrapper;

    template<Archive>
    void serialize(Archive &ar, const unsigned version);
};

const std::string POSFeatureExtractor::FEATURE_UNK_STR = "feature_unk";

const Index POSFeatureExtractor::FEATURE_NONE_IDX = -1;

const size_t POSFeatureExtractor::MAX_CHARS_LENGTH;
const size_t POSFeatureExtractor::PREFIX_SUFFIX_MAX_LEN;
const size_t POSFeatureExtractor::NR_FEATURES;


POSFeatureExtractor::POSFeatureExtractor()
    :prefix_suffix_fdict_wrapper(prefix_suffix_fdict)
    {};

/* *
 *  extract features sequence from raw inputs .
 *  For postagger , there are NR_FEATURES for every words of raw inputs , 
 *  they are storing in the array , ordering like following :
 *          0 -> PREFIX_SUFFIX_MAX_LEN - 1                        : prefix feature index
 *          PREFIX_SUFFIX_MAX_LEN -> PREFIX_SUFFIX_MAX_LEN * 2 -1 : suffix feature index
 *          PREFIX_SUFFIX_MAX_LEN * 2                             : length feature index
*/
void POSFeatureExtractor::extract(const Seq &raw_inputs, POSFeaturesIndexSeq &features_seq)
{
    using std::swap;
    size_t nr_tokens = raw_inputs.size();
    POSFeaturesIndexSeq tmp_features_seq(nr_tokens);

    
    Seq utf8_chars ; utf8_chars.reserve(16);
    for( size_t i = 0; i < nr_tokens; ++i )
    {
        POSFeaturesIndex &cur_f = tmp_features_seq[i];
        const std::string &word = raw_inputs[i] ;
        UTF8Processing::utf8_str2char_seq(word, utf8_chars);
        size_t utf8_chars_len = utf8_chars.size() ;
        // prefix , suffix
        std::string prefix_chars, suffix_chars ;
        size_t min_len = std::min(PREFIX_SUFFIX_MAX_LEN, utf8_chars_len);
        for( size_t len = 1 ; len <= min_len ; ++len )
        {
            prefix_chars += word[len - 1] ;
            suffix_chars = word[utf8_chars_len - len];
            Index prefix_idx = prefix_suffix_fdict_wrapper.Convert("p_" + prefix_chars);
            Index suffix_idx = prefix_suffix_fdict_wrapper.Convert("s_" + suffix_chars);
            cur_f[len - 1] = prefix_idx;
            cur_f[len - 1 + PREFIX_SUFFIX_MAX_LEN] = suffix_idx;
        }
        // padding
        for( size_t len = min_len + 1 ; len <= PREFIX_SUFFIX_MAX_LEN; ++len )
        {
            cur_f[len - 1] = FEATURE_NONE_IDX;
            cur_f[len - 1 + PREFIX_SUFFIX_MAX_LEN] = FEATURE_NONE_IDX;
        }
        size_t length4feature = std::min(MAX_CHARS_LENGTH, utf8_chars_len) ;
        cur_f[NR_FEATURES - 1] = static_cast<Index>(length4feature);
    }
    swap(features_seq, tmp_features_seq);
}

void POSFeatureExtractor::freeze_dict()
{
    prefix_suffix_fdict_wrapper.Freeze();
    prefix_suffix_fdict_wrapper.SetUnk(FEATURE_UNK_STR);
}

template <typename Archive>
void POSFeatureExtractor::serialize(Archive &ar, const unsigned version)
{
    ar & prefix_suffix_fdict ;
}

} // end of namespace slnn


#endif