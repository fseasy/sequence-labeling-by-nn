#ifndef MODELMODULE_CONTEXT_FEATURE_EXTRACTOR_H_
#define MODELMODULE_CONTEXT_FEATURE_EXTRACTOR_H_

#include "context_feature.h"

namespace slnn{

struct ContextFeatureExtractor
{
    void extract(const IndexSeq &sent, ContextFeature::ContexFeatureIndexGroupSeq &context_feature_gp_seq);
};

inline
void ContextFeatureExtractor::extract(const IndexSeq &sent, ContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq)
{
    using std::swap;
    unsigned sent_len = sent.size();
    ContextFeature::ContextFeatureIndexGroupSeq tmp_feature_gp_seq(sent_len,
        ContextFeature::ContextFeatureIndexGroup(ContextFeature::ContextSize));
    for( Index i = 0 ; i < static_cast<Index>(sent_len) ; ++i )
    {
        ContextFeature::ContextFeatureIndexGroup &feature_gp = tmp_feature_gp_seq.at(i);
        unsigned feature_idx = 0 ;
        for( Index left_context_offset = 1 ; left_context_offset <= ContextFeature::ContextLeftSize ; ++left_context_offset )
        {
            Index word_idx = i - left_context_offset;
            feature_gp.at(feature_idx) = (word_idx < 0 ? ContextFeature::WordSOSId : word_idx) ;
            ++feature_idx;
        }
        for( Index right_context_offset = 1 ; right_context_offset <= ContextFeature::ContextRightSize; ++right_context_offset )
        {
            Index word_idx = i + right_context_offset;
            feature_gp.at(feature_idx) = (word_idx >= sent_len ? ContextFeature::WordEOSId : word_idx);
            ++feature_idx;
        }
    }
    swap(context_feature_gp_seq, tmp_feature_gp_seq);
}

}
#endif