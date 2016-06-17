#ifndef MODELMODULE_CONTEXT_FEATURE_EXTRACTOR_H_
#define MODELMODULE_CONTEXT_FEATURE_EXTRACTOR_H_

#include "context_feature.h"

namespace slnn{


struct ContextFeatureExtractor
{
    template <size_t N>
    static void extract(const IndexSeq &sent, typename ContextFeature<N>::ContextFeatureIndexGroupSeq &context_feature_gp_seq);
};

template <size_t N>
inline
void ContextFeatureExtractor::extract(const IndexSeq &sent, typename ContextFeature<N>::ContextFeatureIndexGroupSeq &context_feature_gp_seq)
{
    using std::swap;
    unsigned sent_len = sent.size();
    typename ContextFeature<N>::ContextFeatureIndexGroupSeq tmp_feature_gp_seq(sent_len,
        typename ContextFeature<N>::ContextFeatureIndexGroup(ContextFeature<N>::ContextSize));
    for( Index i = 0 ; i < static_cast<Index>(sent_len) ; ++i )
    {
        typename ContextFeature<N>::ContextFeatureIndexGroup &feature_gp = tmp_feature_gp_seq.at(i);
        unsigned feature_idx = 0 ;
        for( Index left_context_offset = 1 ; left_context_offset <= static_cast<Index>(ContextFeature<N>::ContextLeftSize) ; ++left_context_offset )
        {
            int word_pos = i - left_context_offset;
            feature_gp.at(feature_idx) = (word_pos < 0 ? ContextFeature<N>::WordSOSId : sent.at(word_pos)) ;
            ++feature_idx;
        }
        for( Index right_context_offset = 1 ; right_context_offset <= static_cast<Index>(ContextFeature<N>::ContextRightSize); ++right_context_offset )
        {
            int word_pos = i + right_context_offset;
            feature_gp.at(feature_idx) = ( word_pos >= static_cast<Index>(sent_len) 
                ? ContextFeature<N>::WordEOSId : sent.at(word_pos) );
            ++feature_idx;
        }
    }
    swap(context_feature_gp_seq, tmp_feature_gp_seq);
}

}
#endif