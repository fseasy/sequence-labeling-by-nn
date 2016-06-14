#ifndef MODELMODULE_CONTEXT_FEATURE_H_
#define MODELMODULE_CONTEXT_FEATURE_H_

#include <vector>
#include "utils/typedeclaration.h"
#include "utils/dict_wrapper.hpp"
namespace slnn{

struct ContextFeature
{
    const unsigned ContextSize; // left + right size 
    const unsigned ContextLeftSize ;
    const unsigned ContextRightSize;

    using ContextFeatureIndexGroup = std::vector<Index>;
    using ContextFeatureIndexGroupSeq = std::vector<ContextFeatureIndexGroup>;

    Index WordSOSId = -1;
    Index WordEOSId = -2;
    ContextFeature(unsigned context_size, DictWrapper &word_dict_wrapper);

    unsigned calc_context_feature_dim(unsigned word_embedding_dim);
    void replace_feature_index_group_with_unk(const ContextFeatureIndexGroup &context_feature_gp,
        ContextFeatureIndexGroup &context_feature_replaced_gp);
    void replace_feature_index_group_seq_with_unk(const ContextFeatureIndexGroupSeq &context_feature_gp_seq,
        ContextFeatureIndexGroupSeq &context_feature_unk_replaced_gp_seq);

private:
    DictWrapper &word_dict_wrapper; // for unk_replace
};

inline
unsigned ContextFeature::calc_context_feature_dim(unsigned word_embedding_dim)
{
    return ContextSize * word_embedding_dim;
}

inline
void ContextFeature::replace_feature_index_group_with_unk(const ContextFeatureIndexGroup &context_feature_gp,
    ContextFeatureIndexGroup &context_feature_replaced_gp)
{
    using std::swap;
    ContextFeatureIndexGroup tmp_feature_replaced_gp(context_feature_gp);
    for( unsigned i = 0 ; i < ContextSize; ++i )
    {
        if( word_id == WordSOSId || word_id == WordEOSId ){ continue; }
        else { tmp_feature_replaced_gp.at(i) = word_dict_wrapper.ConvertProbability(tmp_feature_repalced_gp.at(i)); }
    }
    swap(tmp_feature_replaced_gp, context_feature_replaced_gp);
}

inline 
void ContextFeature::replace_feature_index_group_seq_with_unk(const ContextFeatureIndexGroupSeq &context_feature_gp_seq,
    ContextFeatureIndexGroupSeq &context_feature_unk_replaced_gp_seq)
{
    using std::swap;
    unsigned sz = context_feature_gp_seq.size();
    ContextFeatureIndexGroupSeq tmp_replaced_gp_seq(sz);
    for( unsigned i = 0; i < sz; ++i )
    {
        replace_feature_index_group_with_unk(context_feature_gp_seq.at(i), tmp_replaced_gp_seq.at(i));
    }
    swap(context_feature_unk_replaced_gp_seq, tmp_replaced_gp_seq);
}

} // end of namespace slnn

#endif