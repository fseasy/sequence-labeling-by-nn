#ifndef MODELMODULE_CONTEXT_FEATURE_H_
#define MODELMODULE_CONTEXT_FEATURE_H_

#include <vector>
#include <sstream>
#include "cnn/cnn.h"
#include "utils/typedeclaration.h"
#include "utils/dict_wrapper.hpp"
namespace slnn{


template <size_t N>
struct ContextFeature
{
    static constexpr unsigned ContextSize = N; // left + right size 
    static constexpr unsigned ContextLeftSize = N / 2;
    static constexpr unsigned ContextRightSize = N - N / 2;

    using ContextFeatureIndexGroup = std::vector<Index>;
    using ContextFeatureIndexGroupSeq = std::vector<ContextFeatureIndexGroup>;

    static constexpr Index WordSOSId = -1;
    static constexpr Index WordEOSId = -2;
    ContextFeature(DictWrapper &word_dict_wrapper);

    unsigned calc_context_feature_dim(unsigned word_embedding_dim);
    void replace_feature_index_group_with_unk(const ContextFeatureIndexGroup &context_feature_gp,
        ContextFeatureIndexGroup &context_feature_replaced_gp);
    void replace_feature_index_group_seq_with_unk(const ContextFeatureIndexGroupSeq &context_feature_gp_seq,
        ContextFeatureIndexGroupSeq &context_feature_unk_replaced_gp_seq);

    std::string get_context_info();

private:
    DictWrapper &word_dict_wrapper; // for unk_replace
};

template <size_t N>
constexpr unsigned ContextFeature<N>::ContextSize ;

template <size_t N>
constexpr unsigned ContextFeature<N>::ContextLeftSize ;

template <size_t N>
constexpr unsigned ContextFeature<N>::ContextRightSize ;

template <size_t N>
constexpr Index ContextFeature<N>::WordSOSId ;

template <size_t N>
constexpr Index ContextFeature<N>::WordEOSId ;

template <size_t N>
ContextFeature<N>::ContextFeature(DictWrapper &word_dict_wrapper)
    :word_dict_wrapper(word_dict_wrapper)
{}

template <size_t N>
inline
unsigned ContextFeature<N>::calc_context_feature_dim(unsigned word_embedding_dim)
{
    return ContextSize * word_embedding_dim;
}

template <size_t N>
inline
void ContextFeature<N>::replace_feature_index_group_with_unk(const ContextFeatureIndexGroup &context_feature_gp,
    ContextFeatureIndexGroup &context_feature_replaced_gp)
{
    using std::swap;
    ContextFeatureIndexGroup tmp_feature_replaced_gp(context_feature_gp);
    for( unsigned i = 0 ; i < ContextSize; ++i )
    {
        Index word_id = tmp_feature_replaced_gp.at(i);
        if( word_id == WordSOSId || word_id == WordEOSId ){ continue; }
        else { tmp_feature_replaced_gp.at(i) = word_dict_wrapper.ConvertProbability(word_id); }
    }
    swap(tmp_feature_replaced_gp, context_feature_replaced_gp);
}

template <size_t N>
inline 
void ContextFeature<N>::replace_feature_index_group_seq_with_unk(const ContextFeatureIndexGroupSeq &context_feature_gp_seq,
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

template <size_t N>
std::string ContextFeature<N>::get_context_info()
{
    std::ostringstream oss;
    oss << "total context size : " << ContextSize << " , left context size : " << ContextLeftSize
        << " , right context size : " << ContextRightSize ;
    return oss.str();
}

} // end of namespace slnn

#endif
