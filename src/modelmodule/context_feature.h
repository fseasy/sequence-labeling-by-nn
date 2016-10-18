#ifndef MODELMODULE_CONTEXT_FEATURE_H_
#define MODELMODULE_CONTEXT_FEATURE_H_

#include <vector>
#include <sstream>
#include <iostream>
#include "dynet/dynet.h"
#include "utils/typedeclaration.h"
#include "utils/dict_wrapper.hpp"
namespace slnn{

using ContextFeatureData = std::vector<Index>;
using ContextFeatureDataSeq = std::vector<ContextFeatureData>;

class ContextFeature
{
    friend class boost::serialization::access;
public:
    static const Index WordSOSId = -1;
    static const Index WordEOSId = -2;
    // WHY use default ? because in our program structure , the param will be known after the initialization 
    ContextFeature(DictWrapper &dict_wrapper, unsigned context_left_size=0, unsigned context_right_size=0, unsigned word_dim=0);
    void set_parameters(unsigned context_left_size, unsigned context_right_size, unsigned word_dim);
    unsigned get_feature_dim() const { return context_size * word_dim; }
    
    void extract(const IndexSeq &seq, ContextFeatureDataSeq &context_feature_seq);
    void random_replace_with_unk(const ContextFeatureData &context_feature_data, ContextFeatureData &replaced_feature_data);
    void random_replace_with_unk(const ContextFeatureDataSeq &context_feature_data_seq, 
        ContextFeatureDataSeq &replaced_feature_data_seq);
    std::string get_feature_info() const;
    template<typename Archive>
    void serialize(Archive &ar, unsigned version);

    // DEBUG
    void debug_context_feature_seq(const ContextFeatureDataSeq &context_feature_data_seq);
private:
    void replace_wordid_with_unk(Index &wordId);

private:
    int context_size; // because we'll use minus , to avoid unnecessary static cast, we choose int .
    int context_left_size;
    int context_right_size;

    DictWrapper &rwrapper;
    unsigned word_dim;
};

inline
void ContextFeature::replace_wordid_with_unk(Index &wordid)
{
    if( WordSOSId != wordid && WordEOSId != wordid ){ wordid = rwrapper.ConvertProbability(wordid); }
}

inline 
void ContextFeature::random_replace_with_unk(const ContextFeatureData &context_feature_data, ContextFeatureData &replaced_feature_data)
{
    using std::swap;
    ContextFeatureData tmp_rep_data(context_feature_data);
    for( Index &wordid : tmp_rep_data ){ replace_wordid_with_unk(wordid); }
    swap(replaced_feature_data, tmp_rep_data);
}

inline 
void ContextFeature::random_replace_with_unk(const ContextFeatureDataSeq &context_feature_data_seq,
    ContextFeatureDataSeq &replaced_feature_data_seq)
{
    using std::swap;
    ContextFeatureDataSeq tmp_rep_seq(context_feature_data_seq);
    for( ContextFeatureData &fdata : tmp_rep_seq )
    {
        for( Index &wordid : fdata ){ replace_wordid_with_unk(wordid); }
    }
    swap(replaced_feature_data_seq, tmp_rep_seq);
}

template <typename Archive>
void ContextFeature::serialize(Archive &ar, unsigned version)
{
    ar & context_size & context_left_size & context_right_size
        &word_dim; // word dim is somewhat duplicate
}

} // end of namespace slnn

#endif
