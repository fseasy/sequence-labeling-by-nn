#ifndef POS_POS_MODULE_POS_FEATURE_HPP_
#define POS_POS_MODULE_POS_FEATURE_HPP_
#include <string>
#include <boost/serialization/serialization.hpp>
#include "cnn/cnn.h"
#include "utils/dict_wrapper.hpp"
#include "utils/typedeclaration.h"

namespace slnn{

struct POSFeature
{
    friend class boost::serialization::access;
    static const size_t NrFeature = 7 ;
    static const size_t FeatureCharLengthLimit = 5 ;
    static const size_t PrefixSuffixMaxLen = 3;

    static const std::string FeatureEmptyStrPlaceholder;
    static const Index FeatureEmptyIndexPlaceholder;
    static const std::string FeatureUnkStr;

    size_t prefix_suffix_len1_embedding_dim;
    size_t prefix_suffix_len2_embedding_dim;
    size_t prefix_suffix_len3_embedding_dim;
    size_t char_length_embedding_dim;
    size_t concatenated_feature_embedding_dim;

    cnn::Dict prefix_suffix_len1_dict;
    cnn::Dict prefix_suffix_len2_dict;
    cnn::Dict prefix_suffix_len3_dict;

    DictWrapper prefix_suffix_len1_dict_wrapper;
    DictWrapper prefix_suffix_len2_dict_wrapper;
    DictWrapper prefix_suffix_len3_dict_wrapper;

    using POSFeatureIndexGroup = FeaturesIndex<NrFeature>;
    using POSFeatureIndexGroupSeq = FeaturesIndexSeq<NrFeature>;
    using POSFeatureGroup = FeatureGroup<NrFeature>;
    using POSFeatureGroupSeq = FeatureGroupSeq<NrFeature>;
    POSFeature();

    void init_embedding_dim(size_t prefix_suffix_len1_embedding_dim,
                            size_t prefix_suffix_len2_embedding_dim,
                            size_t prefix_suffix_len3_embedding_dim,
                            size_t char_length_embedding_dim);
    // Dict interface
    size_t get_char_length_dict_size(){ return FeatureCharLengthLimit; }
    bool is_dict_frozen();
    bool freeze_dict();

    // replace word with unk interface
    void set_replace_feature_with_unk_threshold(int freq_thres, float prob_thres);
    void do_repalce_feature_with_unk_in_copy(const POSFeatureIndexGroupSeq &gp_seq,
                                          POSFeatureIndexGroupSeq &rep_gp_seq);

    // translate feature str to feature index , and storing feature in dict (in training / dict not frozen)
    void feature_group2feature_index_group(const POSFeatureGroup &feature_gp,
                                           POSFeatureIndexGroup &feature_index_gp);
    void feature_group_seq2feature_index_group_seq(const POSFeatureGroupSeq &feature_gp_seq,
                                                   POSFeatureIndexGroupSeq &feature_index_gp_seq);
private:
    Index prefix_suffix_feature_str2feature_idx_and_adding2dict_in_training(DictWrapper &dw, const std::string &feature_str);
    Index char_length_feature_str2feature_idx_and_adding2dict_in_training(const std::string &feature_str);
    
    template <typename Archive>
    void serialize(Archive &ar, const unsigned version);
};

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
void POSFeature::init_embedding_dim(size_t prefix_suffix_len1_embedding_dim,
                                    size_t prefix_suffix_len2_embedding_dim,
                                    size_t prefix_suffix_len3_embedding_dim,
                                    size_t char_length_embedding_dim)
{
    this->prefix_suffix_len1_embedding_dim = prefix_suffix_len1_embedding_dim ;
    this->prefix_suffix_len2_embedding_dim = prefix_suffix_len2_embedding_dim ;
    this->prefix_suffix_len3_embedding_dim = prefix_suffix_len3_embedding_dim ;
    this->char_length_embedding_dim = char_length_embedding_dim ;
    concatenated_feature_embedding_dim = (
        prefix_suffix_len1_embedding_dim * 2 + prefix_suffix_len2_embedding_dim * 2 +
        prefix_suffix_len3_embedding_dim * 2 + char_length_embedding_dim
        );
}

bool POSFeature::is_dict_frozen()
{
    return (prefix_suffix_len1_dict.is_frozen() && prefix_suffix_len2_dict.is_frozen() &&
            prefix_suffix_len3_dict.is_frozen());
}

bool POSFeature::freeze_dict()
{
    prefix_suffix_len1_dict_wrapper.Freeze(); prefix_suffix_len1_dict_wrapper.SetUnk(FeatureUnkStr);
    prefix_suffix_len2_dict_wrapper.Freeze(); prefix_suffix_len2_dict_wrapper.SetUnk(FeatureUnkStr);
    prefix_suffix_len3_dict_wrapper.Freeze(); prefix_suffix_len3_dict_wrapper.SetUnk(FeatureUnkStr);
}

void POSFeature::set_replace_feature_with_unk_threshold(int freq_thres, float prob_thres)
{
    prefix_suffix_len1_dict_wrapper.set_threshold(freq_thres, prob_thres);
    prefix_suffix_len2_dict_wrapper.set_threshold(freq_thres, prob_thres);
    prefix_suffix_len3_dict_wrapper.set_threshold(freq_thres, prob_thres);
}

void POSFeature::do_repalce_feature_with_unk_in_copy(const POSFeatureIndexGroupSeq &gp_seq,
                                      POSFeatureIndexGroupSeq &rep_gp_seq)
{
    using std::swap;
    static auto word_replace_with_unk = [](DictWrapper &dw, Index idx)->Index
    {
        if( idx != FeatureEmptyIndexPlaceholder ) { return dw.ConvertProbability(idx); }
        else { return idx ; }
    } ;
    size_t seq_len = gp_seq.size();
    POSFeatureIndexGroupSeq tmp_rep_gp_seq(seq_len);
    for( size_t i = 0; i < seq_len; ++i )
    {
        const POSFeatureIndexGroup &ori_gp = gp_seq[i];
        POSFeatureIndexGroup &rep_gp = tmp_rep_gp_seq[i];
        rep_gp[0] = word_replace_with_unk(prefix_suffix_len1_dict_wrapper, ori_gp[0]);
        rep_gp[1] = word_replace_with_unk(prefix_suffix_len2_dict_wrapper, ori_gp[1]);
        rep_gp[2] = word_replace_with_unk(prefix_suffix_len3_dict_wrapper, ori_gp[2]);
        rep_gp[3] = word_replace_with_unk(prefix_suffix_len1_dict_wrapper, ori_gp[3]);
        rep_gp[4] = word_replace_with_unk(prefix_suffix_len2_dict_wrapper, ori_gp[4]);
        rep_gp[5] = word_replace_with_unk(prefix_suffix_len3_dict_wrapper, ori_gp[5]);
        rep_gp[6] = ori_gp[6];
    }
    swap(rep_gp_seq, tmp_rep_gp_seq);
}

Index POSFeature::prefix_suffix_feature_str2feature_idx_and_adding2dict_in_training(DictWrapper &dw, const std::string &feature_str)
{
    if( feature_str == FeatureEmptyStrPlaceholder ){ return FeatureEmptyIndexPlaceholder; }
    else { return dw.Convert(feature_str); }
}
Index POSFeature::char_length_feature_str2feature_idx_and_adding2dict_in_training(const std::string &feature_idx)
{
    int char_len = std::stol(feature_idx);
    if( char_len <= 0 ) throw std::runtime_error("feature char length less equal to 0");
    return std::min(char_len, static_cast<int>(FeatureCharLengthLimit)) - 1 ; // `len -1` as index 
}

void POSFeature::feature_group2feature_index_group(const POSFeatureGroup &feature_gp,
                                                   POSFeatureIndexGroup &feature_index_gp)
{
    // for array , swap costing linear time ! so we just changing the arguments directly.
    
    // feature group :
    // -- [prefix_len1] , [prefix_len2] , [prefix_len3]
    // -- [suffix_len1] , [suffix_len2] , [suffix_len3]
    // -- [char_length_feature]

    feature_index_gp[0] = prefix_suffix_feature_str2feature_idx_and_adding2dict_in_training(prefix_suffix_len1_dict_wrapper, feature_gp[0]);
    feature_index_gp[1] = prefix_suffix_feature_str2feature_idx_and_adding2dict_in_training(prefix_suffix_len2_dict_wrapper, feature_gp[1]);
    feature_index_gp[2] = prefix_suffix_feature_str2feature_idx_and_adding2dict_in_training(prefix_suffix_len3_dict_wrapper, feature_gp[2]);
    feature_index_gp[3] = prefix_suffix_feature_str2feature_idx_and_adding2dict_in_training(prefix_suffix_len1_dict_wrapper, feature_gp[3]);
    feature_index_gp[4] = prefix_suffix_feature_str2feature_idx_and_adding2dict_in_training(prefix_suffix_len2_dict_wrapper, feature_gp[4]);
    feature_index_gp[5] = prefix_suffix_feature_str2feature_idx_and_adding2dict_in_training(prefix_suffix_len3_dict_wrapper, feature_gp[5]);
    feature_index_gp[6] = char_length_feature_str2feature_idx_and_adding2dict_in_training(feature_gp[6]);
}

void POSFeature::feature_group_seq2feature_index_group_seq(const POSFeatureGroupSeq &feature_gp_seq,
                                                           POSFeatureIndexGroupSeq &feature_index_gp_seq)
{
    using std::swap;
    size_t seq_len = feature_gp_seq.size();
    POSFeatureIndexGroupSeq tmp_feature_index_gp_seq(seq_len);
    for( size_t i = 0 ; i < seq_len ; ++i )
    {
        feature_group2feature_index_group(feature_gp_seq[i], tmp_feature_index_gp_seq[i]);
    }
    swap(tmp_feature_index_gp_seq, feature_index_gp_seq);
}


template <typename Archive>
void POSFeature::serialize(Archive &ar, const unsigned versoin) const
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
#endif