#ifndef SLNN_SEGMENTER_BASEMODEL_INPUT1_WITH_FEATURE_MODEL_0628_H_
#define SLNN_SEGMENTER_BASEMODEL_INPUT1_WITH_FEATURE_MODEL_0628_H_

#include <iostream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

#include "dynet/dynet.h"
#include "dynet/dict.h"
#include "utils/typedeclaration.h"
#include "utils/dict_wrapper.hpp"
#include "modelmodule/hyper_layers.h"
#include "segmenter/cws_module/cws_tagging_system.h"
#include "segmenter/cws_module/cws_feature.h"
namespace slnn{

template <typename RNNDerived>
class CWSInput1WithFeatureModel
{
    friend class boost::serialization::access;
public:
    static const std::string UNK_STR;
    static const unsigned SentMaxLen;

public:
    CWSInput1WithFeatureModel() ;
    virtual ~CWSInput1WithFeatureModel() ;
    CWSInput1WithFeatureModel(const CWSInput1WithFeatureModel&) = delete;
    CWSInput1WithFeatureModel& operator=(const CWSInput1WithFeatureModel&) = delete;

    void set_replace_threshold(int freq_threshold, float prob_threshold);
    bool is_dict_frozen();
    void freeze_dict();
    virtual void set_model_param_from_outer(const boost::program_options::variables_map &var_map) = 0;
    virtual void set_model_param_from_inner() = 0;

    virtual void build_model_structure() = 0 ;
    virtual void print_model_info() = 0 ;

    void word_seq2index_seq(const Seq &word_seq,
        IndexSeq &index_sent, IndexSeq &index_tag_seq, CWSFeatureDataSeq &index_feature_data_seq); // for annotated data
    void char_seq2index_seq(const Seq &char_seq,
        IndexSeq &index_sent, CWSFeatureDataSeq &index_feature_data_seq); // for input data
    void replace_word_with_unk(const IndexSeq &sent, const CWSFeatureDataSeq &origin_cws_feature_data_seq,
        IndexSeq &replaced_sent, CWSFeatureDataSeq &replaced_cws_feature_data_seq);
    void char_and_tag2word_seq(const Seq &char_seq, const IndexSeq &tag_seq, Seq &word_seq);

    virtual dynet::expr::Expression  build_loss(dynet::ComputationGraph &cg,
        const IndexSeq &input_seq,
        const CWSFeatureDataSeq &feature_data_seq,
        const IndexSeq &gold_seq) = 0 ;
    virtual void predict(dynet::ComputationGraph &cg,
        const IndexSeq &input_seq,
        const CWSFeatureDataSeq &feature_data_seq,
        IndexSeq &pred_seq) = 0 ;

    size_t get_word_dict_size(){ return word_dict.size(); }
    size_t get_tag_dict_size(){ return CWSTaggingSystem::get_tag_num(); }
    DictWrapper& get_word_dict_wrapper(){ return word_dict_wrapper ; } 
    dynet::Model *get_dynet_model(){ return m ; } ;

    // CWSFeature interface promote to this class
    void count_word_frequency(const Seq &word_seq){ cws_feature.count_word_frequency(word_seq); };
    void build_lexicon(){ cws_feature.build_lexicon(); };

    // DEBUG
    void debug_one_sent(const IndexSeq &index_char_seq, const CWSFeatureDataSeq &feature_seq)
    {
        Seq char_seq(index_char_seq.size());
        std::transform(index_char_seq.begin(), index_char_seq.end(), char_seq.begin(),
            [this](Index word_idx){ return this->word_dict.convert(word_idx); });
        cws_feature.debug_one_sent(char_seq, feature_seq);
    }

protected:
    dynet::Model *m;

    dynet::Dict word_dict;
    DictWrapper word_dict_wrapper;

    CWSFeature cws_feature;
};

template <typename RNNDerived>
const std::string CWSInput1WithFeatureModel<RNNDerived>::UNK_STR = "unk_str";

template<typename RNNDerived>
const unsigned CWSInput1WithFeatureModel<RNNDerived>::SentMaxLen = 256;

template <typename RNNDerived>
CWSInput1WithFeatureModel<RNNDerived>::CWSInput1WithFeatureModel()
    :m(nullptr),
    word_dict_wrapper(word_dict),
    cws_feature(word_dict_wrapper)
{}

template <typename RNNDerived>
CWSInput1WithFeatureModel<RNNDerived>::~CWSInput1WithFeatureModel()
{
    delete m;
}

template <typename RNNDerived>
void CWSInput1WithFeatureModel<RNNDerived>::set_replace_threshold(int freq_threshold, float prob_threshold)
{
    word_dict_wrapper.set_threshold(freq_threshold, prob_threshold);
}

template<typename RNNDerived>
bool CWSInput1WithFeatureModel<RNNDerived>::is_dict_frozen()
{
    return word_dict.is_frozen() ;
}

template <typename RNNDerived>
void CWSInput1WithFeatureModel<RNNDerived>::freeze_dict()
{
    word_dict_wrapper.freeze();
    word_dict_wrapper.set_unk(UNK_STR);
}

template <typename RNNDerived>
void CWSInput1WithFeatureModel<RNNDerived>::word_seq2index_seq(const Seq &word_seq, IndexSeq &word_index_seq, IndexSeq &tag_index_seq,
    CWSFeatureDataSeq &feature_data_seq)
{
    using std::swap;
    IndexSeq tmp_word_index_seq,
        tmp_tag_index_seq;
    Seq tmp_char_seq;
    tmp_word_index_seq.reserve(SentMaxLen);
    tmp_tag_index_seq.reserve(SentMaxLen);
    tmp_char_seq.reserve(SentMaxLen);
    for(const std::string &word : word_seq )
    {
        Seq word_char_seq ;
        IndexSeq word_tag_index_seq;
        CWSTaggingSystem::static_parse_word2chars_indextag(word, word_char_seq, word_tag_index_seq);
        for( size_t i = 0; i < word_char_seq.size(); ++i )
        {
            tmp_tag_index_seq.push_back(word_tag_index_seq[i]);
            Index word_id = word_dict_wrapper.convert(word_char_seq[i]);
            tmp_word_index_seq.push_back(word_id);
            tmp_char_seq.push_back(std::move(word_char_seq[i]));
        }
    }
    cws_feature.extract(tmp_char_seq, tmp_word_index_seq, feature_data_seq);
    swap(word_index_seq, tmp_word_index_seq);
    swap(tag_index_seq, tmp_tag_index_seq);
}


template <typename RNNDerived>
void CWSInput1WithFeatureModel<RNNDerived>::char_seq2index_seq(const Seq &char_seq, IndexSeq &word_index_seq, 
    CWSFeatureDataSeq &feature_data_seq)
{
    using std::swap;
    size_t sz = char_seq.size();
    IndexSeq tmp_word_index_seq(sz);
    for(size_t i = 0; i < sz; ++i )
    {
        tmp_word_index_seq[i] = word_dict.convert(char_seq[i]);
    }
    cws_feature.extract(char_seq, tmp_word_index_seq, feature_data_seq);
    swap(word_index_seq, tmp_word_index_seq);
}

template <typename RNNDerived>
void CWSInput1WithFeatureModel<RNNDerived>::replace_word_with_unk(const IndexSeq &ori_word_seq, 
    const CWSFeatureDataSeq &origin_feature_data_seq, 
    IndexSeq &rep_word_seq,
    CWSFeatureDataSeq &rep_feature_data_seq)
{
    using std::swap;
    size_t sz = ori_word_seq.size();
    IndexSeq tmp_rep_word_seq(sz);
    for( size_t i = 0; i < sz; ++i )
    {
        tmp_rep_word_seq[i] = word_dict_wrapper.unk_replace_probability(ori_word_seq[i]);
    }
    swap(rep_word_seq, tmp_rep_word_seq);
    cws_feature.random_replace_with_unk(origin_feature_data_seq, rep_feature_data_seq);
}

template <typename RNNDerived>
void CWSInput1WithFeatureModel<RNNDerived>::char_and_tag2word_seq(const Seq &char_seq, const IndexSeq &tag_seq,
    Seq &word_seq)
{
    CWSTaggingSystem::parse_word_tag2words(char_seq, tag_seq, word_seq);
}

} // end of namespcace slnn 


#endif
