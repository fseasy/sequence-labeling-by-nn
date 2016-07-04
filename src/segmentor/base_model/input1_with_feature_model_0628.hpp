#ifndef SLNN_SEGMENTOR_BASEMODEL_INPUT1_WITH_FEATURE_MODEL_0628_H_
#define SLNN_SEGMENTOR_BASEMODEL_INPUT1_WITH_FEATURE_MODEL_0628_H_

#include <iostream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

#include "cnn/cnn.h"
#include "cnn/dict.h"
#include "utils/typedeclaration.h"
#include "utils/dict_wrapper.hpp"
#include "modelmodule/hyper_layers.h"
#include "segmentor/cws_module/cws_tagging_system.h"
#include "segmentor/cws_module/cws_feature.h"
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
    virtual void set_model_param(const boost::program_options::variables_map &var_map) = 0;

    virtual void build_model_structure() = 0 ;
    virtual void print_model_info() = 0 ;

    void word_seq2index_seq(const Seq &word_seq,
        IndexSeq &index_sent, IndexSeq &index_postag_seq, CWSFeatureDataSeq &index_feature_data_seq); // for annotated data
    void char_seq2index_seq(const Seq &char_seq,
        IndexSeq &index_sent, CWSFeatureDataSeq &index_feature_data_seq); // for input data
    void replace_word_with_unk(const IndexSeq &sent,
        IndexSeq &replaced_sent);
    void char_and_tag2word_seq(const Seq &char_seq, const IndexSeq &tag_seq, Seq &word_seq);

    virtual cnn::expr::Expression  build_loss(cnn::ComputationGraph &cg,
        const IndexSeq &input_seq,
        const CWSFeatureDataSeq &feature_data_seq,
        const IndexSeq &gold_seq) = 0 ;
    virtual void predict(cnn::ComputationGraph &cg,
        const IndexSeq &input_seq,
        const CWSFeatureDataSeq &feature_data_seq,
        IndexSeq &pred_seq) = 0 ;

    size_t get_word_dict_size(){ return word_dict.size(); }
    size_t get_tag_dict_size(){ return CWSTaggingSystem::get_tag_num(); }
    DictWrapper& get_word_dict_wrapper(){ return word_dict_wrapper ; } 
    cnn::Model *get_cnn_model(){ return m ; } ;

    // CWSFeature interface promote to this class
    void count_word_freqency(const Seq &word_seq){ cws_feature.count_word_freqency(word_seq); };
    void build_lexicon(){ cws_feature.build_lexicon(); };

protected:
    cnn::Model *m;

    cnn::Dict word_dict;
    DictWrapper word_dict_wrapper;

    CWSFeature cws_feature;
};

template <typename RNNDerived>
std::string CWSInput1WithFeatureModel<RNNDerived>::UNK_STR = "unk_str";

template<typename RNNDerived>
unsigned CWSInput1WithFeatureModel<RNNDerived>::SentMaxLen = 256;

template <typename RNNDerived>
CWSInput1WithFeatureModel<RNNDerived>::CWSInput1WithFeatureModel()
    :m(nullptr),
    word_dict_wrapper(word_dict)
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
    return word_dict.is_fronzen() ;
}

template <typename RNNDerived>
void CWSInput1WithFeatureModel<RNNDerived>::freeze_dict()
{
    word_dict_wrapper.Freeze();
    word_dict_wrapper.SetUnk(UNK_STR);
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
        std::string word_char_seq ;
        IndexSeq word_tag_index_seq;
        CWSTaggingSystem::static_parse_word2chars_indextag(word, word_char_seq, word_tag_index_seq);
        CWSTaggingSystem::parse_words2word_tag(word, word_char_seq, word_tag_seq);
        for( size_t i = 0; i < word_char_seq.size(); ++i )
        {
            tmp_tag_index_seq.push_back(word_tag_index_seq[i]);
            Index word_id = word_dict_wrapper.Convert(word_char_seq[i]);
            tmp_word_index_seq.push_back(word_id);
            tmp_char_seq.push_back(word_char_seq[i]);
        }
    }
    swap(word_index_seq, tmp_word_index_seq);
    swap(tag_index_seq, tmp_tag_index_seq);
    cws_feature.extract(tmp_char_seq, feature_data_seq);
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
        tmp_word_index_seq[i] = word_dict.Convert(char_seq[i]);
    }
    swap(word_index_seq, tmp_word_index_seq);
    cws_feature.extract(char_seq, feature_data_seq);
}

template <typename RNNDerived>
void CWSInput1WithFeatureModel<RNNDerived>::replace_word_with_unk(const IndexSeq &ori_word_seq, IndexSeq &rep_word_seq)
{
    using std::swap;
    size_t sz = rep_word_seq.size();
    IndexSeq tmp_rep_word_seq(sz);
    for( size_t i = 0; i < sz; ++i )
    {
        tmp_rep_word_seq[i] = word_dict_wrapper.ConvertProbability(ori_word_seq);
    }
    swap(rep_word_seq, tmp_rep_word_seq);
}

template <typename RNNDerived>
void CWSInput1WithFeatureModel<RNNDerived>::char_and_tag2word_seq(const Seq &char_seq, const IndexSeq &tag_seq,
    Seq &word_seq)
{
    CWSTaggingSystem::parse_word_tag2words(char_seq, tag_seq, word_seq);
}

} // end of namespcace slnn 


#endif
