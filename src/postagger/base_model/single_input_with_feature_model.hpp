#ifndef POS_BASE_MODEL_SINGLE_INPUT_WITH_FEATURE_MODEL_HPP_
#define POS_BASE_MODEL_SINGLE_INPUT_WITH_FEATURE_MODEL_HPP_

#include <boost/program_options.hpp>

#include "dynet/dynet.h"
#include "dynet/dict.h"

#include "postagger/postagger_module/pos_feature.h"
#include "postagger/postagger_module/pos_feature_extractor.h"
#include "postagger/postagger_module/pos_feature_layer.h"
#include "utils/dict_wrapper.hpp"
#include "utils/utf8processing.hpp"
#include "modelmodule/hyper_layers.h"
namespace slnn{

template<typename RNNDerived>
class SingleInputWithFeatureModel
{
    friend class boost::serialization::access;
public :
    static const std::string UNK_STR;
    static const std::string StrOfReplaceNumber ;
    static const size_t LenStrOfRepalceNumber ;

public:
    SingleInputWithFeatureModel();
    virtual ~SingleInputWithFeatureModel();
    SingleInputWithFeatureModel(const SingleInputWithFeatureModel &) = delete;
    SingleInputWithFeatureModel& operator()(const SingleInputWithFeatureModel&) = delete;

    void set_replace_threshold(int freq_threshold, float prob_threshold);
    bool is_dict_frozen();
    void freeze_dict();
    virtual void set_model_param(const boost::program_options::variables_map &var_map) = 0;
    
    virtual void build_model_structure() = 0 ;
    virtual void print_model_info() = 0 ;

    void input_seq2index_seq(const Seq &sent, const Seq &postag_seq, 
                             IndexSeq &index_sent, IndexSeq &index_postag_seq, 
                             POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq); // for annotated data
    void input_seq2index_seq(const Seq &sent, 
                             IndexSeq &index_sent, POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq); // for input data
    void replace_word_with_unk(const IndexSeq &sent, const POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq,
                               IndexSeq &replaced_sent, POSFeature::POSFeatureIndexGroupSeq &replaced_feature_gp_seq);
    void postag_index_seq2postag_str_seq(const IndexSeq &postag_index_seq, Seq &postag_str_seq);

    virtual dynet::expr::Expression  build_loss(dynet::ComputationGraph &cg,
                                              const IndexSeq &input_seq, 
                                              const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
                                              const IndexSeq &gold_seq)  = 0 ;
    virtual void predict(dynet::ComputationGraph &cg ,
                         const IndexSeq &input_seq, 
                         const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
                         IndexSeq &pred_seq) = 0 ;

    dynet::Dict& get_word_dict(){ return word_dict ;  } 
    dynet::Dict& get_postag_dict(){ return postag_dict ; } 
    DictWrapper& get_word_dict_wrapper(){ return word_dict_wrapper ; } 
    dynet::Model *get_dynet_model(){ return m ; } 


protected:
    dynet::Model *m;

    dynet::Dict word_dict;
    dynet::Dict postag_dict;
    DictWrapper word_dict_wrapper;

public:
    POSFeature pos_feature; // also as parameters
};

//template <typename RNNDerived>
//BOOST_SERIALIZATION_ASSUME_ABSTRACT(SingleInputWithFeatureModel)

template<typename RNNDerived>
const std::string SingleInputWithFeatureModel<RNNDerived>::UNK_STR = "unk_str";

template<typename RNNDerived>
const std::string SingleInputWithFeatureModel<RNNDerived>::StrOfReplaceNumber = "##";

template<typename RNNDerived>
const size_t SingleInputWithFeatureModel<RNNDerived>::LenStrOfRepalceNumber = StrOfReplaceNumber.length();

template<typename RNNDerived>
SingleInputWithFeatureModel<RNNDerived>::SingleInputWithFeatureModel() 
    :m(nullptr),
    word_dict_wrapper(word_dict)
{}

template <typename RNNDerived>
SingleInputWithFeatureModel<RNNDerived>::~SingleInputWithFeatureModel()
{
    delete m;
}
template <typename RNNDerived>
void SingleInputWithFeatureModel<RNNDerived>::set_replace_threshold(int freq_threshold, float prob_threshold)
{
    word_dict_wrapper.set_threshold(freq_threshold, prob_threshold);
    pos_feature.set_replace_feature_with_unk_threshold(freq_threshold, prob_threshold);
}

template <typename RNNDerived>
bool SingleInputWithFeatureModel<RNNDerived>::is_dict_frozen()
{
    return (word_dict.is_frozen() && postag_dict.is_frozen() && pos_feature.is_dict_frozen());
}

template <typename RNNDerived>
void SingleInputWithFeatureModel<RNNDerived>::freeze_dict()
{
    word_dict_wrapper.Freeze();
    postag_dict.Freeze();
    word_dict_wrapper.SetUnk(UNK_STR);
    pos_feature.freeze_dict();
}

template <typename RNNDerived>
void SingleInputWithFeatureModel<RNNDerived>::input_seq2index_seq(const Seq &sent, 
                                                                  const Seq &postag_seq,
                                                                  IndexSeq &index_sent, 
                                                                  IndexSeq &index_postag_seq,
                                                                  POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq)
{
    using std::swap;
    assert(sent.size() == postag_seq.size());
    size_t seq_len = sent.size();
    IndexSeq tmp_sent_index_seq(seq_len),
        tmp_postag_index_seq(seq_len);
    for( size_t i = 0 ; i < seq_len; ++i )
    {
        tmp_sent_index_seq[i] = word_dict_wrapper.Convert(
            UTF8Processing::replace_number(sent[i], StrOfReplaceNumber, LenStrOfRepalceNumber)
        );
        tmp_postag_index_seq[i] = postag_dict.Convert(postag_seq[i]);
    }
    POSFeature::POSFeatureGroupSeq feature_gp_str_seq;
    POSFeatureExtractor::extract(sent, feature_gp_str_seq);
    pos_feature.feature_group_seq2feature_index_group_seq(feature_gp_str_seq, feature_gp_seq);

    swap(index_sent, tmp_sent_index_seq);
    swap(index_postag_seq, tmp_postag_index_seq);
}

template <typename RNNDerived>
void SingleInputWithFeatureModel<RNNDerived>::input_seq2index_seq(const Seq &sent, 
                                                                  IndexSeq &index_sent, 
                                                                  POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq)
{
    using std::swap;
    size_t seq_len = sent.size();
    IndexSeq tmp_sent_index_seq(seq_len);
    for( size_t i = 0 ; i < seq_len; ++i )
    {
        tmp_sent_index_seq[i] = word_dict_wrapper.Convert(
            UTF8Processing::replace_number(sent[i], StrOfReplaceNumber, LenStrOfRepalceNumber)
        );
    }
    POSFeature::POSFeatureGroupSeq feature_gp_str_seq;
    POSFeatureExtractor::extract(sent, feature_gp_str_seq);
    pos_feature.feature_group_seq2feature_index_group_seq(feature_gp_str_seq, feature_gp_seq);

    swap(index_sent, tmp_sent_index_seq);
}


template <typename RNNDerived>
void SingleInputWithFeatureModel<RNNDerived>::replace_word_with_unk(const IndexSeq &sent,
                                                                    const POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq,
                                                                    IndexSeq &replaced_sent, 
                                                                    POSFeature::POSFeatureIndexGroupSeq &replaced_feature_gp_seq)
{
    using std::swap;
    size_t seq_len = sent.size();
    IndexSeq tmp_rep_sent(seq_len);
    for( size_t i = 0; i < seq_len; ++i )
    {
        tmp_rep_sent[i] = word_dict_wrapper.ConvertProbability(sent[i]);
    }
    swap(replaced_sent, tmp_rep_sent);
    pos_feature.do_repalce_feature_with_unk_in_copy(feature_gp_seq, replaced_feature_gp_seq);
}

template <typename RNNDerived>
void SingleInputWithFeatureModel<RNNDerived>::postag_index_seq2postag_str_seq(const IndexSeq &postag_index_seq, Seq &postag_str_seq)
{
    size_t seq_len = postag_index_seq.size();
    Seq tmp_str_seq(seq_len);
    for( size_t i = 0; i < seq_len; ++i )
    {
        tmp_str_seq[i] = postag_dict.Convert(postag_index_seq[i]);
    }
    swap(postag_str_seq, tmp_str_seq);
}

} // end of namespace slnn
#endif
