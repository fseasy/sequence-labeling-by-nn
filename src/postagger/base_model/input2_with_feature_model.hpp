#ifndef POS_BASE_MODEL_INPUT2_WITH_FEATURE_MODEL_HPP_
#define POS_BASE_MODEL_INPUT2_WITH_FEATURE_MODEL_HPP_
#include <fstream>
#include <boost/program_options.hpp>

#include "dynet/dynet.h"
#include "dynet/dict.h"

#include "postagger/postagger_module/pos_feature.h"
#include "postagger/postagger_module/pos_feature_extractor.h"
#include "postagger/postagger_module/pos_feature_layer.h"
#include "utils/dict_wrapper.hpp"
#include "utils/utf8processing.hpp"
#include "utils/word2vec_embedding_helper.h"
#include "modelmodule/hyper_layers.h"
namespace slnn{

template<typename RNNDerived>
class Input2WithFeatureModel
{
    friend class boost::serialization::access;
public :
    static const std::string UNK_STR;
    static const std::string StrOfReplaceNumber ;
    static const size_t LenStrOfRepalceNumber ;

public:
    Input2WithFeatureModel();
    virtual ~Input2WithFeatureModel();
    Input2WithFeatureModel(const Input2WithFeatureModel &) = delete;
    Input2WithFeatureModel& operator()(const Input2WithFeatureModel&) = delete;

    void set_replace_threshold(int freq_threshold, float prob_threshold);
    bool is_fixed_dict_frozen(){ return fixed_word_dict.is_frozen(); }
    bool is_dict_frozen();
    void freeze_dict();
    virtual void build_fixed_dict(std::ifstream &is) = 0; // bacause paremeter about size is in derived class
    void print_dynamic_word_hit_info();
    virtual void set_model_param(const boost::program_options::variables_map &var_map) = 0;
    
    virtual void build_model_structure() = 0 ;
    virtual void load_fixed_embedding(std::ifstream &is) = 0;
    virtual void print_model_info() = 0 ;

    void input_seq2index_seq(const Seq &sent, const Seq &postag_seq, 
                             IndexSeq &dynamic_index_sent, IndexSeq &fixed_index_seq, IndexSeq &index_postag_seq, 
                             POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq); // for annotated data
    void input_seq2index_seq(const Seq &sent, 
        IndexSeq &dynamic_index_sent, IndexSeq &fixed_index_sent,
        POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq); // for input data
    void replace_word_with_unk(const IndexSeq &dynamic_sent, const POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq,
                               IndexSeq &replaced_dynamic_sent, POSFeature::POSFeatureIndexGroupSeq &replaced_feature_gp_seq);
    void postag_index_seq2postag_str_seq(const IndexSeq &postag_index_seq, Seq &postag_str_seq);

    virtual dynet::expr::Expression  build_loss(dynet::ComputationGraph &cg,
        const IndexSeq &dynamic_sent,
        const IndexSeq &fixed_sent,
        const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
        const IndexSeq &gold_seq) = 0 ;
    virtual void predict(dynet::ComputationGraph &cg,
        const IndexSeq &dynamic_sent,
        const IndexSeq &fixed_sent,
        const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
        IndexSeq &pred_seq) = 0 ;


    dynet::Dict& get_dynamic_word_dict(){ return dynamic_word_dict ;  } 
    dynet::Dict& get_fixed_word_dict(){ return fixed_word_dict; }
    dynet::Dict& get_postag_dict(){ return postag_dict ; } 
    DictWrapper& get_word_dict_wrapper(){ return dynamic_word_dict_wrapper ; } 
    dynet::Model *get_dynet_model(){ return m ; } 


protected:
    dynet::Model *m;

    dynet::Dict dynamic_word_dict;
    dynet::Dict fixed_word_dict;
    dynet::Dict postag_dict;
    DictWrapper dynamic_word_dict_wrapper;

public:
    POSFeature pos_feature; // also as parameters
};

//template <typename RNNDerived>
//BOOST_SERIALIZATION_ASSUME_ABSTRACT(Input2WithFeatureModel)

template<typename RNNDerived>
const std::string Input2WithFeatureModel<RNNDerived>::UNK_STR = "unk_str";

template<typename RNNDerived>
const std::string Input2WithFeatureModel<RNNDerived>::StrOfReplaceNumber = "##";

template<typename RNNDerived>
const size_t Input2WithFeatureModel<RNNDerived>::LenStrOfRepalceNumber = StrOfReplaceNumber.length();

template<typename RNNDerived>
Input2WithFeatureModel<RNNDerived>::Input2WithFeatureModel() 
    :m(nullptr),
    dynamic_word_dict_wrapper(dynamic_word_dict)
{}

template <typename RNNDerived>
Input2WithFeatureModel<RNNDerived>::~Input2WithFeatureModel()
{
    delete m;
}
template <typename RNNDerived>
void Input2WithFeatureModel<RNNDerived>::set_replace_threshold(int freq_threshold, float prob_threshold)
{
    dynamic_word_dict_wrapper.set_threshold(freq_threshold, prob_threshold);
    pos_feature.set_replace_feature_with_unk_threshold(freq_threshold, prob_threshold);
}

template <typename RNNDerived>
bool Input2WithFeatureModel<RNNDerived>::is_dict_frozen()
{
    return (dynamic_word_dict.is_frozen() && fixed_word_dict.is_frozen() &&
        postag_dict.is_frozen() && pos_feature.is_dict_frozen());
}

template <typename RNNDerived>
void Input2WithFeatureModel<RNNDerived>::freeze_dict()
{
    dynamic_word_dict_wrapper.freeze();
    postag_dict.freeze();
    dynamic_word_dict_wrapper.set_unk(UNK_STR);
    pos_feature.freeze_dict();
}

template <typename RNNDerived>
void Input2WithFeatureModel<RNNDerived>::print_dynamic_word_hit_info()
{
    Word2vecEmbeddingHelper::calc_hit_rate(fixed_word_dict, dynamic_word_dict, UNK_STR);
}


template <typename RNNDerived>
void Input2WithFeatureModel<RNNDerived>::input_seq2index_seq(const Seq &sent,
    const Seq &postag_seq,
    IndexSeq &dynamic_index_sent,
    IndexSeq &fixed_index_sent,
    IndexSeq &index_postag_seq,
    POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq)
{
    using std::swap;
    assert(sent.size() == postag_seq.size());
    size_t seq_len = sent.size();
    IndexSeq tmp_dynamic_sent_index_seq(seq_len),
        tmp_fixed_sent_index_seq(seq_len),
        tmp_postag_index_seq(seq_len);
    for( size_t i = 0 ; i < seq_len; ++i )
    {
        std::string replaced_word = UTF8Processing::replace_number(sent[i], StrOfReplaceNumber, LenStrOfRepalceNumber);
        tmp_dynamic_sent_index_seq[i] = dynamic_word_dict_wrapper.convert(replaced_word);
        tmp_fixed_sent_index_seq.at(i) = fixed_word_dict.convert(replaced_word);
        tmp_postag_index_seq[i] = postag_dict.convert(postag_seq[i]);
    }
    POSFeature::POSFeatureGroupSeq feature_gp_str_seq;
    POSFeatureExtractor::extract(sent, feature_gp_str_seq);
    pos_feature.feature_group_seq2feature_index_group_seq(feature_gp_str_seq, feature_gp_seq);

    swap(dynamic_index_sent, tmp_dynamic_sent_index_seq);
    swap(fixed_index_sent, tmp_fixed_sent_index_seq);
    swap(index_postag_seq, tmp_postag_index_seq);
}

template <typename RNNDerived>
void Input2WithFeatureModel<RNNDerived>::input_seq2index_seq(const Seq &sent, 
                                                                  IndexSeq &dynamic_index_sent, 
    IndexSeq &fixed_index_sent,
                                                                  POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq)
{
    using std::swap;
    size_t seq_len = sent.size();
    IndexSeq tmp_dynamic_index_sent(seq_len),
        tmp_fixed_index_sent(seq_len);
    for( size_t i = 0 ; i < seq_len; ++i )
    {
        std::string replaced_word = UTF8Processing::replace_number(sent[i], StrOfReplaceNumber, LenStrOfRepalceNumber);
        tmp_dynamic_index_sent[i] = dynamic_word_dict_wrapper.convert(replaced_word);
        tmp_fixed_index_sent.at(i) = fixed_word_dict.convert(replaced_word);
    }
    POSFeature::POSFeatureGroupSeq feature_gp_str_seq;
    POSFeatureExtractor::extract(sent, feature_gp_str_seq);
    pos_feature.feature_group_seq2feature_index_group_seq(feature_gp_str_seq, feature_gp_seq);

    swap(dynamic_index_sent, tmp_dynamic_index_sent);
    swap(fixed_index_sent, tmp_fixed_index_sent);
}


template <typename RNNDerived>
void Input2WithFeatureModel<RNNDerived>::replace_word_with_unk(const IndexSeq &dynamic_sent,
                                                                    const POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq,
                                                                    IndexSeq &replaced_dynamic_sent, 
                                                                    POSFeature::POSFeatureIndexGroupSeq &replaced_feature_gp_seq)
{
    using std::swap;
    size_t seq_len = dynamic_sent.size();
    IndexSeq tmp_rep_sent(seq_len);
    for( size_t i = 0; i < seq_len; ++i )
    {
        tmp_rep_sent[i] = dynamic_word_dict_wrapper.unk_replace_probability(dynamic_sent[i]);
    }
    swap(replaced_dynamic_sent, tmp_rep_sent);
    pos_feature.do_repalce_feature_with_unk_in_copy(feature_gp_seq, replaced_feature_gp_seq);
}

template <typename RNNDerived>
void Input2WithFeatureModel<RNNDerived>::postag_index_seq2postag_str_seq(const IndexSeq &postag_index_seq, Seq &postag_str_seq)
{
    size_t seq_len = postag_index_seq.size();
    Seq tmp_str_seq(seq_len);
    for( size_t i = 0; i < seq_len; ++i )
    {
        tmp_str_seq[i] = postag_dict.convert(postag_index_seq[i]);
    }
    swap(postag_str_seq, tmp_str_seq);
}

} // end of namespace slnn
#endif
