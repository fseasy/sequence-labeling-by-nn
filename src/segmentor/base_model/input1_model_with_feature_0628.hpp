#ifndef SLNN_SEGMENTOR_BASEMODEL_INPUT1_WITH_FEATURE_MODEL_H_
#define SLNN_SEGMENTOR_BASEMODEL_INPUT1_WITH_FEATURE_MODEL_H_

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
namespace slnn{

template <typename RNNDerived>
class Input1WithFeatureModel
{
    friend class boost::serialization::access;
public :
    static const std::string UNK_STR;

public:
    Input1WithFeatureModel() ;
    virtual ~Input1WithFeatureModel() ;
    Input1WithFeatureModel(const Input1WithFeatureModel&) = delete;
    Input1WithFeatureModel& operator=(const Input1WithFeatureModel&) = delete;

    void set_replace_threshold(int freq_threshold, float prob_threshold);
    bool is_dict_frozen();
    void freeze_dict();
    virtual void set_model_param(const boost::program_options::variables_map &var_map) = 0;

    virtual void build_model_structure() = 0 ;
    virtual void print_model_info() = 0 ;

    void word_seq2index_seq(const Seq &word_seq, 
        IndexSeq &index_sent, IndexSeq &index_postag_seq); // for annotated data
    void char_seq2index_seq(const Seq &char_seq, 
        IndexSeq &index_sent); // for input data
    void replace_word_with_unk(const IndexSeq &sent, 
        IndexSeq &replaced_sent);
    void char_and_tag2word_seq(const Seq &char_seq, const IndexSeq &tag_seq, Seq &word_seq);

    virtual cnn::expr::Expression  build_loss(cnn::ComputationGraph &cg,
        const IndexSeq &input_seq, 
        const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
        const IndexSeq &gold_seq)  = 0 ;
    virtual void predict(cnn::ComputationGraph &cg ,
        const IndexSeq &input_seq, 
        const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
        IndexSeq &pred_seq) = 0 ;

public :
    
    cnn::Dict& get_word_dict(){ return word_dict ;  } 
    cnn::Dict& get_tag_dict(){ return tag_dict ; } 
    DictWrapper& get_word_dict_wrapper(){ return word_dict_wrapper ; } 
    cnn::Model *get_cnn_model(){ return m ; } ;
    CWSTaggingSystem& get_tag_sys(){ return tag_sys ; }

protected:
    cnn::Model *m;

    cnn::Dict word_dict;
    cnn::Dict tag_dict;
    DictWrapper word_dict_wrapper;


    CWSTaggingSystem tag_sys ;
};

template <typename RNNDerived>
std::string Input1WithFeatureModel<RNNDerived>::UNK_STR = "unk_str";

template <typename RNNDerived>
Input1WithFeatureModel<RNNDerived>::Input1WithFeatureModel()
    :m(nullptr),
    word_dict_wrapper(word_dict)
{}

template <typename RNNDerived>
Input1WithFeatureModel<RNNDerived>::~Input1WithFeatureModel()
{
    delete m;
}

template <typename RNNDerived>
void Input1WithFeatureModel<RNNDerived>::set_replace_threshold(int freq_threshold, float prob_threshold)
{
    word_dict_wrapper.set_threshold(freq_threshold, prob_threshold);
}

template<typename RNNDerived>
bool Input1WithFeatureModel<RNNDerived>::is_dict_frozen()
{
    return word_dict.is_fronzen() && tag_dict.is_fronzen();
}

template <typename RNNDerived>
void Input1WithFeatureModel<RNNDerived>::freeze_dict()
{
    word_dict_wrapper.Freeze();
    tag_dict.Freeze();
    word_dict_wrapper.SetUnk(UNK_STR);
}

template <typename RNNDerived>
void Input1WithFeatureModel<RNNDerived>::word_seq2index_seq(const Seq &word_seq, IndexSeq &word_index_seq, IndexSeq &tag_index_seq)
{

}


template <typename RNNDerived>
void Input1WithFeatureModel<RNNDerived>::char_seq2index_seq(const Seq &char_seq, IndexSeq &word_index_seq)
{

}

template <typename RNNDerived>
void Input1WithFeatureModel<RNNDerived>::replace_word_with_unk(const IndexSeq &ori_word_seq, IndexSeq &rep_word_seq)
{

}

template <typename RNNDerived>
void Input1WithFeatureModel<RNNDerived>::char_and_tag2word_seq(const Seq &char_seq, const IndexSeq &tag_seq,
    Seq &word_seq)
{

}

} // end of namespcace slnn 


#endif
