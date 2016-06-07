#ifndef POS_BASE_MODEL_SINGLE_INPUT_WITH_FEATURE_MODEL_HPP_
#define POS_BASE_MODEL_SINGLE_INPUT_WITH_FEATURE_MODEL_HPP_

#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

#include "cnn/cnn.h"
#include "cnn/dict.h"

#include "postagger/postagger_module/pos_feature.hpp"
#include "postagger/postagger_module/pos_feature_extractor.hpp"
#include "utils/dict_wrapper.hpp"
#include "utils/utf8processing.hpp"
#include "modelmodule/hyper_layers.h"
namespace slnn{

template<typename RNNDerived>
class SingleInputModelWithFeature
{
public :
    static const std::string UNK_STR;

public:
    SingleInputModelWithFeature();
    ~SingleInputModelWithFeature();
    SingleInputModelWithFeature(const SingleInputModelWithFeature &) = delete;
    SingleInputModelWithFeature& operator()(const SingleInputModelWithFeature&) = delete;

    void set_replace_threshold(int freq_threshold, float prob_threshold);
    virtual void set_model_param(const boost::program_options::variables_map &var_map);
    
    virtual void build_model_structure() = 0 ;
    virtual void print_model_info() = 0 ;

    void input_seq2index_seq(const Seq &sent, const Seq &postag_seq, 
                             IndexSeq &index_sent, IndexSeq &index_seq, POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq);
    void repalce_word_with_unk(const IndexSeq &sent, const POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq,
                               IndexSeq &replaced_sent, POSFeature::POSFeatureIndexGroupSeq &replaced_feature_gp_seq);

    virtual cnn::expr::Expression  build_loss(cnn::ComputationGraph &cg,
                                              const IndexSeq &input_seq, 
                                              const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
                                              const IndexSeq &gold_seq) ;
    virtual void predict(cnn::ComputationGraph &cg ,
                         const IndexSeq &input_seq, 
                         const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
                         IndexSeq &pred_seq) ;

    cnn::Dict& get_word_dict(){ return word_dict ;  } 
    cnn::Dict& get_postag_dict(){ return postag_dict ; } 
    DictWrapper& get_word_dict_wrapper(){ return word_dict_wrapper ; } 
    cnn::Model *get_cnn_model(){ return m ; } 
    void set_cnn_model(std::istream &mis){ boost::archive::text_iarchive ti(mis) ; ti >> *m ; } 

    template <typename Archive>
    void save(Archive &ar, const unsigned versoin) const; 
    template <typename Archive>
    void load(Archive &ar, const unsigned version);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

protected:
    cnn::Model *m;
    
    POSFeatureLayer * pos_feature_layer;
    Input1WithFeature *input_layer;
    BIRNNLayer<RNNDerived> *birnn_layer;
    OutputBase *output_layer;

    cnn::Dict word_dict;
    cnn::Dict postag_dict;
    DictWrapper word_dict_wrapper;

public:
    POSFeature pos_feature; // also as parameters
    unsigned word_embedding_dim,
        word_dict_size
        nr_rnn_stacked_layer,
        rnn_x_dim,
        rnn_h_dim,
        hidden_dim,
        output_dim ;

    cnn::real dropout_rate ; 

};

template<typename RNNDerived>
BOOST_SERIALIZATION_ASSUME_ABSTRACT(SingleInputModelWithFeature<RNNDerived>)

template<typename RNNDerived>
const std::string SingleInputModelWithFeature<RNNDerived>::UNK_STR = "unk_str";

template<typename RNNDerived>
SingleInputModelWithFeature<RNNDerived>::SingleInputModelWithFeature() 
    :m(nullptr),
    pos_feature_layer(nullptr),
    input_layer(nullptr),
    birnn_layer(nullptr),
    output_layer(nullptr),
    word_dict_wrapper(word_dict)
{}

template <typename RNNDerived>
SingleInputModelWithFeature<RNNDerived>::~SingleInputModelWithFeature()
{
    delete m;
    delete pos_feature_layer;
    delete input_layer;
    delete birnn_layer;
    delete output_layer;
}
template <typename RNNDerived>
void SingleInputModelWithFeature<RNNDerived>::set_replace_threshold(int freq_threshold, float prob_threshold)
{
    word_dict_wrapper.set_threshold(freq_threshold, prob_threshold);
    pos_feature.set_replace_feature_with_unk_threshold(freq_threshold, prob_threshold);
}

template <typename RNNDerived>
void SingleInputModelWithFeature<RNNDerived>::set_model_param(const boost::program_options::variables_map &var_map)
{
    assert(word_dict.is_frozen() && postag_dict.is_frozen()  && pos_feature->is_dict_frozen()) ;

    word_embedding_dim = var_map["word_embedding_dim"].as<unsigned>() ;
    nr_rnn_stacked_layer = var_map["nr_rnn_stacked_layer"].as<unsigned>() ;
    rnn_x_dim = var_map["rnn_x_dim"].as<unsigned>();
    rnn_h_dim = var_map["rnn_h_dim"].as<unsigned>() ;
    hidden_dim = var_map["tag_layer_hidden_dim"].as<unsigned>() ;

    dropout_rate = var_map["dropout_rate"].as<cnn::real>() ;

    unsigned prefix_suffix_len1_embedding_dim = var_map["prefix_suffix_len1_embedding_dim"].as<unsigned>();
    unsigned prefix_suffix_len2_embedding_dim = var_map["prefix_suffix_len2_embedding_dim"].as<unsigned>();
    unsigned prefix_suffix_len3_embedding_dim = var_map["prefix_suffix_len3_embedding_dim"].as<unsigned>();
    unsigned char_lenth_embedding_dim = var_map["char_length_embedding_dim"].as<unsigned>();
    pos_feature.init_embedding_dim(prefix_suffix_len1_embedding_dim, prefix_suffix_len2_embedding_dim,
                                   prefix_suffix_len3_embedding_dim, char_length_embedding_dim);
    
    word_dict_size = word_dict.size() ;
    output_dim = postag_dict.size() ;
}

template <typename RNNDerived>
void SingleInputModelWithFeature<RNNDerived>::input_seq2index_seq(const Seq &sent, 
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
        tmp_sent_index_seq[i] = word_dict_wrapper.Convert(UTF8Processing::replace_number(sent[i]));
        tmp_postag_index_seq[i] = postag_dict.Convert(postag_seq[i]);
    }
    POSFeature::POSFeatureGroupSeq feature_gp_str_seq;
    POSFeatureExtractor::extract(sent, feature_gp_str_seq);
    pos_feature.feature_group_seq2feature_index_group_seq(feature_gp_str_seq, feature_gp_seq);

    swap(index_sent, tmp_sent_index_seq);
    swap(index_postag_seq, tmp_postag_index_seq);
}

template <typename RNNDerived>
void SingleInputModelWithFeature<RNNDerived>::repalce_word_with_unk(const IndexSeq &sent,
                                                                    const POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq,
                                                                    IndexSeq &replaced_sent, 
                                                                    POSFeature::POSFeatureIndexGroupSeq &replaced_feature_gp_seq)
{
    using std::swap;
    size_t seq_len = sent.size();
    IndexSeq &tmp_rep_sent(seq_len);
    for( size_t i = 0; i < seq_len; ++i )
    {
        tmp_rep_sent[i] = word_dict_wrapper.ConvertProbability(sent[i]);
    }
    swap(replaced_sent, tmp_rep_sent);
    pos_feature.do_repalce_feature_with_unk_in_copy(feature_gp_seq, replaced_feature_gp_seq);
}

template<typename RNNDerived>
cnn::expr::Expression SingleInputModelWithFeature<RNNDerived>::build_loss(cnn::ComputationGraph &cg,
                                                                           const IndexSeq &input_seq, 
                                                                           const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
                                                                           const IndexSeq &gold_seq)
{
    pos_feature_layer->new_graph(cg);
    input_layer->new_graph(cg) ;
    birnn_layer->new_graph(cg) ;
    output_layer->new_graph(cg) ;

    birnn_layer->set_dropout() ;
    birnn_layer->start_new_sequence() ;

    std::vector<cnn::expr::Expression> features_exprs;
    pos_feature_layer->build_feature_exprs(features_gp_seq, features_exprs);

    std::vector<cnn::expr::Expression> inputs_exprs ;
    input_layer->build_inputs(input_seq, features_exprs, inputs_exprs) ;

    std::vector<cnn::expr::Expression> l2r_exprs,
        r2l_exprs ;
    birnn_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;
    return output_layer->build_output_loss(l2r_exprs, r2l_exprs, gold_seq) ;
}

template<typename RNNDerived>
void SingleInputModelWithFeature<RNNDerived>::predict(cnn::ComputationGraph &cg,
                                                      const IndexSeq &input_seq,
                                                      const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
                                                      IndexSeq &pred_seq)
{
    pos_feature_layer->new_graph(cg);
    input_layer->new_graph(cg) ;
    birnn_layer->new_graph(cg) ;
    output_layer->new_graph(cg) ;

    birnn_layer->disable_dropout() ;
    birnn_layer->start_new_sequence();

    std::vector<cnn::expr::Expression> feature_exprs;
    pos_feature_layer->build_feature_exprs(features_gp_seq, features_exprs);

    std::vector<cnn::expr::Expression> inputs_exprs ;
    input_layer->build_inputs(input_seq, features_exprs, inputs_exprs) ;
    std::vector<cnn::expr::Expression> l2r_exprs,
        r2l_exprs ;
    birnn_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;
    output_layer->build_output(l2r_exprs, r2l_exprs , pred_seq) ;
}

template <typename RNNDerived, typename Archive>
void SingleInputModelWithFeature<RNNDerived>::save(Archive &ar, const unsigned version) const
{
    ar & word_dict_size & word_embedding_dim
        & prefix_suffix_feature_dict_size & prefix_suffix_feature_embedding_dim
        & length_feature_dict_size & length_feature_embedding_dim
        & rnn_x_dim & rnn_h_dim & nr_rnn_stacked_layer
        & hidden_dim & output_dim
        & dropout_rate ;
    ar & word_dict & postag_dict & pos_feature ;
    ar & *m ;
}

template <typename RNNDerived, typename Archive>
void SingleInputModelWithFeature<RNNDerived>::load(Archive &ar, const unsigned version)
{
    ar & word_dict_size & word_embedding_dim
        & prefix_suffix_feature_dict_size & prefix_suffix_feature_embedding_dim
        & length_feature_dict_size & length_feature_embedding_dim
        & rnn_x_dim & rnn_h_dim & nr_rnn_stacked_layer
        & hidden_dim & output_dim
        & dropout_rate ;
    ar & input_dict & output_dict & pos_feature;
    assert(word_dict.size() == word_dict_size && postag_dict.size() == output_dim) ;
    build_model_structure() ;
    ar & *m ;
}


} // end of namespace slnn
#endif