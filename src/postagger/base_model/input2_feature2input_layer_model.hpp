#ifndef POS_BASE_MODEL_INPUT2_FEATURE2INPUT_LAYER_MODEL_HPP_
#define POS_BASE_MODEL_INPUT2_FEATURE2INPUT_LAYER_MODEL_HPP_

#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

#include "cnn/cnn.h"
#include "cnn/dict.h"
#include "input2_with_feature_model.hpp"
namespace slnn{

template<typename RNNDerived>
class Input2F2IModel : public Input2WithFeatureModel<RNNDerived>
{
    friend class boost::serialization::access;

public:
    Input2F2IModel();
    virtual ~Input2F2IModel();
    Input2F2IModel(const Input2F2IModel &) = delete;
    Input2F2IModel& operator=(const Input2F2IModel&) = delete;

    virtual void set_model_param(const boost::program_options::variables_map &var_map) override;
    void build_fixed_dict(std::ifstream &is) override;

    virtual void build_model_structure() = 0 ;
    void load_fixed_embedding(std::ifstream &is) override;
    virtual void print_model_info() = 0 ;

   

    virtual cnn::expr::Expression  build_loss(cnn::ComputationGraph &cg,
        const IndexSeq &dynamic_sent,
        const IndexSeq &fixed_sent,
        const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
        const IndexSeq &gold_seq) override;
    virtual void predict(cnn::ComputationGraph &cg,
        const IndexSeq &dynamic_sent,
        const IndexSeq &fixed_sent,
        const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
        IndexSeq &pred_seq) override;

    template <typename Archive>
    void save(Archive &ar, const unsigned versoin) const; 
    template <typename Archive>
    void load(Archive &ar, const unsigned version);
    template <typename Archive>
    void serialize(Archive & ar, const unsigned version);

protected:
    
    POSFeatureLayer * pos_feature_layer;
    Input2WithFeature *input_layer;
    BIRNNLayer<RNNDerived> *birnn_layer;
    OutputBase *output_layer;

public:
    unsigned dynamic_word_embedding_dim,
        dynamic_word_dict_size,
        fixed_word_embedding_dim,
        fixed_word_dict_size,
        nr_rnn_stacked_layer,
        rnn_x_dim,
        rnn_h_dim,
        hidden_dim,
        output_dim ;

    cnn::real dropout_rate ; 

};


template<typename RNNDerived>
Input2F2IModel<RNNDerived>::Input2F2IModel() 
    :Input2WithFeatureModel<RNNDerived>(),
    pos_feature_layer(nullptr),
    input_layer(nullptr),
    birnn_layer(nullptr),
    output_layer(nullptr)
{}

template <typename RNNDerived>
Input2F2IModel<RNNDerived>::~Input2F2IModel()
{
    delete pos_feature_layer;
    delete input_layer;
    delete birnn_layer;
    delete output_layer;
}

template <typename RNNDerived>
void Input2F2IModel<RNNDerived>::set_model_param(const boost::program_options::variables_map &var_map)
{
    assert(this->dynamic_word_dict.is_frozen() && this->fixed_word_dict.is_frozen() &&
        this->postag_dict.is_frozen()  && this->pos_feature.is_dict_frozen()) ;

    unsigned replace_freq_threshold = var_map["replace_freq_threshold"].as<unsigned>();
    float replace_prob_threshold = var_map["replace_prob_threshold"].as<float>();
    this->set_replace_threshold(replace_freq_threshold, replace_prob_threshold);

    dynamic_word_embedding_dim = var_map["dynamic_word_embedding_dim"].as<unsigned>() ;
    nr_rnn_stacked_layer = var_map["nr_rnn_stacked_layer"].as<unsigned>() ;
    rnn_x_dim = var_map["rnn_x_dim"].as<unsigned>();
    rnn_h_dim = var_map["rnn_h_dim"].as<unsigned>() ;
    hidden_dim = var_map["tag_layer_hidden_dim"].as<unsigned>() ;

    dropout_rate = var_map["dropout_rate"].as<cnn::real>() ;

    unsigned prefix_suffix_len1_embedding_dim = var_map["prefix_suffix_len1_embedding_dim"].as<unsigned>();
    unsigned prefix_suffix_len2_embedding_dim = var_map["prefix_suffix_len2_embedding_dim"].as<unsigned>();
    unsigned prefix_suffix_len3_embedding_dim = var_map["prefix_suffix_len3_embedding_dim"].as<unsigned>();
    unsigned char_length_embedding_dim = var_map["char_length_embedding_dim"].as<unsigned>();
    this->pos_feature.init_embedding_dim(prefix_suffix_len1_embedding_dim, prefix_suffix_len2_embedding_dim,
                                   prefix_suffix_len3_embedding_dim, char_length_embedding_dim);
    
    dynamic_word_dict_size = this->dynamic_word_dict.size() ;
    output_dim = this->postag_dict.size() ;
}

template <typename RNNDerived>
void Input2F2IModel<RNNDerived>::build_fixed_dict(std::ifstream &is)
{
    Word2vecEmbeddingHelper::build_fixed_dict(is, this->fixed_word_dict, this->UNK_STR, 
        &fixed_word_dict_size, &fixed_word_embedding_dim);
}

template <typename RNNDerived>
void Input2F2IModel<RNNDerived>::load_fixed_embedding(std::ifstream &is)
{
    Word2vecEmbeddingHelper::load_fixed_embedding(is, this->fixed_word_dict, fixed_word_embedding_dim, input_layer->fixed_lookup_param);
}

template<typename RNNDerived>
cnn::expr::Expression Input2F2IModel<RNNDerived>::build_loss(cnn::ComputationGraph &cg,
    const IndexSeq &dynamic_sent,
    const IndexSeq &fixed_sent,
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
    input_layer->build_inputs(dynamic_sent, fixed_sent, features_exprs, inputs_exprs) ;

    std::vector<cnn::expr::Expression> l2r_exprs,
        r2l_exprs ;
    birnn_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;
    return output_layer->build_output_loss(l2r_exprs, r2l_exprs, gold_seq) ;
}

template<typename RNNDerived>
void Input2F2IModel<RNNDerived>::predict(cnn::ComputationGraph &cg,
    const IndexSeq &dynamic_sent,
    const IndexSeq &fixed_sent,
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
    pos_feature_layer->build_feature_exprs(features_gp_seq, feature_exprs);

    std::vector<cnn::expr::Expression> inputs_exprs ;
    input_layer->build_inputs(dynamic_sent, fixed_sent, feature_exprs, inputs_exprs) ;
    std::vector<cnn::expr::Expression> l2r_exprs,
        r2l_exprs ;
    birnn_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;
    output_layer->build_output(l2r_exprs, r2l_exprs, pred_seq) ;
}

template <typename RNNDerived>template< typename Archive>
void Input2F2IModel<RNNDerived>::save(Archive &ar, const unsigned version) const
{
    ar & dynamic_word_dict_size & dynamic_word_embedding_dim
        & fixed_word_dict_size & fixed_word_embedding_dim
        & rnn_x_dim & rnn_h_dim & nr_rnn_stacked_layer
        & hidden_dim & output_dim
        & dropout_rate ;
    ar & this->dynamic_word_dict & this->fixed_word_dict & this->postag_dict & this->pos_feature ;
    ar & *this->m ;
}

template <typename RNNDerived>template< typename Archive>
void Input2F2IModel<RNNDerived>::load(Archive &ar, const unsigned version)
{
    ar & dynamic_word_dict_size & dynamic_word_embedding_dim
        & fixed_word_dict_size & fixed_word_embedding_dim
        & rnn_x_dim & rnn_h_dim & nr_rnn_stacked_layer
        & hidden_dim & output_dim
        & dropout_rate ;
    ar & this->dynamic_word_dict & this->fixed_word_dict & this->postag_dict & this->pos_feature;
    assert(this->dynamic_word_dict.size() == dynamic_word_dict_size && this->fixed_word_dict.size() == fixed_word_dict_size &&
           this->postag_dict.size() == output_dim) ;
    build_model_structure() ;
    ar & *this->m ;
}

template <typename RNNDerived>
template<typename Archive>
void Input2F2IModel<RNNDerived>::serialize(Archive & ar, const unsigned version)
{
    boost::serialization::split_member(ar, *this, version);
}

} // end of namespace slnn
#endif
