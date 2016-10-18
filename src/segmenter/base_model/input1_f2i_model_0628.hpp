#ifndef SLNN_SEGMENTOR_BASE_MODEL_INPUT1_F2I_MODEL_0628_HPP_
#define SLNN_SEGMENTOR_BASE_MODEL_INPUT1_F2I_MODEL_0628_HPP_

#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

#include "dynet/dynet.h"
#include "dynet/dict.h"
#include "input1_with_feature_model_0628.hpp"
#include "segmenter/cws_module/cws_feature_layer.h"
namespace slnn{

template<typename RNNDerived>
class CWSInput1F2IModel : public CWSInput1WithFeatureModel<RNNDerived>
{
    friend class boost::serialization::access;

public:
    CWSInput1F2IModel();
    virtual ~CWSInput1F2IModel();
    CWSInput1F2IModel(const CWSInput1F2IModel &) = delete;
    CWSInput1F2IModel& operator()(const CWSInput1F2IModel&) = delete;

    virtual void set_model_param_from_outer(const boost::program_options::variables_map &var_map) override;
    virtual void set_model_param_from_inner() override;

    virtual void build_model_structure() = 0 ;
    virtual void print_model_info() = 0 ;

    virtual dynet::expr::Expression  build_loss(dynet::ComputationGraph &cg,
        const IndexSeq &input_seq,
        const CWSFeatureDataSeq &feature_data_seq,
        const IndexSeq &gold_seq) override;
    virtual void predict(dynet::ComputationGraph &cg,
        const IndexSeq &input_seq,
        const CWSFeatureDataSeq &feature_data_seq,
        IndexSeq &pred_seq) override;

    template <typename Archive>
    void save(Archive &ar, const unsigned versoin) const; 
    template <typename Archive>
    void load(Archive &ar, const unsigned version);
    template <typename Archive>
    void serialize(Archive & ar, const unsigned version);

protected:
    
    CWSFeatureLayer * cws_feature_layer;
    Input1WithFeature *input_layer;
    BIRNNLayer<RNNDerived> *birnn_layer;
    OutputBase *output_layer;

public:
    unsigned word_embedding_dim,
        word_dict_size,
        nr_rnn_stacked_layer,
        rnn_x_dim,
        rnn_h_dim,
        hidden_dim,
        output_dim ;

    dynet::real dropout_rate ; 

};


template<typename RNNDerived>
CWSInput1F2IModel<RNNDerived>::CWSInput1F2IModel() 
    :CWSInput1WithFeatureModel<RNNDerived>(),
    cws_feature_layer(nullptr),
    input_layer(nullptr),
    birnn_layer(nullptr),
    output_layer(nullptr)
{}

template <typename RNNDerived>
CWSInput1F2IModel<RNNDerived>::~CWSInput1F2IModel()
{
    delete cws_feature_layer;
    delete input_layer;
    delete birnn_layer;
    delete output_layer;
}

template <typename RNNDerived>
void CWSInput1F2IModel<RNNDerived>::set_model_param_from_outer(const boost::program_options::variables_map &var_map)
{
    unsigned replace_freq_threshold = var_map["replace_freq_threshold"].as<unsigned>();
    float replace_prob_threshold = var_map["replace_prob_threshold"].as<float>();
    this->set_replace_threshold(replace_freq_threshold, replace_prob_threshold);

    word_embedding_dim = var_map["word_embedding_dim"].as<unsigned>() ;
    nr_rnn_stacked_layer = var_map["nr_rnn_stacked_layer"].as<unsigned>() ;
    rnn_x_dim = var_map["rnn_x_dim"].as<unsigned>();
    rnn_h_dim = var_map["rnn_h_dim"].as<unsigned>() ;
    hidden_dim = var_map["tag_layer_hidden_dim"].as<unsigned>() ;

    dropout_rate = var_map["dropout_rate"].as<dynet::real>() ;

    unsigned start_here_embedding_dim = var_map["start_here_embedding_dim"].as<unsigned>();
    unsigned pass_here_embedding_dim = var_map["pass_here_embedding_dim"].as<unsigned>();
    unsigned end_here_embedding_dim = var_map["end_here_embedding_dim"].as<unsigned>();
    unsigned context_left_size = var_map["context_left_size"].as<unsigned>();
    unsigned context_right_size = var_map["context_right_size"].as<unsigned>();
    unsigned chartype_embedding_dim = var_map["chartype_embedding_dim"].as<unsigned>();
    this->cws_feature.set_feature_parameters(start_here_embedding_dim, pass_here_embedding_dim, end_here_embedding_dim,
        context_left_size, context_right_size, word_embedding_dim, chartype_embedding_dim);
}

template <typename RNNDerived>
void CWSInput1F2IModel<RNNDerived>::set_model_param_from_inner()
{
    assert(this->is_dict_frozen());
    word_dict_size = this->get_word_dict_size() ;
    output_dim = this->get_tag_dict_size() ;
}


template<typename RNNDerived>
dynet::expr::Expression CWSInput1F2IModel<RNNDerived>::build_loss(dynet::ComputationGraph &cg,
                                                                           const IndexSeq &input_seq, 
                                                                           const CWSFeatureDataSeq &cws_feature_seq,
                                                                           const IndexSeq &gold_seq) 
{
    cws_feature_layer->new_graph(cg);
    input_layer->new_graph(cg) ;
    birnn_layer->new_graph(cg) ;
    output_layer->new_graph(cg) ;

    birnn_layer->set_dropout() ;
    birnn_layer->start_new_sequence() ;

    std::vector<dynet::expr::Expression> features_exprs;
    cws_feature_layer->build_cws_feature(cws_feature_seq, features_exprs);

    std::vector<dynet::expr::Expression> inputs_exprs ;
    input_layer->build_inputs(input_seq, features_exprs, inputs_exprs) ;

    std::vector<dynet::expr::Expression> l2r_exprs,
        r2l_exprs ;
    birnn_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;
    return output_layer->build_output_loss(l2r_exprs, r2l_exprs, gold_seq) ;
}

template<typename RNNDerived>
void CWSInput1F2IModel<RNNDerived>::predict(dynet::ComputationGraph &cg,
                                                      const IndexSeq &input_seq,
                                                      const CWSFeatureDataSeq &cws_feature_seq,
                                                      IndexSeq &pred_seq)
{
    cws_feature_layer->new_graph(cg);
    input_layer->new_graph(cg) ;
    birnn_layer->new_graph(cg) ;
    output_layer->new_graph(cg) ;

    birnn_layer->disable_dropout() ;
    birnn_layer->start_new_sequence();

    std::vector<dynet::expr::Expression> feature_exprs;
    cws_feature_layer->build_cws_feature(cws_feature_seq, feature_exprs);

    std::vector<dynet::expr::Expression> inputs_exprs ;
    input_layer->build_inputs(input_seq, feature_exprs, inputs_exprs) ;
    std::vector<dynet::expr::Expression> l2r_exprs,
        r2l_exprs ;
    birnn_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;
    output_layer->build_output(l2r_exprs, r2l_exprs , pred_seq) ;
}

template <typename RNNDerived>template< typename Archive>
void CWSInput1F2IModel<RNNDerived>::save(Archive &ar, const unsigned version) const
{
    ar & word_dict_size & word_embedding_dim
        & rnn_x_dim & rnn_h_dim & nr_rnn_stacked_layer
        & hidden_dim & output_dim
        & dropout_rate ;
    ar & this->word_dict & this->cws_feature ;
    ar & *this->m ;
}

template <typename RNNDerived>template< typename Archive>
void CWSInput1F2IModel<RNNDerived>::load(Archive &ar, const unsigned version)
{
    ar & word_dict_size & word_embedding_dim
        & rnn_x_dim & rnn_h_dim & nr_rnn_stacked_layer
        & hidden_dim & output_dim
        & dropout_rate ;
    ar & this->word_dict & this->cws_feature;
    assert(this->get_word_dict_size() == word_dict_size && this->get_tag_dict_size() == output_dim) ;
    build_model_structure() ;
    ar & *this->m ;
}

template <typename RNNDerived>
template<typename Archive>
void CWSInput1F2IModel<RNNDerived>::serialize(Archive & ar, const unsigned version)
{
    boost::serialization::split_member(ar, *this, version);
}

} // end of namespace slnn
#endif
