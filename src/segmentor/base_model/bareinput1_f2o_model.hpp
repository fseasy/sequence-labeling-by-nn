#ifndef SLNN_SEGMENTOR_BASE_MODEL_BAREINPUT1_F2O_MODEL_0628_HPP_
#define SLNN_SEGMENTOR_BASE_MODEL_BAREINPUT1_F2O_MODEL_0628_HPP_

#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

#include "cnn/cnn.h"
#include "cnn/dict.h"
#include "input1_with_feature_model_0628.hpp"
#include "segmentor/cws_module/cws_feature_layer.h"

namespace slnn{

template<typename RNNDerived>
class CWSBareInput1F2OModel : public CWSInput1WithFeatureModel<RNNDerived>
{
    friend class boost::serialization::access;

public:
    CWSBareInput1F2OModel();
    virtual ~CWSBareInput1F2OModel();
    CWSBareInput1F2OModel(const CWSBareInput1F2OModel &) = delete;
    CWSBareInput1F2OModel& operator()(const CWSBareInput1F2OModel&) = delete;

    virtual void set_model_param(const boost::program_options::variables_map &var_map) override;

    virtual void build_model_structure() = 0 ;
    virtual void print_model_info() = 0 ;

    virtual cnn::expr::Expression  build_loss(cnn::ComputationGraph &cg,
        const IndexSeq &input_seq,
        const CWSFeatureDataSeq &feature_data_seq,
        const IndexSeq &gold_seq) override;
    virtual void predict(cnn::ComputationGraph &cg,
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

    CWSFeatureLayer *cws_feature_layer;
    Index2ExprLayer *word_expr_layer;
    BIRNNLayer<RNNDerived> *birnn_layer;
    BareOutputBase *output_layer;

public:
    unsigned word_embedding_dim,
        word_dict_size,
        nr_rnn_stacked_layer,
        rnn_h_dim,
        softmax_layer_input_dim,
        output_dim ;

    cnn::real dropout_rate ; 

};


template<typename RNNDerived>
CWSBareInput1F2OModel<RNNDerived>::CWSBareInput1F2OModel() 
    :CWSInput1WithFeatureModel<RNNDerived>(),
    cws_feature_layer(nullptr),
    word_expr_layer(nullptr),
    birnn_layer(nullptr),
    output_layer(nullptr)
{}

template <typename RNNDerived>
CWSBareInput1F2OModel<RNNDerived>::~CWSBareInput1F2OModel()
{
    delete cws_feature_layer;
    delete word_expr_layer;
    delete birnn_layer;
    delete output_layer;
}

template <typename RNNDerived>
void CWSBareInput1F2OModel<RNNDerived>::set_model_param(const boost::program_options::variables_map &var_map)
{
    assert(this->word_dict.is_frozen()) ;

    unsigned replace_freq_threshold = var_map["replace_freq_threshold"].as<unsigned>();
    float replace_prob_threshold = var_map["replace_prob_threshold"].as<float>();
    this->set_replace_threshold(replace_freq_threshold, replace_prob_threshold);

    word_embedding_dim = var_map["word_embedding_dim"].as<unsigned>() ;
    nr_rnn_stacked_layer = var_map["nr_rnn_stacked_layer"].as<unsigned>() ;
    rnn_h_dim = var_map["rnn_h_dim"].as<unsigned>() ;

    dropout_rate = var_map["dropout_rate"].as<cnn::real>() ;

    unsigned start_here_embedding_dim = var_map["start_here_embedding_dim"].as<unsigned>();
    unsigned pass_here_embedding_dim = var_map["pass_here_embedding_dim"].as<unsigned>();
    unsigned end_here_embedding_dim = var_map["end_here_embedding_dim"].as<unsigned>();
    this->cws_feature.set_dim(start_here_embedding_dim, pass_here_embedding_dim, end_here_embedding_dim);

    word_dict_size = this->get_word_dict_size() ;
    output_dim = this->get_tag_dict_size() ;
    softmax_layer_input_dim = rnn_h_dim * 2 + this->cws_feature.get_feature_dim();
}


template<typename RNNDerived>
cnn::expr::Expression CWSBareInput1F2OModel<RNNDerived>::build_loss(cnn::ComputationGraph &cg,
    const IndexSeq &input_seq, 
    const CWSFeatureDataSeq &feature_seq,
    const IndexSeq &gold_seq)
{
    cws_feature_layer->new_graph(cg);
    word_expr_layer->new_graph(cg) ;
    birnn_layer->new_graph(cg) ;
    output_layer->new_graph(cg) ;

    birnn_layer->set_dropout() ;
    birnn_layer->start_new_sequence() ;

    std::vector<cnn::expr::Expression> word_exprs ;
    word_expr_layer->index_seq2expr_seq(input_seq, word_exprs) ;

    std::vector<cnn::expr::Expression> l2r_exprs,
        r2l_exprs ;
    birnn_layer->build_graph(word_exprs, l2r_exprs, r2l_exprs) ;

    std::vector<cnn::expr::Expression> feature_exprs;
    cws_feature_layer->build_cws_feature(feature_seq, feature_exprs);

    return output_layer->build_output_loss(std::vector<std::vector<cnn::expr::Expression> *>({
           &l2r_exprs, &r2l_exprs, &feature_exprs}), gold_seq) ;
}

template<typename RNNDerived>
void CWSBareInput1F2OModel<RNNDerived>::predict(cnn::ComputationGraph &cg,
    const IndexSeq &input_seq,
    const CWSFeatureDataSeq &feature_seq,
    IndexSeq &pred_seq)
{
    cws_feature_layer->new_graph(cg);
    word_expr_layer->new_graph(cg) ;
    birnn_layer->new_graph(cg) ;
    output_layer->new_graph(cg) ;

    birnn_layer->set_dropout() ;
    birnn_layer->start_new_sequence() ;

    std::vector<cnn::expr::Expression> word_exprs ;
    word_expr_layer->index_seq2expr_seq(input_seq, word_exprs) ;

    std::vector<cnn::expr::Expression> l2r_exprs,
        r2l_exprs ;
    birnn_layer->build_graph(word_exprs, l2r_exprs, r2l_exprs) ;

    std::vector<cnn::expr::Expression> feature_exprs;
    cws_feature_layer->build_cws_feature(feature_seq, feature_exprs);

    output_layer->build_output(std::vector<std::vector<cnn::expr::Expression> *>({ &l2r_exprs, &r2l_exprs, &feature_exprs }), 
                 pred_seq) ;
}

template <typename RNNDerived>template< typename Archive>
void CWSBareInput1F2OModel<RNNDerived>::save(Archive &ar, const unsigned version) const
{
    ar & word_dict_size & word_embedding_dim
        & rnn_h_dim & nr_rnn_stacked_layer
        & softmax_layer_input_dim & output_dim
        & dropout_rate ;
    ar & this->word_dict & this->cws_feature ;
    ar & *this->m ;
}

template <typename RNNDerived>template< typename Archive>
void CWSBareInput1F2OModel<RNNDerived>::load(Archive &ar, const unsigned version)
{
    ar & word_dict_size & word_embedding_dim
        & rnn_h_dim & nr_rnn_stacked_layer
        & softmax_layer_input_dim & output_dim
        & dropout_rate ;
    ar & this->word_dict & this->cws_feature;
    assert(this->get_word_dict_size() == word_dict_size && this->get_tag_dict_size() == output_dim) ;
    build_model_structure() ;
    ar & *this->m ;
}

template <typename RNNDerived>
template<typename Archive>
void CWSBareInput1F2OModel<RNNDerived>::serialize(Archive & ar, const unsigned version)
{
    boost::serialization::split_member(ar, *this, version);
}

} // end of namespace slnn
#endif
