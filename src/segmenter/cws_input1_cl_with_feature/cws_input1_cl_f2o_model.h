#ifndef SLNN_SEGMENTOR_CWS_INPUT1_CL_F2O_MODEL_HPP_
#define SLNN_SEGMENTOR_CWS_INPUT1_CL_F2O_MODEL_HPP_

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

#include "dynet/dynet.h"

#include "segmenter/base_model/input1_f2o_model_0628.hpp"
#include "segmenter/cws_module/cws_output_layer.h"
namespace slnn{

template<typename RNNDerived>
class CWSInput1CLF2OModel : public CWSInput1F2OModel<RNNDerived>
{
public:

    CWSInput1CLF2OModel();
    ~CWSInput1CLF2OModel() ;

    void build_model_structure() override;
    void print_model_info() override;
};

template <typename RNNDerived>
CWSInput1CLF2OModel<RNNDerived>::CWSInput1CLF2OModel()
    :CWSInput1F2OModel<RNNDerived>()
{}

template <typename RNNDerived>
CWSInput1CLF2OModel<RNNDerived>::~CWSInput1CLF2OModel(){}

template <typename RNNDerived>
void CWSInput1CLF2OModel<RNNDerived>::build_model_structure()
{
    this->m = new dynet::Model() ;
    this->input_layer = new Input1(this->m, this->word_dict_size, this->word_embedding_dim) ;
    this->cws_feature_layer = new CWSFeatureLayer(this->m, this->cws_feature, this->input_layer->get_lookup_param());
    this->birnn_layer = new BIRNNLayer<RNNDerived>(this->m, this->nr_rnn_stacked_layer, this->word_embedding_dim, this->rnn_h_dim, 
        this->dropout_rate) ;
    this->output_layer = new CWSSimpleOutputWithFeature(this->m, this->rnn_h_dim, this->rnn_h_dim, this->cws_feature.get_feature_dim(),
        this->hidden_dim, this->output_dim, this->dropout_rate) ;
}

template <typename RNNDerived>
void CWSInput1CLF2OModel<RNNDerived>::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- CWS Input1 Classification F2O Model -----------------\n"
        << "vocabulary size : " << this->word_dict_size << " with dimension : " << this->word_embedding_dim << "\n"
        << "birnn x dim : " << this->word_embedding_dim << " , h dim : " << this->rnn_h_dim
        << " , stacked layer num : " << this->nr_rnn_stacked_layer << "\n"
        << "tag hidden layer dim : " << this->hidden_dim << "\n"
        << "output dim : " << this->output_dim << "\n"
        << "feature info : \n"
        << this->cws_feature.get_feature_info() ;
}
} // end of namespace slnn 
#endif 
