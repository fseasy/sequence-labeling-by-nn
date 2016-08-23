#ifndef SLNN_SEGMENTOR_CWS_INPUT1_CL_F2I_MODEL_H_
#define SLNN_SEGMENTOR_CWS_INPUT1_CL_F2I_MODEL_H_

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

#include "cnn/cnn.h"

#include "segmentor/base_model/input1_f2i_model_0628.hpp"
#include "segmentor/cws_module/cws_output_layer.h"
namespace slnn{

template<typename RNNDerived>
class CWSInput1CLF2IModel : public CWSInput1F2IModel<RNNDerived>
{
public:
    
    CWSInput1CLF2IModel() ;
    ~CWSInput1CLF2IModel() ;

    void build_model_structure() override;
    void print_model_info() override;
};

template <typename RNNDerived>
CWSInput1CLF2IModel<RNNDerived>::CWSInput1CLF2IModel()
    :CWSInput1F2IModel<RNNDerived>()
{}

template <typename RNNDerived>
CWSInput1CLF2IModel<RNNDerived>::~CWSInput1CLF2IModel(){}

template <typename RNNDerived>
void CWSInput1CLF2IModel<RNNDerived>::build_model_structure()
{
    this->m = new cnn::Model() ;
    this->input_layer = new Input1WithFeature(this->m, this->word_dict_size, this->word_embedding_dim, 
        this->cws_feature.get_feature_dim(), this->rnn_x_dim) ;
    this->cws_feature_layer = new CWSFeatureLayer(this->m, this->cws_feature,this->input_layer->get_lookup_param());
    this->birnn_layer = new BIRNNLayer<RNNDerived>(this->m, this->nr_rnn_stacked_layer, this->rnn_x_dim, this->rnn_h_dim, 
        this->dropout_rate) ;
    this->output_layer = new CWSSimpleOutputNew(this->m, this->rnn_h_dim, this->rnn_h_dim, 
        this->hidden_dim, this->output_dim, this->dropout_rate) ;
}

template <typename RNNDerived>
void CWSInput1CLF2IModel<RNNDerived>::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- CWS Input1 Classification F2I Model -----------------\n"
        << "vocabulary size : " << this->word_dict_size << " with dimension : " << this->word_embedding_dim << "\n"
        << "birnn x dim : " << this->rnn_x_dim << " , h dim : " << this->rnn_h_dim
        << " , stacked layer num : " << this->nr_rnn_stacked_layer << "\n"
        << "tag hidden layer dim : " << this->hidden_dim << "\n"
        << "output dim : " << this->output_dim << "\n"
        << "feature info : \n"
        << this->cws_feature.get_feature_info() ;
}
} // end of namespace slnn 
#endif 
