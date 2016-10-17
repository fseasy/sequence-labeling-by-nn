#ifndef SLNN_SEGMENTOR_CWS_BAREINPUT1_CL_F2O_MODEL_HPP_
#define SLNN_SEGMENTOR_CWS_BAREINPUT1_CL_F2O_MODEL_HPP_

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

#include "cnn/cnn.h"

#include "segmenter/base_model/bareinput1_f2o_model.hpp"
#include "segmenter/cws_module/cws_output_layer.h"
namespace slnn{

template<typename RNNDerived>
class CWSBareInput1CLF2OModel : public CWSBareInput1F2OModel<RNNDerived>
{
public:

    CWSBareInput1CLF2OModel();
    ~CWSBareInput1CLF2OModel() ;

    void build_model_structure() override;
    void print_model_info() override;
};

template <typename RNNDerived>
CWSBareInput1CLF2OModel<RNNDerived>::CWSBareInput1CLF2OModel()
    :CWSBareInput1F2OModel<RNNDerived>()
{}

template <typename RNNDerived>
CWSBareInput1CLF2OModel<RNNDerived>::~CWSBareInput1CLF2OModel(){}


template <typename RNNDerived>
void CWSBareInput1CLF2OModel<RNNDerived>::build_model_structure()
{
    this->m = new cnn::Model() ;
    this->word_expr_layer = new Index2ExprLayer(this->m, this->word_dict_size, this->word_embedding_dim) ;
    this->cws_feature_layer = new CWSFeatureLayer(this->m, this->cws_feature, this->word_expr_layer->get_lookup_param());
    this->birnn_layer = new BIRNNLayer<RNNDerived>(this->m, this->nr_rnn_stacked_layer, this->word_embedding_dim, this->rnn_h_dim, 
        this->dropout_rate) ;
    this->output_layer = new CWSSimpleBareOutput(this->m, this->softmax_layer_input_dim , this->output_dim) ;
}

template <typename RNNDerived>
void CWSBareInput1CLF2OModel<RNNDerived>::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- CWS BareInput1 Classification F2O Model -----------------\n"
        << "vocabulary size : " << this->word_dict_size << " with dimension : " << this->word_embedding_dim << "\n"
        << "birnn x dim : " << this->word_embedding_dim << " , h dim : " << this->rnn_h_dim
        << " , stacked layer num : " << this->nr_rnn_stacked_layer << "\n"
        << "softmax layer input dim: " << this->softmax_layer_input_dim << "\n"
        << "output dim : " << this->output_dim << "\n"
        << "feature info : \n"
        << this->cws_feature.get_feature_info() ;
}
} // end of namespace slnn 
#endif 
