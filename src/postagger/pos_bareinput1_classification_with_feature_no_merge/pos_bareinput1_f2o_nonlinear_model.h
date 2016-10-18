#ifndef SLNN_POSTAGGER_POS_BAREINPUT1_CLASSIFICATION_FEATURE2OUTPUT_LAYER_NONLINEAR_MODEL_H_
#define SLNN_POSTAGGER_POS_BAREINPUT1_CLASSIFICATION_FEATURE2OUTPUT_LAYER_NONLINEAR_MODEL_H_

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "dynet/dynet.h"

#include "postagger/base_model/bareinput1_f2o_nonlinear_model.hpp"
namespace slnn{
template<typename RNNDerived>
class POSBareInput1ClassificationF2ONonlinearModel : public BareInput1F2ONonlinearModel<RNNDerived>
{
    friend class boost::serialization::access;
public:
   
    POSBareInput1ClassificationF2ONonlinearModel() ;
    ~POSBareInput1ClassificationF2ONonlinearModel();

    void set_model_param(const boost::program_options::variables_map &var_map) ;
    void build_model_structure() ;
    void print_model_info() ;
};

template <typename RNNDerived>
POSBareInput1ClassificationF2ONonlinearModel<RNNDerived>::POSBareInput1ClassificationF2ONonlinearModel()
    : BareInput1F2ONonlinearModel<RNNDerived>()
{}

template <typename RNNDerived>
POSBareInput1ClassificationF2ONonlinearModel<RNNDerived>::~POSBareInput1ClassificationF2ONonlinearModel()
{}

template <typename RNNDerived>
void POSBareInput1ClassificationF2ONonlinearModel<RNNDerived>::set_model_param(const boost::program_options::variables_map &var_map)
{
    POSBareInput1ClassificationF2ONonlinearModel<RNNDerived>::BareInput1F2ONonlinearModel::set_model_param(var_map);
}

template <typename RNNDerived>
void POSBareInput1ClassificationF2ONonlinearModel<RNNDerived>::build_model_structure()
{
    this->m = new dynet::Model() ;
    this->pos_feature_layer = new POSFeatureLayer(this->m, this->pos_feature);
    this->pos_feature_hidden_layer = new DenseLayer(this->m, this->pos_feature.get_pos_feature_dim(), this->pos_feature_hidden_layer_dim);
    this->input_layer = new Input1(this->m, this->word_dict_size, this->word_embedding_dim) ;
    this->birnn_layer = new BIRNNLayer<RNNDerived>(this->m, this->nr_rnn_stacked_layer, 
                                                   this->word_embedding_dim, this->rnn_h_dim, this->dropout_rate) ;
    this->output_layer = new SimpleBareOutput(this->m, this->softmax_input_dim, this->output_dim) ;
}

template <typename RNNDerived>
void POSBareInput1ClassificationF2ONonlinearModel<RNNDerived>::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- POS Bare Input1 Classification F2O NonLinear Model -----------------\n"
        << "vocabulary size : " << this->word_dict_size << " with dimension : " << this->word_embedding_dim << "\n"
        << "birnn x dim : " << this->word_embedding_dim << " , h dim : " << this->rnn_h_dim
        << " , stacked layer num : " << this->nr_rnn_stacked_layer << "\n"
        << "postag feature hidden layer dimension : " << this->pos_feature_hidden_layer_dim << "\n"
        << "softmax layer input dim : " << this->softmax_input_dim << "\n"
        << "output dim : " << this->output_dim << "\n"
        << "feature info : \n"
        << this->pos_feature.get_feature_info() ;
}
} // end of namespace slnn 
#endif 
