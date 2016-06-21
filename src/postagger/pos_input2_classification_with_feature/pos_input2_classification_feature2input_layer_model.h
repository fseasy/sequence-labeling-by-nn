#ifndef SLNN_POSTAGGER_POS_INPUT1_CLASSIFICATION_FEATURE2INPUT_LAYER_MODEL_H_
#define SLNN_POSTAGGER_POS_INPUT1_CLASSIFICATION_FEATURE2INPUT_LAYER_MODEL_H_

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "cnn/cnn.h"

#include "postagger/base_model/input1_feature2input_layer_model.hpp"
namespace slnn{
template<typename RNNDerived>
class POSInput1ClassificationF2IModel : public Input2F2IModel<RNNDerived>
{
    friend class boost::serialization::access;
public:
   
    POSInput1ClassificationF2IModel() ;
    ~POSInput1ClassificationF2IModel();

    void set_model_param(const boost::program_options::variables_map &var_map) ;
    void build_model_structure() ;
    void print_model_info() ;
};

template <typename RNNDerived>
POSInput1ClassificationF2IModel<RNNDerived>::POSInput1ClassificationF2IModel()
    : Input2F2IModel<RNNDerived>()
{}

template <typename RNNDerived>
POSInput1ClassificationF2IModel<RNNDerived>::~POSInput1ClassificationF2IModel()
{}

template <typename RNNDerived>
void POSInput1ClassificationF2IModel<RNNDerived>::set_model_param(const boost::program_options::variables_map &var_map)
{
    POSInput1ClassificationF2IModel<RNNDerived>::Input2F2IModel::set_model_param(var_map);
}

template <typename RNNDerived>
void POSInput1ClassificationF2IModel<RNNDerived>::build_model_structure()
{
    this->m = new cnn::Model() ;
    this->pos_feature_layer = new POSFeatureLayer(this->m, this->pos_feature);
    this->input_layer = new Input1WithFeature(this->m, this->word_dict_size, this->word_embedding_dim, 
                                              this->pos_feature.concatenated_feature_embedding_dim,
                                              this->rnn_x_dim) ;
    this->birnn_layer = new BIRNNLayer<RNNDerived>(this->m, this->nr_rnn_stacked_layer, 
                                                   this->rnn_x_dim, this->rnn_h_dim, this->dropout_rate) ;
    this->output_layer = new SimpleOutput(this->m, this->rnn_h_dim, this->rnn_h_dim, this->hidden_dim, this->output_dim,
        this->dropout_rate) ;
}

template <typename RNNDerived>
void POSInput1ClassificationF2IModel<RNNDerived>::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- POS Input1 Classification F2I Model -----------------\n"
        << "vocabulary size : " << this->word_dict_size << " with dimension : " << this->word_embedding_dim << "\n"
        << "birnn x dim : " << this->rnn_x_dim << " , h dim : " << this->rnn_h_dim
        << " , stacked layer num : " << this->nr_rnn_stacked_layer << "\n"
        << "tag hidden layer dim : " << this->hidden_dim << "\n"
        << "output dim : " << this->output_dim << "\n"
        << "feature info : \n"
        << this->pos_feature.get_feature_info() ;
}
} // end of namespace slnn 
#endif 
