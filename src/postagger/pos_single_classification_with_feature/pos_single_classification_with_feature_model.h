#ifndef SLNN_POSTAGGER_POS_SINGLE_INPUT_CLASSIFICATION_WITH_FEATURE_MODEL_H_
#define SLNN_POSTAGGER_POS_SINGLE_INPUT_CLASSIFICATION_WITH_FEATURE_MODEL_H_

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "cnn/cnn.h"

#include "postagger/base_model/input1_feature2input_layer_model.hpp"
namespace slnn{
template<typename RNNDerived>
class POSSingleClassificationWithFeatureModel : public Input1F2IModel<RNNDerived>
{
    friend class boost::serialization::access;
public:
   
    POSSingleClassificationWithFeatureModel() ;
    ~POSSingleClassificationWithFeatureModel();

    void set_model_param(const boost::program_options::variables_map &var_map) ;
    void build_model_structure() ;
    void print_model_info() ;
};

template <typename RNNDerived>
POSSingleClassificationWithFeatureModel<RNNDerived>::POSSingleClassificationWithFeatureModel()
    : Input1F2IModel<RNNDerived>()
{}

template <typename RNNDerived>
POSSingleClassificationWithFeatureModel<RNNDerived>::~POSSingleClassificationWithFeatureModel()
{}

template <typename RNNDerived>
void POSSingleClassificationWithFeatureModel<RNNDerived>::set_model_param(const boost::program_options::variables_map &var_map)
{
    POSSingleClassificationWithFeatureModel<RNNDerived>::Input1F2IModel::set_model_param(var_map);
}

template <typename RNNDerived>
void POSSingleClassificationWithFeatureModel<RNNDerived>::build_model_structure()
{
    this->m = new cnn::Model() ;
    this->pos_feature_layer = new POSFeatureLayer(this->m, this->pos_feature);
    this->input_layer = new Input1WithFeature(this->m, this->word_dict_size, this->word_embedding_dim, 
                                              this->pos_feature.concatenated_feature_embedding_dim,
                                              this->rnn_x_dim) ;
    this->birnn_layer = new BIRNNLayer<RNNDerived>(this->m, this->nr_rnn_stacked_layer, 
                                                   this->rnn_x_dim, this->rnn_h_dim, this->dropout_rate) ;
    this->output_layer = new SimpleOutput(this->m, this->rnn_h_dim, this->rnn_h_dim, this->hidden_dim, this->output_dim) ;
}

template <typename RNNDerived>
void POSSingleClassificationWithFeatureModel<RNNDerived>::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- Single Input Classification With Feature Model -----------------\n"
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
