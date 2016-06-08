#include "pos_single_classification_with_feature_model.h"
#include "modelmodule/hyper_layers.h"

namespace slnn{

template <typename RNNDerived>
POSSingleClassificationWithFeatureModel<RNNDerived>::POSSingleClassificationWithFeatureModel()
    :SingleInputWithFeatureModel<RNNDerived>() 
{}

template <typename RNNDerived>
POSSingleClassificationWithFeatureModel<RNNDerived>::~POSSingleClassificationWithFeatureModel()
{}


template <typename RNNDerived>
void POSSingleClassificationWithFeatureModel<RNNDerived>::set_model_param(const boost::program_options::variables_map &var_map)
{
    POSSingleClassificationWithFeatureModel<RNNDerived>::SingleInputModelWithFeature<RNNDerived>::set_model_param(var_map);
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
                                                   this->rnn_x_dim, this->lstm_h_dim, this->dropout_rate) ;
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
        << "output dim : " << this->output_dim "\n"
        << "feature info : \n"
        << this->pos_feature.get_feature_info() ;
}


}
