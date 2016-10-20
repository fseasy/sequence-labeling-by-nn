#ifndef SLNN_POSTAGGER_POS_INPUT2_CRF_FEATURE2OUTPUT_LAYER_MODEL_H_
#define SLNN_POSTAGGER_POS_INPUT2_CRF_FEATURE2OUTPUT_LAYER_MODEL_H_

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "dynet/dynet.h"

#include "postagger/base_model/input2_feature2output_layer_model.hpp"
namespace slnn{
template<typename RNNDerived>
class POSInput2CRFF2OModel : public Input2F2OModel<RNNDerived>
{
    friend class boost::serialization::access;
public:
   
    POSInput2CRFF2OModel() ;
    ~POSInput2CRFF2OModel();

    void set_model_param(const boost::program_options::variables_map &var_map) ;
    void build_model_structure() ;
    void print_model_info() ;
    template<typename Archive>
    void serialize(Archive &ar, const unsigned version);
public :
    unsigned tag_embedding_dim;
};

template <typename RNNDerived>
POSInput2CRFF2OModel<RNNDerived>::POSInput2CRFF2OModel()
    : Input2F2OModel<RNNDerived>()
{}

template <typename RNNDerived>
POSInput2CRFF2OModel<RNNDerived>::~POSInput2CRFF2OModel()
{}

template <typename RNNDerived>
void POSInput2CRFF2OModel<RNNDerived>::set_model_param(const boost::program_options::variables_map &var_map)
{
    tag_embedding_dim = var_map["tag_embedding_dim"].as<unsigned>();
    POSInput2CRFF2OModel<RNNDerived>::Input2F2OModel::set_model_param(var_map);
}

template <typename RNNDerived>
void POSInput2CRFF2OModel<RNNDerived>::build_model_structure()
{
    this->m = new dynet::Model() ;
    this->pos_feature_layer = new POSFeatureLayer(this->m, this->pos_feature);
    this->input_layer = new Input2(this->m, this->dynamic_word_dict_size, this->dynamic_word_embedding_dim, 
        this->fixed_word_dict_size, this->fixed_word_embedding_dim, this->rnn_x_dim) ;
    this->birnn_layer = new BIRNNLayer<RNNDerived>(this->m, this->nr_rnn_stacked_layer, 
                                                   this->rnn_x_dim, this->rnn_h_dim, this->dropout_rate) ;
    this->output_layer = new CRFOutputWithFeature(this->m, tag_embedding_dim, this->rnn_h_dim, this->rnn_h_dim, 
        this->pos_feature.concatenated_feature_embedding_dim,
        this->hidden_dim, this->output_dim,
        this->dropout_rate) ;
}

template <typename RNNDerived>
void POSInput2CRFF2OModel<RNNDerived>::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- POS Input2 CRF F2O Model -----------------\n"
        << "dynamic vocabulary size : " << this->dynamic_word_dict_size << " with dimension : " << this->dynamic_word_embedding_dim << "\n"
        << "fixed vocabulary size : " << this->fixed_word_dict_size << " with dimension : " << this->fixed_word_embedding_dim << "\n"
        << "tag dict size : " << this->output_dim << " with dimension : " << tag_embedding_dim << "\n"
        << "birnn x dim : " << this->rnn_x_dim << " , h dim : " << this->rnn_h_dim
        << " , stacked layer num : " << this->nr_rnn_stacked_layer << "\n"
        << "tag hidden layer dim : " << this->hidden_dim << "\n"
        << "output dim : " << this->output_dim << "\n"
        << "feature info : \n"
        << this->pos_feature.get_feature_info() ;
}
template <typename RNNDerived>
template <typename Archive>
void POSInput2CRFF2OModel<RNNDerived>::serialize(Archive &ar, unsigned version)
{
    ar & tag_embedding_dim;
    ar & boost::serialization::base_object<Input2F2OModel<RNNDerived>>(*this);
}

} // end of namespace slnn 
#endif 
