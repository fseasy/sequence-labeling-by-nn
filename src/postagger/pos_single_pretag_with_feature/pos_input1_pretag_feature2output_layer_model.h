#ifndef SLNN_POSTAGGER_POS_INPUT1_PRETAG_FEATURE2OUTPUT_LAYER_MODEL_H_
#define SLNN_POSTAGGER_POS_INPUT1_PRETAG_FEATURE2OUTPUT_LAYER_MODEL_H_

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "dynet/dynet.h"

#include "postagger/base_model/input1_feature2output_layer_model.hpp"
namespace slnn{
template<typename RNNDerived>
class POSInput1PretagF2OModel : public Input1F2OModel<RNNDerived>
{
    friend class boost::serialization::access;
public:
   
    POSInput1PretagF2OModel() ;
    ~POSInput1PretagF2OModel();

    void set_model_param(const boost::program_options::variables_map &var_map) ;
    void build_model_structure() ;
    void print_model_info() ;

    template<typename Archive>
    void serialize(Archive &ar, const unsigned version);
public :
    unsigned tag_embedding_dim;

};

template <typename RNNDerived>
POSInput1PretagF2OModel<RNNDerived>::POSInput1PretagF2OModel()
    : Input1F2OModel<RNNDerived>()
{}

template <typename RNNDerived>
POSInput1PretagF2OModel<RNNDerived>::~POSInput1PretagF2OModel()
{}

template <typename RNNDerived>
void POSInput1PretagF2OModel<RNNDerived>::set_model_param(const boost::program_options::variables_map &var_map)
{
    tag_embedding_dim = var_map["tag_embedding_dim"].as<unsigned>();
    POSInput1PretagF2OModel<RNNDerived>::Input1F2OModel::set_model_param(var_map);
}

template <typename RNNDerived>
void POSInput1PretagF2OModel<RNNDerived>::build_model_structure()
{
    this->m = new dynet::Model() ;
    this->pos_feature_layer = new POSFeatureLayer(this->m, this->pos_feature);
    this->input_layer = new Input1(this->m, this->word_dict_size, this->word_embedding_dim) ;
    this->birnn_layer = new BIRNNLayer<RNNDerived>(this->m, this->nr_rnn_stacked_layer, 
                                                   this->word_embedding_dim, this->rnn_h_dim, this->dropout_rate) ;
    this->output_layer = new PretagOutputWithFeature(this->m, tag_embedding_dim, this->rnn_h_dim, this->rnn_h_dim, 
        this->pos_feature.concatenated_feature_embedding_dim,
        this->hidden_dim, this->output_dim,
        this->dropout_rate) ;
}

template <typename RNNDerived>
void POSInput1PretagF2OModel<RNNDerived>::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- Input1 Pretag F2O Model -----------------\n"
        << "vocabulary size : " << this->word_dict_size << " with dimension : " << this->word_embedding_dim << "\n"
        << "tag dict size : " << this->output_dim << " with dimension : " << tag_embedding_dim << "\n"
        << "birnn x dim : " << this->word_embedding_dim << " , h dim : " << this->rnn_h_dim
        << " , stacked layer num : " << this->nr_rnn_stacked_layer << "\n"
        << "tag hidden layer dim : " << this->hidden_dim << "\n"
        << "output dim : " << this->output_dim << "\n"
        << "feature info : \n"
        << this->pos_feature.get_feature_info() ;
}

template <typename RNNDerived>
template <typename Archive>
void POSInput1PretagF2OModel<RNNDerived>::serialize(Archive &ar, unsigned version)
{
    ar & tag_embedding_dim;
    ar & boost::serialization::base_object<Input1F2OModel<RNNDerived>>(*this);
}

} // end of namespace slnn 
#endif 
