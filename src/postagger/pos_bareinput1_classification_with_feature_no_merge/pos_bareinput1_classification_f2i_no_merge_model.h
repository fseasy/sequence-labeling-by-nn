#ifndef SLNN_POSTAGGER_POS_BAREINPUT1_CLASSIFICATION_FEATURE2INPUT_LAYER_NO_MERGE_MODEL_H_
#define SLNN_POSTAGGER_POS_BAREINPUT1_CLASSIFICATION_FEATURE2INPUT_LAYER_NO_MERGE_MODEL_H_

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "dynet/dynet.h"

#include "postagger/base_model/bareinput1_f2i_no_merge_model.hpp"
namespace slnn{
template<typename RNNDerived>
class POSBareInput1ClassificationF2IModel : public BareInput1F2IModel<RNNDerived>
{
    friend class boost::serialization::access;
public:
   
    POSBareInput1ClassificationF2IModel() ;
    ~POSBareInput1ClassificationF2IModel();

    void set_model_param(const boost::program_options::variables_map &var_map) ;
    void build_model_structure() ;
    void print_model_info() ;
};

template <typename RNNDerived>
POSBareInput1ClassificationF2IModel<RNNDerived>::POSBareInput1ClassificationF2IModel()
    : BareInput1F2IModel<RNNDerived>()
{}

template <typename RNNDerived>
POSBareInput1ClassificationF2IModel<RNNDerived>::~POSBareInput1ClassificationF2IModel()
{}

template <typename RNNDerived>
void POSBareInput1ClassificationF2IModel<RNNDerived>::set_model_param(const boost::program_options::variables_map &var_map)
{
    POSBareInput1ClassificationF2IModel<RNNDerived>::BareInput1F2IModel::set_model_param(var_map);
}

template <typename RNNDerived>
void POSBareInput1ClassificationF2IModel<RNNDerived>::build_model_structure()
{
    this->m = new dynet::Model() ;
    this->pos_feature_layer = new POSFeatureLayer(this->m, this->pos_feature);
    this->input_layer = new AnotherBareInput1(this->m, this->word_dict_size, this->word_embedding_dim) ;
    this->birnn_layer = new BIRNNLayer<RNNDerived>(this->m, this->nr_rnn_stacked_layer, 
                                                   this->rnn_x_dim, this->rnn_h_dim, this->dropout_rate) ;
    this->output_layer = new SimpleBareOutput(this->m, this->softmax_input_dim, this->output_dim) ;
}

template <typename RNNDerived>
void POSBareInput1ClassificationF2IModel<RNNDerived>::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- POS BareInput1 Classification F2I No Merge Model -----------------\n"
        << "vocabulary size : " << this->word_dict_size << " with dimension : " << this->word_embedding_dim << "\n"
        << "birnn x dim : " << this->rnn_x_dim << " , h dim : " << this->rnn_h_dim
        << " , stacked layer num : " << this->nr_rnn_stacked_layer << "\n"
        << "softmax layer input dim : " << this->softmax_input_dim << "\n"
        << "output dim : " << this->output_dim << "\n"
        << "feature info : \n"
        << this->pos_feature.get_feature_info() ;
}
} // end of namespace slnn 
#endif 
