#ifndef SLNN_NER_NER_SINGLE_INPUT_CLASSIFICATION_H_
#define SLNN_NER_NER_SINGLE_INPUT_CLASSIFICATION_H_

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "dynet/dynet.h"

#include "ner/base_model/input2D_model.h"
namespace slnn{

class NERSingleClassificationModel : public Input2DModel
{
public:
    unsigned word_embedding_dim,
        word_dict_size,
        postag_embedding_dim,
        postag_dict_size,
        lstm_nr_stacked_layer,
        lstm_x_dim,
        lstm_h_dim,
        hidden_dim,
        output_dim ;

    dynet::real dropout_rate ; // only for bilstm (output doesn't enable dropout)

    NERSingleClassificationModel() ;
    ~NERSingleClassificationModel() ;

    void set_model_param(const boost::program_options::variables_map &var_map) ;
    void build_model_structure() ;
    void print_model_info() ;

    void save_model(std::ostream &os) ;
    void load_model(std::istream &is) ;
};


} // end of namespace slnn 
#endif 
