#ifndef SLNN_SEGMENTOR_CWS_SINGLE_INPUT_CLASSIFICATION_HPP_
#define SLNN_SEGMENTOR_CWS_SINGLE_INPUT_CLASSIFICATION_HPP_

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "cnn/cnn.h"

#include "segmentor/base_model/single_input_model.h"
#include "segmentor/cws_module/cws_tagging_system.h"
namespace slnn{

class CWSSingleClassificationModel : public SingleInputModel
{
public:
    unsigned word_embedding_dim,
        word_dict_size,
        lstm_nr_stacked_layer,
        lstm_h_dim,
        hidden_dim,
        output_dim ;

    cnn::real dropout_rate ; // only for bilstm (output doesn't enable dropout)

    cnn::Dict &word_dict ;
    cnn::Dict &tag_dict ;

    CWSSingleClassificationModel() ;
    ~CWSSingleClassificationModel() ;

    void set_model_param(const boost::program_options::variables_map &var_map) ;
    void build_model_structure() ;
    void print_model_info() ;

    void save_model(std::ostream &os) ;
    void load_model(std::istream &is) ;
};


} // end of namespace slnn 
#endif 
