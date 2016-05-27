#ifndef SLNN_SEGMENTOR_CWS_DOUBLE_CLASSIFICATION_MODEL_H_
#define SLNN_SEGMENTOR_CWS_DOUBLE_CLASSIFICATION_MODEL_H_

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "cnn/cnn.h"

#include "segmentor/base_model/input2_model.h"
#include "segmentor/cws_module/cws_tagging_system.h"
namespace slnn{

class CWSDoubleClassificationModel : public Input2Model
{
    friend class boost::serialization::access;
public:
    unsigned dynamic_word_dim,
        fixed_word_dim,
        dynamic_dict_size,
        fixed_dict_size,
        lstm_nr_stacked_layer,
        lstm_x_dim,
        lstm_h_dim,
        hidden_dim,
        output_dim ;

    cnn::real dropout_rate ; // only for bilstm (output doesn't enable dropout)


    CWSDoubleClassificationModel() ;
    ~CWSDoubleClassificationModel() ;

    void set_model_param(const boost::program_options::variables_map &var_map) ;
    void build_model_structure() ;
    void print_model_info() ;
    
    template <typename Archive>
    void save(Archive &ar, const unsigned version) const ;
    template <typename Archive>
    void load(Archive &ar, const unsigned verison) ;
    BOOST_SERIALIZATION_SPLIT_MEMBER() ;
};


} // end of namespace slnn 
#endif 
