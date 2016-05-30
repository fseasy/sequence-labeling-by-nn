#ifndef SLNN_SEGMENTOR_CWS_SINGLE_INPUT_PRETAG_H_
#define SLNN_SEGMENTOR_CWS_SINGLE_INPUT_PRETAG_H_

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "cnn/cnn.h"

#include "segmentor/base_model/single_input_model.h"
#include "segmentor/cws_module/cws_tagging_system.h"
namespace slnn{

class CWSSinglePretagModel : public SingleInputModel
{
    friend class boost::serialization::access;
public:
    unsigned tag_embedding_dim ;
    
    cnn::Dict &word_dict ;
    cnn::Dict &tag_dict ;

    CWSSinglePretagModel() ;
    ~CWSSinglePretagModel() ;

    void set_model_param(const boost::program_options::variables_map &var_map) ;
    void build_model_structure() ;
    void print_model_info() ;

    template<typename Archive>
    void serialize(Archive &ar, const unsigned version);
};

template<typename Archive>
void CWSSinglePretagModel::serialize(Archive &ar, const unsigned version)
{
    ar & tag_embedding_dim;
    ar & boost::serialization::base_object<SingleInputModel>(*this);
}

} // end of namespace slnn 
#endif 
