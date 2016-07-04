#ifndef SLNN_SEGMENTOR_CWS_SINGLE_INPUT_CLASSIFICATION_HPP_
#define SLNN_SEGMENTOR_CWS_SINGLE_INPUT_CLASSIFICATION_HPP_

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

#include "cnn/cnn.h"

#include "segmentor/base_model/single_input_model.h"
#include "segmentor/cws_module/cws_tagging_system.h"
namespace slnn{

class CWSSingleClassificationModel : public CWSInput1WithFeatureModel
{
    friend class boost::serialization::access;
public:
    
    cnn::Dict &word_dict ;
    cnn::Dict &tag_dict ;

    CWSSingleClassificationModel() ;
    ~CWSSingleClassificationModel() ;

    void set_model_param(const boost::program_options::variables_map &var_map) ;
    void build_model_structure() ;
    void print_model_info() ;

    void save_model(std::ostream &os) ;
    void load_model(std::istream &is) ;

    template<typename Archive>
    void serialize(Archive &ar, const unsigned version);
};

template<typename Archive>
void CWSSingleClassificationModel::serialize(Archive &ar, const unsigned version)
{
    ar & boost::serialization::base_object<CWSInput1WithFeatureModel>(*this);
}

} // end of namespace slnn 
#endif 
