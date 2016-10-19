#ifndef SLNN_SEGMENTER_CWS_DOUBLE_CRF_MODEL_H_
#define SLNN_SEGMENTER_CWS_DOUBLE_CRF_MODEL_H_

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "dynet/dynet.h"

#include "segmenter/base_model/input2_model.h"
#include "segmenter/cws_module/cws_tagging_system.h"
namespace slnn{

class CWSDoubleCRFModel : public Input2Model
{
    friend class boost::serialization::access;
public:

    CWSDoubleCRFModel() ;
    ~CWSDoubleCRFModel() ;

    void set_model_param(const boost::program_options::variables_map &var_map) ;
    void build_model_structure() ;
    void print_model_info() ;
    
    template <typename Archive>
    void save(Archive &ar, const unsigned version) const ;
    template <typename Archive>
    void load(Archive &ar, const unsigned verison) ;
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    unsigned tag_dim;
};

/*************  Template Implementation *****************/

template <typename Archive>
void CWSDoubleCRFModel::save(Archive &ar, const unsigned version) const
{
    ar & tag_dim;
    ar & boost::serialization::base_object<Input2Model>(*this) ;
}

template <typename Archive>
void CWSDoubleCRFModel::load(Archive &ar, const unsigned version)
{
    ar & tag_dim ;
    ar & boost::serialization::base_object<Input2Model>(*this) ;
}
} // end of namespace slnn 
#endif 
