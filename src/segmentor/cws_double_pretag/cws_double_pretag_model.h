#ifndef SLNN_SEGMENTOR_CWS_DOUBLE_PRETAG_MODEL_H_
#define SLNN_SEGMENTOR_CWS_DOUBLE_PRETAG_MODEL_H_

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

class CWSDoublePretagModel : public Input2Model
{
    friend class boost::serialization::access;
public:

    CWSDoublePretagModel() ;
    ~CWSDoublePretagModel() ;

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
void CWSDoublePretagModel::save(Archive &ar, const unsigned version) const
{
    ar & tag_dim;
    boost::serialization::base_object<Input2Model>(*this) ;
}

template <typename Archive>
void CWSDoublePretagModel::load(Archive &ar, const unsigned version)
{
    ar & tag_dim ;
    boost::serialization::base_object<Input2Model>(*this) ;
}
} // end of namespace slnn 
#endif 
