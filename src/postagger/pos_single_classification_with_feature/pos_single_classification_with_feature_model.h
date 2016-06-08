#ifndef SLNN_POSTAGGER_POS_SINGLE_INPUT_CLASSIFICATION_WITH_FEATURE_MODEL_H_
#define SLNN_POSTAGGER_POS_SINGLE_INPUT_CLASSIFICATION_WITH_FEATURE_MODEL_H_

#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "cnn/cnn.h"

#include "postagger/base_model/single_input_with_feature_model.hpp"
namespace slnn{
template<typename RNNDerived>
class POSSingleClassificationWithFeatureModel : public SingleInputModelWithFeature<RNNDerived>
{
    friend class boost::serialization::access;
public:
   
    POSSingleClassificationWithFeatureModel() ;
    ~POSSingleClassificationWithFeatureModel() ;

    void set_model_param(const boost::program_options::variables_map &var_map) ;
    void build_model_structure() ;
    void print_model_info() ;
};


} // end of namespace slnn 
#endif 
