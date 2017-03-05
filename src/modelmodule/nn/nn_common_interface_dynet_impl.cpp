#include <iostream>
#include "nn_common_interface_dynet_impl.h"

namespace slnn{
namespace module{
namespace nn{

void 
NeuralNetworkCommonInterface<nn_framework::NN_DyNet, dynet::expr::Expression, dynet::Tensor>::
set_update_method(const std::string &optmization_name)
{
    std::string opt_norm_name(optmization_name);
    for( char &c : opt_norm_name ){ c = ::tolower(c); }
    if( opt_norm_name == "sgd" )
    {
        trainer = new dynet::SimpleSGDTrainer(dynet_model);
    }
    else if( opt_norm_name == "adagrad" )
    {
        trainer = new dynet::AdagradTrainer(dynet_model);
    }
    else if( opt_norm_name == "momentum" )
    {
        trainer = new dynet::MomentumSGDTrainer(dynet_model);
    }
    else if( opt_norm_name == "adadelta" )
    {
        trainer = new dynet::AdadeltaTrainer(dynet_model);
    }
    else if( opt_norm_name == "rmsprop" )
    {
        trainer = new dynet::RmsPropTrainer(dynet_model);
    }
    else if(opt_norm_name == "adam" )
    {
        trainer = new dynet::AdamTrainer(dynet_model);
    }
    else
    {
        throw std::invalid_argument(std::string("un-supported optimization method : '") + optmization_name + std::string("'"));
    }
}

} // end of namespace nn
} // end of namespace module
} // end of namespace slnn