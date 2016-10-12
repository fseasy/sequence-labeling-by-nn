#include <iostream>
#include "nn_common_interface_cnn_impl.h"

namespace slnn{
namespace segmentor{
namespace nn_module{

void NeuralNetworkCommonInterfaceCnnImpl::set_update_method(const std::string &optmization_name)
{
    std::string opt_norm_name(optmization_name);
    for( char &c : opt_norm_name ){ c = ::tolower(c); }
    if( opt_norm_name == "sgd" )
    {
        trainer = new cnn::SimpleSGDTrainer(cnn_model);
    }
    else if( opt_norm_name == "adagrad" )
    {
        trainer = new cnn::AdagradTrainer(cnn_model);
    }
    else
    {
        throw std::invalid_argument(std::string("un-supported optimization method : '") + optmization_name + std::string("'"));
    }
}

} // end of namespace nn-module
} // end of namespace segmentor
} // end of namespace slnn