#ifndef SLNN_MODULE_NN_NN_INTERFACE_H_
#define SLNN_MODULE_NN_NN_INTERFACE_H_
#include <vector>
#include "utils/typedeclaration.h"
namespace slnn{
namespace module{
namespace nn{

namespace nn_framework{
    // using namespace instead of Enum for extend by user(Although no other users , 2333);
    using NnFrameworkTagT = int;
    constexpr NnFrameworkTagT NN_DyNet = 0; 

} // end of namespace nn-framework

/**
* neural network common interface for training and predict. 
* not for polymorphism but for build interface for common NN operation and different NN framework.
*/
template <nn_framework::NnFrameworkTagT nn_tag, typename NnExprType, typename NnValueType>
class NeuralNetworkCommonInterface
{
public:
    // Type
    using NnExprT = NnExprType;
    using NnValueT = NnValueType;
    // training
    void set_update_method(const std::string &optmization_name);
    void update(slnn::type::real scale);
    void update_epoch();
    const NnValueT& forward(const NnExprType&);
    slnn::type::real as_scalar(const NnValueT&);
    std::vector<slnn::type::real> as_vector(const NnValueT&);
    void backward(const NnExprT&);
    // stash model
    void stash_model();
    bool stash_model_when_best(slnn::type::real current_score);
    bool reset2stashed_model();
};

} // end of namespace nn
} // end of namespace module
} // end of namespace slnn

#endif