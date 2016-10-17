#ifndef SLNN_SEGMENTOR_CWS_MODULE_NN_MODULE_NN_INTERFACE_H_
#define SLNN_SEGMENTOR_CWS_MODULE_NN_MODULE_NN_INTERFACE_H_
#include <vector>
#include "utils/typedeclaration.h"
namespace slnn{
namespace segmenter{
namespace nn_module{
/**
 * neural network common interface for training and predict. 
 * not for polymorphism but for build interface for common NN operation and different NN framework.
 */
class NeuralNetworkCommonInterface
{
public:
    // training
    virtual void set_update_method(const std::string &optmization_name) = 0;
    virtual void update(slnn::type::real scale) = 0;
    virtual void update_epoch() = 0;
    virtual void forward() = 0;
    virtual slnn::type::real forward_as_scalar() = 0;
    virtual std::vector<slnn::type::real> forward_as_vector() = 0;
    virtual void backward() = 0;
    // stash model
    virtual void stash_model() = 0;
    virtual bool stash_model_when_best(slnn::type::real current_score) = 0;
    virtual bool reset2stashed_model() = 0;
    virtual ~NeuralNetworkCommonInterface() {};
};



} // end of namespace nn-module
} // end of namespace segmenter
} // end of namespace slnn

#endif