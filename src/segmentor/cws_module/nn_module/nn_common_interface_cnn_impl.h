#ifndef SLNN_SEGMENTOR_CWS_MODULE_NN_MODULE_NN_COMMON_INTERFACE_CNN_IMPL_H_
#define SLNN_SEGMENTOR_CWS_MODULE_NN_MODULE_NN_COMMON_INTERFACE_CNN_IMPL_H_
#include <sstream>
#include "cnn/cnn.h"
#include "nn_common_interface.h"
namespace slnn{
namespace segmentor{
namespace nn_module{

class NeuralNetworkCommonInterfaceCnnImpl : public NeuralNetworkCommonInterface
{
public:
    // training
    virtual void update(slnn::type::real scale) override;
    virtual void update_epoch() override;
    virtual void forward() override;
    virtual slnn::type::real forward_as_scalar() override;
    virtual std::vector<slnn::tpye::real> void forward_as_vector() override;
    virtual void backward() override;
    // stash model
    virtual void stash_model() override;
    virtual void stath_model_when_best(slnn::type::real current_score) override;
    virtual void reset2stashed_model() override;
private:
    slnn::type::real best_score;
    std::stringstream best_model_tmp_ss;
    cnn::Trainer *trainer;
    cnn::ComputationGraph *pcg;
};


} // end of namespace nn-module
} // end of namespace segmentor
} // end of namespace slnn



#endif