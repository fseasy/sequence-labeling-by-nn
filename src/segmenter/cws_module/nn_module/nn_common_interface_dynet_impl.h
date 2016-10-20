#ifndef SLNN_SEGMENTER_CWS_MODULE_NN_MODULE_NN_COMMON_INTERFACE_CNN_IMPL_H_
#define SLNN_SEGMENTER_CWS_MODULE_NN_MODULE_NN_COMMON_INTERFACE_CNN_IMPL_H_
#include <sstream>
#include <boost/log/trivial.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/access.hpp>
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "nn_common_interface.h"
namespace slnn{
namespace segmenter{
namespace nn_module{

template <>
class NeuralNetworkCommonInterface<nn_framework::NN_DyNet, dynet::expr::Expression, dynet::Tensor>
{
    friend class boost::serialization::access;
public:
    using NnExprT = dynet::expr::Expression;
    using NnValueT = dynet::Tensor;
public:
    NeuralNetworkCommonInterface(int argc, char**argv, unsigned seed);
    ~NeuralNetworkCommonInterface();
    NeuralNetworkCommonInterface(const NeuralNetworkCommonInterface &) = delete; // here param is the tempalte instance, not template
    NeuralNetworkCommonInterface& operator=(const NeuralNetworkCommonInterface &) = delete;
    // training
    void set_update_method(const std::string &optmization_name);
    void update(slnn::type::real scale);
    void update_epoch();
    const NnValueT& forward(const NnExprT&);
    slnn::type::real as_scalar(const NnValueT&);
    std::vector<slnn::type::real> as_vector(const NnValueT&);
    void backward(const NnExprT&);
    // stash model
    void stash_model();
    bool stash_model_when_best(slnn::type::real current_score);
    bool reset2stashed_model();
public:
    void clear_cg(){ pcg->clear(); }; // ! BUG: using clear() will cause Error -> DyNet get dim error!(BUG for it.)
    void reset_cg(){ delete pcg; pcg = new dynet::ComputationGraph(); }
protected:
    dynet::ComputationGraph* get_cg(){ return pcg; }
    dynet::Model* get_dynet_model(){ return dynet_model; }
private:
    template <typename Archive>
    void serialize(Archive &ar, const unsigned version);
private:
    slnn::type::real best_score;
    std::stringstream best_model_tmp_ss;
    dynet::Trainer *trainer;
    dynet::ComputationGraph *pcg;
    dynet::Model *dynet_model;
    unsigned dynet_rng_seed;
};

using NeuralNetworkCommonInterfaceCnnImpl = NeuralNetworkCommonInterface<nn_framework::NN_DyNet, 
    dynet::expr::Expression, dynet::Tensor>;

/*********************************************
 * Inline Implementation
 *********************************************/

inline 
NeuralNetworkCommonInterface<nn_framework::NN_DyNet, dynet::expr::Expression, dynet::Tensor>::
NeuralNetworkCommonInterface(int argc, char **argv, unsigned seed)
    :best_score(0.f),
    best_model_tmp_ss(""),
    trainer(nullptr),
    pcg(new dynet::ComputationGraph()),
    dynet_model(new dynet::Model())
{
    dynet::initialize(argc, argv, seed); 
}

inline
NeuralNetworkCommonInterface<nn_framework::NN_DyNet, dynet::expr::Expression, dynet::Tensor>::
~NeuralNetworkCommonInterface()
{
    delete trainer;
    delete pcg;
    delete dynet_model;
}

inline
void 
NeuralNetworkCommonInterface<nn_framework::NN_DyNet, dynet::expr::Expression, dynet::Tensor>::
update(slnn::type::real scale)
{
    trainer->update(scale);
}

inline
void 
NeuralNetworkCommonInterface<nn_framework::NN_DyNet, dynet::expr::Expression, dynet::Tensor>::
update_epoch()
{
    trainer->update_epoch();
}

inline
const NeuralNetworkCommonInterface<nn_framework::NN_DyNet, dynet::expr::Expression, dynet::Tensor>::NnValueT&
NeuralNetworkCommonInterface<nn_framework::NN_DyNet, dynet::expr::Expression, dynet::Tensor>::
forward(const NnExprT& expr)
{
    return pcg->incremental_forward(expr);
}

inline
slnn::type::real 
NeuralNetworkCommonInterface<nn_framework::NN_DyNet, dynet::expr::Expression, dynet::Tensor>::
as_scalar(const NnValueT& tensor_val)
{
    return dynet::as_scalar(tensor_val);
}

inline
std::vector<slnn::type::real> 
NeuralNetworkCommonInterface<nn_framework::NN_DyNet, dynet::expr::Expression, dynet::Tensor>::
as_vector(const NnValueT& tensor_val)
{
    return dynet::as_vector(tensor_val);
}

inline
void 
NeuralNetworkCommonInterface<nn_framework::NN_DyNet, dynet::expr::Expression, dynet::Tensor>::
backward(const NnExprT& expr)
{
    pcg->backward(expr);
}

inline 
void 
NeuralNetworkCommonInterface<nn_framework::NN_DyNet, dynet::expr::Expression, dynet::Tensor>::
stash_model()
{
    best_model_tmp_ss.str(""); // first , clear it's content !
    boost::archive::text_oarchive to(best_model_tmp_ss); // to construct a text_oarchive using stringstream
    to << *dynet_model;
}

inline 
bool 
NeuralNetworkCommonInterface<nn_framework::NN_DyNet, dynet::expr::Expression, dynet::Tensor>::
stash_model_when_best(slnn::type::real current_score)
{
    if( current_score > best_score )
    {
        best_score = current_score;
        stash_model();
        std::cerr << " * better model found and stashed done.\n";
        return true;
    }
    else { return false; }
}

inline 
bool 
NeuralNetworkCommonInterface<nn_framework::NN_DyNet, dynet::expr::Expression, dynet::Tensor>::
reset2stashed_model()
{
    if( best_model_tmp_ss.rdbuf()->in_avail() != 0 )
    {
        boost::archive::text_iarchive ti(best_model_tmp_ss);
        ti >> *dynet_model;
        return true;
    }
    else { return false; }
}

template <typename Archive>
inline
void
NeuralNetworkCommonInterface<nn_framework::NN_DyNet, dynet::expr::Expression, dynet::Tensor>::
serialize(Archive &ar, const unsigned version)
{
    ar & *dynet_model;
}

} // end of namespace nn-module
} // end of namespace segmenter
} // end of namespace slnn
#endif
