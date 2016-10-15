#ifndef SLNN_SEGMENTOR_CWS_MODULE_NN_MODULE_NN_COMMON_INTERFACE_CNN_IMPL_H_
#define SLNN_SEGMENTOR_CWS_MODULE_NN_MODULE_NN_COMMON_INTERFACE_CNN_IMPL_H_
#include <sstream>
#include <boost/log/trivial.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/access.hpp>
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "nn_common_interface.h"
namespace slnn{
namespace segmentor{
namespace nn_module{

class NeuralNetworkCommonInterfaceCnnImpl : public NeuralNetworkCommonInterface
{
    friend class boost::serialization::access;
public:
    NeuralNetworkCommonInterfaceCnnImpl(int argc, char**argv, unsigned seed);
    ~NeuralNetworkCommonInterfaceCnnImpl();
    NeuralNetworkCommonInterfaceCnnImpl(const NeuralNetworkCommonInterfaceCnnImpl &) = delete;
    NeuralNetworkCommonInterfaceCnnImpl& operator=(const NeuralNetworkCommonInterfaceCnnImpl &) = delete;
    // training
    virtual void set_update_method(const std::string &optmization_name) override;
    virtual void update(slnn::type::real scale) override;
    virtual void update_epoch() override;
    virtual void forward() override;
    virtual slnn::type::real forward_as_scalar() override;
    virtual std::vector<slnn::type::real> forward_as_vector() override;
    virtual void backward() override;
    // stash model
    virtual void stash_model() override;
    virtual bool stash_model_when_best(slnn::type::real current_score) override;
    virtual bool reset2stashed_model() override;
public:
    void clear_cg(){ pcg->clear(); }; // ! BUG: using clear() will cause Error -> CNN get dim error!(BUG for it.)
    void reset_cg(){ delete pcg; pcg = new cnn::ComputationGraph(); }
protected:
    cnn::ComputationGraph* get_cg(){ return pcg; }
    cnn::Model* get_cnn_model(){ return cnn_model; }
private:
    template <typename Archive>
    void serialize(Archive &ar, const unsigned version);
private:
    slnn::type::real best_score;
    std::stringstream best_model_tmp_ss;
    cnn::Trainer *trainer;
    cnn::ComputationGraph *pcg;
    cnn::Model *cnn_model;
    unsigned cnn_rng_seed;
};

/*********************************************
 * Inline Implementation
 *********************************************/

inline 
NeuralNetworkCommonInterfaceCnnImpl::NeuralNetworkCommonInterfaceCnnImpl(int argc, char **argv, unsigned seed)
    :best_score(0.f),
    best_model_tmp_ss(""),
    trainer(nullptr),
    pcg(new cnn::ComputationGraph()),
    cnn_model(new cnn::Model())
{
    cnn::Initialize(argc, argv, seed); 
}

inline
NeuralNetworkCommonInterfaceCnnImpl::~NeuralNetworkCommonInterfaceCnnImpl()
{
    delete trainer;
    delete pcg;
    delete cnn_model;
}

inline
void NeuralNetworkCommonInterfaceCnnImpl::update(slnn::type::real scale)
{
    trainer->update(scale);
}

inline
void NeuralNetworkCommonInterfaceCnnImpl::update_epoch()
{
    trainer->update_epoch();
}

inline
void NeuralNetworkCommonInterfaceCnnImpl::forward()
{
    pcg->forward();
}

inline
slnn::type::real NeuralNetworkCommonInterfaceCnnImpl::forward_as_scalar()
{
    return cnn::as_scalar(pcg->forward());
}

inline
std::vector<slnn::type::real> NeuralNetworkCommonInterfaceCnnImpl::forward_as_vector()
{
    return cnn::as_vector(pcg->forward());
}

inline
void NeuralNetworkCommonInterfaceCnnImpl::backward()
{
    pcg->backward();
}

inline 
void NeuralNetworkCommonInterfaceCnnImpl::stash_model()
{
    best_model_tmp_ss.str(""); // first , clear it's content !
    boost::archive::text_oarchive to(best_model_tmp_ss); // to construct a text_oarchive using stringstream
    to << *cnn_model;
}

inline 
bool NeuralNetworkCommonInterfaceCnnImpl::stash_model_when_best(slnn::type::real current_score)
{
    if( current_score > best_score )
    {
        best_score = current_score;
        stash_model();
        return true;
    }
    else { return false; }
}

inline 
bool NeuralNetworkCommonInterfaceCnnImpl::reset2stashed_model()
{
    if( best_model_tmp_ss.rdbuf()->in_avail() != 0 )
    {
        boost::archive::text_iarchive ti(best_model_tmp_ss);
        ti >> *cnn_model;
        return true;
    }
    else { return false; }
}

template <typename Archive>
inline
void NeuralNetworkCommonInterfaceCnnImpl::serialize(Archive &ar, const unsigned version)
{
    ar & *cnn_model;
}

} // end of namespace nn-module
} // end of namespace segmentor
} // end of namespace slnn
#endif
