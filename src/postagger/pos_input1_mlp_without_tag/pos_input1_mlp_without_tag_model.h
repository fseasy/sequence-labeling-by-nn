#ifndef POSTAGGER_POS_INPUT1_MLP_WITHOUT_TAG_POSTAG_INPUT1_MLP_WITHOUT_TAG_MODEL_H_
#define POSTAGGER_POS_INPUT1_MLP_WITHOUT_TAG_POSTAG_INPUT1_MLP_WITHOUT_TAG_MODEL_H_
#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include "postagger/base_model/input1_mlp_model.h"
#include "modelmodule/hyper_layers.h"
namespace slnn{

class Input1MLPWithoutTagModel : public Input1MLPModel
{
    friend class boost::serialization::access;
public :
    Input1MLPWithoutTagModel();
    ~Input1MLPWithoutTagModel();
    Input1MLPWithoutTagModel(const Input1MLPWithoutTagModel &) = delete ;
    Input1MLPWithoutTagModel &operator=(const Input1MLPWithoutTagModel &) = delete ;

    void set_model_param_from_outer(const boost::program_options::variables_map &var_map) override;
    void build_model_structure() override;
    void print_model_info() override;

    dynet::expr::Expression  build_loss(dynet::ComputationGraph &cg,
        const IndexSeq &input_seq, 
        const ContextFeatureDataSeq &context_feature_gp_seq,
        const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
        const IndexSeq &gold_seq)  override ;
    void predict(dynet::ComputationGraph &cg ,
        const IndexSeq &input_seq, 
        const ContextFeatureDataSeq &context_feature_gp_seq,
        const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
        IndexSeq &pred_seq) override ;

    template <typename Archive>
    void serialize(Archive &ar, unsigned version);

private :
    BareInput1 *input_layer;
    MLPHiddenLayer *mlp_hidden_layer;
    SoftmaxLayer *output_layer;
    POSFeatureLayer *pos_feature_layer;
    ContextFeatureLayer *pos_context_feature_layer;
};

template <typename Archive>
void Input1MLPWithoutTagModel::serialize(Archive &ar, unsigned version)
{
    ar & boost::serialization::base_object<Input1MLPModel>(*this);
}

}


#endif