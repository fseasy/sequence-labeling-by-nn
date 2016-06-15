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
public :
    Input1MLPWithoutTagModel();
    ~Input1MLPWithoutTagModel();
    Input1MLPWithoutTagModel(const Input1MLPWithoutTagModel &) = delete ;
    Input1MLPWithoutTagModel &operator=(const Input1MLPWithoutTagModel &) = delete ;

    void set_model_param(const boost::program_options::variables_map &var_map) override;
    void build_model_structure() override;
    void print_model_info() override;

    cnn::expr::Expression  build_loss(cnn::ComputationGraph &cg,
        const IndexSeq &input_seq, 
        const POSContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq,
        const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
        const IndexSeq &gold_seq)  override ;
    void predict(cnn::ComputationGraph &cg ,
        const IndexSeq &input_seq, 
        const POSContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq,
        const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
        IndexSeq &pred_seq) override ;
private :
    BareInput1 *input_layer;
    MLPHiddenLayer *mlp_hidden_layer;
    SoftmaxLayer *output_layer;
    POSFeatureLayer *pos_feature_layer;
    ContextFeatureLayer *pos_context_feature_layer;
};


}


#endif