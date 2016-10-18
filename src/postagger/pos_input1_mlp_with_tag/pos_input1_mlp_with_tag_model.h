#ifndef POSTAGGER_POS_INPUT1_MLP_WITH_TAG_POSTAG_INPUT1_MLP_WITH_TAG_MODEL_H_
#define POSTAGGER_POS_INPUT1_MLP_WITH_TAG_POSTAG_INPUT1_MLP_WITH_TAG_MODEL_H_
#include <boost/log/trivial.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include "postagger/base_model/input1_mlp_model.h"
#include "modelmodule/hyper_layers.h"
namespace slnn{

class POSInput1MLPWithTagModel : public Input1MLPModel
{
    friend class boost::serialization::access;
public :
    POSInput1MLPWithTagModel();
    ~POSInput1MLPWithTagModel();
    POSInput1MLPWithTagModel(const POSInput1MLPWithTagModel &) = delete ;
    POSInput1MLPWithTagModel &operator=(const POSInput1MLPWithTagModel &) = delete ;

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
    Index2ExprLayer *word_expr_layer;
    ShiftedIndex2ExprLayer *tag_expr_layer;
    POSFeatureLayer *pos_feature_layer;
    ContextFeatureLayer *pos_context_feature_layer;
    MLPHiddenLayer *mlp_hidden_layer;
    SoftmaxLayer *output_layer;

    unsigned tag_embedding_dim;
    NonLinearFunc *nonlinear_func;
    std::string nonlinear_func_indicate;
};

template <typename Archive>
void POSInput1MLPWithTagModel::serialize(Archive &ar, unsigned version)
{
    ar & tag_embedding_dim;
    ar & nonlinear_func_indicate;
    ar & boost::serialization::base_object<Input1MLPModel>(*this);
}

}


#endif