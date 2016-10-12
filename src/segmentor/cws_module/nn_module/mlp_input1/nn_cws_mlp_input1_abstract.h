#ifndef SLNN_SEGMENTOR_CWS_MODULE_NN_MODULE_CWS_MLP_INPUT1_ABSTRACT_H_
#define SLNN_SEGMENTOR_CWS_MODULE_NN_MODULE_CWS_MLP_INPUT1_ABSTRACT_H_
#include <functional>
#include "utils/typedeclaration.h"
#include "segmentor/cws_module/nn_module/nn_common_interface_cnn_impl.h"
#include "segmentor/cws_module/cws_feature_layer.h"
namespace slnn{
namespace segmentor{
namespace nn_module{

class NnSegmentorInput1Abstract : public NeuralNetworkCommonInterfaceCnnImpl
{
public:
    NnSegmentorInput1Abstract(int argc, char **argv, unsigned seed);
    template <typename AnnotatedDataProcessedT>
    cnn::expr::Expression build_training_graph(const AnnotatedDataProcessedT &ann_processed_data);
    template <typename UnannotatedDataProcessedT>
    std::vector<Index> predict(const UnannotatedDataProcessedT &unann_processed_data);
protected:
    cnn::expr::Expression build_training_graph(const std::vector<Index> &charseq, const std::vector<Index> &tagseq);
    std::vector<Index> predict(const std::vector<Index> &charseq);
protected:
    std::shared_ptr<Index2ExprLayer> word_expr_layer;
    std::shared_ptr<WindowExprGenerateLayer> window_expr_generate_layer;
    std::shared_ptr<MLPHiddenLayer> mlp_hidden_layer;
    std::shared_ptr<BareOutputBase> output_layer;
};

std::function<cnn::expr::Expression(const cnn::expr::Expression &)> 
get_nonlinear_function_from_name(const std::string &name);


/**************************************************
 * Inline Implementation
 **************************************************/

template <typename AnnotatedDataProcessedT>
inline
cnn::expr::Expression 
NnSegmentorInput1Abstract::build_training_graph(const AnnotatedDataProcessedT &ann_processed_data)
{
    build_training_graph(*ann_processed_data.pcharseq, *ann_processed_data.ptagseq);
}
template <typename UnannotatedDataProcessedT>
inline
std::vector<Index> 
NnSegmentorInput1Abstract::predict(const UnannotatedDataProcessedT &unann_processed_data)
{
    // here Template type equals to non-template function's type. According to function-call match rule,
    // this template function should never be called. we write here for specification?
    predict(unann_processed_data);  
}


} // end of namespace nn_module
} // end of namespace segmentor
} // end of namespace slnn




#endif