#ifndef SLNN_SEGMENTER_CWS_MODULE_NN_MODULE_CWS_MLP_INPUT1_ABSTRACT_H_
#define SLNN_SEGMENTER_CWS_MODULE_NN_MODULE_CWS_MLP_INPUT1_ABSTRACT_H_
#include <functional>
#include "utils/typedeclaration.h"
#include "segmenter/cws_module/nn_module/nn_common_interface_dynet_impl.h"
#include "segmenter/cws_module/token_module/cws_tag_definition.h"
#include "segmenter/cws_module/cws_output_layer.h"
#include "segmenter/cws_module/nn_module/experiment_layer/nn_window_expr_processing_layer.h"
#include "segmenter/cws_module/nn_module/experiment_layer/nn_cws_specific_output_layer.h"
#include "utils/nn_utility.h"
namespace slnn{
namespace segmenter{
namespace nn_module{

class NnSegmenterInput1Abstract : public NeuralNetworkCommonInterfaceCnnImpl
{
public:
    NnSegmenterInput1Abstract(int argc, char **argv, unsigned seed);
    template <typename StructureParamT>
    void build_model_structure(const StructureParamT &param);
    template <typename AnnotatedDataProcessedT>
    dynet::expr::Expression build_training_graph(const AnnotatedDataProcessedT &ann_processed_data);
    template <typename UnannotatedDataProcessedT>
    std::vector<Index> predict(const UnannotatedDataProcessedT &unann_processed_data);
protected:
    dynet::expr::Expression build_training_graph_impl(const std::vector<Index> &charseq, const std::vector<Index> &tagseq);
    std::vector<Index> predict_impl(const std::vector<Index> &charseq);
protected:
    std::shared_ptr<Index2ExprLayer> word_expr_layer;
    std::shared_ptr<WindowExprGenerateLayer> window_expr_generate_layer;
    std::shared_ptr<experiment::WindowExprProcessingLayerAbstract> window_expr_processing_layer;
    std::shared_ptr<MLPHiddenLayer> mlp_hidden_layer;
    std::shared_ptr<BareOutputBase> output_layer;
};




/**************************************************
 * Inline / Template Implementation
 **************************************************/

template <typename StructureParamT>
void NnSegmenterInput1Abstract::build_model_structure(const StructureParamT &param)
{
    word_expr_layer.reset(new Index2ExprLayer(this->get_dynet_model(), param.corpus_token_dict_size,
        param.corpus_token_embedding_dim));
    window_expr_generate_layer.reset(new WindowExprGenerateLayer(this->get_dynet_model(), param.window_size,
        param.corpus_token_embedding_dim));
    window_expr_processing_layer = experiment::create_window_expr_processing_layer(param.window_process_method, this->get_dynet_model(),
        param.corpus_token_embedding_dim, param.window_size);
    mlp_hidden_layer.reset(new MLPHiddenLayer(this->get_dynet_model(), window_expr_processing_layer->get_output_dim(),
        param.mlp_hidden_dim_list, param.mlp_dropout_rate, 
        utils::get_nonlinear_function_from_name(param.mlp_nonlinear_function_str)));
    output_layer = experiment::create_segmenter_output_layer(param.output_layer_type, this->get_dynet_model(),
        mlp_hidden_layer->get_output_dim(), param.output_dim);
}

template <typename AnnotatedDataProcessedT>
inline
dynet::expr::Expression 
NnSegmenterInput1Abstract::build_training_graph(const AnnotatedDataProcessedT &ann_processed_data)
{
    return build_training_graph_impl(*ann_processed_data.pcharseq, *ann_processed_data.ptagseq);
}
template <typename UnannotatedDataProcessedT>
inline
std::vector<Index> 
NnSegmenterInput1Abstract::predict(const UnannotatedDataProcessedT &unann_processed_data)
{
    return predict_impl(unann_processed_data);  
}


} // end of namespace nn_module
} // end of namespace segmenter
} // end of namespace slnn




#endif