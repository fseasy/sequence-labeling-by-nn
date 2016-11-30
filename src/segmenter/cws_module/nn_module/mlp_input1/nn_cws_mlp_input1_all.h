#ifndef SLNN_SEGMENTER_CWS_MODULE_NN_MODULE_MLP_INPUT1_ALL_H_
#define SLNN_SEGMNETER_CWS_MODULE_NN_MODULE_MLP_INPUT1_ALL_H_
#include <vector>
#include "dynet/expr.h"
#include "segmenter/cws_module/nn_module/nn_common_interface_dynet_impl.h"
#include "segmenter/cws_module/cws_output_layer.h"
#include "segmenter/cws_module/nn_module/experiment_layer/nn_window_expr_processing_layer.h"
#include "segmenter/cws_module/nn_module/experiment_layer/nn_cws_specific_output_layer.h"
#include "utils/typedeclaration.h"
namespace slnn{
namespace segmenter{
namespace nn_module{

class NnSegmenterMlpInput1All : public NeuralNetworkCommonInterfaceDynetImpl
{
public:
    NnSegmenterMlpInput1All(int argc, char **argv, unsigned seed);
    template <typename StructureParamT>
    void build_model_structure(const StructureParamT &param);
    template <typename AnnotatedDataProcessedT>
    dynet::expr::Expression build_training_graph(const AnnotatedDataProcessedT &ann_processed_data);
    template <typename UnannotatedDataProcessedT>
    std::vector<Index> predict(const UnannotatedDataProcessedT &unann_processed_data);
protected:
    dynet::expr::Expression build_training_graph_impl(const std::shared_ptr<std::vector<Index>>& punigram_seq,
        const std::shared_ptr<std::vector<Index>>& pbigram_seq,
        const std::shared_ptr<std::vector<std::vector<Index>>>& plexicon_seq, 
        const std::shared_ptr<std::vector<Index>>& ptype_seq, 
        const std::shared_ptr<std::vector<Index>>& ptag_seq);
    std::vector<Index> predict_impl(const std::shared_ptr<std::vector<Index>>& punigram_seq,
        const std::shared_ptr<std::vector<Index>>& pbigram_seq,
        const std::shared_ptr<std::vector<std::vector<Index>>>& plexicon_seq, 
        const std::shared_ptr<std::vector<Index>>& ptype_seq);
private:
    void new_graph();
    std::vector<dynet::expr::Expression> concat_all_feature_as_expr(unsigned seq_len,
        const std::shared_ptr<std::vector<Index>>& punigram_seq,
        const std::shared_ptr<std::vector<Index>>& pbigram_seq,
        const std::shared_ptr<std::vector<std::vector<Index>>>& plexicon_seq,
        const std::shared_ptr<std::vector<Index>>& ptype_seq);
private:
    std::shared_ptr<Index2ExprLayer> unigram_embed_layer;
    std::shared_ptr<Index2ExprLayer> bigram_embed_layer;
    std::shared_ptr<std::vector<Index2ExprLayer>> lexicon_embed_layer_group; // lexicon feature group has 3 feature
    std::shared_ptr<Index2ExprLayer> type_embed_layer;

    std::shared_ptr<WindowExprGenerateLayer> window_expr_generate_layer;
    std::shared_ptr<experiment::WindowExprProcessingLayerAbstract> window_expr_processing_layer;

    std::shared_ptr<MLPHiddenLayer> mlp_hidden_layer;
    std::shared_ptr<BareOutputBase> output_layer;
};


/*********************
 * Inline/template Implementation
 *****************/

template<typename StructureParamT>
void NnSegmenterMlpInput1All::build_model_structure(const StructureParamT& param)
{
    // we concat all the features to one embedding.
    unsigned total_embed_dim = 0;
    if( param.enable_unigram )
    {
        unigram_embed_layer.reset(new Index2ExprLayer(this->get_dynet_model(), 
            param.unigram_dict_sz, param.unigram_embedding_dim));
        total_embed_dim += param.unigram_embedding_dim;
    }
    if( param.enable_bigram )
    {
        bigram_embed_layer.reset(new Index2ExprLayer(this->get_dynet_model(),
            param.bigram_dict_sz, param.bigram_embedding_dim));
        total_embed_dim += param.bigram_embedding_dim;
    }
    if( param.enable_lexicon )
    {
        lexicon_embed_layer_group.reset(new std::vector<Index2ExprLayer>(3, 
            Index2ExprLayer(this->get_dynet_model(), param.lexicon_dict_sz, param.lexicon_embedding_dim)));
        total_embed_dim += param.lexicon_embedding_dim * 3;
    }
    if( param.enable_type )
    {
        type_embed_layer.reset(new Index2ExprLayer(this->get_dynet_model(),
            param.type_dict_sz, param.type_embedding_dim));
        total_embed_dim += param.type_embedding_dim;
    }
    window_expr_generate_layer.reset(new WindowExprGenerateLayer(this->get_dynet_model(), param.window_sz,
        total_embed_dim));
    window_expr_processing_layer = experiment::create_window_expr_processing_layer(param.window_process_method, this->get_dynet_model(),
        total_embed_dim, param.window_sz);

    unsigned output_layer_input_dim = window_expr_processing_layer->get_output_dim();
    if( param.mlp_hidden_dim_list.size() > 0 )
    {
        mlp_hidden_layer.reset(new MLPHiddenLayer(this->get_dynet_model(), window_expr_processing_layer->get_output_dim(),
            param.mlp_hidden_dim_list, param.mlp_dropout_rate,
            utils::get_nonlinear_function_from_name(param.mlp_nonlinear_func_str)));
        output_layer_input_dim = mlp_hidden_layer->get_output_dim();
    }
    output_layer = experiment::create_segmenter_output_layer(param.output_layer_type, this->get_dynet_model(),
        output_layer_input_dim, param.tag_dict_sz);

}

template <typename AnnotatedDataProcessedT>
dynet::expr::Expression NnSegmenterMlpInput1All::build_training_graph(const AnnotatedDataProcessedT &ann_processed_data)
{
    return build_training_graph_impl(ann_processed_data.punigramseq, ann_processed_data.pbigramseq,
        ann_processed_data.plexiconseq, ann_processed_data.ptypeseq, ann_processed_data.ptagseq);
}


template <typename UnannotatedDataProcessedT>
std::vector<Index> NnSegmenterMlpInput1All::predict(const UnannotatedDataProcessedT &unann_processed_data)
{
    return predict_impl(unann_processed_data.punigramseq, unann_processed_data.pbigramseq,
        unann_processed_data.plexiconseq, unann_processed_data.ptypeseq);
}

} // end of namespace nn module
} // end of namespace segmenter
} // end of namespace slnn


#endif