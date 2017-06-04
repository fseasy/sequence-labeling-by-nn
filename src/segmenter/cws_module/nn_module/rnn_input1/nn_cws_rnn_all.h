#ifndef SLNN_SEGMENTER_CWS_MODULE_NN_MODULE_RNN_ALL_H_
#define SLNN_SEGMENTER_CWS_MODULE_NN_MODULE_RNN_ALL_H_

#include <vector>
#include "dynet/expr.h"
#include "segmenter/cws_module/nn_module/nn_common_interface_dynet_impl.h"
#include "segmenter/cws_module/cws_output_layer.h"
#include "segmenter/cws_module/nn_module/experiment_layer/nn_cws_specific_output_layer.h"
#include "utils/typedeclaration.h"
#include "utils/nn_utility.h"
namespace slnn{
namespace segmenter{
namespace nn_module{

class NnSegmenterRnnAll : public NeuralNetworkCommonInterfaceDynetImpl
{
public:
    NnSegmenterRnnAll(int argc, char **argv, unsigned seed);
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

    std::shared_ptr<BILSTMLayer> birnn_layer;
    //std::shared_ptr<MLPHiddenLayer> mlp_hidden_layer;
    std::shared_ptr<BareOutputBase> output_layer;
};


/*********************
 * Inline/template Implementation
 *****************/

template<typename StructureParamT>
void NnSegmenterRnnAll::build_model_structure(const StructureParamT& param)
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
    birnn_layer.reset(new BILSTMLayer(this->get_dynet_model(), param.rnn_nr_stack_layer, total_embed_dim,
        param.rnn_h_dim, param.rnn_dropout_rate));
    //mlp_hidden_layer.reset(new MLPHiddenLayer(this->get_dynet_model(), window_expr_processing_layer->get_output_dim(),
    //    param.mlp_hidden_dim_list, param.mlp_dropout_rate, 
    //    utils::get_nonlinear_function_from_name(param.mlp_nonlinear_function_str)));
    output_layer = experiment::create_segmenter_output_layer(param.output_layer_type, this->get_dynet_model(),
        param.rnn_h_dim * 2, param.output_dim); // concatenate hidden layer
}

template <typename AnnotatedDataProcessedT>
inline
dynet::expr::Expression NnSegmenterRnnAll::build_training_graph(const AnnotatedDataProcessedT &ann_processed_data)
{
    return build_training_graph_impl(ann_processed_data.punigramseq, ann_processed_data.pbigramseq,
        ann_processed_data.plexiconseq, ann_processed_data.ptypeseq, ann_processed_data.ptagseq);
}


template <typename UnannotatedDataProcessedT>
inline
std::vector<Index> NnSegmenterRnnAll::predict(const UnannotatedDataProcessedT &unann_processed_data)
{
    return predict_impl(unann_processed_data.punigramseq, unann_processed_data.pbigramseq,
        unann_processed_data.plexiconseq, unann_processed_data.ptypeseq);
}

} // end of namespace nn module
} // end of namespace segmenter
} // end of namespace slnn


#endif
