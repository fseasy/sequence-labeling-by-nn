#ifndef SLNN_NER_MLP_NN_INCLUDE_
#define SLNN_NER_MLP_NN_INCLUDE_

#include <random>
#include <functional>

#include "structure_param_module.h"
#include "token_module.h"
#include "modelmodule/nn/nn_common_interface_dynet_impl.h"
#include "modelmodule/nn/experiment_layer/nn_window_expr_processing_layer.h"
#include "modelmodule/hyper_layers.h"
#include "utils/typedeclaration.h"
#include "utils/nn_utility.h"
namespace slnn{
namespace ner{
namespace nn{

class NnMlp : public slnn::module::nn::NeuralNetworkCommonInterfaceDynetImpl
{
public:
    NnMlp(int argc, char **argv, unsigned seed, 
        std::shared_ptr<structure_param::StructureParam> struct_param,
        std::shared_ptr<token_module::WordFeatInfo> word_feat_info=nullptr);
    void build_model_structure();
    template <typename InstanceFeatT, typename TagIndexSeq>
    dynet::expr::Expression build_training_graph(const InstanceFeatT &feat, const TagIndexSeq&);
    template <typename InstanceFeatT>
    std::vector<Index> predict(const InstanceFeatT &unann_processed_data);
    unsigned get_rng_seed() const { return rng_seed; }
    std::mt19937* get_mt19937_rng(){ return &rng; }
    using slnn::module::nn::NeuralNetworkCommonInterfaceDynetImpl::get_dynet_model;
protected:
    dynet::expr::Expression build_training_graph_impl(const std::shared_ptr<std::vector<Index>>& pword_seq,
        const std::shared_ptr<std::vector<Index>>& ppos_tag_seq,
        const std::vector<Index>& ner_tag_seq);
    std::vector<Index> predict_impl(const std::shared_ptr<std::vector<Index>>& pword_seq,
        const std::shared_ptr<std::vector<Index>>& ppos_tag_seq);
private:
    void new_graph();
    std::vector<dynet::expr::Expression> concat_all_feature_as_expr(unsigned seq_len,
        const std::shared_ptr<std::vector<Index>>& pword_seq,
        const std::shared_ptr<std::vector<Index>>& ppos_tag_seq);
    std::shared_ptr<std::vector<Index>> unk_replace(std::shared_ptr<std::vector<Index>> pword_seq);
private:
    std::shared_ptr<structure_param::StructureParam> struct_param;
    std::shared_ptr<token_module::WordFeatInfo> word_feat_info;
    std::mt19937 rng;
    unsigned rng_seed;

    std::shared_ptr<Index2ExprLayer> word_embed_layer;
    std::shared_ptr<Index2ExprLayer> pos_tag_embed_layer;

    std::shared_ptr<WindowExprGenerateLayer> window_expr_generate_layer;
    std::shared_ptr<slnn::module::nn::experiment::WindowExprProcessingLayerAbstract> window_expr_processing_layer;

    std::shared_ptr<MLPHiddenLayer> mlp_hidden_layer;
    std::shared_ptr<BareOutputBase> output_layer;
};


/*********************
* Inline/template Implementation
*****************/

void NnMlp::build_model_structure()
{
    // we concat all the features to one embedding.
    word_embed_layer.reset(new Index2ExprLayer(this->get_dynet_model(),
        struct_param->word_dict_sz, struct_param->word_embed_dim));
    pos_tag_embed_layer.reset(new Index2ExprLayer(this->get_dynet_model(),
        struct_param->pos_tag_dict_sz, struct_param->pos_tag_embed_dim));
    unsigned total_embed_dim = struct_param->word_embed_dim + struct_param->pos_tag_embed_dim;
    window_expr_generate_layer.reset(new WindowExprGenerateLayer(this->get_dynet_model(), struct_param->window_sz,
        total_embed_dim));
    window_expr_processing_layer = slnn::module::nn::experiment::create_window_expr_processing_layer(
        struct_param->window_process_method, this->get_dynet_model(),
        total_embed_dim, struct_param->window_sz);

    unsigned output_layer_input_dim = window_expr_processing_layer->get_output_dim();
    if( struct_param->mlp_hidden_dim_list.size() > 0 )
    {
        mlp_hidden_layer.reset(new MLPHiddenLayer(this->get_dynet_model(), window_expr_processing_layer->get_output_dim(),
            struct_param->mlp_hidden_dim_list, struct_param->mlp_dropout_rate,
            utils::get_nonlinear_function_from_name(struct_param->mlp_nonlinear_func_str)));
        output_layer_input_dim = mlp_hidden_layer->get_output_dim();
    }
    output_layer = create_output_layer(struct_param->output_layer_type, this->get_dynet_model(),
        output_layer_input_dim, struct_param->ner_tag_dict_sz);
}

template <typename InstanceFeatT, typename TagIndexSeq>
dynet::expr::Expression NnMlp::build_training_graph(const InstanceFeatT &feat, const TagIndexSeq& ner_tag_index_seq)
{
    return build_training_graph_impl(feat.word_seq, feat.pos_tag_seq, ner_tag_index_seq);
}


template <typename InstanceFeatT>
std::vector<Index> NnMlp::predict(const InstanceFeatT &feat)
{
    return predict_impl(feat.word_seq, feat.pos_tag_seq);
}



} // end of namespace nn
} // end of namespace ner
} // end of nemespace slnn
#endif