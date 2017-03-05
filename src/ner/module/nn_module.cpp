#include "nn_module.h"
namespace slnn{
namespace ner{
namespace nn{

NnMlp::NnMlp(int argc, char **argv, unsigned seed,
    std::shared_ptr<structure_param::StructureParam> struct_param,
    std::shared_ptr<token_module::WordFeatInfo> word_feat_info)
    :slnn::module::nn::NeuralNetworkCommonInterfaceDynetImpl(argc, argv, seed),
    struct_param(struct_param),
    word_feat_info(word_feat_info),
    rng(seed))
{
}

void NnMlp::new_graph()
{
    reset_cg();
    word_embed_layer->new_graph(*get_cg());
    pos_tag_embed_layer->new_graph(*get_cg());
    window_expr_generate_layer->new_graph(*get_cg());
    window_expr_processing_layer->new_graph(*get_cg());
    if( mlp_hidden_layer ){ mlp_hidden_layer->new_graph(*get_cg()); }
    output_layer->new_graph(*get_cg());
}

std::vector<dynet::expr::Expression>
NnMlp::concat_all_feature_as_expr(
    unsigned seq_len,
    const std::shared_ptr<std::vector<Index>>& pword_seq,
    const std::shared_ptr<std::vector<Index>>& ppos_tag_seq)
{
    std::vector<std::vector<dynet::expr::Expression>> all_feature_list(seq_len);
    auto append_feature = [&all_feature_list, seq_len](const std::vector<dynet::expr::Expression>& feature_list)
    {
        for( unsigned i = 0; i < seq_len; ++i )
        {
            all_feature_list[i].push_back(feature_list[i]);
        }
    };

    std::vector<dynet::expr::Expression> word_feature_list;
    word_embed_layer->index_seq2expr_seq(*pword_seq, word_feature_list);
    append_feature(word_feature_list);

    std::vector<dynet::expr::Expression> pos_tag_feature_list;
    pos_tag_embed_layer->index_seq2expr_seq(*ppos_tag_seq, pos_tag_feature_list);
    append_feature(pos_tag_feature_list);
  
    std::vector<dynet::expr::Expression> all_feature_concat_list(seq_len);
    for( unsigned i = 0; i < seq_len; ++i )
    {
        all_feature_concat_list[i] = dynet::expr::concatenate(all_feature_list[i]);
    }
    return all_feature_concat_list;
}

std::shared_ptr<std::vector<Index>>
NnMlp::unk_replace(std::shared_ptr<std::vector<Index>> pword_seq)
{
    // copy the orign data.
    std::shared_ptr<std::vector<Index>> replaced_seq = std::make_shared<std::vector<Index>>(*pword_seq);
    // replace
    for( Index &word_idx : *replaced_seq )
    {
        std::size_t cnt = word_feat_info->count(word_idx);
        if( cnt <= struct_param->replace_freq_threshold &&
            std::uniform_real_distribution<float>(0, 1)(rng) <= struct_param->replace_prob_threshold )
        {
            word_idx = word_feat_info->get_unk_index();
        }
    }
    return replaced_seq;
}

dynet::expr::Expression 
NnMlp::build_training_graph_impl(const std::shared_ptr<std::vector<Index>>& pword_seq,
    const std::shared_ptr<std::vector<Index>>& ppos_tag_seq,
    const std::vector<Index>& ner_tag_seq)
{
    new_graph();
    if( mlp_hidden_layer ){ mlp_hidden_layer->enable_dropout(); }

    unsigned seq_len = ppos_tag_seq->size();
    std::vector<dynet::expr::Expression> all_feature_concat_expr_list = concat_all_feature_as_expr(seq_len,
        pword_seq, ppos_tag_seq);

    // generate window expr
    std::vector<std::vector<dynet::expr::Expression>> input_window_expr_list =
        window_expr_generate_layer->generate_window_expr_list(all_feature_concat_expr_list);
    // processing window expr
    std::vector<dynet::expr::Expression> input_exprs = window_expr_processing_layer->process(input_window_expr_list);

    std::vector<dynet::expr::Expression> output_exprs;
    if( mlp_hidden_layer ){ mlp_hidden_layer->build_graph(input_exprs, output_exprs); }
    else{ output_exprs = std::move(input_exprs); }
    return output_layer->build_output_loss(output_exprs, ner_tag_seq);
}

std::vector<Index>
NnMlp::predict_impl(const std::shared_ptr<std::vector<Index>>& pword_seq,
    const std::shared_ptr<std::vector<Index>>& ppos_tag_seq)
{
    new_graph();
    if( mlp_hidden_layer ){ mlp_hidden_layer->enable_dropout(); }

    unsigned seq_len = pword_seq->size();
    std::vector<dynet::expr::Expression> all_feature_concat_expr_list = concat_all_feature_as_expr(seq_len,
        pword_seq, ppos_tag_seq);

    // generate window expr
    std::vector<std::vector<dynet::expr::Expression>> input_window_expr_list =
        window_expr_generate_layer->generate_window_expr_list(all_feature_concat_expr_list);
    // processing window expr
    std::vector<dynet::expr::Expression> input_exprs = window_expr_processing_layer->process(input_window_expr_list);

    std::vector<dynet::expr::Expression> output_exprs;
    if( mlp_hidden_layer ){ mlp_hidden_layer->build_graph(input_exprs, output_exprs); }
    else{ output_exprs = std::move(input_exprs); }
    std::vector<Index> pred_tagseq;
    output_layer->build_output(output_exprs, pred_tagseq);
    return pred_tagseq;
}



} // end of namespace nn
} // end of namespace ner
} // end of namespace slnn