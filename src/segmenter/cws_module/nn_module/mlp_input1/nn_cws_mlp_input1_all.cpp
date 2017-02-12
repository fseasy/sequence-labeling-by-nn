#include "nn_cws_mlp_input1_all.h"

namespace slnn{
namespace segmenter{
namespace nn_module{

NnSegmenterMlpInput1All::NnSegmenterMlpInput1All(int argc, char* argv[], unsigned seed)
    :NeuralNetworkCommonInterfaceDynetImpl(argc, argv, seed)
{}


void NnSegmenterMlpInput1All::new_graph()
{
    reset_cg();
    if( unigram_embed_layer ){ unigram_embed_layer->new_graph(*get_cg()); }
    if( bigram_embed_layer ){ bigram_embed_layer->new_graph(*get_cg()); }
    if( lexicon_embed_layer_group )
    { 
        for( unsigned i = 0; i < lexicon_embed_layer_group->size(); ++i )
        {
            (*lexicon_embed_layer_group)[i].new_graph(*get_cg()); 
        }
    }
    if( type_embed_layer ){ type_embed_layer->new_graph(*get_cg()); }
    window_expr_generate_layer->new_graph(*get_cg());
    window_expr_processing_layer->new_graph(*get_cg());
    if( mlp_hidden_layer ){ mlp_hidden_layer->new_graph(*get_cg()); }
    output_layer->new_graph(*get_cg());
}

std::vector<dynet::expr::Expression> NnSegmenterMlpInput1All::concat_all_feature_as_expr(
    unsigned seq_len,
    const std::shared_ptr<std::vector<Index>>& punigram_seq,
    const std::shared_ptr<std::vector<Index>>& pbigram_seq,
    const std::shared_ptr<std::vector<std::vector<Index>>>& plexicon_seq,
    const std::shared_ptr<std::vector<Index>>& ptype_seq)
{
    std::vector<std::vector<dynet::expr::Expression>> all_feature_list(seq_len);
    auto append_feature = [&all_feature_list, seq_len](const std::vector<dynet::expr::Expression>& feature_list)
    {
        for( unsigned i = 0; i < seq_len; ++i )
        {
            all_feature_list[i].push_back(feature_list[i]);
        }
    };
    if( unigram_embed_layer )
    {
        std::vector<dynet::expr::Expression> unigram_feature_list;
        unigram_embed_layer->index_seq2expr_seq(*punigram_seq, unigram_feature_list);
        append_feature(unigram_feature_list);
    }
    if( bigram_embed_layer )
    {
        std::vector<dynet::expr::Expression> bigram_feature_list;
        bigram_embed_layer->index_seq2expr_seq(*pbigram_seq, bigram_feature_list);
        append_feature(bigram_feature_list);
    }
    if( lexicon_embed_layer_group )
    {
        for( unsigned i = 0; i < lexicon_embed_layer_group->size(); ++i )
        {
            std::vector<dynet::expr::Expression> lexicon_feature_list;
            (*lexicon_embed_layer_group)[i].index_seq2expr_seq((*plexicon_seq)[i], lexicon_feature_list);
            append_feature(lexicon_feature_list);
        }
    }
    if( type_embed_layer )
    {
        std::vector<dynet::expr::Expression> type_feature_list;
        type_embed_layer->index_seq2expr_seq(*ptype_seq, type_feature_list);
        append_feature(type_feature_list);
    }
    std::vector<dynet::expr::Expression> all_feature_concat_list(seq_len);
    for( unsigned i = 0; i < seq_len; ++i )
    {
        all_feature_concat_list[i] = dynet::expr::concatenate(all_feature_list[i]);
    }
    return all_feature_concat_list;
}


dynet::expr::Expression NnSegmenterMlpInput1All::build_training_graph_impl(const std::shared_ptr<std::vector<Index>>& punigram_seq,
    const std::shared_ptr<std::vector<Index>>& pbigram_seq,
    const std::shared_ptr<std::vector<std::vector<Index>>>& plexicon_seq,
    const std::shared_ptr<std::vector<Index>>& ptype_seq,
    const std::shared_ptr<std::vector<Index>>& ptag_seq)
{
    new_graph();
    if( mlp_hidden_layer ){ mlp_hidden_layer->enable_dropout(); }

    unsigned seq_len = ptag_seq->size();
    std::vector<dynet::expr::Expression> all_feature_concat_expr_list = concat_all_feature_as_expr(seq_len,
        punigram_seq, pbigram_seq, plexicon_seq, ptype_seq);

    // generate window expr
    std::vector<std::vector<dynet::expr::Expression>> input_window_expr_list =
        window_expr_generate_layer->generate_window_expr_list(all_feature_concat_expr_list);
    // processing window expr
    std::vector<dynet::expr::Expression> input_exprs = window_expr_processing_layer->process(input_window_expr_list);

    std::vector<dynet::expr::Expression> output_exprs;
    if( mlp_hidden_layer ){ mlp_hidden_layer->build_graph(input_exprs, output_exprs); }
    else{ output_exprs = std::move(input_exprs); }
    return output_layer->build_output_loss(output_exprs, *ptag_seq);
}

std::vector<Index> NnSegmenterMlpInput1All::predict_impl(const std::shared_ptr<std::vector<Index>>& punigram_seq,
    const std::shared_ptr<std::vector<Index>>& pbigram_seq,
    const std::shared_ptr<std::vector<std::vector<Index>>>& plexicon_seq,
    const std::shared_ptr<std::vector<Index>>& ptype_seq)
{
    new_graph();
    if( mlp_hidden_layer ){ mlp_hidden_layer->enable_dropout(); }

    unsigned seq_len = 0;
    if( punigram_seq ){ seq_len = punigram_seq->size(); }
    else if( pbigram_seq ){ seq_len = pbigram_seq->size(); }
    else{ throw std::logic_error("at least one of {unigram, bigram} should be enable."); }
    std::vector<dynet::expr::Expression> all_feature_concat_expr_list = concat_all_feature_as_expr(seq_len,
        punigram_seq, pbigram_seq, plexicon_seq, ptype_seq);

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


} // end of namespace nn-module
} // end of namespace segmenter
} // end of namespace slnn