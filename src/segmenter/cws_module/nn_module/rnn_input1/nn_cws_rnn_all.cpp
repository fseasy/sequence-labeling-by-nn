#include "nn_cws_rnn_all.h"

namespace slnn{
namespace segmenter{
namespace nn_module{

NnSegmenterRnnAll::NnSegmenterRnnAll(int argc, char* argv[], unsigned seed)
    :NeuralNetworkCommonInterfaceDynetImpl(argc, argv, seed)
{}


void NnSegmenterRnnAll::new_graph()
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
    birnn_layer->new_graph(*get_cg());
    output_layer->new_graph(*get_cg()) ;
}

std::vector<dynet::expr::Expression> NnSegmenterRnnAll::concat_all_feature_as_expr(
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


dynet::expr::Expression NnSegmenterRnnAll::build_training_graph_impl(const std::shared_ptr<std::vector<Index>>& punigram_seq,
    const std::shared_ptr<std::vector<Index>>& pbigram_seq,
    const std::shared_ptr<std::vector<std::vector<Index>>>& plexicon_seq,
    const std::shared_ptr<std::vector<Index>>& ptype_seq,
    const std::shared_ptr<std::vector<Index>>& ptag_seq)
{
    new_graph();
    
    birnn_layer->set_dropout();
    birnn_layer->start_new_sequence();

    unsigned seq_len = ptag_seq->size();
    std::vector<dynet::expr::Expression> all_feature_concat_expr_list = concat_all_feature_as_expr(seq_len,
        punigram_seq, pbigram_seq, plexicon_seq, ptype_seq);

    // to bi-rnn
    std::vector<dynet::expr::Expression> l2r_output_exprs, r2l_output_exprs;
    birnn_layer->build_graph(all_feature_concat_expr_list, l2r_output_exprs, r2l_output_exprs);
    
    // concatenate & build loss
    std::vector<dynet::expr::Expression> concated_exprs(sent_len);
    for( unsigned i = 0; i < sent_len; ++i )
    { 
        concated_exprs[i] = dynet::expr::concatenate({ l2r_output_exprs[i], r2l_output_exprs[i] }); 
    }
    return output_layer->build_output_loss(concated_exprs, *ptag_seq);

}

std::vector<Index> NnSegmenterRnnAll::predict_impl(const std::shared_ptr<std::vector<Index>>& punigram_seq,
    const std::shared_ptr<std::vector<Index>>& pbigram_seq,
    const std::shared_ptr<std::vector<std::vector<Index>>>& plexicon_seq,
    const std::shared_ptr<std::vector<Index>>& ptype_seq)
{
    new_graph();
    birnn_layer->disable_dropout();
    birnn_layer->start_new_sequence();

    unsigned seq_len = 0;
    if( punigram_seq ){ seq_len = punigram_seq->size(); }
    else if( pbigram_seq ){ seq_len = pbigram_seq->size(); }
    else{ throw std::logic_error("at least one of {unigram, bigram} should be enable."); }
    std::vector<dynet::expr::Expression> all_feature_concat_expr_list = concat_all_feature_as_expr(seq_len,
        punigram_seq, pbigram_seq, plexicon_seq, ptype_seq);

    // to bi-rnn
    std::vector<dynet::expr::Expression> l2r_output_exprs, r2l_output_exprs;
    birnn_layer->build_graph(all_feature_concat_expr_list, l2r_output_exprs, r2l_output_exprs);
    
    // concatenate & build loss
    std::vector<dynet::expr::Expression> concated_exprs(sent_len);
    for( unsigned i = 0; i < sent_len; ++i )
    { 
        concated_exprs[i] = dynet::expr::concatenate({ l2r_output_exprs[i], r2l_output_exprs[i] }); 
    }
    std::vector<Index> pred_tagseq;
    output_layer->build_output(output_exprs, pred_tagseq);
    return pred_tagseq;
}


} // end of namespace nn-module
} // end of namespace segmenter
} // end of namespace slnn