#include "nn_cws_rnn_input1_abstract.h"
namespace slnn{
namespace segmenter{
namespace nn_module{



NnSegmenterRnnInput1Abstract::NnSegmenterRnnInput1Abstract(int argc, char **argv, unsigned seed) : 
    NeuralNetworkCommonInterfaceDynetImpl(argc, argv, seed)
{}

dynet::expr::Expression 
NnSegmenterRnnInput1Abstract::build_training_graph_impl(const std::vector<Index> &charseq, 
    const std::vector<Index> &tagseq)
{
    //clear_cg(); // !! ATTENTION !! dynet's implementation is not successful. so abandon it.
    reset_cg();
    word_expr_layer->new_graph(*get_cg());
    birnn_layer->new_graph(*get_cg());
    output_layer->new_graph(*get_cg()) ;
    
    birnn_layer->set_dropout();
    birnn_layer->start_new_sequence();

    unsigned sent_len = charseq.size();

    std::vector<dynet::expr::Expression> word_exprs(sent_len);
    word_expr_layer->index_seq2expr_seq(charseq, word_exprs);
    // to bi-rnn
    std::vector<dynet::expr::Expression> l2r_output_exprs, r2l_output_exprs;
    birnn_layer->build_graph(word_exprs, l2r_output_exprs, r2l_output_exprs);
    // concatenate & build loss
    std::vector<dynet::expr::Expression> concated_exprs(sent_len);
    for( unsigned i = 0; i < sent_len; ++i )
    { 
        concated_exprs[i] = dynet::expr::concatenate({ l2r_output_exprs[i], r2l_output_exprs[i] }); 
    }
    return output_layer->build_output_loss(concated_exprs, tagseq);
}

std::vector<Index> 
NnSegmenterRnnInput1Abstract::predict_impl(const std::vector<Index> &charseq)
{
    //clear_cg(); // !!!!
    reset_cg();
    word_expr_layer->new_graph(*get_cg());
    birnn_layer->new_graph(*get_cg());
    output_layer->new_graph(*get_cg()) ;
    
    birnn_layer->disable_dropout();
    birnn_layer->start_new_sequence();

    unsigned sent_len = charseq.size();

    std::vector<dynet::expr::Expression> word_exprs(sent_len);
    word_expr_layer->index_seq2expr_seq(charseq, word_exprs);
    // to bi-rnn
    std::vector<dynet::expr::Expression> l2r_output_exprs, r2l_output_exprs;
    birnn_layer->build_graph(word_exprs, l2r_output_exprs, r2l_output_exprs);
    // concatenate & build loss
    std::vector<dynet::expr::Expression> concated_exprs(sent_len);
    for( unsigned i = 0; i < sent_len; ++i )
    { 
        concated_exprs[i] = dynet::expr::concatenate({ l2r_output_exprs[i], r2l_output_exprs[i] }); 
    }
    std::vector<Index> pred_tagseq;
    output_layer->build_output(concated_exprs, pred_tagseq);
    return pred_tagseq;
}

} // end of namespace nn_module
} // end of namespace segmenter
} // end of namespace slnn
